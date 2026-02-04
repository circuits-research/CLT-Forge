"""
Compressed Activations Store with multiple quantization and compression strategies.

Storage Comparison (per token with 12 layers, 768d):
- float16: ~37 KB/token → 37 TB for 1B tokens
- int8: ~18.5 KB/token → 18.5 TB for 1B tokens (2x compression)
- int4: ~9.25 KB/token → 9.25 TB for 1B tokens (4x compression)
- int8 + zstd: ~5-8 KB/token → 5-8 TB for 1B tokens (4-7x compression)
- int4 + zstd: ~3-5 KB/token → 3-5 TB for 1B tokens (7-12x compression)
"""

from __future__ import annotations
from typing import Literal, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import os
import struct

import torch
import numpy as np
from safetensors.torch import save_file, load_file
import zstandard as zstd

from clt import logger


@dataclass
class CompressionConfig:
    """Configuration for activation compression."""
    
    # Quantization settings
    quantization: Literal["none", "int8", "int4", "int2"] = "int8"
    
    # Compression settings
    compression: Literal["none", "zstd", "lz4"] = "zstd"
    compression_level: int = 3  # 1-22 for zstd, higher = better compression but slower
    
    # Per-layer or per-tensor quantization
    per_layer_scale: bool = True
    
    # Whether to use symmetric quantization (around 0) or asymmetric
    symmetric: bool = True
    
    def __post_init__(self):
        if self.compression == "zstd" and not (1 <= self.compression_level <= 22):
            raise ValueError("zstd compression_level must be between 1 and 22")


class QuantizedTensor:
    """A quantized tensor with scale/zero-point for dequantization."""
    
    def __init__(
        self,
        data: np.ndarray,  # quantized data (int8, int4, etc.)
        scale: np.ndarray,  # scale factor(s)
        zero_point: Optional[np.ndarray] = None,  # zero point for asymmetric quantization
        original_shape: tuple = None,
        original_dtype: torch.dtype = torch.float16,
        scale_shape: tuple = None,  # NEW: shape of scale for proper broadcasting
    ):
        self.data = data
        self.scale = scale
        self.zero_point = zero_point
        self.original_shape = original_shape or data.shape
        self.original_dtype = original_dtype
        self.scale_shape = scale_shape  # Store scale shape for proper broadcasting
    
    def dequantize(self) -> torch.Tensor:
        """Convert back to floating point."""
        # Convert to float
        if self.data.dtype == np.uint8:
            if hasattr(self, '_is_int4') and self._is_int4:
                data_float = self._unpack_int4(self.data).astype(np.float32)
            elif hasattr(self, '_is_int2') and self._is_int2:
                data_float = self._unpack_int2(self.data).astype(np.float32)
            else:
                data_float = self.data.astype(np.float32)
        else:
            data_float = self.data.astype(np.float32)
        
        # FIXED: Reshape to original shape FIRST, then apply scale
        data_reshaped = data_float.reshape(self.original_shape)
        
        # Reshape scale if needed for proper broadcasting
        scale = self.scale
        if self.scale_shape is not None:
            scale = scale.reshape(self.scale_shape)
        
        # Now multiply (broadcasting works correctly)
        if self.zero_point is not None:
            result = (data_reshaped - self.zero_point) * scale
        else:
            result = data_reshaped * scale
        
        # Convert to torch tensor
        tensor = torch.from_numpy(result)
        
        # Convert to original dtype
        return tensor.to(self.original_dtype)
    
    def _unpack_int4(self, packed: np.ndarray) -> np.ndarray:
        """Unpack int4 values stored in uint8."""
        # Each uint8 contains two int4 values
        low = (packed & 0x0F).astype(np.int8)
        high = ((packed >> 4) & 0x0F).astype(np.int8)

        # Reverse the +8 offset applied during packing
        low = low - 8
        high = high - 8

        # Interleave
        result = np.empty(packed.size * 2, dtype=np.int8)
        result[0::2] = low
        result[1::2] = high
        total = 1
        for s in self.original_shape:
            total *= s
        return result[:total]

    def _unpack_int2(self, packed: np.ndarray) -> np.ndarray:
        """Unpack int2 values stored in uint8."""
        # Each uint8 contains four int2 values
        v0 = ((packed >> 6) & 0x03).astype(np.int8)
        v1 = ((packed >> 4) & 0x03).astype(np.int8)
        v2 = ((packed >> 2) & 0x03).astype(np.int8)
        v3 = (packed & 0x03).astype(np.int8)

        # Reverse the +2 offset applied during packing
        v0 -= 2; v1 -= 2; v2 -= 2; v3 -= 2

        result = np.empty(packed.size * 4, dtype=np.int8)
        result[0::4] = v0
        result[1::4] = v1
        result[2::4] = v2
        result[3::4] = v3
        total = 1
        for s in self.original_shape:
            total *= s
        return result[:total]


class CompressedActivationsStore:
    """
    Enhanced activation storage with quantization and compression.
    
    Key features:
    - Multiple quantization levels (int8, int4, int2)
    - Optional compression (zstd, lz4)
    - Per-layer or per-tensor scale factors
    - Minimal accuracy loss with proper calibration
    """
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.compressor = None
        
        if config.compression == "zstd":
            self.compressor = zstd.ZstdCompressor(level=config.compression_level)
            self.decompressor = zstd.ZstdDecompressor()
        elif config.compression == "lz4":
            try:
                import lz4.frame
                self.compressor = lz4.frame
            except ImportError:
                logger.warning("lz4 not installed, falling back to no compression")
                self.config.compression = "none"
    
    def quantize_activations(
        self,
        act_in: torch.Tensor,
        act_out: torch.Tensor,
    ) -> Tuple[QuantizedTensor, QuantizedTensor]:
        """
        Quantize activation tensors.
        
        Args:
            act_in: Input activations [tokens, layers, d_model]
            act_out: Output activations [tokens, layers, d_model]
        
        Returns:
            Tuple of quantized input and output activations
        """
        return (
            self._quantize_tensor(act_in),
            self._quantize_tensor(act_out),
        )
    
    def _quantize_tensor(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize a single tensor."""
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        
        # Move to numpy for quantization
        data = tensor.cpu().numpy()
        
        if self.config.quantization == "none":
            scale = np.array([1.0], dtype=np.float32)
            return QuantizedTensor(
                data=data.astype(np.float16),
                scale=scale,
                original_shape=original_shape,
                original_dtype=original_dtype,
                scale_shape=scale.shape,
            )
        
        # Calculate scale factors
        if self.config.per_layer_scale:
            # Per-layer quantization (preserves layer-wise statistics)
            # Shape: [tokens, layers, d_model] -> scale per layer
            scale = self._compute_scale(data, axis=(0, 2), keepdims=True)
        else:
            # Global quantization (single scale for entire tensor)
            scale = self._compute_scale(data)
        
        # Quantize
        if self.config.quantization == "int8":
            q_data, zero_point = self._quantize_to_int8(data, scale)
        elif self.config.quantization == "int4":
            q_data, zero_point = self._quantize_to_int4(data, scale)
        elif self.config.quantization == "int2":
            q_data, zero_point = self._quantize_to_int2(data, scale)
        else:
            raise ValueError(f"Unknown quantization: {self.config.quantization}")
        
        qt = QuantizedTensor(
            data=q_data,
            scale=scale,
            zero_point=zero_point if not self.config.symmetric else None,
            original_shape=original_shape,
            original_dtype=original_dtype,
            scale_shape=scale.shape,
        )
        
        # FIX: Set quantization type flags
        if self.config.quantization == "int4":
            qt._is_int4 = True
        elif self.config.quantization == "int2":
            qt._is_int2 = True
        
        return qt
    
    def _compute_scale(
        self,
        data: np.ndarray,
        axis: Optional[tuple] = None,
        keepdims: bool = False,
    ) -> np.ndarray:
        """Compute quantization scale factor."""
        if self.config.symmetric:
            # Symmetric: scale based on max absolute value
            abs_max = np.abs(data).max(axis=axis, keepdims=keepdims)
            # Avoid division by zero
            abs_max = np.maximum(abs_max, 1e-8)
            
            # Determine quantization range
            if self.config.quantization == "int8":
                q_max = 127.0
            elif self.config.quantization == "int4":
                q_max = 7.0
            elif self.config.quantization == "int2":
                q_max = 1.0
            else:
                q_max = 127.0
            
            scale = abs_max / q_max
        else:
            # Asymmetric: scale based on min/max range
            data_min = data.min(axis=axis, keepdims=keepdims)
            data_max = data.max(axis=axis, keepdims=keepdims)
            
            if self.config.quantization == "int8":
                q_range = 255.0
            elif self.config.quantization == "int4":
                q_range = 15.0
            elif self.config.quantization == "int2":
                q_range = 3.0
            else:
                q_range = 255.0
            
            scale = (data_max - data_min) / q_range
            # Avoid division by zero
            scale = np.maximum(scale, 1e-8)
        
        return scale
    
    def _quantize_to_int8(
        self, data: np.ndarray, scale: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Quantize to int8."""
        if self.config.symmetric:
            # Map to [-127, 127]
            q_data = np.round(data / scale).astype(np.int8)
            q_data = np.clip(q_data, -127, 127)
            return q_data, None
        else:
            # Map to [0, 255]
            data_min = data.min(axis=(0, 2), keepdims=True) if self.config.per_layer_scale else data.min()
            zero_point = np.round(-data_min / scale).astype(np.uint8)
            q_data = np.round(data / scale + zero_point).astype(np.uint8)
            q_data = np.clip(q_data, 0, 255)
            return q_data, zero_point
    
    def _quantize_to_int4(
        self, data: np.ndarray, scale: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Quantize to int4 and pack into uint8."""
        if self.config.symmetric:
            # Map to [-7, 7]
            q_data = np.round(data / scale).astype(np.int8)
            q_data = np.clip(q_data, -7, 7)
            
            # Pack two int4 values into one uint8
            flat = q_data.flatten()
            # Pad if odd length
            if len(flat) % 2 != 0:
                flat = np.concatenate([flat, np.array([0], dtype=np.int8)])
            
            # Convert to unsigned for packing
            flat_unsigned = (flat + 8).astype(np.uint8)  # Shift to [0, 15]
            
            # Pack: low nibble = even indices, high nibble = odd indices
            packed = (flat_unsigned[1::2] << 4) | flat_unsigned[0::2]
            return packed, None
        else:
            # Asymmetric int4
            data_min = data.min(axis=(0, 2), keepdims=True) if self.config.per_layer_scale else data.min()
            zero_point = np.round(-data_min / scale).astype(np.uint8)
            q_data = np.round(data / scale + zero_point).astype(np.uint8)
            q_data = np.clip(q_data, 0, 15)
            
            # Pack similarly
            flat = q_data.flatten()
            if len(flat) % 2 != 0:
                flat = np.concatenate([flat, np.array([0], dtype=np.uint8)])
            
            packed = (flat[1::2] << 4) | flat[0::2]
            return packed, zero_point
    
    def _quantize_to_int2(
        self, data: np.ndarray, scale: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Quantize to int2 (extreme compression, higher error)."""
        # Map to [-1, 1] (or [0, 3] for asymmetric)
        if self.config.symmetric:
            q_data = np.round(data / scale).astype(np.int8)
            q_data = np.clip(q_data, -1, 1)
            
            # Pack 4 int2 values into one uint8
            flat = q_data.flatten()
            # Pad to multiple of 4
            remainder = len(flat) % 4
            if remainder != 0:
                flat = np.concatenate([flat, np.zeros(4 - remainder, dtype=np.int8)])
            
            # Convert to unsigned [0, 3]
            flat_unsigned = (flat + 2).astype(np.uint8)
            
            # Pack: 2 bits per value
            packed = (
                (flat_unsigned[0::4] << 6) |
                (flat_unsigned[1::4] << 4) |
                (flat_unsigned[2::4] << 2) |
                flat_unsigned[3::4]
            )
            return packed, None
        else:
            # Asymmetric not implemented for int2 (rarely needed)
            raise NotImplementedError("Asymmetric int2 quantization not implemented")
    
    def compress_bytes(self, data: bytes) -> bytes:
        """Apply compression to byte data."""
        if self.config.compression == "none":
            return data
        elif self.config.compression == "zstd":
            return self.compressor.compress(data)
        elif self.config.compression == "lz4":
            return self.compressor.compress(data)
        else:
            return data
    
    def decompress_bytes(self, data: bytes) -> bytes:
        """Decompress byte data."""
        if self.config.compression == "none":
            return data
        elif self.config.compression == "zstd":
            return self.decompressor.decompress(data)
        elif self.config.compression == "lz4":
            return self.compressor.decompress(data)
        else:
            return data
    
    def save_compressed(
        self,
        path: Path,
        act_in: torch.Tensor,
        act_out: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save activations with quantization and compression.
        
        Args:
            path: Path to save file
            act_in: Input activations
            act_out: Output activations
            tokens: Optional token IDs
            metadata: Optional metadata dictionary
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Quantize
        q_act_in, q_act_out = self.quantize_activations(act_in, act_out)
        
        # Prepare metadata
        meta = {
            "quantization": self.config.quantization,
            "compression": self.config.compression,
            "compression_level": self.config.compression_level,
            "per_layer_scale": self.config.per_layer_scale,
            "symmetric": self.config.symmetric,
            "original_shape_in": list(act_in.shape),
            "original_shape_out": list(act_out.shape),
            "original_dtype": str(act_in.dtype),
            "scale_shape_in": list(q_act_in.scale.shape),
            "scale_shape_out": list(q_act_out.scale.shape),
        }
        if metadata:
            meta.update(metadata)
        
        # Save using custom format (not safetensors for better compression)
        # Format: metadata_size (4 bytes) | metadata (json) | compressed_data
        import json
        
        data_dict = {
            "act_in_data": q_act_in.data.tobytes(),
            "act_in_scale": q_act_in.scale.flatten().astype(np.float32).tobytes(),
            "act_out_data": q_act_out.data.tobytes(),
            "act_out_scale": q_act_out.scale.flatten().astype(np.float32).tobytes(),
        }
        
        # Add zero points if asymmetric
        if q_act_in.zero_point is not None:
            data_dict["act_in_zero_point"] = q_act_in.zero_point.tobytes()
        if q_act_out.zero_point is not None:
            data_dict["act_out_zero_point"] = q_act_out.zero_point.tobytes()
        
        # Add tokens if provided
        if tokens is not None:
            data_dict["tokens"] = tokens.cpu().numpy().tobytes()
            meta["has_tokens"] = True
        else:
            meta["has_tokens"] = False
        
        # Serialize
        meta_bytes = json.dumps(meta).encode("utf-8")
        
        # Compress each data array separately for better compression
        compressed_data = {}
        for key, value in data_dict.items():
            compressed_data[key] = self.compress_bytes(value)
        
        # Write to file
        with open(path, "wb") as f:
            # Write metadata size
            f.write(struct.pack("I", len(meta_bytes)))
            # Write metadata
            f.write(meta_bytes)
            
            # Write number of data items
            f.write(struct.pack("I", len(compressed_data)))
            
            # Write each compressed item
            for key, value in compressed_data.items():
                key_bytes = key.encode("utf-8")
                f.write(struct.pack("I", len(key_bytes)))
                f.write(key_bytes)
                f.write(struct.pack("Q", len(value)))  # Use Q for large sizes
                f.write(value)
        
        # Log compression stats
        original_size = (
            act_in.element_size() * act_in.numel() +
            act_out.element_size() * act_out.numel()
        )
        if tokens is not None:
            original_size += tokens.element_size() * tokens.numel()
        
        compressed_size = path.stat().st_size
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        logger.info(
            f"Saved {path.name}: {original_size / 1024 / 1024:.2f} MB → "
            f"{compressed_size / 1024 / 1024:.2f} MB ({ratio:.2f}x compression)"
        )
        
        return compressed_size  # Return file size in bytes
    
    def load_compressed(
        self, path: Path
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
        """
        Load and dequantize activations.
        
        Returns:
            (act_in, act_out, tokens, metadata)
        """
        import json
        
        with open(path, "rb") as f:
            # Read metadata
            meta_size = struct.unpack("I", f.read(4))[0]
            meta_bytes = f.read(meta_size)
            metadata = json.loads(meta_bytes.decode("utf-8"))
            
            # Read compressed data
            num_items = struct.unpack("I", f.read(4))[0]

            # Use the file's own metadata to pick the right decompressor,
            # so loading works regardless of the config this instance was
            # constructed with.
            file_compression = metadata.get("compression", "none")
            if file_compression == "zstd":
                _decompress = zstd.ZstdDecompressor().decompress
            elif file_compression == "lz4":
                import lz4.frame
                _decompress = lz4.frame.decompress
            else:
                _decompress = lambda d: d

            data_dict = {}
            for _ in range(num_items):
                key_size = struct.unpack("I", f.read(4))[0]
                key = f.read(key_size).decode("utf-8")
                data_size = struct.unpack("Q", f.read(8))[0]
                compressed_data = f.read(data_size)
                data_dict[key] = _decompress(compressed_data)
        
        # Reconstruct quantized tensors
        original_shape_in = tuple(metadata["original_shape_in"])
        original_shape_out = tuple(metadata["original_shape_out"])
        scale_shape_in = tuple(metadata.get("scale_shape_in", [1]))
        scale_shape_out = tuple(metadata.get("scale_shape_out", [1]))
        
        # Determine quantized data shape
        if metadata["quantization"] == "int4":
            q_shape_in = (np.prod(original_shape_in) + 1) // 2  # Packed
            q_shape_out = (np.prod(original_shape_out) + 1) // 2
        elif metadata["quantization"] == "int2":
            q_shape_in = (np.prod(original_shape_in) + 3) // 4
            q_shape_out = (np.prod(original_shape_out) + 3) // 4
        else:
            q_shape_in = original_shape_in
            q_shape_out = original_shape_out
        
        # Load act_in
        if metadata["quantization"] == "none":
            # No quantization: data is stored as float16
            act_in_data = np.frombuffer(data_dict["act_in_data"], dtype=np.float16)
        elif metadata["quantization"] in ["int4", "int2"] or not metadata["symmetric"]:
            # Packed or asymmetric: use uint8
            act_in_data = np.frombuffer(data_dict["act_in_data"], dtype=np.uint8)
        else:
            # Symmetric int8: use int8
            act_in_data = np.frombuffer(data_dict["act_in_data"], dtype=np.int8)
            
        act_in_scale = np.frombuffer(data_dict["act_in_scale"], dtype=np.float32)
        act_in_zero_point = None
        if "act_in_zero_point" in data_dict:
            act_in_zero_point = np.frombuffer(data_dict["act_in_zero_point"], dtype=np.uint8)
        
        q_act_in = QuantizedTensor(
            data=act_in_data,
            scale=act_in_scale,
            zero_point=act_in_zero_point,
            original_shape=original_shape_in,
            original_dtype=getattr(torch, metadata["original_dtype"].split(".")[-1]),
            scale_shape=scale_shape_in,
        )
        if metadata["quantization"] == "int4":
            q_act_in._is_int4 = True
        elif metadata["quantization"] == "int2":
            q_act_in._is_int2 = True
        
        # Load act_out
        if metadata["quantization"] == "none":
            # No quantization: data is stored as float16
            act_out_data = np.frombuffer(data_dict["act_out_data"], dtype=np.float16)
        elif metadata["quantization"] in ["int4", "int2"] or not metadata["symmetric"]:
            # Packed or asymmetric: use uint8
            act_out_data = np.frombuffer(data_dict["act_out_data"], dtype=np.uint8)
        else:
            # Symmetric int8: use int8
            act_out_data = np.frombuffer(data_dict["act_out_data"], dtype=np.int8)
            
        act_out_scale = np.frombuffer(data_dict["act_out_scale"], dtype=np.float32)
        act_out_zero_point = None
        if "act_out_zero_point" in data_dict:
            act_out_zero_point = np.frombuffer(data_dict["act_out_zero_point"], dtype=np.uint8)
        
        q_act_out = QuantizedTensor(
            data=act_out_data,
            scale=act_out_scale,
            zero_point=act_out_zero_point,
            original_shape=original_shape_out,
            original_dtype=getattr(torch, metadata["original_dtype"].split(".")[-1]),
            scale_shape=scale_shape_out,
        )
        if metadata["quantization"] == "int4":
            q_act_out._is_int4 = True
        elif metadata["quantization"] == "int2":
            q_act_out._is_int2 = True
        
        # Dequantize
        act_in = q_act_in.dequantize()
        act_out = q_act_out.dequantize()
        
        # Load tokens if present
        tokens = None
        if metadata.get("has_tokens", False):
            tokens_data = np.frombuffer(data_dict["tokens"], dtype=np.int64).copy()  # Copy to make writable
            tokens = torch.from_numpy(tokens_data)
        
        return act_in, act_out, tokens, metadata


def estimate_compression_savings(
    num_tokens: int,
    num_layers: int,
    d_model: int,
    config: CompressionConfig,
) -> Dict[str, float]:
    """
    Estimate storage savings for different compression configurations.
    
    Args:
        num_tokens: Number of tokens to store
        num_layers: Number of transformer layers
        d_model: Model dimension
        config: Compression configuration
    
    Returns:
        Dictionary with size estimates in GB
    """
    # Calculate base size (float16, 2 bytes per value)
    # Factor of 2 for act_in and act_out
    base_size_bytes = num_tokens * num_layers * d_model * 2 * 2
    base_size_gb = base_size_bytes / (1024 ** 3)
    
    # Quantization reduction
    if config.quantization == "int8":
        quant_factor = 0.5  # 1 byte vs 2 bytes
    elif config.quantization == "int4":
        quant_factor = 0.25  # 0.5 bytes vs 2 bytes
    elif config.quantization == "int2":
        quant_factor = 0.125  # 0.25 bytes vs 2 bytes
    else:
        quant_factor = 1.0
    
    # Add overhead for scale factors (negligible for large datasets)
    if config.per_layer_scale:
        scale_overhead_bytes = num_layers * 4 * 2  # float32 per layer, in + out
    else:
        scale_overhead_bytes = 4 * 2  # float32 global
    
    quantized_size_bytes = base_size_bytes * quant_factor + scale_overhead_bytes
    
    # Compression reduction (empirical estimates)
    if config.compression == "zstd":
        # zstd typically achieves 1.5-2.5x on quantized activations
        if config.quantization == "int8":
            compression_factor = 0.5 if config.compression_level >= 10 else 0.6
        elif config.quantization == "int4":
            compression_factor = 0.6 if config.compression_level >= 10 else 0.7
        else:
            compression_factor = 0.7
    elif config.compression == "lz4":
        # lz4 is faster but less effective
        compression_factor = 0.7 if config.quantization != "none" else 0.85
    else:
        compression_factor = 1.0
    
    final_size_bytes = quantized_size_bytes * compression_factor
    final_size_gb = final_size_bytes / (1024 ** 3)
    
    return {
        "base_size_gb": base_size_gb,
        "quantized_size_gb": quantized_size_bytes / (1024 ** 3),
        "final_size_gb": final_size_gb,
        "compression_ratio": base_size_gb / final_size_gb if final_size_gb > 0 else 0,
        "savings_gb": base_size_gb - final_size_gb,
        "savings_percent": ((base_size_gb - final_size_gb) / base_size_gb * 100) if base_size_gb > 0 else 0,
    }

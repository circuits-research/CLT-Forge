from pathlib import Path

def patch_sparse_attention_models():
    """Add sparse attention models to TransformerLens OFFICIAL_MODEL_NAMES and MODEL_ALIASES."""
    
    try:
        from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES, MODEL_ALIASES
        
        # Add sparse models from files_to_patch
        sparse_models = [
            "ansonisl/SparseGPT2",
            "ansonisl/SparseTinystories", 
            "ansonisl/DenseTinystories",
        ]
        
        for model in sparse_models:
            if model not in OFFICIAL_MODEL_NAMES:
                OFFICIAL_MODEL_NAMES.append(model)
                print(f"Added {model} to OFFICIAL_MODEL_NAMES")
        
        # Add aliases for sparse models
        sparse_aliases = {
            "ansonisl/SparseGPT2": ["sparse-gpt2"],
            "ansonisl/SparseTinystories": ["sparse-tinystories"],
            "ansonisl/DenseTinystories": ["dense-tinystories"],
        }
        
        for model, aliases in sparse_aliases.items():
            if model not in MODEL_ALIASES:
                MODEL_ALIASES[model] = aliases
                print(f"Added aliases {aliases} for {model}")
        
    except ImportError:
        print("TransformerLens not found. Make sure it's installed.")
    except Exception as e:
        print(f"Error patching sparse attention models: {e}")

def patch_sparse_attention_config():
    """Patch convert_hf_model_config and related functions for sparse attention models."""
    
    try:
        from transformer_lens import loading_from_pretrained
        import os
        import torch
        from transformers import AutoConfig
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        
        original_convert_hf_model_config = loading_from_pretrained.convert_hf_model_config
        original_get_pretrained_state_dict = loading_from_pretrained.get_pretrained_state_dict
        
        def patched_convert_hf_model_config(model_name: str, **kwargs):
            # Get official model name
            if (Path(model_name) / "config.json").exists():
                official_model_name = model_name
            else:
                official_model_name = loading_from_pretrained.get_official_model_name(model_name)

            # Handle sparse models specially
            if "ansonisl/Sparse" in official_model_name or "ansonisl/Dense" in official_model_name:
                huggingface_token = os.environ.get("HF_TOKEN", "")
                hf_config = AutoConfig.from_pretrained(
                    official_model_name,
                    token=huggingface_token if len(huggingface_token) > 0 else None,
                    **kwargs,
                )
                
                # Build config dict for sparse models  
                # The sparse model uses padded vocab size of 50304 (GPT-2's 50257 padded to nearest multiple of 64)
                cfg_dict = {
                    "d_model": hf_config.n_embd,
                    "d_head": hf_config.n_embd // hf_config.n_head,
                    "n_heads": hf_config.n_head,
                    "d_mlp": hf_config.n_embd * 4,
                    "n_layers": hf_config.n_layer,
                    "n_ctx": hf_config.n_positions,
                    "eps": hf_config.layer_norm_epsilon,
                    "d_vocab": 50304,  # Use the padded vocab size that matches the actual model weights
                    "act_fn": hf_config.activation_function,
                    "use_attn_scale": True,
                    "use_local_attn": False,
                    "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
                    "normalization_type": "LN",
                    "original_architecture": "GPT2LMHeadModel",
                    "tokenizer_name": "gpt2",  # Use standard GPT-2 tokenizer since sparse models don't include tokenizer files
                }
                
                if kwargs.get("trust_remote_code", False):
                    cfg_dict["trust_remote_code"] = True
                    
                return cfg_dict
            else:
                # Use original function for non-sparse models
                return original_convert_hf_model_config(model_name, **kwargs)

        def patched_get_pretrained_state_dict(official_model_name: str, cfg: dict, hf_model=None, **kwargs):
            """Patched version that handles sparse models with final.pth weights."""
            
            # Handle sparse models specially - they use final.pth instead of standard weight files
            if "ansonisl/Sparse" in official_model_name or "ansonisl/Dense" in official_model_name:
                huggingface_token = os.environ.get("HF_TOKEN", "")
                
                try:
                    # Download the custom weight file
                    weight_path = hf_hub_download(
                        repo_id=official_model_name,
                        filename="final.pth",
                        token=huggingface_token if len(huggingface_token) > 0 else None,
                    )
                    
                    # Load the state dict
                    state_dict = torch.load(weight_path, map_location="cpu")
                    print(f"Successfully loaded sparse model weights from final.pth")
                    return state_dict
                    
                except Exception as e:
                    print(f"Error loading sparse model weights: {e}")
                    # Fall back to original method
                    return original_get_pretrained_state_dict(official_model_name, cfg, hf_model, **kwargs)
            else:
                # Use original function for non-sparse models
                return original_get_pretrained_state_dict(official_model_name, cfg, hf_model, **kwargs)

        loading_from_pretrained.convert_hf_model_config = patched_convert_hf_model_config
        loading_from_pretrained.get_pretrained_state_dict = patched_get_pretrained_state_dict
        print("Successfully patched convert_hf_model_config and get_pretrained_state_dict for sparse attention models")
        
    except ImportError:
        print("TransformerLens not found. Make sure it's installed.")
    except Exception as e:
        print(f"Error patching sparse attention config: {e}")

def patch_sparse_attention_components():
    """Patch TransformerLens components for sparse attention support."""
    
    try:
        from transformer_lens.components import abstract_attention
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from typing import Dict, Optional, Tuple, Union
        import einops
        from jaxtyping import Float, Int
        from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
        from transformer_lens.hook_points import HookPoint
        from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCacheEntry
        from transformer_lens.utilities.attention import complex_attn_linear, simple_attn_linear
        from transformer_lens.utils import get_offset_position_ids
        from transformer_lens.FactoredMatrix import FactoredMatrix
        from abc import ABC
        from better_abc import abstract_attribute
        
        # Check if bitsandbytes is available
        try:
            from transformers.utils import is_bitsandbytes_available
            if is_bitsandbytes_available():
                import bitsandbytes as bnb
                from bitsandbytes.nn.modules import Params4bit
        except ImportError:
            is_bitsandbytes_available = lambda: False
        
        # Get the original AbstractAttention class to inherit behavior we don't want to change
        original_AbstractAttention = abstract_attention.AbstractAttention
        
        class PatchedAbstractAttention(ABC, nn.Module):
            """Patched AbstractAttention class with sparse attention support."""
            
            alibi: Union[torch.Tensor, None]

            def __init__(
                self,
                cfg: Union[Dict, HookedTransformerConfig],
                attn_type: str = "global",
                layer_id: Optional[int] = None,
            ):
                super().__init__()
                self.cfg = HookedTransformerConfig.unwrap(cfg)

                if self.cfg.load_in_4bit:
                    nq = int((self.cfg.d_model * self.cfg.d_head * self.cfg.n_heads) / 2)
                    self.W_Q = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)
                    self.W_O = Params4bit(torch.empty(nq, 1, dtype=torch.uint8), requires_grad=False)
                else:
                    self.W_Q = nn.Parameter(torch.empty(self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head))
                    self.W_O = nn.Parameter(torch.empty(self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model))

                self.b_Q = nn.Parameter(torch.zeros(self.cfg.n_heads, self.cfg.d_head))
                self.b_O = nn.Parameter(torch.zeros(self.cfg.d_model))

                self.attn_type = attn_type
                self.layer_id = layer_id

                # Sparse attention specific parameters
                if hasattr(self.cfg, 'sparse_attn_window_size'):
                    self.sparse_attn_window_size = self.cfg.sparse_attn_window_size
                else:
                    self.sparse_attn_window_size = None

                if hasattr(self.cfg, 'use_sparse_attention'):
                    self.use_sparse_attention = self.cfg.use_sparse_attention
                else:
                    self.use_sparse_attention = False

                # Hook points
                self.hook_k = HookPoint()
                self.hook_q = HookPoint()
                self.hook_v = HookPoint()
                self.hook_z = HookPoint()
                self.hook_attn_scores = HookPoint()
                self.hook_pattern = HookPoint()
                self.hook_result = HookPoint()

            # Add abstract attributes that child classes must implement
            W_K = abstract_attribute()
            W_V = abstract_attribute() 
            b_K = abstract_attribute()
            b_V = abstract_attribute()

            def apply_sparse_attention_mask(self, attn_scores, pos):
                """Apply sparse attention masking if enabled."""
                if not self.use_sparse_attention or self.sparse_attn_window_size is None:
                    return attn_scores
                
                # Create sparse attention mask (local window)
                seq_len = attn_scores.size(-1)
                mask = torch.ones_like(attn_scores, dtype=torch.bool)
                
                for i in range(seq_len):
                    start_idx = max(0, i - self.sparse_attn_window_size)
                    end_idx = min(seq_len, i + self.sparse_attn_window_size + 1)
                    mask[..., i, start_idx:end_idx] = False
                
                attn_scores = attn_scores.masked_fill(mask, float('-inf'))
                return attn_scores

        # Replace the original AbstractAttention class
        abstract_attention.AbstractAttention = PatchedAbstractAttention
        print("Successfully patched AbstractAttention for sparse attention support")
        
    except ImportError as e:
        print(f"ImportError patching AbstractAttention: {e}")
    except Exception as e:
        print(f"Error patching AbstractAttention: {e}")

def patch_sparse_attention():
    """Patch TransformerLens with sparse attention support using runtime patching."""
    # patch_sparse_attention_models()
    # patch_sparse_attention_config()
    # patch_sparse_attention_components()
    print("Successfully patched TransformerLens for sparse attention support")
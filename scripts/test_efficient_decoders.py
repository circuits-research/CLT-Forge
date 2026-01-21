"""
Test script for efficient decoder parameterizations (LoRA and linear_transform).
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

import torch
# Import only CLTConfig directly to avoid wandb dependency
from clt.config.clt_config import CLTConfig
from clt.clt import CLT


def test_decoder_type(decoder_type: str, n_layers: int = 4, d_in: int = 64, d_latent: int = 256, decoder_rank: int = 16):
    """Test a specific decoder type."""
    print(f"\n{'='*60}")
    print(f"Testing decoder_type='{decoder_type}'")
    print(f"  n_layers={n_layers}, d_in={d_in}, d_latent={d_latent}, decoder_rank={decoder_rank}")
    print(f"{'='*60}")

    cfg = CLTConfig(
        device="cpu",
        dtype="float32",
        seed=42,
        model_name="test",
        d_in=d_in,
        d_latent=d_latent,
        n_layers=n_layers,
        jumprelu_bandwidth=0.001,
        jumprelu_init_threshold=0.001,
        normalize_decoder=False,
        dead_feature_window=250,
        cross_layer_decoders=True,
        context_size=16,
        l0_coefficient=1e-3,
        decoder_type=decoder_type,
        decoder_rank=decoder_rank,
    )

    clt = CLT(cfg)

    # Print parameter counts
    param_counts = clt.get_decoder_param_count()
    print(f"\nParameter counts:")
    for k, v in param_counts.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Test forward pass
    batch_size = 8
    acts_in = torch.randn(batch_size, n_layers, d_in)
    acts_out = torch.randn(batch_size, n_layers, d_in)

    print(f"\nTesting forward pass...")
    z, hidden_pre = clt.encode(acts_in)
    print(f"  Encoded shape: {z.shape}")

    recon = clt.decode(z)
    print(f"  Decoded shape: {recon.shape}")

    # Test loss computation
    print(f"\nTesting loss computation...")
    loss, metrics = clt.forward(acts_in, acts_out, l0_coef=1e-3, df_coef=0)
    print(f"  Total loss: {loss.item():.6f}")
    print(f"  MSE loss: {metrics.mse_loss.item():.6f}")
    print(f"  L0 loss: {metrics.l0_loss.item():.6f}")

    # Test gradients flow correctly
    print(f"\nTesting gradient flow...")
    loss.backward()

    if decoder_type == "full":
        print(f"  W_dec grad norm: {clt.W_dec.grad.norm().item():.6f}")
    elif decoder_type == "lora":
        print(f"  W_dec_base grad norm: {clt.W_dec_base.grad.norm().item():.6f}")
        print(f"  lora_A grad norm: {clt.lora_A.grad.norm().item():.6f}")
        print(f"  lora_B grad norm: {clt.lora_B.grad.norm().item():.6f}")
    elif decoder_type == "linear_transform":
        print(f"  W_dec_base grad norm: {clt.W_dec_base.grad.norm().item():.6f}")
        print(f"  transform_A_left grad norm: {clt.transform_A_left.grad.norm().item():.6f}")
        print(f"  transform_B_left grad norm: {clt.transform_B_left.grad.norm().item():.6f}")

    print(f"\n[PASS] decoder_type='{decoder_type}' works correctly!")
    return clt, param_counts


def compare_parameter_efficiency():
    """Compare parameter counts across decoder types."""
    print("\n" + "="*60)
    print("PARAMETER EFFICIENCY COMPARISON")
    print("="*60)

    n_layers = 12  # GPT-2 style
    d_in = 768
    d_latent = d_in * 16  # 16x expansion
    decoder_rank = 64

    results = {}
    for decoder_type in ["full", "lora", "linear_transform"]:
        cfg = CLTConfig(
            device="cpu",
            dtype="float32",
            seed=42,
            model_name="test",
            d_in=d_in,
            d_latent=d_latent,
            n_layers=n_layers,
            jumprelu_bandwidth=0.001,
            jumprelu_init_threshold=0.001,
            normalize_decoder=False,
            dead_feature_window=250,
            cross_layer_decoders=True,
            context_size=16,
            l0_coefficient=1e-3,
            decoder_type=decoder_type,
            decoder_rank=decoder_rank,
        )
        clt = CLT(cfg)
        results[decoder_type] = clt.get_decoder_param_count()

    print(f"\nConfiguration: n_layers={n_layers}, d_in={d_in}, d_latent={d_latent}, rank={decoder_rank}")
    print(f"N_dec (full cross-layer pairs): {n_layers * (n_layers + 1) // 2}")

    print(f"\nDecoder weight parameters:")
    print(f"  full:             {results['full']['W_dec']:,}")
    print(f"  lora:             {results['lora']['W_dec_base'] + results['lora']['lora_A'] + results['lora']['lora_B']:,}")
    print(f"    - W_dec_base:   {results['lora']['W_dec_base']:,}")
    print(f"    - lora_A:       {results['lora']['lora_A']:,}")
    print(f"    - lora_B:       {results['lora']['lora_B']:,}")
    print(f"  linear_transform: {results['linear_transform']['W_dec_base'] + results['linear_transform']['transform_A_left'] + results['linear_transform']['transform_B_left']:,}")
    print(f"    - W_dec_base:   {results['linear_transform']['W_dec_base']:,}")
    print(f"    - transform_A:  {results['linear_transform']['transform_A_left']:,}")
    print(f"    - transform_B:  {results['linear_transform']['transform_B_left']:,}")

    print(f"\nCompression ratios (vs full):")
    print(f"  lora:             {results['lora']['compression_ratio']:.2f}x")
    print(f"  linear_transform: {results['linear_transform']['compression_ratio']:.2f}x")


if __name__ == "__main__":
    # Test each decoder type
    for decoder_type in ["full", "lora", "linear_transform"]:
        test_decoder_type(decoder_type)

    # Compare parameter efficiency at scale
    compare_parameter_efficiency()

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)

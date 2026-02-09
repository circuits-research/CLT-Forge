import os
import sys
from pathlib import Path
import time
import random
import numpy as np
import torch
import torch.distributed as dist
import gc

from clt.config.clt_config import CLTConfig
from clt.clt import CLT

project_root = Path(__file__).parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))


def set_seed(seed, rank):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed + rank)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def cleanup_memory():
    """Clean up GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def test_batch_size(args, model, batch_size, device, mode="ddp"):
    cleanup_memory()
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    success = True
    x = act_out = loss = None

    try:
        if mode == "ddp":
            local_bs = batch_size // world_size
            if local_bs < 1:
                success = False
            else:
                x = torch.randn(local_bs, args.n_layers, args.d_in, device=device)
                act_out = torch.randn_like(x)
                loss = model(x, act_out, l0_coef=args.l0_coefficient, df_coef=args.df_coefficient, return_metrics=False)
        else:
            x = torch.randn(batch_size, args.n_layers, args.d_in, device=device)
            act_out = torch.randn_like(x)
            loss = model(x, act_out, l0_coef=args.l0_coefficient, df_coef=args.df_coefficient, return_metrics=False)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            success = False
        else:
            raise

    # all ranks must agree whether to proceed to backward
    if dist.is_initialized():
        flag = torch.tensor([1 if success else 0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        success = (flag.item() == 1)

    if not success:
        # IMPORTANT: nobody runs backward
        if mode == "ddp":
            model.zero_grad(set_to_none=True)
        for var in ['x', 'act_out', 'loss']:
            if var in locals():
                del locals()[var]
        cleanup_memory()
        return False
    
    # now safe: everybody runs backward
    try:
        model.zero_grad(set_to_none=True)  # good hygiene for repeated tests
        loss.backward()
        torch.cuda.synchronize()
        return True
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            model.zero_grad(set_to_none=True)
            return False
        else:
            raise
    finally:
        for var in ['x', 'act_out', 'loss']:
            if var in locals():
                del locals()[var]
        cleanup_memory()

def binary_search_max_batch_size(args, model, device, mode="ddp", min_batch=1, max_batch=2048):
    """
    Binary search to find maximum batch size that fits in memory.
    Both DDP and Feature Sharding require all ranks to participate:
    - DDP: All ranks process different data splits (requires sync for gradient averaging)
    - Feature Sharding: All ranks process same data but with sharded weights (requires sync for distributed ops)
    For DDP, result is multiplied by world_size since each GPU handles a portion of the batch.
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    left, right = min_batch, max_batch
    max_working = 0
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Binary searching max batch size for {mode.upper()}. Search range: [{left}, {right}]")
        print(f"{'='*60}\n")
    
    while left <= right:
        mid = (left + right) // 2
        
        if rank == 0:
            print(f"Testing batch_size={mid}", end=" ", flush=True)
        
        success = test_batch_size(args, model, mid, device, mode)

        if success:
            max_working = mid
            if rank == 0:
                print(f"✓ SUCCESS")
            left = mid + 1
        else:
            if rank == 0:
                print(f"✗ OOM")
            right = mid - 1

    return max_working


def test_max_batch_size_comparison(args):
    """
    Compare maximum batch size between DDP and feature sharding.
    
    Args:
        seed: Random seed
        args: Arguments with model configuration
    """

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    set_seed(args.seed, rank)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"GPU Memory Test: DDP vs Feature Sharding")
        print(f"{'='*60}")
        print(f"World size: {world_size}")
        print(f"Model: {args.model_name} (d_in={args.d_in}, d_latent={args.d_latent})")
        print(f"{'='*60}\n")
    
    # Test Feature Sharding first
    if rank == 0:
        print("\n" + "="*60)
        print("TESTING FEATURE SHARDING MODE")
        print("="*60)
    
    cfg_fs = CLTConfig(
        device=device,
        dtype=args.dtype,
        seed=args.seed + rank,
        model_name=args.model_name,
        n_layers=args.n_layers,
        d_in=args.d_in,
        d_latent=args.d_latent,
        jumprelu_init_threshold=args.jumprelu_init_threshold,
        jumprelu_bandwidth=args.jumprelu_bandwidth,
        normalize_decoder=args.normalize_decoder,
        dead_feature_window=args.dead_feature_window,
        cross_layer_decoders=args.cross_layer_decoders,
        context_size=args.context_size,
        l0_coefficient=args.l0_coefficient,
        debug=False,
        ddp=False,
        feature_sharding=True,
    )
    
    clt_fs = CLT(cfg_fs, rank=rank, world_size=world_size) # here, whether parameters are synced is not relevant
    clt_fs.eval()
    
    # if rank == 0:
    #     print(f"Feature sharding W_enc shape per rank: {clt_fs.W_enc.shape}")
    #     print(f"Feature sharding W_dec shape per rank: {clt_fs.W_dec.shape}")
    #     print(f"Local d_latent per rank: {clt_fs.local_d_latent}")
    
    max_batch_fs = binary_search_max_batch_size(
        args, clt_fs, device, mode="feature_sharding", 
        min_batch=1, max_batch=args.max_batch
    )
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Feature Sharding Max Batch Size: {max_batch_fs}")
        print(f"{'='*60}\n")

    del clt_fs
    cleanup_memory()
    dist.barrier()
    
    # Test DDP mode
    if rank == 0:
        print("\n" + "="*60)
        print("TESTING DDP MODE")
        print("="*60)
    
    cfg_ddp = CLTConfig(
        device=device,
        dtype=args.dtype,
        seed=args.seed + rank,
        model_name=args.model_name,
        n_layers=args.n_layers,
        d_in=args.d_in,
        d_latent=args.d_latent,
        jumprelu_init_threshold=args.jumprelu_init_threshold,
        jumprelu_bandwidth=args.jumprelu_bandwidth,
        normalize_decoder=args.normalize_decoder,
        dead_feature_window=args.dead_feature_window,
        cross_layer_decoders=args.cross_layer_decoders,
        context_size=args.context_size,
        l0_coefficient=args.l0_coefficient,
        debug=False,
        ddp=True,
        feature_sharding=False,
    )
    
    clt_ddp = CLT(cfg_ddp, rank=rank, world_size=world_size)
    clt_ddp.eval()
    
    clt_ddp = torch.nn.parallel.DistributedDataParallel(
        clt_ddp,
        device_ids=[rank],
        output_device=rank,
        broadcast_buffers=False,
        static_graph=True
    )
    
    # if rank == 0:
    #     print(f"DDP W_enc shape (full): {clt_ddp.module.W_enc.shape}")
    #     print(f"DDP W_dec shape (full): {clt_ddp.module.W_dec.shape}")
    #     print(f"Total d_latent: {clt_ddp.module.cfg.d_latent}")
    
    max_batch_ddp = binary_search_max_batch_size(
        args, clt_ddp, device, mode="ddp",
        min_batch=1, max_batch=args.max_batch
    )
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DDP Max Batch Size (total): {max_batch_ddp}")
        print(f"DDP Max Batch Size per GPU: {max_batch_ddp // world_size}")
        print(f"{'='*60}\n")
    
    del clt_ddp
    cleanup_memory()
    dist.barrier()
    
    # Final comparison
    if rank == 0:
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        print(f"Number of GPUs: {world_size}")
        print(f"\nFeature Sharding:")
        print(f"  - Max batch size (total): {max_batch_fs}")
        print(f"  - Batch per GPU: {max_batch_fs} (all GPUs process same data)")
        print(f"  - Effective throughput: {max_batch_fs} samples/step")
        print(f"\nDDP:")
        print(f"  - Max batch size (total): {max_batch_ddp}")
        print(f"  - Batch per GPU: {max_batch_ddp // world_size}")
        print(f"  - Effective throughput: {max_batch_ddp} samples/step")
        print(f"\nMemory Efficiency:")
        ratio = max_batch_ddp / max_batch_fs if max_batch_fs > 0 else 0
        print(f"  DDP / Feature Sharding max-batch-size ratio: {ratio:.2f}x")
        
        print(f"\nNotes:")
        print(f"  - Feature Sharding splits model parameters across GPUs")
        print(f"    but processes the SAME data on all GPUs")
        print(f"  - DDP replicates full model on each GPU but SPLITS")
        print(f"    the batch across GPUs")
        print(f"  - Feature Sharding saves parameter memory but duplicates")
        print(f"    activation memory")
        print(f"  - DDP duplicates parameter memory but saves activation memory")
        print(f"{'='*60}\n")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    # Test configuration
    parser.add_argument("--max_batch", type=int, default=32768, help="Maximum batch size to test. 32768 is proper for A100-80G")
    # Model configuration
    # controlling n_layers can manifest difference between DDP and FS. Larger n_layers leads to poorer DDP compared with FS.
    parser.add_argument("--n_layers", type=int, default=24, help="Number of layers. As the model becomes larger, the DDP will need more time to clean the state. If it didn't make it before the next forward pass, an error will be raised. With static_graph=False, n_layers can be 12 but not 24; with True, n_layers can be 24 but not 36.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name")
    parser.add_argument("--context_size", type=int, default=16, help="Context size for the buffer")
    parser.add_argument("--d_in", type=int, default=768, help="Input dimension")
    parser.add_argument("--d_latent", type=int, default=768*32, help="Latent dimension")
    parser.add_argument("--jumprelu_init_threshold", type=float, default=0.03, help="JumpReLU init threshold")
    parser.add_argument("--jumprelu_bandwidth", type=float, default=1.0, help="JumpReLU bandwidth")
    parser.add_argument("--normalize_decoder", action="store_true", help="Normalize decoder")
    parser.add_argument("--dead_feature_window", type=int, default=250, help="Dead feature window")
    parser.add_argument("--cross_layer_decoders", action="store_true", help="Use cross layer decoders")
    parser.add_argument("--l0_coefficient", type=float, default=1.15, help="L0 coefficient")
    parser.add_argument("--df_coefficient", type=float, default=1e-4, help="Dead feature coefficient")
    args = parser.parse_args()
    
    test_max_batch_size_comparison(args)

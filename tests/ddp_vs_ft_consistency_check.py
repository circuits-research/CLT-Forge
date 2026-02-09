import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
import torch.distributed as dist

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

def test_ddp_vs_feature_sharding(args):
    
    """
    Compare DDP vs feature sharding on toy data.
    We compare the dimensionality of parameters, forward outputs, losses, and gradients.
    feature sharding shards the dim across ranks, while DDP keeps full params on each rank.
    feature sharding processes the same data on each rank, while DDP splits the data into mini-batches across ranks.
    """
    
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    set_seed(args.seed, rank)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    
    if rank == 0:
        print(f"Feature Sharding vs DDP Comparison")
    
    # Feature sharding: each rank has local_d_latent features
    cfg_fs = CLTConfig(
        device=device,
        dtype=args.dtype,
        seed=args.seed+rank,
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
        ddp=False,
        feature_sharding=True,
        debug=False,
        functional_loss=args.functional_loss
    )
    
    clt_fs = CLT(cfg_fs, rank=rank, world_size=world_size)
    with torch.no_grad():
        dist.broadcast(clt_fs.b_dec, src=0)
    
    clt_fs.eval()
    
    if rank == 0:
        print(f"Feature sharding W_enc shape (per rank): {clt_fs.W_enc.shape}")
        print(f"Feature sharding W_dec shape (per rank): {clt_fs.W_dec.shape}\n")
    
    dist.barrier()
    
    local_d_latent = clt_fs.local_d_latent
    
    # Gather parameters from all ranks to rank 0
    W_enc_list = [torch.zeros(args.n_layers, args.d_in, local_d_latent, device=device) for _ in range(world_size)]
    dist.all_gather(W_enc_list, clt_fs.W_enc)
    
    W_dec_list = [torch.zeros(args.n_layers, local_d_latent, args.d_in, device=device) for _ in range(world_size)]
    dist.all_gather(W_dec_list, clt_fs.W_dec)
    
    b_enc_list = [torch.zeros(args.n_layers, local_d_latent, device=device) for _ in range(world_size)]
    dist.all_gather(b_enc_list, clt_fs.b_enc)
    
    log_threshold_list = [torch.zeros(args.n_layers, local_d_latent, device=device) for _ in range(world_size)]
    dist.all_gather(log_threshold_list, clt_fs.log_threshold)
    
    dist.barrier()
    
    if rank == 0:
        # Verify concatenated shapes
        W_enc_full = torch.cat(W_enc_list, dim=-1)
        W_dec_full = torch.cat(W_dec_list, dim=1)
        b_enc_full = torch.cat(b_enc_list, dim=-1)
        log_threshold_full = torch.cat(log_threshold_list, dim=-1)
        
        expected_d_latent = local_d_latent * world_size
        print(f"Expected d_latent after gather: {expected_d_latent}")
        print(f"W_enc_full shape: {W_enc_full.shape} (expected [{args.n_layers}, {args.d_in}, {expected_d_latent}])")
        print(f"W_dec_full shape: {W_dec_full.shape} (expected [{args.n_layers}, {expected_d_latent}, {args.d_in}])")
        print(f"b_enc_full shape: {b_enc_full.shape} (expected [{args.n_layers}, {expected_d_latent}])\n")
        
    # Create DDP model with full parameters. DDP should be initialized and conduct wrapping on all devices.
    cfg_ddp = CLTConfig(
        device=device,
        dtype=args.dtype,
        seed=args.seed+rank,
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
        ddp=True,
        feature_sharding=False,
        debug=False,
        functional_loss=args.functional_loss
    )
    
    clt = CLT(cfg_ddp, rank=rank, world_size=world_size)
    clt.eval()
    
    clt_ddp = torch.nn.parallel.DistributedDataParallel(
            clt,
            device_ids=[rank],
            output_device=rank,
            broadcast_buffers=False
        )
    
    print(f"DDP W_enc shape (full): {clt_ddp.module.W_enc.shape}")
    print(f"DDP W_dec shape (full): {clt_ddp.module.W_dec.shape}\n")
    
    # Copy feature sharding params into DDP
    W_enc_full = torch.cat(W_enc_list, dim=-1)
    W_dec_full = torch.cat(W_dec_list, dim=1)
    b_enc_full = torch.cat(b_enc_list, dim=-1)
    log_threshold_full = torch.cat(log_threshold_list, dim=-1)
    
    with torch.no_grad():
        clt_ddp.module.W_enc.copy_(W_enc_full)
        clt_ddp.module.W_dec.copy_(W_dec_full)
        clt_ddp.module.b_enc.copy_(b_enc_full)
        clt_ddp.module.log_threshold.copy_(log_threshold_full)
    
    # Generate full batch data on rank 0 and broadcast to all ranks
    if rank == 0:
        print("Generating shared data batch")
        x_full = torch.randn(args.batch_size, args.n_layers, args.d_in, device=device)
        act_out_full = torch.randn_like(x_full)
    else:
        x_full = torch.zeros(args.batch_size, args.n_layers, args.d_in, device=device)
        act_out_full = torch.zeros_like(x_full)
    
    dist.barrier()
    dist.broadcast(x_full, src=0)
    dist.broadcast(act_out_full, src=0)
    dist.barrier()
    
    # DDP forward - each rank processes its portion of the batch
    if rank == 0:
        print("DDP Forward Pass")
    
    # Split data across ranks for DDP
    assert args.batch_size % world_size == 0, "Batch size must be divisible by world size for DDP."
    local_batch_size = args.batch_size // world_size
    start_idx = rank * local_batch_size
    end_idx = start_idx + local_batch_size
    x_ddp = x_full[start_idx:end_idx]
    act_out_ddp = act_out_full[start_idx:end_idx]
    
    _, metrics_ddp = clt_ddp(x_ddp, act_out_ddp, l0_coef=args.l0_coefficient, df_coef=args.df_coefficient)
    
    # different gpus should have different losses before all-reduce
    print(f"rank {rank} DDP MSE loss before all-reduce: {metrics_ddp.mse_loss.item():.6f}\n")
    print(f"rank {rank} DDP L0 loss before all-reduce: {metrics_ddp.l0_loss.item():.6f}\n")
    print(f"rank {rank} DDP DF loss before all-reduce: {metrics_ddp.dead_feature_loss.item():.6f}\n")
    
    dist.barrier()
    
    # Feature sharding forward - all ranks process the full batch
    if rank == 0:
        print("Feature Sharding Forward Pass")
    
    metrics_fs = clt_fs.loss(x_full, act_out_full, l0_coef=args.l0_coefficient, df_coef=args.df_coefficient)
    
    if rank == 0:
        print(f"Rank 0 MSE loss: {metrics_fs.mse_loss.item():.6f}")
        print(f"Rank 0 L0 loss: {metrics_fs.l0_loss.item():.6f}")
        print(f"Rank 0 DF loss: {metrics_fs.dead_feature_loss.item():.6f}\n")
        
        print("Forward Comparison")
        print(f"MSE. ddp: {metrics_ddp.mse_loss.item():.6f}, fs: {metrics_fs.mse_loss.item():.6f}, match: {torch.allclose(metrics_ddp.mse_loss, metrics_fs.mse_loss, atol=1e-4)}")
        print(f"L0. ddp: {metrics_ddp.l0_loss.item():.6f},  fs: {metrics_fs.l0_loss.item():.6f}, match: {torch.allclose(metrics_ddp.l0_loss, metrics_fs.l0_loss, atol=1e-4)}")
        print(f"DF. ddp: {metrics_ddp.dead_feature_loss.item():.6f},  fs: {metrics_fs.dead_feature_loss.item():.6f}, match: {torch.allclose(metrics_ddp.dead_feature_loss, metrics_fs.dead_feature_loss, atol=1e-4)}\n")
    
    dist.barrier()
    
    # DDP backward pass - ALL ranks must call backward for DDP to work correctly
    if rank == 0:
        print("DDP Backward Pass")
    
    # DDP requires all ranks to call backward (it synchronizes gradients automatically)
    loss_ddp = metrics_ddp.mse_loss + metrics_ddp.l0_loss + metrics_ddp.dead_feature_loss
    loss_ddp.backward()
    
    dist.barrier()
    
    # Create optimizer for DDP model
    optimizer_ddp = torch.optim.Adam(clt_ddp.parameters(), lr=1e-3)

    optimizer_ddp.step()
    
    if rank == 0:
        print("DDP Backward Pass Complete - Checking Gradient Synchronization")

    # Gather W_enc gradients from all ranks
    W_enc_grad_list_ddp = [torch.zeros_like(clt_ddp.module.W_enc.grad) for _ in range(world_size)]
    dist.all_gather(W_enc_grad_list_ddp, clt_ddp.module.W_enc.grad)

    dist.barrier()

    if rank == 0:
        # Check if all ranks have identical gradients
        all_same = all(torch.allclose(W_enc_grad_list_ddp[0], W_enc_grad_list_ddp[i], atol=1e-4) 
                    for i in range(1, world_size))
        
        if all_same:
            print("✓ DDP gradients are synchronized across all ranks")
        else:
            print("✗ DDP gradients differ across ranks")
        
        # Print sample values
        print(f"\nDDP gradient sample from rank 0: {W_enc_grad_list_ddp[0][0, 0, :5]}")
        print(f"DDP gradient sample from rank 1: {W_enc_grad_list_ddp[1][0, 0, :5]}")
        print(f"\nDDP Rank 0 W_enc.grad norm: {clt_ddp.module.W_enc.grad.norm().item():.6f}")
        print(f"DDP Rank 1 W_enc.grad norm: {W_enc_grad_list_ddp[1].norm().item():.6f}")
        
    dist.barrier()
    
    # Feature sharding backward pass
    if rank == 0:
        print("\n=== Feature Sharding Backward Pass ===")
    
    # Create optimizer for feature sharding model
    optimizer_fs = torch.optim.Adam(clt_fs.parameters(), lr=1e-3)
    
    loss_fs = metrics_fs.mse_loss + metrics_fs.l0_loss + metrics_fs.dead_feature_loss
    loss_fs.backward()
    
    if rank == 0:
        fs_w_enc_grad_norm = clt_fs.W_enc.grad.norm().item() if clt_fs.W_enc.grad is not None else 0.0
        fs_w_dec_grad_norm = clt_fs.W_dec.grad.norm().item() if clt_fs.W_dec.grad is not None else 0.0
        
        print(f"Feature Sharding Rank 0 W_enc.grad norm: {fs_w_enc_grad_norm:.6f}")
        print(f"Feature Sharding Rank 0 W_dec.grad norm: {fs_w_dec_grad_norm:.6f}")

    dist.barrier()
    
    # Optimizer step for feature sharding
    optimizer_fs.step()
    
    if rank == 0:
        print("Feature sharding optimizer step complete\n")
    
    dist.barrier()
    
    # Gather gradients from all ranks for comparison
    if rank == 0:
        print("=== Gradient Comparison ===")
    
    W_enc_grad_list = [torch.zeros(args.n_layers, args.d_in, local_d_latent, device=device) for _ in range(world_size)]
    W_dec_grad_list = [torch.zeros(args.n_layers, local_d_latent, args.d_in, device=device) for _ in range(world_size)]
    
    if clt_fs.W_enc.grad is not None:
        dist.all_gather(W_enc_grad_list, clt_fs.W_enc.grad)
    if clt_fs.W_dec.grad is not None:
        dist.all_gather(W_dec_grad_list, clt_fs.W_dec.grad)
    
    dist.barrier()
    
    if rank == 0:
        if all(g is not None for g in W_enc_grad_list):
            W_enc_grad_full = torch.cat(W_enc_grad_list, dim=-1)
            print(f"DDP W_enc.grad sample: {clt_ddp.module.W_enc.grad[0,0,:5]}")
            print(f"FS W_enc.grad sample: {W_enc_grad_full[0,0,:5]}")
            
            # Check if they match (they should if everything is correct)
            enc_match = torch.allclose(clt_ddp.module.W_enc.grad, W_enc_grad_full, atol=1e-3)
            print(f"W_enc.grad match: {enc_match}")
            
        if all(g is not None for g in W_dec_grad_list):
            W_dec_grad_full = torch.cat(W_dec_grad_list, dim=1)
            dec_match = torch.allclose(clt_ddp.module.W_dec.grad, W_dec_grad_full, atol=1e-3)
            print(f"W_dec.grad match: {dec_match}")
            
    # check if updated parameters are close
    dist.barrier()
    if rank == 0:
        print("\n=== Parameter Update Comparison ===")
    
    W_enc_list_after = [torch.zeros(args.n_layers, args.d_in, local_d_latent, device=device) for _ in range(world_size)]
    dist.all_gather(W_enc_list_after, clt_fs.W_enc)
    
    if rank == 0:
        W_enc_full_after = torch.cat(W_enc_list_after, dim=-1)
        enc_param_match = torch.allclose(clt_ddp.module.W_enc, W_enc_full_after, atol=1e-3)
        print(f"W_enc parameter match after update: {enc_param_match}")
        print('Test completed.\n')
        
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Test configuration
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for comparison test")
    # Model configuration
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--dtype", type=str, default="float32", help="Data type")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name")
    parser.add_argument("--context_size", type=int, default=16, help="Context size for the buffer")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of layers")
    parser.add_argument("--d_in", type=int, default=768, help="Input dimension")
    parser.add_argument("--d_latent", type=int, default=768*32, help="Latent dimension")
    parser.add_argument("--jumprelu_init_threshold", type=float, default=0.03, help="JumpReLU init threshold")
    parser.add_argument("--jumprelu_bandwidth", type=float, default=1.0, help="JumpReLU bandwidth")
    parser.add_argument("--normalize_decoder", action="store_true", help="Normalize decoder")
    parser.add_argument("--dead_feature_window", type=int, default=250, help="Dead feature window")
    parser.add_argument("--cross_layer_decoders", action="store_true", help="Use cross layer decoders")
    parser.add_argument("--l0_coefficient", type=float, default=1.15, help="L0 coefficient")
    parser.add_argument("--df_coefficient", type=float, default=1e-4, help="Dead feature coefficient")
    parser.add_argument("--functional_loss", type=str, default="kl", help="Functional loss type")
    args = parser.parse_args()
    
    test_ddp_vs_feature_sharding(args)
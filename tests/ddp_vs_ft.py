import os
import torch
import torch.distributed as dist
import sys
sys.path.insert(0, "/scratch/abir19/CLT")

from clt.config.clt_config import CLTConfig
from clt.clt import CLT


def test_ddp_vs_feature_sharding():
    """Compare DDP vs feature sharding on same random data."""
    
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    
    # Generate data on rank 0, broadcast to all
    batch_size = 4
    if rank == 0:
        x = torch.randn(batch_size, 12, 768, device=device)
        act_out = torch.randn_like(x)
    else:
        x = torch.zeros(batch_size, 12, 768, device=device)
        act_out = torch.zeros_like(x)
    
    dist.barrier()
    dist.broadcast(x, src=0)
    dist.broadcast(act_out, src=0)
    
    if rank == 0:
        print(f"Feature Sharding vs DDP Comparison")
    
    # Feature sharding: each rank has local_d_latent features
    cfg_fs = CLTConfig(
        device=device,
        dtype="float32",
        seed=42,
        model_name="gpt2",
        n_layers=12,
        d_in=768,
        d_latent=768*32, 
        jumprelu_init_threshold=0.03,
        jumprelu_bandwidth=1.0,
        normalize_decoder=False,
        dead_feature_window=250,
        cross_layer_decoders=False,
        context_size=16,
        l0_coefficient=1.15,
        ddp=False,
        functional_loss="kl"
    )
    
    clt_fs = CLT(cfg_fs, rank=rank, world_size=world_size)
    clt_fs.train() 
    
    if rank == 0:
        print(f"Feature sharding W_enc shape (per rank): {clt_fs.W_enc.shape}")
        print(f"Feature sharding W_dec shape (per rank): {clt_fs.W_dec.shape}\n")
    
    dist.barrier()
    
    local_d_latent = clt_fs.local_d_latent
    
    # Gather parameters from all ranks to rank 0
    W_enc_list = [torch.zeros(12, 768, local_d_latent, device=device) for _ in range(world_size)]
    dist.all_gather(W_enc_list, clt_fs.W_enc)
    
    W_dec_list = [torch.zeros(12, local_d_latent, 768, device=device) for _ in range(world_size)]
    dist.all_gather(W_dec_list, clt_fs.W_dec)
    
    b_enc_list = [torch.zeros(12, local_d_latent, device=device) for _ in range(world_size)]
    dist.all_gather(b_enc_list, clt_fs.b_enc)
    
    log_threshold_list = [torch.zeros(12, local_d_latent, device=device) for _ in range(world_size)]
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
        print(f"W_enc_full shape: {W_enc_full.shape} (expected [12, 768, {expected_d_latent}])")
        print(f"W_dec_full shape: {W_dec_full.shape} (expected [12, {expected_d_latent}, 768])")
        print(f"b_enc_full shape: {b_enc_full.shape} (expected [12, {expected_d_latent}])\n")
        
        # Create DDP model on rank 0 with full parameters
        cfg_ddp = CLTConfig(
            device=device,
            dtype="float32",
            seed=42,
            model_name="gpt2",
            n_layers=12,
            d_in=768,
            d_latent=768*32, 
            jumprelu_init_threshold=0.03,
            jumprelu_bandwidth=1.0,
            normalize_decoder=False,
            dead_feature_window=250,
            cross_layer_decoders=False,
            context_size=16,
            l0_coefficient=1.15,
            ddp=True,
            functional_loss="kl"
        )
        
        clt_ddp = CLT(cfg_ddp, rank=0, world_size=1)
        clt_ddp.train()  
        
        print(f"DDP W_enc shape (full): {clt_ddp.W_enc.shape}")
        print(f"DDP W_dec shape (full): {clt_ddp.W_dec.shape}\n")
        
        # Copy feature sharding params into DDP
        W_enc_full = torch.cat(W_enc_list, dim=-1)  # [12, 768, 256]
        W_dec_full = torch.cat(W_dec_list, dim=1)   # [12, 256, 768]
        b_enc_full = torch.cat(b_enc_list, dim=-1)  # [12, 256]
        log_threshold_full = torch.cat(log_threshold_list, dim=-1)  # [12, 256]
        
        with torch.no_grad():
            clt_ddp.W_enc.copy_(W_enc_full)
            clt_ddp.W_dec.copy_(W_dec_full)
            clt_ddp.b_enc.copy_(b_enc_full)
            clt_ddp.log_threshold.copy_(log_threshold_full)
        
        # DDP forward
        print("DDP Forward Pass")
        feat_ddp, hidden_ddp = clt_ddp.encode(x)
        recon_ddp = clt_ddp.decode(feat_ddp)
        metrics_ddp = clt_ddp.loss(x, act_out, l0_coef=1.15, df_coef=1e-4)
        
        print(f"DDP MSE loss: {metrics_ddp.mse_loss.item():.6f}")
        print(f"DDP L0 loss: {metrics_ddp.l0_loss.item():.6f}")
        print(f"DDP DF loss: {metrics_ddp.dead_feature_loss.item():.6f}\n")
    
    dist.barrier()
    
    # Feature sharding forward
    if rank == 0:
        print("Feature Sharding Forward Pass")
    
    feat_fs, hidden_fs = clt_fs.encode(x)
    recon_fs = clt_fs.decode(feat_fs)
    metrics_fs = clt_fs.loss(x, act_out, l0_coef=1.15, df_coef=1e-4)
    
    if rank == 0:
        print(f"Rank 0 MSE loss: {metrics_fs.mse_loss.item():.6f}")
        print(f"Rank 0 L0 loss: {metrics_fs.l0_loss.item():.6f}")
        print(f"Rank 0 DF loss: {metrics_fs.dead_feature_loss.item():.6f}\n")
        
        print("Forward Comparison")
        print(f"MSE match: {torch.allclose(metrics_ddp.mse_loss, metrics_fs.mse_loss, atol=1e-4)}")
        print(f"L0 match: {torch.allclose(metrics_ddp.l0_loss, metrics_fs.l0_loss, atol=1e-4)}")
        print(f"DF match: {torch.allclose(metrics_ddp.dead_feature_loss, metrics_fs.dead_feature_loss, atol=1e-4)}\n")
    
    dist.barrier()
    
    # Backward pass
    if rank == 0:
        print("DDP Backward Pass")
        loss_ddp = metrics_ddp.mse_loss + metrics_ddp.l0_loss + metrics_ddp.dead_feature_loss
        loss_ddp.backward()
        
        ddp_w_enc_grad_norm = clt_ddp.W_enc.grad.norm().item() if clt_ddp.W_enc.grad is not None else 0.0
        ddp_w_dec_grad_norm = clt_ddp.W_dec.grad.norm().item() if clt_ddp.W_dec.grad is not None else 0.0
        
        print(f"DDP W_enc.grad norm: {ddp_w_enc_grad_norm:.6f}")
        print(f"DDP W_dec.grad norm: {ddp_w_dec_grad_norm:.6f}\n")
    
    dist.barrier()
    
    if rank == 0:
        print("Feature Sharding Backward Pass")
    
    loss_fs = metrics_fs.mse_loss + metrics_fs.l0_loss + metrics_fs.dead_feature_loss
    loss_fs.backward()
    
    if rank == 0:
        fs_w_enc_grad_norm = clt_fs.W_enc.grad.norm().item() if clt_fs.W_enc.grad is not None else 0.0
        fs_w_dec_grad_norm = clt_fs.W_dec.grad.norm().item() if clt_fs.W_dec.grad is not None else 0.0
        
        print(f"Rank 0 W_enc.grad norm: {fs_w_enc_grad_norm:.6f}")
        print(f"Rank 0 W_dec.grad norm: {fs_w_dec_grad_norm:.6f}\n")
        
        print("Backward Comparison")
        print(f"W_enc.grad exists: {clt_fs.W_enc.grad is not None}")
        print(f"W_dec.grad exists: {clt_fs.W_dec.grad is not None}\n")
    
    dist.barrier()
    
    # Gather gradients from all ranks
    W_enc_grad_list = [torch.zeros(12, 768, local_d_latent, device=device) for _ in range(world_size)]
    W_dec_grad_list = [torch.zeros(12, local_d_latent, 768, device=device) for _ in range(world_size)]
    
    if clt_fs.W_enc.grad is not None:
        dist.all_gather(W_enc_grad_list, clt_fs.W_enc.grad)
    if clt_fs.W_dec.grad is not None:
        dist.all_gather(W_dec_grad_list, clt_fs.W_dec.grad)
    
    dist.barrier()
    
    if rank == 0:
        if all(g is not None for g in W_enc_grad_list):
            W_enc_grad_full = torch.cat(W_enc_grad_list, dim=-1)
            enc_match = torch.allclose(clt_ddp.W_enc.grad, W_enc_grad_full, atol=1e-3)
            print(f"W_enc.grad match: {enc_match}")
        
        if all(g is not None for g in W_dec_grad_list):
            W_dec_grad_full = torch.cat(W_dec_grad_list, dim=1)
            dec_match = torch.allclose(clt_ddp.W_dec.grad, W_dec_grad_full, atol=1e-3)
            print(f"W_dec.grad match: {dec_match}")
        
        print("Test complete!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    test_ddp_vs_feature_sharding()
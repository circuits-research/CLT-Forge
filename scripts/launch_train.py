import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))
data_root = project_root / "data"
if str(data_root) not in sys.path:
    sys.path.insert(0, str(data_root))

import torch
import torch.distributed as dist
from clt.config.clt_training_runner_config import CLTTrainingRunnerConfig
from clt.clt_training_runner import CLTTrainingRunner
import wandb

def main():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ: # catch interactive jobs and torchrun on local servers
        print(f"Detected distributed launch via torchrun (Local Rank: {os.environ.get('LOCAL_RANK')})")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ: # catch sbatch jobs
        print("Detected launch via srun (SLURM)")
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ["SLURM_LOCALID"])
    else: # catch single-process launch
        print("Running in single-process mode")
        rank = 0
        world_size = 1
        local_rank = 0
    print(f"[Rank {rank}] Initialized on GPU {local_rank} (World Size: {world_size})")
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    
    # set up environment variables for this process
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    total_training_steps = 75_000
    train_batch_size_tokens = 4096
    total_training_tokens = train_batch_size_tokens * total_training_steps
    lr_decay_steps = (total_training_steps // 20) - 1
    final_lr_scale = 0.0
    lr_warm_up_steps = 10
    l0_waiting_steps = 0
    l0_warm_up_steps = int(0.2 * total_training_steps) - l0_waiting_steps - 1
    decay_stable_steps = total_training_steps - l0_warm_up_steps - lr_decay_steps

    functional_loss = "kl"
    fc_coefficient = 0
    fc_warm_up_steps = 1000
    fc_waiting_steps = total_training_steps - fc_warm_up_steps - 1

    if rank == 0:
        wandb_id = wandb.util.generate_id()
    else:
        wandb_id = None 

    wandb_id_list = [wandb_id]
    dist.broadcast_object_list(wandb_id_list, src=0)
    wandb_id = wandb_id_list[0]

    # Get absolute path to data directory

    cfg = CLTTrainingRunnerConfig(
        # device=f"cuda:{local_rank}",
        device="cuda", 
        dtype="float32",
        seed=42,
        n_checkpoints=5,
        checkpoint_path="checkpoints/gpt2",
        logger_verbose=True,
        model_class_name="HookedTransformer",
        model_name="gpt2",
        context_size=16,
        from_pretrained_path=None,
        d_in=768,
        expansion_factor=32,
        jumprelu_init_threshold=0.03,
        jumprelu_bandwidth=1.,
        cached_activations_path=str(data_root / "activations_gpt2_multilingual_20"),
        n_train_batch_per_buffer=36,
        total_training_tokens=total_training_tokens,
        train_batch_size_tokens=train_batch_size_tokens,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr=2e-4,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        final_lr_scale=final_lr_scale,
        l0_coefficient=4.0,
        dead_penalty_coef=1e-4,
        dead_feature_window=250,
        l0_warm_up_steps=l0_warm_up_steps,
        l0_waiting_steps=l0_waiting_steps,
        decay_stable_steps=decay_stable_steps,
        log_to_wandb=True,
        wandb_project="gpt2-clt2",
        wandb_id=wandb_id,
        wandb_log_frequency=10,
        eval_every_n_wandb_logs=100,
        run_name=None,
        wandb_entity=None,
        fsdp=False,
        ddp=True,
        functional_loss=functional_loss,
        fc_coefficient=fc_coefficient,
        fc_warm_up_steps=fc_warm_up_steps,
        fc_waiting_steps=fc_waiting_steps
    )

    trainer = CLTTrainingRunner(cfg, rank=rank, world_size=world_size)
    trainer.run()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
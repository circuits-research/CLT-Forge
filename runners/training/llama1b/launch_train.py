from circuitlab.clt_training_runner import CLTTrainingRunner
from circuitlab import logger
from pathlib import Path
import sys
import os
import torch
import torch.distributed as dist

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from runners.training.llama1b.config import clt_training_runner_config

def main():
    local_rank = int(os.environ["LOCAL_RANK"])

    # distributed computing
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # local_rank = dist.get_node_local_rank()
    logger.info(f"Initialized on GPU {local_rank} (World Size: {world_size})")

    cfg = clt_training_runner_config(rank=rank, world_size=world_size)
    trainer = CLTTrainingRunner(cfg, rank=rank, world_size=world_size)
    trainer.run()

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
    

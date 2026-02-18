
from clt.clt_training_runner import CLTTrainingRunner
from runners.gpt2.config import clt_training_runner_config
from clt import logger

import os
import torch
import torch.distributed as dist

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

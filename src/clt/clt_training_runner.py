import wandb
from typing import Any, cast, Union
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from transformers import AutoConfig

from clt.transformer_lens.multilingual_patching import patch_official_model_names, patch_convert_hf_model_config
from clt.config import CLTTrainingRunnerConfig, CLTConfig
from clt.utils import DTYPE_MAP, DummyModel
from clt.clt import CLT
from clt.load_model import load_model
from clt.training.activations_store import ActivationsStore
from clt.training.clt_trainer import CLTTrainer
import os

_missing = object()

class CLTTrainingRunner:
    """
    * Initialize the model, the clt, the activations_store 
    * Run the training
    * Save checkpoints
    """
    cfg: CLTTrainingRunnerConfig
    dtype: torch.dtype
    device: torch.device

    def __init__(self, cfg: CLTTrainingRunnerConfig, rank: Union[int, object] = _missing, world_size: Union[int, object] = _missing):
        self.cfg = cfg
        self.dtype = DTYPE_MAP[cfg.dtype]
        self.ddp = cfg.ddp
        self.fsdp = cfg.fsdp
        torch.manual_seed(cfg.seed)
        if self.ddp or self.fsdp:
            if rank is _missing or world_size is _missing:
                raise ValueError("DDP is enabled but 'rank' and/or 'world_size' were not provided.")
            if not dist.is_initialized():
                raise RuntimeError("Distributed training requested but process group not initialized.")
            self.rank = cast(int, rank)
            self.world_size = cast(int, world_size)
            self.cfg.device = f"cuda:{self.rank}"
        else:
            # Feature sharding or single GPU
            if rank is not _missing and world_size is not _missing:
                # Feature sharding mode
                if not dist.is_initialized():
                    raise RuntimeError("Feature sharding requested but process group not initialized.")
                self.rank = cast(int, rank)
                self.world_size = cast(int, world_size)
                self.cfg.device = f"cuda:{self.rank}"
            else:
                # Single GPU mode
                self.rank = 0
                self.world_size = 1

        self.is_main_process = True if self.rank == 0 else False
        self.device = torch.device(self.cfg.device)

        # For multlingual models added to transformer-lens
        if "CausalNLP" in self.cfg.model_name or "meta-llama" in self.cfg.model_name: 
            print("Adding names to Transformer Lens")
            patch_official_model_names()
            patch_convert_hf_model_config()

        if self.ddp or self.fsdp:
            self.cfg.train_batch_size_tokens = cfg.train_batch_size_tokens // self.world_size
            self.cfg.total_training_tokens = cfg.total_training_tokens // self.world_size

        # no need to load the model if the activations are saved, just the number of layers
        if self.cfg.cached_activations_path is not None:

            if "sparse" in self.cfg.model_name:
                n_layers = 12  # sparse GPT-2 has 12 layers, TODO: to remove
            else:
                model_cfg = AutoConfig.from_pretrained(self.cfg.model_name)
                n_layers = (
                    getattr(model_cfg, "n_layer", None)
                    or getattr(model_cfg, "num_hidden_layers", None)
                    or getattr(model_cfg, "num_layers", None)
                )
                if n_layers is None:
                    raise ValueError(
                        f"Could not infer number of layers for model '{self.cfg.model_name}'."
                    )

            self.model = DummyModel(
                cfg=SimpleNamespace(
                    n_layers=n_layers,
                    use_hook_mlp_in=True,
                )
            )

        else:  
            self.model = load_model(
                self.cfg.model_class_name,
                self.cfg.model_name,
                device=self.device, 
                model_from_pretrained_kwargs=self.cfg.model_from_pretrained_kwargs,
            )

        self.activations_store = ActivationsStore(
            self.model,
            self.cfg, 
            rank=self.rank, 
            world_size=self.world_size
        )

        if self.cfg.from_pretrained_path is not None:
            self.clt = CLT.load_from_pretrained(
                self.cfg.from_pretrained_path, self.cfg.device
            )
            self.clt = self.clt.to(self.device) # could be on device from previous run
        else: 
            self.clt = CLT(
                cfg.create_sub_config(
                    CLTConfig,
                    n_layers=self.model.cfg.n_layers
                ),
                rank=self.rank,
                world_size=self.world_size
            )

        if self.ddp:
            self.clt = torch.nn.parallel.DistributedDataParallel(
                self.clt,
                device_ids=[self.rank],
                output_device=self.rank,
            )

        elif self.fsdp: 
            self.clt.to(self.device)
            # cpu_offload = CPUOffload(offload_params=True)
            self.clt = FSDP(
                self.clt.to(self.device),
                device_id=self.device,
            )
        else:
            #feature sharding or single GPU - no wrapper needed
            pass
        self.update_clt_norm_scaling_factor()

    def run(self): 
        """
        Run the training of the CLT
        """

        if self.cfg.log_to_wandb and self.is_main_process:
            wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                config=cast(Any, self.cfg),
                name=self.cfg.run_name,
                id=self.cfg.wandb_id,
            )
  
        if self.is_main_process:
            print(f"Norm scaling in: {self.clt.estimated_norm_scaling_factor_in if hasattr(self.clt, 'estimated_norm_scaling_factor_in') else 'N/A'}")
            print(f"l0_coefficient: {self.cfg.l0_coefficient}")
            print(f"l0_warm_up_steps: {self.cfg.l0_warm_up_steps}")
            print(f"lr: {self.cfg.lr}")
            print(f"dead_penalty_coef: {self.cfg.dead_penalty_coef}")

        trainer = CLTTrainer(
            clt=self.clt,
            activations_store=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg,
            rank=self.rank, 
            world_size=self.world_size
        )

        clt = trainer.fit()

        if self.cfg.log_to_wandb and self.is_main_process:
            wandb.finish()

        return clt
    
    def save_checkpoint(self, trainer: CLTTrainer, checkpoint_name: str) -> None: 
        """
        Saves the model state. Uses centralized saving for DDP/FSDP (if Rank 0 has memory) 
        and switches to OOM-safe decentralized saving for Feature Sharding/Single GPU.
        """


        base_path = Path(trainer.cfg.checkpoint_path) / checkpoint_name
        
        # 1. Handle DDP/FSDP
        if self.fsdp or self.ddp:
            if self.is_main_process:
                base_path.mkdir(exist_ok=True, parents=True)
                
            dist.barrier() # Ensure directory is created before FSDP/DDP proceed

            if self.fsdp:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import StateDictType, FullStateDictConfig

                # Wrap FSDP for state_dict
                with FSDP.state_dict_type(
                    trainer.clt,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True),
                ):
                    if self.is_main_process:
                        trainer.clt.module.save_model(str(base_path))
            elif self.ddp:
                # Just unwrap and save
                if self.is_main_process:
                    trainer.clt.module.save_model(str(base_path))
            else:
                # Fallback for single GPU if somehow DDP/FSDP is False
                if self.is_main_process:
                    trainer.clt.save_model(str(base_path))
            
        # 2. Handle Feature Sharding / Single GPU (OOM-Safe Decentralized Save)
        else:
            if self.rank == 0:
                base_path.mkdir(exist_ok=True, parents=True)
                print(f"Saving sharded checkpoint to directory: {base_path}")
            
            # Wait for Rank 0 to create the directory
            dist.barrier() 

            # A. Save Model Parameter Shards (All Ranks)
            local_model_state = {
                'W_enc': trainer.clt.W_enc.detach().cpu(), 
                'W_dec': trainer.clt.W_dec.detach().cpu(),
                'b_enc': trainer.clt.b_enc.detach().cpu(),
                'b_dec': trainer.clt.b_dec.detach().cpu(),
            }
            shard_filename = base_path / f"model_shard_{self.rank}.pt"
            torch.save(local_model_state, str(shard_filename))

            #  B. Save Optimizer State Shards (All Ranks) 
            opt_filename = base_path / f"optimizer_shard_{self.rank}.pt"
            torch.save(trainer.optimizer.state_dict(), str(opt_filename))

            #  C. Save Global Metadata (Rank 0 Only) 
            if self.rank == 0:
                global_state = {
                    'config': trainer.cfg,
                    'steps': trainer.n_training_steps,
                    'world_size': self.world_size,
                    'sharding_dim_W_enc': 1, 
                    'sharding_dim_W_dec': 0, 
                }
                global_filename = base_path / "global_metadata.pt"
                torch.save(global_state, str(global_filename))
            
            dist.barrier()
            if self.rank == 0:
                print(f"Completed saving {self.world_size} model and optimizer shards.")


    def update_clt_norm_scaling_factor(self): 
        """ update the CLTs norm scaling factor from the activation store"""
        if self.fsdp or self.ddp: 
            self.clt.module.estimated_norm_scaling_factor_in = self.activations_store.estimated_norm_scaling_factor_in.to(self.device)
            self.clt.module.estimated_norm_scaling_factor_out = self.activations_store.estimated_norm_scaling_factor_out.to(self.device)
        else: 
            self.clt.estimated_norm_scaling_factor_in = self.activations_store.estimated_norm_scaling_factor_in.to(self.device)
            self.clt.estimated_norm_scaling_factor_out = self.activations_store.estimated_norm_scaling_factor_out.to(self.device)

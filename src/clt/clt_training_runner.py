import os
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
from clt import logger

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
        self.feature_sharding = cfg.feature_sharding
        torch.manual_seed(cfg.seed)

        if self.cfg.uses_process_group:
            if rank is _missing or world_size is _missing:
                raise ValueError("Parallel computing is enabled but 'rank' and/or 'world_size' were not provided.")
            if not dist.is_initialized():
                raise RuntimeError("Parallel computing requested but process group not initialized.")
            self.rank = cast(int, rank)
            self.world_size = cast(int, world_size)
            self.cfg.device = f"cuda:{self.rank}"
        else:
            self.rank = 0
            self.world_size = 1

        self.is_main_process = True if self.rank == 0 else False
        self.device = torch.device(self.cfg.device)
        
        # Log to verify
        logger.info(f"Rank {self.rank}: Initializing CLT on device {self.cfg.device}")

        # For multlingual models added to transformer-lens
        if self.cfg.is_multilingual_split_dataset: 
            logger.info("Adding names to Transformer Lens")
            patch_official_model_names()
            patch_convert_hf_model_config()

        if self.cfg.is_distributed:
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
                self.cfg.from_pretrained_path,
                self.cfg.device,
                is_sharded=self.cfg.is_sharded,
                rank=self.rank,
                world_size=self.world_size,
            )
            self.clt = self.clt.to(self.device)
        else:
            self.clt = CLT(
                cfg.create_sub_config(
                    CLTConfig,
                    n_layers=self.model.cfg.n_layers
                ),
                rank=self.rank,
                world_size=self.world_size
            )
            # Ensure it's on the correct device
            self.clt = self.clt.to(self.device)
            logger.info(f"Rank {self.rank}: CLT created and moved to {self.device}")

        if self.ddp:
            logger.info(f"Rank {self.rank}: Wrapping CLT with DDP")
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
            logger.info(f"Norm scaling in: {self.clt.estimated_norm_scaling_factor_in if hasattr(self.clt, 'estimated_norm_scaling_factor_in') else 'N/A'}")
            logger.info(f"l0_coefficient: {self.cfg.l0_coefficient}")
            logger.info(f"l0_warm_up_steps: {self.cfg.l0_warm_up_steps}")
            logger.info(f"lr: {self.cfg.lr}")
            logger.info(f"dead_penalty_coef: {self.cfg.dead_penalty_coef}")

        trainer = CLTTrainer(
            clt=self.clt,
            activations_store=self.activations_store,
            save_checkpoint_fn=self.save_checkpoint,
            cfg=self.cfg,
            rank=self.rank, 
            world_size=self.world_size
        )
        
        logger.info("Start training...")
        clt = trainer.fit()

        if self.cfg.log_to_wandb and self.is_main_process:
            wandb.finish()

        return clt


    def save_checkpoint(self, trainer: CLTTrainer, checkpoint_name: str) -> None:
        base_path = Path(trainer.cfg.checkpoint_path) / checkpoint_name

        if self.rank == 0:
            base_path.mkdir(exist_ok=True, parents=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        def _unwrap_clt(m):
            return m.module if hasattr(m, "module") else m

        clt_obj = _unwrap_clt(trainer.clt)

        if self.cfg.is_sharded:
            # Each rank writes rank{r}_weights.safetensors
            clt_obj.save_model(str(base_path), save_cfg=(self.rank == 0), rank=self.rank)

            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            if self.rank == 0:
                logger.info(f"Saved sharded checkpoint with {self.world_size} shards to {base_path}")
            return

        if self.rank == 0:
            if self.fsdp:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
                from torch.distributed.fsdp import StateDictType, FullStateDictConfig

                with FSDP.state_dict_type(
                    trainer.clt,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True),
                ):
                    # In this context, trainer.clt.state_dict() is full; your save_model uses self.state_dict()
                    clt_obj.save_model(str(base_path), save_cfg=True, rank=None)

            elif self.ddp:
                clt_obj.save_model(str(base_path), save_cfg=True, rank=None)

            else:
                # single GPU
                clt_obj.save_model(str(base_path), save_cfg=True, rank=None)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def update_clt_norm_scaling_factor(self): 
        """ update the CLTs norm scaling factor from the activation store"""
        if self.cfg.is_distributed: 
            self.clt.module.estimated_norm_scaling_factor_in = self.activations_store.estimated_norm_scaling_factor_in.to(self.device)
            self.clt.module.estimated_norm_scaling_factor_out = self.activations_store.estimated_norm_scaling_factor_out.to(self.device)
        else: 
            self.clt.estimated_norm_scaling_factor_in = self.activations_store.estimated_norm_scaling_factor_in.to(self.device)
            self.clt.estimated_norm_scaling_factor_out = self.activations_store.estimated_norm_scaling_factor_out.to(self.device)
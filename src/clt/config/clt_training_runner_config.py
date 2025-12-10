import torch
import wandb

from pydantic import BaseModel, field_validator, model_validator, ConfigDict
from typing import Any, Dict, Optional, TypeVar
from clt import logger
from clt.config import CLTConfig

T = TypeVar("T", bound=BaseModel)

class CLTTrainingRunnerConfig(BaseModel): 
    # -----MISC------------------------------
    device : str = "cuda"
    dtype: str = "float32"
    seed: int = 42
    n_checkpoints: int = 4
    checkpoint_path: str = "checkpoints"
    logger_verbose: bool = True 

    # -----Model & Data-----------------------
    model_class_name: str = "HookedTransformer"
    model_name: str = "gpt2"
    model_kwargs: Optional[Dict[str, Any]] = None
    model_from_pretrained_kwargs: Optional[Dict[str, Any]] = None
    dataset_path: str = "" # Hugging face path
    is_dataset_tokenized: bool = True

    # -----CLT parameters---------------------
    from_pretrained_path: str | None = None
    d_in: int = 512
    expansion_factor: Optional[int] = None
    d_latent: Optional[int] = None
    jumprelu_init_threshold: float = 0.001
    jumprelu_bandwidth: float = 0.001
    normalize_decoder: bool = False
    
    # -----ActivationStore Parameters---------
    context_size: int = 128
    n_batches_in_buffer: int = 20
    store_batch_size_prompts: int = 32
    cached_activations_path: Optional[str] = None
    n_train_batch_per_buffer: Optional[int] = None # Folder where activations splits are saved

    # -----Training/Optimization--------------
    total_training_tokens: int = 100_000_000
    train_batch_size_tokens: int = 4096
    adam_beta1: float = 0.0
    adam_beta2: float = 0.999
    lr: float = 1e-5
    lr_warm_up_steps: int = 1000
    lr_decay_steps: int = 1000
    decay_stable_steps: int = 0
    final_lr_scale: float = 0.0
    cross_layer_decoders: bool = True
    lr_warm_up_type: str = "cosine"
    use_mixed_precision: bool = True

    # ------Functional Loss------------------
    functional_loss: Optional[str] = None
    fc_warm_up_type: str = "cosine"
    fc_coefficient: float = 1e-3
    fc_warm_up_steps: Optional[int] = None
    fc_waiting_steps: Optional[int] = None

    # -----Sparsity---------------------------
    l0_coefficient: float = 1e-4
    l0_warm_up_steps: int = 5000
    l0_waiting_steps: int = 0
    l0_warm_up_type: str = "linear"
    dead_penalty_coef: float =  0.0 #7.5 * 1e-8
    optimal_l0: Optional[float] = None
    checkpoint_l0: Optional[list[int]] = None

    # -----Metrics----------------------------
    dead_feature_window: int = 250
    # n_eval_batches: int = 10

    # -----WANDB------------------------------
    log_to_wandb: bool = True
    wandb_project: str = "CLT training"
    wandb_id: str | None = None 
    wandb_log_frequency: int = 10
    eval_every_n_wandb_logs: int = 100 
    run_name: str | None = None
    wandb_entity: str | None = None

    # -----DDP------------------------------
    ddp: bool = False
    fsdp: bool = False
    
    model_config = ConfigDict(
        validate_assignment = False, # re-run assigment after field value change
        extra = "forbid",  # avoid unknown fields
        json_encoders= { # make JSON‑safe for to_dict 
            torch.device: str,  
            torch.dtype: str, 
        }
    )

    @model_validator(mode="after")
    def validate_ddp_and_device(cls, cfg: "CLTTrainingRunnerConfig") -> "CLTTrainingRunnerConfig":
        if cfg.ddp:
            if not torch.cuda.is_available() or not cfg.device == "cuda":
                raise ValueError(
                    "DDP is enabled but CUDA is not available or not selected."
                )
        return cfg
    
    @field_validator("functional_loss", mode="before")
    def validate_functional_loss(cls, v: Optional[str]) -> Optional[str]:
        valid_losses = ["argmax", "kl"]
        if v is None: 
            return None
        if v not in valid_losses:
            raise ValueError(
                f"Invalid functional_loss '{v}'. Must be one of {valid_losses}."
            )
        return v
    
    @field_validator("device", mode="before")
    def fallback_to_cpu(cls, v: str) -> str:
        if v.lower().startswith("cuda") and not torch.cuda.is_available():
            print("CUDA requested but not available, using CPU.")
            return "cpu"
        elif v.lower().startswith("mps") and not torch.mps.is_available():
            print("MPS requested but not available, using CPU.")
            return "cpu"

        if v.lower() not in ["mps", "cpu", "cuda"]+[f"cuda:{i}" for i in range(8)]: 
            raise ValueError(
                f"Invalid device {v}. Must be cpu, cuda or mps."
            )
        return v
    
    @model_validator(mode="before")
    def wandb_id_check(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        wandb_id = values.get("wandb_id")
        if not wandb_id:
            raise ValueError("wandb_id must be provided")  # Force explicit
        
        base_path = values.get("checkpoint_path", "checkpoints")
        values["checkpoint_path"] = f"{base_path}/{wandb_id}"
        return values
    
    @model_validator(mode="before")
    def check_cached_activations(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        n_batches_in_buffer = values.get("n_batches_in_buffer")
        store_batch_size_prompts = values.get("store_batch_size_prompts")
        cached_activations_path = values.get("cached_activations_path")
        n_train_batch_per_buffer = values.get("n_train_batch_per_buffer")

        using_fresh = any(v is not None for v in (n_batches_in_buffer, store_batch_size_prompts))
        using_cached = any(v is not None for v in (cached_activations_path, n_train_batch_per_buffer))

        if using_fresh and using_cached:
            raise ValueError(
                "Invalid configuration: you cannot set both cached_activations_path / n_train_batch_per_buffer "
                "and n_batches_in_buffer / store_batch_size_prompts."
            )
        
        if cached_activations_path is not None and n_train_batch_per_buffer is None: 
            raise ValueError(
                "If you set cached_activations_path, you must also set n_train_batch_per_buffer."
            )
        return values
            
    @model_validator(mode="before")
    def check_latent_vs_expansion(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        d_latent = values.get("d_latent")
        expansion = values.get("expansion_factor")
        d_latent_given = d_latent is not None
        expansion_given = expansion is not None

        if d_latent_given and expansion_given:
            raise ValueError("You can't set both d_latent and expansion_factor.")

        if not d_latent_given:
            expansion = expansion or 16
            d_in = values.get("d_in", 512)
            values["d_latent"] = d_in * expansion

        return values
    
    def model_post_init(self, __context):
        if not self.logger_verbose: 
            return 
        logger.info("-------- CLT training run -------")
        logger.info("d_latent        : %d", self.d_latent)
        logger.info("total tokens    : %.3e", self.total_training_tokens)
        logger.info("batch (tokens)  : %d", self.train_batch_size_tokens)
        total_steps = self.total_training_tokens // self.train_batch_size_tokens
        logger.info("total steps     : %d", total_steps)
        n_tokens_per_buffer = (
            self.store_batch_size_prompts
            * self.context_size
            * self.n_batches_in_buffer
        )
        logger.info(
            f"n_tokens_per_buffer (millions): {n_tokens_per_buffer / 10**6}"
        )
        logger.info("checkpoint dir  : %s", self.checkpoint_path)
        logger.info("wandb project   : %s  (id=%s)", self.wandb_project, self.wandb_id)
        logger.info("---------------------------------")

    def create_sub_config(self, sub_config_class: type[T], **overrides) -> T:
        # Instantiate CLT using the overlapping fields of `parent`.
        data = self.model_dump(include=sub_config_class.model_fields.keys(), mode="python")
        data.update(overrides)
        if sub_config_class is CLTConfig and "n_layers" not in overrides:
            raise ValueError("n_layers is required when instantiating CLTConfig")

        return sub_config_class.model_validate(data)
        
    # one‑liner to get a json‑safe dict
    def to_dict(self, *, exclude_none: bool = True,**kw) -> Dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=exclude_none)
    
    @property
    def total_training_steps(self) -> int:
        return int(self.total_training_tokens // self.train_batch_size_tokens)

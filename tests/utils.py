from clt.config import CLTTrainingRunnerConfig
from typing import Any
import torch 
from clt.training.activations_store import ActivationsStore

TINYSTORIES_MODEL = "tiny-stories-1M"
NEEL_NANDA_C4_10K_DATASET = "data/NeelNanda_c4_10k_tokenized"

def build_clt_training_runner_cfg(**kwargs: Any) -> CLTTrainingRunnerConfig:
    """
    Helper to create a mock instance of CLTTrainingRunnerConfig.
    """
    mock_config_dict = { 
        "device": "cpu", 
        "model_name": TINYSTORIES_MODEL, 
        "dataset_path": NEEL_NANDA_C4_10K_DATASET, 
        "d_in": 12,
        "l0_coefficient": 2e-3,
        "lr": 1e-2,
        "d_latent": 4,
        "train_batch_size_tokens": 4,
        "context_size": 4,
        "n_batches_in_buffer": 4,
        "total_training_tokens": 100,
        "l0_warm_up_steps": 0,
        "lr_decay_steps": 10,
        "lr_warm_up_steps": 10,
        "store_batch_size_prompts": 4,
        "log_to_wandb": False,
        "wandb_project": "test_project",
        "wandb_entity": "test_entity",
        "wandb_log_frequency": 5,
        "checkpoint_path": "test/checkpoints",
        "n_checkpoints": 1,
        "dead_feature_window": 1, 
    }

    for key, value in kwargs.items():
        mock_config_dict[key] = value

    print(mock_config_dict)

    mock_config = CLTTrainingRunnerConfig(**mock_config_dict)

    # reset checkpoint path (as we add an id to each each time)
    mock_config.checkpoint_path = kwargs.get("checkpoint_path", "test/checkpoints")

    return mock_config

class FakeActivationsStore(ActivationsStore):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y

    def __iter__(self):
        yield self.x, self.y

### COMMAND TO RUN THE DDP TESTS: 
# poetry run torchrun   --nproc_per_node=2   --rdzv_backend=c10d   --rdzv_endpoint=localhost:29501   -m pytest ddp_vs_ft.py
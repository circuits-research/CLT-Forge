from pathlib import Path
from circuitlab.config.clt_training_runner_config import CLTTrainingRunnerConfig
from circuitlab.infra.wandb_utils import get_synced_wandb_id

import circuitlab
STORAGE_ROOT = Path(circuitlab.__file__).resolve().parents[2] / "storage" # symlink to scratch

def clt_training_runner_config(rank: int = 0, world_size: int = 1, generation: bool = False):
    MODEL = "gpt2"

    # Training type in ["None", "ddp", "sfdp", "feature_sharding"]
    distributed_setup = "feature_sharding" if not generation else "None"
    
    ### IMPORTANT, where activations will be stored (around 1-2TB)
    activations_root = STORAGE_ROOT / Path("activations") / f"{MODEL}"
    checkpoints_root = STORAGE_ROOT / Path("checkpoints") / f"{MODEL}"

    gradient_accumulation_steps = 1
    total_training_steps = 300_000 // world_size
    train_batch_size_tokens = world_size * 1024
    total_training_tokens = gradient_accumulation_steps * train_batch_size_tokens * total_training_steps

    lr_decay_steps = (total_training_steps // 20) - 1
    final_lr_scale = 0.0
    lr_warm_up_steps = 1000

    l0_waiting_steps = 0
    l0_warm_up_steps = int(0.7 * total_training_steps) - l0_waiting_steps - 1
    decay_stable_steps = total_training_steps - l0_warm_up_steps - lr_decay_steps

    cfg = CLTTrainingRunnerConfig(
        device="cuda",
        dtype="bfloat16",
        seed=42,
        n_checkpoints=0,
        checkpoint_path=str(checkpoints_root),
        logger_verbose=(rank==0),
        model_class_name="HookedTransformer",
        model_name=MODEL,
        dataset_path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        context_size=16,
        from_pretrained_path=None,
        d_in=768,
        expansion_factor=32,
        jumprelu_init_threshold=0.03,
        jumprelu_bandwidth=1.0,
        cached_activations_path=str(activations_root),
        n_train_batch_per_buffer=16, # depends on the size of your buffer
        total_training_tokens=total_training_tokens,
        train_batch_size_tokens=train_batch_size_tokens,
        gradient_accumulation_steps=gradient_accumulation_steps,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr=4e-4,
        lr_warm_up_steps=lr_warm_up_steps,
        lr_decay_steps=lr_decay_steps,
        final_lr_scale=final_lr_scale,
        l0_coefficient=2.0,
        dead_penalty_coef=1e-5,
        dead_feature_window=250,
        l0_warm_up_steps=l0_warm_up_steps,
        l0_waiting_steps=l0_waiting_steps,
        decay_stable_steps=decay_stable_steps,
        checkpoint_l0=[10], # which l0 checkpoint to save
        optimal_l0=10, # when to stop training, this is per layer 0, so total l0 is 120
        log_to_wandb=True,
        wandb_project=f"{MODEL}-clt",
        wandb_id=get_synced_wandb_id(rank),
        wandb_log_frequency=10,
        eval_every_n_wandb_logs=100,
        run_name=None,
        wandb_entity=None,
        distributed_setup=distributed_setup
    )

    return cfg

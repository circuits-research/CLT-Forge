import torch
from clt.config.clt_training_runner_config import CLTTrainingRunnerConfig
from clt.transformer_lens.multilingual_patching import patch_official_model_names, patch_convert_hf_model_config
from clt.training.activations_store import ActivationsStore
from clt.load_model import load_model
import sys 

total_training_steps = 1000
train_batch_size_tokens = 2048
total_training_tokens = train_batch_size_tokens * total_training_steps
lr_decay_steps = total_training_steps // 10
lr_warm_up_steps = 1000
l0_warm_up_steps = 10000

jobID = int(sys.argv[1])
MODEL_PERCENTAGE = sys.argv[2] if len(sys.argv) > 2 else None
assert MODEL_PERCENTAGE in ["20", "50", "70", "90"]
TOTAL_SPLITS = 1024 # what should be the right number of splits ? 
TOTAL_JOBS = 8
SPLITS_PER_JOB = TOTAL_SPLITS // TOTAL_JOBS 
split_begin_idx = jobID * SPLITS_PER_JOB
split_end_idx = (jobID + 1) * SPLITS_PER_JOB
if jobID == TOTAL_JOBS - 1:
    split_end_idx = TOTAL_SPLITS

print(f"Job {jobID}: Processing splits {split_begin_idx} to {split_end_idx-1}")

# apollo-research/roneneldan-TinyStories-tokenizer-gpt2
# apollo-research/Skylion007-openwebtext-tokenizer-gpt2

cfg = CLTTrainingRunnerConfig(
    device="cuda",  # will be updated in _ddp_worker
    dtype="torch.bfloat16",
    seed=42,
    n_checkpoints=4,
    checkpoint_path="checkpoints/gpt2_multilingual_"+MODEL_PERCENTAGE,
    logger_verbose=True,
    model_class_name="HookedTransformer",
    model_name="CausalNLP/gpt2-hf_multilingual-"+MODEL_PERCENTAGE,
    dataset_path="abir-hr196/clt_gpt2_tokenized",
    context_size=16, # changed to 16
    from_pretrained_path=None,
    d_in=768,
    expansion_factor=32,
    jumprelu_init_threshold=0.002,
    jumprelu_bandwidth=0.001,
    n_batches_in_buffer=16,
    store_batch_size_prompts=32, # changed to 64
    total_training_tokens=total_training_tokens,
    train_batch_size_tokens=train_batch_size_tokens,
    adam_beta1=0.0,
    adam_beta2=0.999,
    lr=7e-5,
    lr_warm_up_steps=lr_warm_up_steps,
    lr_decay_steps=lr_decay_steps,
    l0_coefficient=0.0005,
    l0_warm_up_steps=l0_warm_up_steps,
    log_to_wandb=True,
    wandb_project="gpt2-clt-multilingual-"+MODEL_PERCENTAGE,
    wandb_id=None,
    wandb_log_frequency=10,
    eval_every_n_wandb_logs=100,
    run_name=None,
    wandb_entity=None,
    ddp=False, 
    fsdp=False
)

patch_official_model_names()
patch_convert_hf_model_config()

model = load_model(
    cfg.model_class_name,
    cfg.model_name,
    device=torch.device(cfg.device), 
    model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
)

activations_store = ActivationsStore(
    model,
    cfg
)

activations_store.generate_and_save_activations(
    path="/home/abir19/scratch/data/featflow/activations_gpt2_multilingual_"+MODEL_PERCENTAGE, 
    split_count=TOTAL_SPLITS, 
    number_of_tokens=150994944,  # 1024^2 x 144
    split_begin_idx=split_begin_idx,  # Add this
    split_end_idx=split_end_idx       # Add this
)

print("Finished activations generation and saving.")
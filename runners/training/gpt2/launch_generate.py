import torch
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from circuitlab.transformer_lens.multilingual_patching import patch_official_model_names, patch_convert_hf_model_config
from circuitlab.training.activations_store import ActivationsStore
from circuitlab.training.compressed_activations_store import CompressionConfig
from circuitlab.infra.jobs_id import compute_job_split_range
from runners.training.gpt2.config import clt_training_runner_config

from sae_lens.load_model import load_model

def main(job_id: int, total_jobs: int):
    """
    Generate and saving the activations is highly parallelizable. Ideally this script should ran on my instances in parallel
    """
    cfg = clt_training_runner_config(generation=True)
    cached_activations_path = cfg.cached_activations_path
    cfg.cached_activations_path = None

    total_splits = 4196
    split_begin_idx, split_end_idx = compute_job_split_range(
        job_id=job_id,
        total_jobs=total_jobs,
        total_splits=total_splits,
    )

    # number of token activations to save, could more than total_training_tokens 
    number_of_tokens = total_splits * cfg.train_batch_size_tokens * 8 * 8
    print(f"Job {job_id}: Processing splits {split_begin_idx} to {split_end_idx-1}")

    if cfg.is_multilingual_split_dataset: 
        patch_official_model_names()
        patch_convert_hf_model_config()

    model = load_model(
        cfg.model_class_name,
        cfg.model_name,
        device=torch.device(cfg.device), 
        model_from_pretrained_kwargs=cfg.model_from_pretrained_kwargs,
    )

    compression_config = CompressionConfig(
        quantization="int8",
        compression="zstd", 
        compression_level=3,
    )

    activations_store = ActivationsStore(
        model,
        cfg
    )

    activations_store.generate_and_save_activations(
        path=cached_activations_path,
        split_count=total_splits, 
        number_of_tokens=number_of_tokens,
        split_begin_idx=split_begin_idx,
        split_end_idx=split_end_idx,
        use_compression=False,           
        compression_config=compression_config, 
    )

    print("Finished activations generation and saving.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_id", type=int, required=True)
    parser.add_argument("--total_jobs", type=int, required=True)
    args = parser.parse_args()

    main(args.job_id, args.total_jobs)

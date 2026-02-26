"""
Run a single job:
    python launch_autointerp.py --job_id 0 --total_jobs 32

With LLM explanations:
    python launch_autointerp.py --job_id 0 --total_jobs 32 --generate_explanations

Launch all jobs in parallel:
    for i in $(seq 0 31); do
        python launch_autointerp.py --job_id $i --total_jobs 32 &
    done
"""

import argparse
from pathlib import Path

import torch  # noqa: F401 — used for cuda availability checks

from circuitlab.config.autointerp_config import AutoInterpConfig
from circuitlab.autointerp.pipeline_new import AutoInterp


def build_config() -> AutoInterpConfig:
    d_in = 768
    return AutoInterpConfig(
        device="cuda",
        dtype="bfloat16",
        model_name="gpt2",
        clt_path="/fast/fdraye/data/featflow/cache/checkpoints/gpt2/d1s3fw30/middle_22137856",
        latent_cache_path="/fast/fdraye/data/featflow/autointerp/gpt2",
        dataset_path="apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        context_size=16,
        total_autointerp_tokens=12 * 16 * 4096,   # ~786k tokens
        train_batch_size_tokens=4096,
        n_batches_in_buffer=32,
        store_batch_size_prompts=32,
        d_in=d_in,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AutoInterp for one job (feature slice).")
    parser.add_argument("--job_id",    type=int, required=True, help="Job index [0, total_jobs).")
    parser.add_argument("--total_jobs", type=int, required=True, help="Total number of parallel jobs.")
    parser.add_argument(
        "--generate_explanations",
        action="store_true",
        default=False,
        help="Run vLLM to generate LLM explanations (optional, slow).",
    )
    args = parser.parse_args()

    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    print(f"CUDA device count: {torch.cuda.device_count()}", flush=True)
    print(f"[Job {args.job_id}/{args.total_jobs}] Starting…", flush=True)

    cfg = build_config()
    autointerp = AutoInterp(cfg)
    autointerp.run(
        job_id=args.job_id,
        total_jobs=args.total_jobs,
        top_k=100,
        save_dir=Path(cfg.latent_cache_path),
        generate_explanations=args.generate_explanations,
    )

    print(f"[Job {args.job_id}] DONE", flush=True)


if __name__ == "__main__":
    main()

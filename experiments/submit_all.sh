#!/bin/bash
# Submit all CLT decoder comparison experiments

cd /home/jiaruil5/CLT
mkdir -p /data/group_data/r3lit_rl/clt_experiments/logs

echo "Submitting CLT decoder comparison experiments..."

# Baseline: full decoder
sbatch --job-name=clt_full experiments/sbatch_clt.sh full 0

# LoRA with different ranks
sbatch --job-name=clt_lora_r16 experiments/sbatch_clt.sh lora 16
sbatch --job-name=clt_lora_r32 experiments/sbatch_clt.sh lora 32
sbatch --job-name=clt_lora_r64 experiments/sbatch_clt.sh lora 64
sbatch --job-name=clt_lora_r128 experiments/sbatch_clt.sh lora 128

echo ""
echo "All jobs submitted. Check status with: squeue -u $(whoami)"
echo "Results logged to wandb project: gpt2-clt-lora-comparison"

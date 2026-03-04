#!/bin/bash
#SBATCH --partition=general
#SBATCH --output=/data/group_data/r3lit_rl/clt_experiments/logs/%x_%j.out
#SBATCH --error=/data/group_data/r3lit_rl/clt_experiments/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=8:00:00
#SBATCH --mem=48G
#SBATCH --mail-user=jiaruiliu999@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
conda activate clt

DECODER_TYPE=$1
DECODER_RANK=$2

cd /home/jiaruil5/CLT
export PYTHONUNBUFFERED=1

echo "Starting experiment: decoder_type=$DECODER_TYPE, decoder_rank=$DECODER_RANK"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python -m runners.decoder_comparison.launch_train \
    --decoder_type "$DECODER_TYPE" \
    --decoder_rank "$DECODER_RANK"

echo "Experiment completed."

#!/bin/bash
#SBATCH --job-name=resnet50-ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/resnet50-ddp-%j.log
#SBATCH --account=research
#SBATCH --partition=gpu_dedicated

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p results

# IMPORTANT: activate YOUR existing GPU-enabled Python/PyTorch environment here.
# Examples (replace the path or module with your actual installation):
# module load pytorch/<version>
# source /opt/apps/venvs/pytorch/bin/activate
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate myenv_313

python -c 'import torch, torchvision; print("torch",torch.__version__,"torchvision",torchvision.__version__,"CUDA",torch.version.cuda); assert torch.cuda.is_available()'

# The batch script runs on one allocated compute node. Find its private 10GbE IP.
MASTER_ADDR=$(ip -o -4 addr show | awk '$4 ~ /^10[.]210[.]45[.]/ {sub(/\/.*/, "", $4); print $4; }' | head -n 1)
if [[ -z "$MASTER_ADDR" ]]; then
  echo 'ERROR: No 10.210.45.x IPv4 address on this node. Check private cluster network.' >&2
  exit 2
fi
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_ADDR MASTER_PORT
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

# Configurable without editing the scripts. Use the same values across the scaling run.
export BENCH_BATCH_SIZE="${BENCH_BATCH_SIZE:-32}"
export BENCH_STEPS="${BENCH_STEPS:-200}"
export BENCH_WARMUP="${BENCH_WARMUP:-20}"

echo "job=$SLURM_JOB_ID nodes=$SLURM_JOB_NUM_NODES master=$MASTER_ADDR:$MASTER_PORT"
echo "batch/gpu=$BENCH_BATCH_SIZE warmup=$BENCH_WARMUP steps=$BENCH_STEPS"
srun --ntasks="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
  bash ./resnet50_ddp_node.sh

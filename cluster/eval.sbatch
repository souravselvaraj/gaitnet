#!/usr/bin/env bash
# Deterministic evaluation of GaitNet checkpoints at fixed void densities (gaitnet/gaitnet/eval_policy.py)
# on one GPU of Turing. Arguments go to `python -m gaitnet.gaitnet.eval_policy`:
#
#   sbatch cluster/eval.sbatch --checkpoints <a.pt> <b.pt> --labels A B --out <dir>
#SBATCH --job-name=gaitnet-eval
#SBATCH --account=mmaghelih
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# gpu-6-05: CUDA driver initialisation fails there (2026-10-12)
#SBATCH --exclude=gpu-6-05
#SBATCH --output=/scratch/sselvaraj/projects/gait-net-train/gaitnet/cluster/logs/%x-%j.out

set -euo pipefail
export ISAAC_ROOT=${ISAAC_ROOT:-/scratch/sselvaraj/isaac-gaitnet}
source "$ISAAC_ROOT/env.sh"
cd /scratch/sselvaraj/projects/gait-net-train/gaitnet
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "node $(hostname); + python -m gaitnet.gaitnet.eval_policy --headless $*"
python -m gaitnet.gaitnet.eval_policy --headless "$@"
echo GAITNET-EVAL-DONE

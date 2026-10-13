#!/usr/bin/env bash
# Fit the footstep forward/inverse models (gaitnet/gaitnet/idm.py) for both controllers on one GPU.
#   sbatch cluster/idm_fit.sbatch <idm_data dir> <out dir>
#SBATCH --job-name=gaitnet-idmfit
#SBATCH --account=mmaghelih
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/sselvaraj/projects/gait-net-train/gaitnet/cluster/logs/%x-%j.out
set -euo pipefail
export ISAAC_ROOT=${ISAAC_ROOT:-/scratch/sselvaraj/isaac-gaitnet}
source "$ISAAC_ROOT/env.sh"
cd /scratch/sselvaraj/projects/gait-net-train/gaitnet
DATA=$1; OUT=$2; mkdir -p "$OUT"
for v in gaitnet gaitnet_core; do
  for dir in forward inverse; do
    python -m gaitnet.gaitnet.idm --no-log-file --events "$DATA/$v"/*.events.pt --direction $dir --out "$OUT/${v}_${dir}.pt" --epochs 60
  done
done
echo GAITNET-IDMFIT-DONE

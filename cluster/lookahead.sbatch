#!/usr/bin/env bash
# Lookahead footstep selection vs the plain argmax at fixed void densities (gaitnet/gaitnet/eval_lookahead.py)
# on one GPU of Turing. Arguments go to `python -m gaitnet.gaitnet.eval_lookahead`:
#
#   sbatch cluster/lookahead.sbatch --checkpoints <a.pt> --labels A --modes argmax d1 d2 --out <dir>
#SBATCH --job-name=gaitnet-lookahead
#SBATCH --account=mmaghelih
#SBATCH --partition=short
#SBATCH --qos=short
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/sselvaraj/projects/gait-net-train/gaitnet/cluster/logs/%x-%j.out

set -euo pipefail
export ISAAC_ROOT=${ISAAC_ROOT:-/scratch/sselvaraj/isaac-gaitnet}
source "$ISAAC_ROOT/env.sh"
cd /scratch/sselvaraj/projects/gait-net-train/gaitnet
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ -n "${LOOKAHEAD_BENCH:-}" ]]; then   # planner time per tick on synthetic inputs, CPU (1 thread) and GPU
    python -m gaitnet.gaitnet.lookahead --no-log-file
fi
echo "node $(hostname); + python -m gaitnet.gaitnet.eval_lookahead --headless $*"
python -m gaitnet.gaitnet.eval_lookahead --headless "$@"
echo GAITNET-LOOKAHEAD-DONE

#!/usr/bin/env bash
# GaitNet PPO training (gaitnet/gaitnet/train.py) on one GPU of WPI's Turing cluster, with Isaac Sim 5.1 +
# Isaac Lab 2.3.2 + rsl-rl-lib 3.1.2 installed by cluster/setup_train_env.sh into $ISAAC_ROOT.
# By default the MPC runs in one CPU process per allocated CPU (VectorPool), so --cpus-per-task is the
# MPC parallelism; Isaac Sim itself needs the GPU. With --gpu_mpc the MPC runs batched on the GPU
# (gaitnet/sim2real/gpu_mpc.py) and the job needs only a few CPUs: pass -c 8 to sbatch.
#
#   fresh:     sbatch cluster/train.sbatch --num_envs 256 --max_iterations 700
#   fine-tune: sbatch cluster/train.sbatch --num_envs 256 --max_iterations 100 --resume <model.pt>
#   smoke:     sbatch -t 00:45:00 cluster/train.sbatch --num_envs 128 --max_iterations 3
#   GPU MPC:   sbatch -c 8 cluster/train.sbatch --gpu_mpc --num_envs 1024 --max_iterations 3
#
# Arguments go to `python -m gaitnet.gaitnet` (train.py). Runs and checkpoints (model_<it>.pt every 5
# iterations, TensorBoard events) land in gaitnet/training/gaitnet/runs/gaitnet_<timestamp>/.
# Override any #SBATCH line on the command line, e.g. sbatch -t 2-00:00:00 cluster/train.sbatch ...
#SBATCH --job-name=gaitnet-train
#SBATCH --account=mmaghelih
#SBATCH --partition=long
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/sselvaraj/projects/gait-net-train/gaitnet/cluster/logs/%x-%j.out

set -euo pipefail
export ISAAC_ROOT=${ISAAC_ROOT:-/scratch/sselvaraj/isaac-gaitnet}
source "$ISAAC_ROOT/env.sh"
GAITNET_ROOT=/scratch/sselvaraj/projects/gait-net-train/gaitnet
cd "$GAITNET_ROOT"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "node $(hostname), $SLURM_CPUS_PER_TASK cpus (affinity $(nproc)), in $PWD"
echo "gait-net $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD), $(git status --porcelain | wc -l) changed files"
echo "+ python -m gaitnet.gaitnet --headless $*"

python -m gaitnet.gaitnet --headless "$@"
echo GAITNET-TRAIN-DONE

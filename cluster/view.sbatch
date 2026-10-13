#!/usr/bin/env bash
# Watch a GaitNet policy live in your browser (Viser), on Isaac Lab 2.3.2, by playing a training
# run's newest checkpoint (gaitnet/gaitnet/view_policy.py). Follows the isaac-turing README's viewer
# recipe: the log prints the ssh tunnel to run on your computer and the address to open.
#
#   sbatch cluster/view.sbatch --run SC-idm --idm_inverse $PWD/training/gaitnet/idm_models/gaitnet_inverse.pt
#
# Runs on `quick` (1 GPU at a time, so scancel it when done) for up to 4 h.
#SBATCH --job-name=gaitnet-view
#SBATCH --account=mmaghelih
#SBATCH --partition=quick
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/sselvaraj/projects/gait-net-train/gaitnet/cluster/logs/%x-%j.out

set -euo pipefail
export ISAAC_ROOT=${ISAAC_ROOT:-/scratch/sselvaraj/isaac-gaitnet}
source "$ISAAC_ROOT/env.sh"
# viser and the few packages it needs at newer versions, kept out of the training venv
export PYTHONPATH=$ISAAC_ROOT/viser-pkgs${PYTHONPATH:+:$PYTHONPATH}
cd /scratch/sselvaraj/projects/gait-net-train/gaitnet
isaac_viser_setup   # env.sh: free port + the ssh command for your computer
python -m gaitnet.gaitnet.view_policy --headless "$@"

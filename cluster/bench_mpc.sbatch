#!/usr/bin/env bash
# Training throughput with the CPU MPC pool vs the batched GPU MPC, on one node, one run after
# another. Each argument is NAME:ARGS for one short `python -m gaitnet.gaitnet --headless` run
# (two PPO iterations of 32 steps unless ARGS say otherwise); GPU memory is sampled with
# nvidia-smi every 0.5 s during each run. Per-run logs land in cluster/logs/bench-<jobid>/.
#
#   sbatch cluster/bench_mpc.sbatch "cpu256:--num_envs 256" "gpu256:--gpu_mpc --num_envs 256"
#
# Steps/s are rsl-rl's "Computation" line of the LAST iteration (the first includes warm-up).
# A NAME ending in "prof" sets GAITNET_PROFILE_MPC=1: the GPU controller then synchronises around
# each compute_torques and reports its mean time per physics step (which slows the run a little).
#SBATCH --job-name=gaitnet-bench
#SBATCH --account=mmaghelih
#SBATCH --partition=long
#SBATCH --qos=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/sselvaraj/projects/gait-net-train/gaitnet/cluster/logs/%x-%j.out

set -uo pipefail
export ISAAC_ROOT=${ISAAC_ROOT:-/scratch/sselvaraj/isaac-gaitnet}
source "$ISAAC_ROOT/env.sh"
GAITNET_ROOT=/scratch/sselvaraj/projects/gait-net-train/gaitnet
cd "$GAITNET_ROOT"
out=$GAITNET_ROOT/cluster/logs/bench-${SLURM_JOB_ID:-local}
mkdir -p "$out"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "node $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus (affinity $(nproc)), logs in $out"
echo "gait-net $(git rev-parse --short HEAD) on $(git rev-parse --abbrev-ref HEAD), $(git status --porcelain | wc -l) changed files"

summary=()
for spec in "$@"; do
    name=${spec%%:*}
    args=${spec#*:}
    echo "== $name: python -m gaitnet.gaitnet --headless --max_iterations 2 --num_steps_per_env 32 $args"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 500 > "$out/$name.mem" &
    smi=$!
    start=$(date +%s)
    # shellcheck disable=SC2086
    profile=0; [[ $name == *prof ]] && profile=1
    GAITNET_PROFILE_MPC=$profile python -m gaitnet.gaitnet --headless --max_iterations 2 --num_steps_per_env 32 $args \
        > "$out/$name.log" 2>&1 || echo "$name FAILED (exit $?)"
    kill "$smi" 2>/dev/null
    wall=$(( $(date +%s) - start ))
    peak=$(sort -n "$out/$name.mem" | tail -1)
    comp=$(grep "Computation:" "$out/$name.log" | tail -1 | sed 's/^ *//')
    profile=$(grep -o "GPU MPC profile:.*\|torch peak allocated.*" "$out/$name.log" | tr '\n' ' ')
    line="$name | $comp | GPU memory peak ${peak} MiB | wall ${wall}s | $profile"
    echo "$line"
    summary+=("$line")
done

echo "== summary"
printf '%s\n' "${summary[@]}"
echo GAITNET-BENCH-DONE

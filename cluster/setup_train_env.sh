#!/usr/bin/env bash
# Build the environment cluster/train.sbatch trains in, on WPI's Turing cluster (run on the LOGIN node,
# it downloads ~30 GB the first time; safe to re-run):
#
#   bash cluster/setup_train_env.sh
#
# 1. Isaac Sim 5.1 + Isaac Lab 2.3.2 (isaaclab 0.54.2, isaaclab_rl 0.4.7 -> rsl-rl-lib 3.1.2), Python 3.11,
#    torch 2.7.0+cu128, via isaac-turing's setup-lab2.sh, into its own $ISAAC_ROOT. That script also
#    supplies libGLU.so.1, which the GPU nodes lack (Isaac Sim hangs without it).
# 2. gaitnet's extra runtime deps (nptyping for the MPC; pybind11 to build it).
# 3. gaitnet/gaitnet/control: rl-mpc-locomotion @ dbe115f, a gitlink without a .gitmodules entry, so a
#    fresh checkout/worktree has it empty. Cloned here from the main gait-net checkout, then its
#    pybind11 extension mpc_osqp is compiled in place.
# 4. gaitnet itself, editable and --no-deps: setup.py still pins rsl-rl v1.0.2, which would overwrite
#    rsl_rl 3.1.2 (the version the code's TensorDict/obs_groups API needs).
# 5. gaitnet-core (the batched GPU MPC behind train.py --gpu_mpc), from the gaitnet repo at a pinned
#    commit, as a snapshot (not editable: later edits there must not change training here), --no-deps
#    (it needs only torch and numpy, both already installed).
set -euo pipefail
export ISAAC_ROOT=${ISAAC_ROOT:-/scratch/sselvaraj/isaac-gaitnet}
ISAAC_TURING=${ISAAC_TURING:-/scratch/sselvaraj/projects/isaac-turing}
GAITNET_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONTROL_SRC=${CONTROL_SRC:-/scratch/sselvaraj/projects/gait-net/gaitnet/gaitnet/control}
CONTROL_COMMIT=dbe115f7d53647d211d2cb4bc8144e4b0ef61656
GAITNET_CORE_REPO=${GAITNET_CORE_REPO:-/scratch/sselvaraj/projects/gaitnet}
GAITNET_CORE_COMMIT=c79ab670a1bef7261aeee350475eaca81500e446

bash "$ISAAC_TURING/setup-lab2.sh"
source "$ISAAC_ROOT/env.sh"
pip=(uv pip install --python "$VIRTUAL_ENV/bin/python")

echo "== gaitnet extra deps"
"${pip[@]}" nptyping==2.5.0 pybind11==2.13.6

echo "== MPC (rl-mpc-locomotion @ ${CONTROL_COMMIT:0:7}) -> $GAITNET_ROOT/gaitnet/control"
control=$GAITNET_ROOT/gaitnet/control
if [[ ! -e $control/.git ]]; then
    rmdir "$control" 2>/dev/null || true
    git clone -q --no-checkout "$CONTROL_SRC" "$control"
fi
git -C "$control" checkout -q "$CONTROL_COMMIT"
for sub in eigen3 qpoases; do
    git -C "$control" config "submodule.extern/$sub.url" "$CONTROL_SRC/.git/modules/extern/$sub"
done
git -C "$control" -c protocol.file.allow=always submodule update --init extern/eigen3 extern/qpoases
(cd "$control" && CPLUS_INCLUDE_PATH=$(python -c "import pybind11; print(pybind11.get_include())") \
    python setup.py build_ext --inplace > "$ISAAC_ROOT/mpc-build.log" 2>&1) || { tail -30 "$ISAAC_ROOT/mpc-build.log"; exit 1; }

mkdir -p "$GAITNET_ROOT/cluster/logs"  # train.sbatch writes its Slurm log here (gitignored)

echo "== gaitnet (editable, no deps)"
"${pip[@]}" --no-deps -e "$GAITNET_ROOT"

echo "== gaitnet-core @ ${GAITNET_CORE_COMMIT:0:7} (GPU MPC), snapshot, no deps"
core_src=$ISAAC_ROOT/src/gaitnet-core-${GAITNET_CORE_COMMIT:0:7}
if [[ ! -d $core_src/packages/gaitnet-core ]]; then
    mkdir -p "$core_src"
    git -C "$GAITNET_CORE_REPO" archive "$GAITNET_CORE_COMMIT" packages/gaitnet-core | tar -x -C "$core_src"
fi
"${pip[@]}" --no-deps "$core_src/packages/gaitnet-core"

python -c "import gaitnet.control.mpc_osqp as m, rsl_rl, isaaclab; print('mpc', m.__file__)"
python -c "import gaitnet_core.control as c; print('gpu mpc', c.__file__)"
echo "Done. Next: sbatch $GAITNET_ROOT/cluster/train.sbatch --num_envs 128 --max_iterations 3"

"""Coarse check that the GPU controller stands in for the CPU one: same inputs, compare outputs.

Drives `PooledRobotControllers` (the CPU path training has always used) and
`GpuRobotControllers` through the same scripted sequence of robot states, velocity
commands and footstep commands, both through the batched API the env uses with
Isaac-format inputs (joint-type-major joints, wxyz quaternion, world twist), so the
conversions on each side are part of what is compared. No simulator: the states are
scripted, so this checks the controllers, not closed-loop behaviour.

    python -m gaitnet.sim2real.compare_controllers [--steps 120] [--device cpu]
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from gaitnet.sim2real.robot_controllers import (
    GpuMpcCfg,
    GpuRobotControllers,
    PooledRobotControllers,
)

DT = 0.004
ITERATIONS_BETWEEN_MPC = 5
FOOTSTEPS = {3: (0, 0.2013), 11: (3, 0.1871), 60: (1, 0.2137), 75: (2, 0.1659)}
"""Control step -> (controller leg (FL, FR, RL, RR), swing duration (s)) started on it.

Durations are off the 4 ms grid, as the policy's are: a touchdown landing exactly on a
control step is a float tie the two controllers break differently (the CPU one adds
float32 start and duration and compares with a float64 clock), which flips one step."""


def scripted_state(step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Isaac-format (joint_pos (12,), joint_vel (12,), root_state_w (13,)) at a step.

    A body that rolls, pitches and yaws while moving, legs that keep moving: not a
    trajectory the robot would follow, but one where frame and sign errors show.
    """
    time = step * DT
    roll, pitch, yaw = 0.05 * math.sin(3 * time), 0.04 * math.cos(2 * time), 0.02 * time
    (cr, sr), (cp, sp), (cy, sy) = [(math.cos(a / 2), math.sin(a / 2)) for a in (roll, pitch, yaw)]
    quat_wxyz = [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]
    root = np.array([0.1 * time, 0.0, 0.3, *quat_wxyz, 0.3, 0.02, 0.0, 0.01, 0.02, 0.05])
    knee = math.acos(0.26 / (2 * 0.213))
    joints = np.arange(12).reshape(4, 3)  # (leg, joint)
    pos = np.tile([0.0, knee, -2 * knee], (4, 1)) + 0.01 * np.sin(step + joints)
    vel = 0.02 * np.cos(step + joints)
    # leg-major (leg, joint) -> Isaac's joint-type-major index leg + 4 * joint
    to_isaac = lambda a: a.T.reshape(12)  # noqa: E731
    return to_isaac(pos), to_isaac(vel), root


def run(controllers, steps: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    torques, contacts = [], []
    command = torch.tensor([[0.4, 0.0, 0.1]], device=device)
    for step in range(steps):
        if step in FOOTSTEPS:
            leg, duration = FOOTSTEPS[step]
            controllers.initiate_footsteps(
                active=torch.tensor([True], device=device),
                leg=torch.tensor([leg], device=device),
                location_hip=torch.tensor([[0.12, 0.09 if leg in (0, 2) else -0.09]], device=device),
                duration=torch.tensor([duration], device=device),
            )
        pos, vel, root = scripted_state(step)
        tensor = lambda a: torch.tensor(a, dtype=torch.float32, device=device).view(1, -1)  # noqa: E731
        torques.append(
            controllers.compute_torques(tensor(pos), tensor(vel), tensor(root), command)
            .cpu()
            .numpy()[0]
        )
        contacts.append(controllers.contact_state().cpu().numpy()[0])
    return np.array(torques), np.array(contacts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cpu = PooledRobotControllers(1, "cpu", DT, ITERATIONS_BETWEEN_MPC)
    try:
        cpu_torques, cpu_contacts = run(cpu, args.steps, "cpu")
    finally:
        cpu.close()
    peak = np.abs(cpu_torques).max()
    print(f"CPU controller: {args.steps} steps of {DT} s, peak |torque| {peak:.2f} N m")
    print(f"{'GPU controller':38s} {'max |dtau|':>10s} {'(% peak)':>9s} {'median':>8s} "
          f"{'mean':>8s} {'contact agree':>14s}")
    cases = [
        ("gaitnet", 80, False),
        ("gaitnet", 400, False),
        ("gaitnet", 400, True),
        ("gaitnet_core", 80, False),
    ]
    for variant, iterations, double in cases:
        gpu = GpuRobotControllers(
            1,
            args.device,
            DT,
            ITERATIONS_BETWEEN_MPC,
            GpuMpcCfg(variant=variant, solver_iterations=iterations, double_precision=double),
        )
        gpu_torques, gpu_contacts = run(gpu, args.steps, args.device)
        step_error = np.abs(gpu_torques - cpu_torques).max(axis=1)
        agree = (gpu_contacts == cpu_contacts).all(axis=1).mean()
        name = f"{variant}, {iterations} ADMM it, {'float64' if double else 'float32'}"
        print(
            f"{name:38s} {step_error.max():10.3f} {100 * step_error.max() / peak:8.1f}% "
            f"{np.median(step_error):8.3f} {step_error.mean():8.3f} {100 * agree:13.1f}%"
        )
        if variant == "gaitnet" and iterations == 80:
            worst = np.argsort(step_error)[::-1][:5]
            print(f"  worst steps {worst.tolist()}: {np.round(step_error[worst], 3).tolist()} N m")


if __name__ == "__main__":
    main()

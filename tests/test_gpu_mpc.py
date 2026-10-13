"""The GPU controller path: defaults reproduce the controllers it stands in for, knobs act.

Runs on CPU torch without Isaac Lab (the CPU-pool comparison needs the compiled
gaitnet.control.mpc_osqp extension, which the training venv has):

    python -m pytest -q tests/test_gpu_mpc.py
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from gaitnet_core.control import AdmmSettings, BatchedMpcConfig, BatchedMpcController, MpcSettings
from gaitnet_core.interfaces import FootstepCommand

from gaitnet.sim2real import compare_controllers as cc
from gaitnet.sim2real.gpu_mpc import VARIANTS, GaitnetMpcController
from gaitnet.sim2real.robot_controllers import (
    ControllerRandomizationCfg,
    GpuMpcCfg,
    GpuRobotControllers,
    PooledRobotControllers,
)

DT = cc.DT
N = 3


def controller_inputs(step: int, num: int = N):
    """Controller-convention inputs: leg-major joints, xyzw pose, world twist."""
    pos, vel, root = cc.scripted_state(step)
    leg_major = lambda a: a.reshape(3, 4).T.reshape(12)  # noqa: E731
    pose = np.concatenate([root[:3], root[[4, 5, 6, 3]]])
    rows = lambda a: torch.tensor(a, dtype=torch.float32).repeat(num, 1)  # noqa: E731
    # make the robots differ a little so a batching bug would show
    spread = torch.linspace(0.0, 0.02, num).unsqueeze(-1)
    return (
        rows(leg_major(pos)) + spread,
        rows(leg_major(vel)),
        rows(pose),
        rows(root[7:13]),
        rows([0.4, 0.0, 0.1]),
    )


def footstep(step: int, num: int = N) -> FootstepCommand | None:
    if step not in cc.FOOTSTEPS:
        return None
    leg, duration = cc.FOOTSTEPS[step]
    command = FootstepCommand.none(num)
    command.active[:] = True
    command.active[1] = False  # robot 1 never steps on this tick
    command.leg[:] = leg
    command.target[:] = torch.tensor([0.12, 0.09 if leg in (0, 2) else -0.09, -0.28])
    command.duration[:] = duration
    return command


def drive(controllers: list, steps: int = 90) -> list[list[torch.Tensor]]:
    out: list[list[torch.Tensor]] = [[] for _ in controllers]
    for step in range(steps):
        command = footstep(step)
        for record, controller in zip(out, controllers):
            if command is not None:
                controller.command_footsteps(dataclasses.replace(command))
            record.append(controller.compute_torques(*controller_inputs(step)))
            record.append(controller._gait.in_contact().float())
    return out


def config(iterations: int = 80) -> BatchedMpcConfig:
    return BatchedMpcConfig(
        mpc=MpcSettings(control_steps_per_update=5, admm=AdmmSettings(iterations=iterations))
    )


def test_default_knobs_reproduce_gaitnet_core_exactly():
    plain = BatchedMpcController(config(), N, DT, "cpu")
    knobbed = GaitnetMpcController(config(), VARIANTS["gaitnet_core"], N, DT, "cpu")
    rng_before = torch.random.get_rng_state()
    reference, ours = drive([plain, knobbed])
    for a, b in zip(reference, ours):
        assert torch.equal(a, b)
    assert torch.equal(rng_before, torch.random.get_rng_state())


def test_gaitnet_variant_matches_the_cpu_controller_at_convergence():
    cpu = PooledRobotControllers(1, "cpu", DT, 5)
    try:
        cpu_torques, cpu_contacts = cc.run(cpu, 120, "cpu")
    finally:
        cpu.close()
    gpu = GpuRobotControllers(1, "cpu", DT, 5, GpuMpcCfg(solver_iterations=400))
    gpu_torques, gpu_contacts = cc.run(gpu, 120, "cpu")
    assert np.abs(gpu_torques - cpu_torques).max() < 0.2  # N m, peak ~22
    assert (gpu_contacts == cpu_contacts).all()


def run_knobbed(set_knobs, steps: int = 90) -> tuple[GaitnetMpcController, list[torch.Tensor]]:
    controller = GaitnetMpcController(config(), VARIANTS["gaitnet"], N, DT, "cpu")
    set_knobs(controller.knobs)
    (record,) = drive([controller], steps)
    return controller, record


def test_knobs_only_change_the_robots_they_are_set_for():
    _, base = run_knobbed(lambda k: None)

    def set_robot_2(knobs):
        knobs.command_tracking[2] = 0.0
        knobs.planar_weight_scale[2] = 0.3
        knobs.swing_apex[2] = 0.12
        knobs.touchdown_bias[2] = 0.02

    _, changed = run_knobbed(set_robot_2)
    torques = [i for i in range(0, len(base), 2)]
    assert all(torch.equal(base[i][:2], changed[i][:2]) for i in torques)
    assert max((base[i][2] - changed[i][2]).abs().max() for i in torques) > 1.0


def test_zero_command_tracking_is_a_zero_command():
    def no_tracking(knobs):
        knobs.command_tracking[:] = 0.0

    _, tracked_zero = run_knobbed(no_tracking)

    controller = GaitnetMpcController(config(), VARIANTS["gaitnet"], N, DT, "cpu")
    record = []
    for step in range(90):
        command = footstep(step)
        if command is not None:
            controller.command_footsteps(command)
        *state, velocity = controller_inputs(step)
        record.append(controller.compute_torques(*state, torch.zeros_like(velocity)))
    for a, b in zip(tracked_zero[::2], record):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-5)


def test_swing_apex_raises_the_swing():
    heights = {}
    for apex in (0.07, 0.12):
        controller = GaitnetMpcController(config(), VARIANTS["gaitnet"], N, DT, "cpu")
        controller.knobs.swing_apex[:] = apex
        command = FootstepCommand.none(N)
        command.active[:] = True
        command.duration[:] = 0.2
        command.target[:] = torch.tensor([0.12, 0.09, -0.28])
        controller.command_footsteps(command)
        for step in range(26):  # half of the 50-step swing
            controller.compute_torques(*controller_inputs(step))
        heights[apex] = controller._foot_target[:, 0, 2]
    rise = heights[0.12] - heights[0.07]
    torch.testing.assert_close(rise, torch.full_like(rise, 0.05), atol=2e-3, rtol=0)


def test_touchdown_bias_and_noise_shift_the_target_not_the_command():
    controller = GaitnetMpcController(config(), VARIANTS["gaitnet"], N, DT, "cpu")
    controller.knobs.touchdown_bias[1, 0] = torch.tensor([0.03, -0.01])
    controller.knobs.touchdown_noise[2] = 0.05
    controller.knobs.noise_enabled = True
    controller.generator = torch.Generator().manual_seed(0)
    command = FootstepCommand.none(N)
    command.active[:] = True
    command.duration[:] = 0.2
    command.target[:] = torch.tensor([0.12, 0.09, -0.28])
    controller.command_footsteps(command)
    assert torch.equal(controller.commanded_foothold[:, 0], command.target[:, :2])
    offsets = controller._footstep_offsets[:, 0, :2] - command.target[:, :2]
    assert offsets[0].abs().max() == 0
    torch.testing.assert_close(offsets[1], torch.tensor([0.03, -0.01]))
    assert offsets[2].abs().max() > 0


def gpu_controllers(randomization: ControllerRandomizationCfg) -> GpuRobotControllers:
    return GpuRobotControllers(
        N, "cpu", DT, 5, GpuMpcCfg(randomization=randomization, seed=1)
    )


def isaac_state(step: int):
    pos, vel, root = cc.scripted_state(step)
    rows = lambda a: torch.tensor(a, dtype=torch.float32).repeat(N, 1)  # noqa: E731
    return rows(pos), rows(vel), rows(root), rows([0.4, 0.0, 0.1])


def tick(controllers: GpuRobotControllers, step: int, legs: list[int] | None) -> None:
    """One policy tick: footsteps (None = no step), then ten physics steps."""
    active = torch.tensor([legs is not None] * N)
    controllers.initiate_footsteps(
        active,
        torch.tensor(legs if legs is not None else [0] * N),
        torch.tensor([[0.12, 0.09]] * N),
        torch.full((N,), 0.2013),
    )
    for physics in range(10):
        controllers.compute_torques(*isaac_state(10 * step + physics))


def test_liftoff_delay_holds_the_command_and_hides_the_leg():
    delayed = gpu_controllers(ControllerRandomizationCfg(liftoff_delay_ticks=(2, 2)))
    tick(delayed, 0, [0, 1, 2])
    scheduled = delayed.controller.contact_state()
    assert scheduled.all()  # nothing has lifted yet
    assert not delayed.contact_state()[torch.arange(N), torch.tensor([0, 1, 2])].any()
    tick(delayed, 1, None)
    assert delayed.controller.contact_state().all()
    tick(delayed, 2, None)  # fires now
    scheduled = delayed.controller.contact_state()
    assert not scheduled[torch.arange(N), torch.tensor([0, 1, 2])].any()
    assert torch.equal(delayed.contact_state(), scheduled)


def test_no_delay_starts_the_swing_on_the_tick():
    plain = gpu_controllers(ControllerRandomizationCfg())
    tick(plain, 0, [0, 1, 2])
    assert not plain.contact_state()[torch.arange(N), torch.tensor([0, 1, 2])].any()


def test_reset_restores_one_env_and_resamples_its_knobs():
    rnd = ControllerRandomizationCfg(
        command_tracking=(0.0, 1.0), swing_apex=(0.07, 0.12), touchdown_bias=0.03
    )
    controllers = gpu_controllers(rnd)
    knobs = controllers.controller.knobs
    assert ((knobs.swing_apex >= 0.07) & (knobs.swing_apex <= 0.12)).all()
    tick(controllers, 0, [0, 1, 2])
    before = {name: getattr(knobs, name).clone() for name in ("swing_apex", "command_tracking")}
    swing_start = controllers.controller._gait.swing_start.clone()

    controllers.reset(torch.tensor([1]))
    gait = controllers.controller._gait
    assert controllers.contact_state()[1].all()
    assert gait.swing_duration[1].abs().max() == 0 and gait.iteration[1] == 0
    assert torch.equal(gait.swing_start[[0, 2]], swing_start[[0, 2]])
    for name, value in before.items():
        assert torch.equal(getattr(knobs, name)[[0, 2]], value[[0, 2]])
        assert not torch.equal(getattr(knobs, name)[1], value[1])


def test_mix_draws_each_env_from_one_randomisation_or_the_other():
    other = ControllerRandomizationCfg(command_tracking=(0.2, 0.2), liftoff_delay_ticks=(4, 4))
    base = ControllerRandomizationCfg(command_tracking=(0.9, 0.9), liftoff_delay_ticks=(1, 1))
    every = gpu_controllers(dataclasses.replace(base, mix=other, mix_fraction=1.0))
    assert torch.allclose(every.controller.knobs.command_tracking, torch.tensor(0.2))
    assert (every.liftoff_delay == 4).all() and every._max_delay == 4
    never = gpu_controllers(dataclasses.replace(base, mix=other, mix_fraction=0.0))
    assert torch.allclose(never.controller.knobs.command_tracking, torch.tensor(0.9))
    assert (never.liftoff_delay == 1).all()
    half = gpu_controllers(dataclasses.replace(base, mix=other, mix_fraction=0.5))
    from_other = torch.isclose(half.controller.knobs.command_tracking, torch.tensor(0.2))
    assert torch.equal(from_other, half.liftoff_delay == 4)          # an env is one or the other, never a blend
    assert (from_other | torch.isclose(half.controller.knobs.command_tracking, torch.tensor(0.9))).all()


def test_travel_compensation_spans_this_controller_to_gaitnet_core():
    """alpha 0 is the gaitnet variant's target; alpha 1 takes the body's travel off it, as
    gaitnet_core does; in between is in between."""
    controller = GaitnetMpcController(config(), VARIANTS["gaitnet"], N, DT, "cpu")
    controller._gait.swing_duration[:] = 0.2
    velocity = torch.tensor([[0.2, 0.05, 0.0]] * N)
    position = torch.zeros(N, 3)
    controller.knobs.travel_compensation[:] = torch.tensor([0.0, 0.5, 1.0])
    targets = controller._touchdown_targets(velocity, position)
    plain = controller._hip_offsets + controller._footstep_offsets + position.unsqueeze(1)
    torch.testing.assert_close(targets[0, :, :2], plain[0, :, :2])   # z: this variant puts it on the stance plane
    shift = plain[:, :, :2] - targets[:, :, :2]
    torch.testing.assert_close(shift[2], torch.tensor([[0.04, 0.01]] * shift.shape[1]))
    torch.testing.assert_close(shift[1], shift[2] / 2)


def test_along_and_outboard_offsets_have_the_right_signs():
    rnd = ControllerRandomizationCfg(touchdown_along=(-0.04, -0.04), touchdown_outboard=(0.03, 0.03))
    bias = gpu_controllers(rnd).controller.knobs.touchdown_bias       # (N, FL FR RL RR, xy)
    torch.testing.assert_close(bias[..., 0], torch.full((N, 4), -0.04))
    torch.testing.assert_close(bias[..., 1], torch.tensor([[0.03, -0.03, 0.03, -0.03]] * N))

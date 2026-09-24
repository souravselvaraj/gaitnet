"""Teacher-student distillation (`gaitnet_sim.rl.distillation`) on a fake environment: the loss,
the DAgger rollout, checkpoints, the teacher's contract, and exporting the student.

Needs rsl_rl, and isaaclab's configclass for reading the teacher run's params; not the
simulator.
"""

from __future__ import annotations

import pytest
import torch
import yaml

pytest.importorskip("rsl_rl")
pytest.importorskip("isaaclab")

from tensordict import TensorDict  # noqa: E402

from gaitnet_core import action_layout  # noqa: E402
from gaitnet_core.action_layout import EnvAction  # noqa: E402
from gaitnet_core.candidates import Candidates  # noqa: E402
from gaitnet_core.features import DEFAULT_FEATURES, state_vector  # noqa: E402
from gaitnet_core.rounds import TickSelection  # noqa: E402
from gaitnet_core.state import RobotState  # noqa: E402
from gaitnet_sim.rl.distillation import CandidateDistillation, tick_kl  # noqa: E402
from gaitnet_sim.rl.model import GaitNetActor  # noqa: E402

N, L, K, T = 16, 4, 12, 6
TEACHER_NET = {"class_name": "CandidateScorer", "shared_sizes": [32, 32], "candidate_sizes": [16, 16], "trunk_sizes": [32, 32]}
STUDENT_NET = {"class_name": "CandidateScorer", "shared_sizes": [16], "candidate_sizes": [8], "trunk_sizes": [16]}


def make_obs(rounds: int = 2, seed: int | None = None, hidden: float = 0.3) -> TensorDict:
    """A tick as the `distill` preset gives it: the student's noisy state and candidates, and
    the teacher's true state and the same candidates judged on the truth (a subset stays
    valid, at other heights). `hidden` of the student's candidates are invalid in truth, at
    random: nothing the student sees tells it which."""
    generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def rand(*shape):
        return torch.rand(*shape, generator=generator)

    def state(noise: float) -> torch.Tensor:
        robot = RobotState(
            foot_pos=(rand(N, L, 3) - 0.5) * 0.1 + noise * (rand(N, L, 3) - 0.5),
            foot_vel=torch.zeros(N, L, 3),
            base_lin_vel=torch.zeros(N, 3),
            base_ang_vel=torch.zeros(N, 3),
            projected_gravity=torch.tensor([0.0, 0.0, -1.0]).expand(N, 3).clone(),
            contact=torch.ones(N, L, dtype=torch.bool),
            # every leg in stance, so two may lift in one tick
            gait_timing=torch.zeros(N, L, 3),
            command=torch.full((N, 3), 0.1),
            base_command=torch.full((N, 3), 0.1),
        )
        return state_vector(robot, DEFAULT_FEATURES)

    xyz = (rand(N, L, K, 3) - 0.5) * 0.2
    valid = rand(N, L, K) < 0.8
    student = Candidates(xyz=xyz, valid=valid, log_q=torch.zeros(N, L, K))
    truth_valid = valid & (rand(N, L, K) >= hidden)
    truth_xyz = torch.where(truth_valid.unsqueeze(-1), xyz + torch.tensor([0.0, 0.0, 0.02]), torch.zeros_like(xyz))
    teacher = Candidates(xyz=truth_xyz, valid=truth_valid, log_q=torch.zeros(N, L, K))
    torch.manual_seed(0 if seed is None else seed)
    return TensorDict(
        {
            "state": state(0.01),
            "candidates": student.pack(),
            "teacher_state": state(0.0),
            "teacher_candidates": teacher.pack(),
        },
        batch_size=[N],
    )


class FakeEnv:
    num_envs = N
    cfg: dict = {}

    def __init__(self, rounds: int = 2):
        self.num_actions = action_layout.dim(rounds)
        self.rounds = rounds

    def get_observations(self) -> TensorDict:
        return make_obs(self.rounds)


def teacher_actor(obs: TensorDict, rounds: int = 2, network: dict = TEACHER_NET) -> GaitNetActor:
    """A 'trained' teacher, as PPO builds its actor."""
    ppo_obs = TensorDict({"state": obs["teacher_state"], "candidates": obs["teacher_candidates"]}, batch_size=[N])
    return GaitNetActor(ppo_obs, {"actor": ["state"]}, "actor", action_layout.dim(rounds), network=dict(network), duration_std_floor=0.01)


def write_teacher_run(tmp_path, teacher: GaitNetActor, rounds: int = 2, features=DEFAULT_FEATURES, network: dict = TEACHER_NET):
    run = tmp_path / "teacher_run"
    (run / "params").mkdir(parents=True)
    env = {
        "gaitnet": {"robot": "go1", "max_steps_per_tick": rounds, "min_stance_after_step": 2},
        "observations": {"state": {"robot_state": {"params": {"features": list(features)}}}},
    }
    agent = {
        "obs_groups": {"actor": ["state"], "critic": ["state"]},
        "actor": {"network": dict(network), "duration_std_floor": 0.01, "state_features": list(features), "min_stance_after_step": 2},
    }
    (run / "params" / "env.yaml").write_text(yaml.dump(env))
    (run / "params" / "agent.yaml").write_text(yaml.dump(agent))
    torch.save({"actor_state_dict": teacher.state_dict(), "iter": 100, "infos": None}, run / "model_100.pt")
    return run / "model_100.pt"


def distill_cfg(checkpoint, **algorithm) -> dict:
    return {
        "algorithm": {
            "class_name": "gaitnet_sim.rl.distillation:CandidateDistillation",
            "teacher_checkpoint": str(checkpoint),
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
            "learning_rate": 3e-3,
            "teacher_action_prob": 0.5,
            "teacher_action_iterations": 4,
            **algorithm,
        },
        "student": {"class_name": "gaitnet_sim.rl.model:GaitNetActor", "network": dict(STUDENT_NET), "distribution_cfg": None, "duration_std_floor": 0.01},
        # the network is replaced by the teacher run's
        "teacher": {"class_name": "gaitnet_sim.rl.model:GaitNetActor", "network": {"class_name": "CandidateScorer"},
                    "candidates_group": "teacher_candidates", "distribution_cfg": None},
        "obs_groups": {"student": ["state"], "teacher": ["teacher_state"]},
        "num_steps_per_env": T,
        "multi_gpu": None,
    }


def make_distillation(tmp_path, rounds: int = 2, **algorithm) -> tuple[CandidateDistillation, GaitNetActor]:
    torch.manual_seed(0)
    obs = make_obs(rounds)
    teacher = teacher_actor(obs, rounds)
    checkpoint = write_teacher_run(tmp_path, teacher, rounds)
    alg = CandidateDistillation.construct_algorithm(obs, FakeEnv(rounds), distill_cfg(checkpoint, **algorithm), "cpu")
    return alg, teacher


def rollout(alg: CandidateDistillation, rounds: int = 2, seed: int | None = None, hidden: float = 0.3) -> list[torch.Tensor]:
    """T steps of acting; with a seed, every step sees the same tick."""
    executed = []
    obs = make_obs(rounds, seed, hidden)
    for _ in range(T):
        with torch.inference_mode():
            executed.append(alg.act(obs).clone())
            obs = make_obs(rounds, seed, hidden)
            alg.process_env_step(obs, torch.randn(N), torch.zeros(N, dtype=torch.bool), {})
    return executed


# --- the loss ---


def both(obs: TensorDict, student: GaitNetActor, teacher: GaitNetActor):
    with torch.no_grad():
        selection_actions = student(obs, stochastic_output=True)
    action = EnvAction.decode(selection_actions)
    selection = TickSelection(index=action.choice_index.long(), duration=action.duration)
    return teacher.tick_distribution(obs), student.tick_distribution(obs), selection


def test_the_kl_is_zero_for_a_student_that_is_the_teacher():
    torch.manual_seed(0)
    obs = make_obs()
    # both see the teacher's view: identical distributions
    same = TensorDict({"state": obs["teacher_state"], "candidates": obs["teacher_candidates"]}, batch_size=[N])
    actor = teacher_actor(obs)
    teacher, student, selection = both(same, actor, actor)
    categorical, duration = tick_kl(teacher, student, selection)
    assert torch.allclose(categorical, torch.zeros(N), atol=1e-5)
    assert torch.allclose(duration, torch.zeros(N), atol=1e-5)


def test_the_kl_is_positive_finite_and_trains_only_the_student():
    torch.manual_seed(0)
    obs = make_obs()
    teacher = teacher_actor(obs)
    teacher_view = TensorDict({"state": obs["teacher_state"], "candidates": obs["teacher_candidates"]}, batch_size=[N])
    student = GaitNetActor(obs, {"student": ["state"]}, "student", action_layout.dim(2), network=dict(STUDENT_NET), duration_std_floor=0.01)
    with torch.no_grad():
        t = teacher.tick_distribution(teacher_view)
    s = student.tick_distribution(obs)
    with torch.no_grad():
        action = EnvAction.decode(student(obs, stochastic_output=True))
    categorical, duration = tick_kl(t, s, TickSelection(index=action.choice_index.long(), duration=action.duration))
    assert torch.isfinite(categorical).all() and torch.isfinite(duration).all()
    assert (categorical > 0).all() and (duration >= 0).all()
    (categorical + duration).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_the_teacher_never_puts_mass_where_the_student_cannot_step():
    """Footholds valid only in the teacher's view (eligibility noise) are dropped from its
    target, and the KL stays finite."""
    torch.manual_seed(0)
    obs = make_obs()
    teacher = teacher_actor(obs)
    student_candidates = Candidates.unpack(obs["candidates"])
    student_candidates.valid[:, 0] = False  # the student sees nothing for leg 0
    obs["candidates"] = student_candidates.pack()
    view = TensorDict({"state": obs["teacher_state"], "candidates": obs["teacher_candidates"]}, batch_size=[N])
    view_candidates = Candidates.unpack(view["candidates"])
    view_candidates.valid[:, 0] = True  # the teacher would step leg 0
    view["candidates"] = view_candidates.pack()
    student = GaitNetActor(obs, {"student": ["state"]}, "student", action_layout.dim(2), network=dict(STUDENT_NET), duration_std_floor=0.01)
    with torch.no_grad():
        t = teacher.tick_distribution(view)
        action = EnvAction.decode(student(obs, stochastic_output=True))
    s = student.tick_distribution(obs)
    categorical, duration = tick_kl(t, s, TickSelection(index=action.choice_index.long(), duration=action.duration))
    assert torch.isfinite(categorical).all() and torch.isfinite(duration).all()


def test_later_rounds_count_only_while_the_robot_is_still_stepping():
    torch.manual_seed(0)
    obs = make_obs()
    view = TensorDict({"state": obs["teacher_state"], "candidates": obs["teacher_candidates"]}, batch_size=[N])
    teacher = teacher_actor(obs)
    student = GaitNetActor(obs, {"student": ["state"]}, "student", action_layout.dim(2), network=dict(STUDENT_NET), duration_std_floor=0.01)
    noop = Candidates.unpack(obs["candidates"]).noop_index
    # every robot holds: only the first round is a choice
    hold = TickSelection(index=torch.full((N, 2), noop), duration=torch.zeros(N, 2))
    with torch.no_grad():
        both_rounds = tick_kl(teacher.tick_distribution(view), student.tick_distribution(obs), hold)
        t, s = teacher.tick_distribution(view), student.tick_distribution(obs)
        t.log_prob(hold)
        s.log_prob(hold)
        from gaitnet_sim.rl.distillation import round_kl

        first = round_kl(t.path[0], s.path[0])
    assert torch.allclose(both_rounds[0], first[0]) and torch.allclose(both_rounds[1], first[1])


# --- the algorithm ---


def test_the_teacher_is_built_and_loaded_from_its_run(tmp_path):
    alg, trained = make_distillation(tmp_path)
    assert alg.teacher_loaded
    for (name, a), b in zip(alg.teacher.state_dict().items(), trained.state_dict().values()):
        assert torch.equal(a, b), name
    # the teacher's network is its run's, not the cfg's placeholder
    assert [m.out_features for m in alg.teacher.network.shared_encoder if hasattr(m, "out_features")] == [32, 32]
    assert not any(p.requires_grad for p in alg.teacher.parameters())
    assert alg.student.rounds == alg.teacher.rounds == 2


def test_dagger_executes_the_teacher_for_a_decaying_fraction(tmp_path):
    alg, _ = make_distillation(tmp_path, teacher_action_prob=1.0, teacher_action_iterations=2)
    obs = make_obs()
    with torch.inference_mode():
        executed = alg.act(obs)
        teacher = alg.teacher(obs)
    assert torch.equal(executed, teacher)
    alg.transition.clear()
    assert alg.beta == 1.0
    alg.num_updates = 1
    assert alg.beta == pytest.approx(0.5)
    alg.num_updates = 2
    assert alg.beta == 0.0
    with torch.inference_mode():
        executed = alg.act(obs)
    # the student's own choice, from its candidate set
    assert (EnvAction.decode(executed).choice_index <= Candidates.unpack(obs["candidates"]).noop_index).all()


def test_updates_are_finite_and_the_student_learns_the_teacher(tmp_path):
    alg, _ = make_distillation(tmp_path, num_learning_epochs=4, learning_rate=1e-2)
    kls = []
    for _ in range(12):
        # the same ticks every time, and a truth the student can infer, which it can fit
        rollout(alg, seed=1, hidden=0.0)
        stats = alg.update()
        assert all(torch.isfinite(torch.tensor(v)) for v in stats.values()), stats
        kls.append(stats["kl"])
    assert kls[-1] < 0.5 * kls[0], kls
    assert 0.0 <= stats["agreement_gate"] <= 1.0 and stats["teacher_action_prob"] == 0.0
    assert stats["teacher_executed"] == 0.0


def test_one_footstep_per_tick(tmp_path):
    alg, _ = make_distillation(tmp_path, rounds=1)
    rollout(alg, rounds=1)
    stats = alg.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values()), stats


def test_a_checkpoint_resumes_student_teacher_and_schedule(tmp_path):
    alg, _ = make_distillation(tmp_path)
    rollout(alg)
    alg.update()
    saved = alg.save()
    assert saved["distillation_updates"] == 1

    fresh, _ = make_distillation(tmp_path / "again")
    assert fresh.load(saved, None, strict=True)  # the iteration is restored too
    assert fresh.num_updates == 1
    for a, b in zip(fresh.student.state_dict().values(), alg.student.state_dict().values()):
        assert torch.equal(a, b)


def test_refuses_a_teacher_with_another_contract(tmp_path):
    obs = make_obs()
    # trained with one footstep per tick, distilled with two
    checkpoint = write_teacher_run(tmp_path, teacher_actor(obs, rounds=1), rounds=1)
    with pytest.raises(ValueError, match="footsteps per tick"):
        CandidateDistillation.construct_algorithm(obs, FakeEnv(2), distill_cfg(checkpoint), "cpu")

    class EnvWithFeatures(FakeEnv):
        class cfg:
            class observations:
                class teacher_state:
                    class robot_state:
                        params = {"features": ["foot_pos", "command"]}

    checkpoint = write_teacher_run(tmp_path / "b", teacher_actor(obs))
    with pytest.raises(ValueError, match="features"):
        CandidateDistillation.construct_algorithm(obs, EnvWithFeatures(2), distill_cfg(checkpoint), "cpu")
    with pytest.raises(ValueError, match="teacher_checkpoint"):
        CandidateDistillation.construct_algorithm(obs, FakeEnv(2), distill_cfg(""), "cpu")


# --- export ---


def test_a_distillation_run_exports_its_student(tmp_path):
    from gaitnet_sim.rl.export import bundle_from_run

    alg, _ = make_distillation(tmp_path)
    run = tmp_path / "student_run"
    (run / "params").mkdir(parents=True)
    env = {
        "gaitnet": {"robot": "go1", "max_steps_per_tick": 2, "min_stance_after_step": 2},
        "observations": {
            "state": {"robot_state": {"params": {"features": list(DEFAULT_FEATURES)}}},
            "candidates": {"candidates": {"params": {"sampler": "uniform_jitter", "sampler_kwargs": {"per_leg": K}}}},
        },
    }
    agent = {
        "obs_groups": {"student": ["state"], "teacher": ["teacher_state"]},
        "student": {"network": dict(STUDENT_NET), "duration_std_floor": 0.01, "state_features": list(DEFAULT_FEATURES),
                    "min_stance_after_step": 2, "candidates_group": "candidates"},
        "teacher": {"network": dict(TEACHER_NET), "candidates_group": "teacher_candidates"},
    }
    (run / "params" / "env.yaml").write_text(yaml.dump(env))
    (run / "params" / "agent.yaml").write_text(yaml.dump(agent))
    torch.save({**alg.save(), "iter": 7, "infos": None}, run / "model_7.pt")

    bundle = bundle_from_run(run)
    assert bundle.extra["policy"] == "student" and bundle.extra["iteration"] == 7
    assert bundle.duration_std == pytest.approx(alg.student.duration_std.item())
    obs = make_obs()
    candidates = Candidates.unpack(obs["candidates"])
    with torch.no_grad():
        expected = alg.student.network(obs["state"], candidates)
        got = bundle.actor(obs["state"], candidates)
    assert torch.equal(expected.step_logits, got.step_logits) and torch.equal(expected.duration, got.duration)

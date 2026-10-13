"""DAgger distillation (distill.py): the loss terms, and a full DistillationPPO update loop on an
rsl-rl rollout storage showing the student moves towards the teacher.

    python -m pytest -q tests/test_distill.py
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

import gaitnet.constants as const
from gaitnet.gaitnet.components import privileged
from gaitnet.gaitnet.components.policy_history import HISTORY_DIM
from gaitnet.gaitnet.distill import DistillationPPO, distillation_terms, mask_option_logits
from gaitnet.gaitnet.gaitnet import NUM_OPTIONS, GaitnetActorCritic
from gaitnet.gaitnet.gaitnet_t import GaitnetTActor
from gaitnet.gaitnet.teacher import PrivilegedActor
from tests.test_gaitnet_t import observation, old_actor
from tests.test_policy_history import critic

T = 25
STUDENT_DIM = const.gait_net.robot_state_dim + NUM_OPTIONS * 8 + T * HISTORY_DIM


def env_obs(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 100)
    base = observation(n, padded=5, history=T * HISTORY_DIM, seed=seed)
    return torch.cat([base, torch.randn(n, privileged.privileged_dim(NUM_OPTIONS), generator=g)], dim=1)


def test_terms():
    logits = torch.randn(4, NUM_OPTIONS)
    obs = env_obs(4)
    masked = mask_option_logits(logits, obs)
    assert torch.isinf(masked[:, :5]).all() and torch.isfinite(masked[:, 5:]).all()
    duration = torch.rand(4, NUM_OPTIONS)
    same = distillation_terms(masked, duration, masked, duration)
    assert same["kl"].abs().max() < 1e-6 and same["duration_mse"].max() == 0 and same["agreement"].min() == 1
    other = distillation_terms(masked, duration, mask_option_logits(torch.randn(4, NUM_OPTIONS), obs), duration + 0.1)
    assert (other["kl"] > 0).all() and (other["ce"] >= other["kl"]).all() and torch.isfinite(other["ce"]).all()
    torch.testing.assert_close(other["duration_mse"], torch.full((4,), 0.01))


def teacher() -> PrivilegedActor:
    torch.manual_seed(0)
    t = PrivilegedActor.widen(GaitnetTActor.from_gaitnet_actor(old_actor(True), history_len=T, history_dim=HISTORY_DIM))
    with torch.no_grad():  # make the privileged inputs matter, as a trained teacher's would
        t.inner.shared_encoder[0].weight[:, const.gait_net.robot_state_dim :].normal_(0, 0.3)
        t.inner.unique_encoder[0].weight[:, 8:].normal_(0, 0.3)
    return t


@pytest.mark.parametrize("mode", ["dagger_bc", "dagger_ppo"])
def test_student_moves_towards_the_teacher(mode):
    student = GaitnetTActor.from_gaitnet_actor(old_actor(True), history_len=T, history_dim=HISTORY_DIM)
    policy = GaitnetActorCritic(0, 0, 2, actor=student, critic=critic(), actor_obs_dim=STUDENT_DIM, actor_chunk_size=16)
    alg = DistillationPPO(
        policy,
        teacher=teacher(),
        distill_mode=mode,
        beta_iterations=6,
        num_learning_epochs=2,
        num_mini_batches=2,
        learning_rate=1e-3,
        schedule="fixed",
        device="cpu",
    )
    envs, steps = 16, 4
    alg.init_storage("rl", envs, steps, TensorDict({"policy": env_obs(envs)}, batch_size=[envs]), [2])
    kls, betas = [], []
    for iteration in range(6):
        for step in range(steps):
            obs = TensorDict({"policy": env_obs(envs, seed=step)}, batch_size=[envs])
            alg.act(obs)
            alg.process_env_step(obs, torch.zeros(envs), torch.zeros(envs, dtype=torch.long), {})
        alg.compute_returns(obs)
        losses = alg.update()
        kls.append(losses["distill_kl"])
        betas.append(losses["distill_beta"])
    # (dagger_ppo: while beta > 0; it is 1/6 on the last update here)
    assert kls[-1] < 0.5 * kls[0], kls
    if mode == "dagger_ppo":
        assert betas == pytest.approx([1.0, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6]) and "surrogate" in losses
    else:
        assert set(betas) == {1.0} and "surrogate" not in losses

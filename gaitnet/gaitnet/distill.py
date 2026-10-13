"""DAgger distillation of the privileged GaitNet-T teacher (teacher.py) into a deployable GaitNet-T.

The student is a GaitNet-T with history and without privileged inputs: it reads the observation
up to the end of the history ([state | candidates | history]); the environment still appends the
teacher's privileged features after that. Rollouts are driven by the STUDENT's actions (DAgger),
so the states are the student's own; in every PPO minibatch the frozen teacher is evaluated on
the same stored observations, with its privileged inputs and the same candidate set, and the
student is pulled towards it:

- dagger_bc:  loss = KL(teacher || student) on the masked categorical over the candidates (the
              cross-entropy with soft labels, up to the teacher's entropy) + duration_coef * MSE of
              the swing duration at the teacher's argmax option. No reward or value learning; the
              runner still collects and logs rewards and episodes.
- dagger_ppo: the PPO loss + beta * (the same distillation loss), beta annealed linearly from 1
              to 0 over the first `beta_iterations` updates.

`DistillationPPO` replaces rsl-rl's PPO.update for this repository's configuration only (fixed
learning rate schedule, no RND, no symmetry, one GPU, a feed-forward policy), and asserts that.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO

from gaitnet.gaitnet.gaitnet import candidate_block

MODES = ("dagger_bc", "dagger_ppo")


def mask_option_logits(logits: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
    """GaitnetActorCritic.update_distribution's masking: padding no-ops get -inf, the real no-op
    (last slot) stays."""
    no_op = candidate_block(observations, logits.shape[1])[:, :, 0] == 1
    masked = logits.masked_fill(no_op, float("-inf"))
    masked[:, -1] = logits[:, -1]
    return masked


def chunked(actor: nn.Module, observations: torch.Tensor, chunk: int) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = [actor(part) for part in observations.split(chunk)]
    return torch.cat([o[0] for o in outputs]), torch.cat([o[1] for o in outputs])


def distillation_terms(
    teacher_logits: torch.Tensor,
    teacher_duration: torch.Tensor,
    student_logits: torch.Tensor,
    student_duration: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """KL(teacher || student) and cross-entropy per sample, over masked logits (-inf = masked),
    and the duration MSE at the teacher's argmax."""
    log_t = torch.log_softmax(teacher_logits, dim=-1)
    log_s = torch.log_softmax(student_logits, dim=-1)
    p_t = log_t.exp()
    valid = p_t > 0
    zero = torch.zeros_like(p_t)
    kl = torch.where(valid, p_t * (log_t - log_s), zero).sum(dim=-1)
    ce = torch.where(valid, -p_t * log_s, zero).sum(dim=-1)
    best = teacher_logits.argmax(dim=-1, keepdim=True)
    duration_mse = (student_duration.gather(1, best) - teacher_duration.gather(1, best)).squeeze(1).pow(2)
    agreement = (student_logits.argmax(dim=-1, keepdim=True) == best).float().squeeze(1)
    return {"kl": kl, "ce": ce, "duration_mse": duration_mse, "agreement": agreement}


class DistillationPPO(PPO):
    def __init__(
        self,
        policy,
        *,
        teacher: nn.Module,
        distill_mode: str,
        beta_iterations: int = 200,
        duration_coef: float = 1.0,
        teacher_chunk: int = 16384,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        assert distill_mode in MODES, distill_mode
        assert self.rnd is None and self.symmetry is None and not self.is_multi_gpu
        assert not policy.is_recurrent and self.schedule == "fixed"
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.distill_mode = distill_mode
        self.beta_iterations = beta_iterations
        self.duration_coef = duration_coef
        self.teacher_chunk = teacher_chunk
        self.distill_updates = 0

    @property
    def beta(self) -> float:
        if self.distill_mode == "dagger_bc":
            return 1.0
        return max(0.0, 1.0 - self.distill_updates / self.beta_iterations)

    def update(self) -> dict[str, float]:
        beta = self.beta
        sums: dict[str, float] = defaultdict(float)
        batches = 0
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            _old_mu,
            _old_sigma,
            hidden_states_batch,
            masks_batch,
        ) in generator:
            observations = obs_batch["policy"]
            with torch.no_grad():
                t_logits, t_duration = chunked(self.teacher, observations, self.teacher_chunk)
                t_logits = mask_option_logits(t_logits, observations)

            # the student's (masked) distribution and durations on the same states
            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            s_logits = self.policy.discrete_distribution.logits
            s_duration = self.policy._cached_duration_means
            terms = distillation_terms(t_logits, t_duration, s_logits, s_duration)
            distill_loss = terms["kl"].mean() + self.duration_coef * terms["duration_mse"].mean()
            entropy = self.policy.entropy

            if self.distill_mode == "dagger_bc":
                loss = distill_loss
            else:
                log_prob = self.policy.get_actions_log_prob(actions_batch)
                values = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1])
                if self.normalize_advantage_per_mini_batch:
                    with torch.no_grad():
                        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
                ratio = torch.exp(log_prob - torch.squeeze(old_actions_log_prob_batch))
                advantages = torch.squeeze(advantages_batch)
                surrogate = torch.max(
                    -advantages * ratio,
                    -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param),
                ).mean()
                if self.use_clipped_value_loss:
                    clipped = target_values_batch + (values - target_values_batch).clamp(-self.clip_param, self.clip_param)
                    value_loss = torch.max((values - returns_batch).pow(2), (clipped - returns_batch).pow(2)).mean()
                else:
                    value_loss = (returns_batch - values).pow(2).mean()
                loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
                loss = loss + beta * distill_loss
                sums["surrogate"] += surrogate.item()
                sums["value_function"] += value_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            sums["entropy"] += entropy.mean().item()
            sums["distill_kl"] += terms["kl"].mean().item()
            sums["distill_ce"] += terms["ce"].mean().item()
            sums["distill_duration_mse"] += terms["duration_mse"].mean().item()
            sums["teacher_agreement"] += terms["agreement"].mean().item()
            batches += 1

        self.storage.clear()
        self.distill_updates += 1
        losses = {key: value / max(batches, 1) for key, value in sums.items()}
        losses["distill_beta"] = beta
        return losses

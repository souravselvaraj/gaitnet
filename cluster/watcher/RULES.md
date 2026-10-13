# Training watcher: decision rules

Written 2026-10-12, before the first watched decision. The watcher (`watch.sh`) wakes Claude; Claude
appends to `decisions.md` before acting on anything.

## Runs watched

| job | run | arch / recipe | rollout |
| --- | --- | --- | --- |
| 2361793 | WB500-walk-track | GaitNet / step | 500 steps x 4 minibatches (control) |
| 2361789 | WB-walk-fast-track | GaitNet / step | 100 x 8 |
| 2361790 | WC-walk-fast-track | GaitNet-T / step | 100 x 8 |
| 2361791 | WD-walk-fast-track | GaitNet-T teacher | 100 x 8 |
| 2361792 | WDc-walk-fast-track-critic | GaitNet-T teacher + critic tail + no-op fix | 100 x 8 |

All: walk curriculum, tracking std 0.1 m/s / yaw 0.2 rad/s, resumed from random-sampling_1035.pt.

From scratch (added 2026-10-12 05:12 at the user's request, run alongside the fine-tunes until the
evals say which family to keep): same settings, no `--resume`, and `--critic_noop_fix` on all five
(no checkpoint to stay compatible with); the critic tail is SDc's only difference from SD.

| job | run | arch / recipe | rollout |
| --- | --- | --- | --- |
| 2361957 | SB500-scratch | GaitNet / step | 500 x 4 (control), 780 iterations |
| 2361958 | SB-scratch-fast | GaitNet / step | 100 x 8, 3400 iterations |
| 2361959 | SC-scratch-fast | GaitNet-T / step | 100 x 8, 2500 iterations |
| 2361960 | SD-scratch-fast | GaitNet-T teacher | 100 x 8, 2400 iterations |
| 2361961 | SDc-scratch-fast-critic | GaitNet-T teacher + critic tail | 100 x 8, 2400 iterations |

Scratch runs start from a random policy, so rule 5's "gain over 1035" is the bar they must clear
eventually, not in their first evals; judge them on their own trend for the first ~6 h.
Choosing between the fine-tune and scratch families is the user's call once both have evals.

## Instruments

- `health.py` every 10 min: job state, last iteration, log staleness, NaNs, last-20-iteration
  means of falls by cause, walked fraction, terrain level.
- `eval_policy.py` every ~2 h, latest checkpoint of every run: deterministic, per void fraction
  (0-0.4) x speed (0.1, 0.2 m/s), 150 robots per cell, seeded terrain. Compared with the 1035
  baseline and the previous eval. **Main metric: success (survived and walked >= half the commanded
  distance) at voids 0.1-0.3**, then survival and distance fraction.

## Wake immediately

- a job is gone without `GAITNET-TRAIN-DONE` in its log (crash, time limit, node failure);
- no new iteration logged for 20 min while RUNNING;
- NaN/inf in any loss;
- tipping (`bad_orientation`) > 0.30 averaged over the last 20 iterations in any run.

## What Claude may do without asking

1. **Crash / stall / time limit:** relaunch the same command, resumed from that run's latest
   checkpoint (`--resume <run>/model_<last>.pt`), same run name with a `-r<k>` suffix.
2. **Update schedule:** if tipping > 0.30 (last 20 iterations) in fast runs while WB500 stays
   < 0.05 after a comparable number of policy updates, the schedule is the suspect: restart those
   runs from their latest checkpoint with `--num_mini_batches 4`.
3. **Reward-caused tipping:** if WB500 tips too (> 0.15), add the teacher recipe's mild stability
   terms (lin_vel_z_l2 -0.5, ang_vel_xy_l2 -0.05, flat_orientation_l2 -2.0) to the step runs, all
   step runs including WB500 together, restart from 1035.
4. **Curriculum:** if the mean terrain level sinks below 0.5 and stays there for 2 h, lower the
   walk bar (`required_fraction` 0.3 -> 0.2) or `p_down_given_failure` (0.5 -> 0.3), all runs
   together, restart from latest checkpoints.
5. **Not learning:** only after a run is >= 2 h old, and only when **two consecutive evals >= 4 h
   apart** show no gain in success at voids 0.1-0.3 over both the previous eval and 1035 (beyond
   the Wilson intervals). First response: rules 2-4 if they apply; otherwise stop that run to free
   its GPU and wait for the user.

Always: one change per restart; WB500 gets every reward/curriculum change the others get;
never cancel jobs that are not these (or their relaunches); never touch other users' jobs.

## Waits for the user

Reward redesign beyond rule 3, architecture changes, new arms, anything touching the deployed
stack (gait-net repo, Gazebo), and deleting any run or checkpoint.

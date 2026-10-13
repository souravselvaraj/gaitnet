# SC (GaitNet-T @645) falls on Gazebo/OCS2 but walks in Isaac: root-cause report

2026-10-12, written 23:40-23:50 UTC. Analysis only: no repository code was changed. Rates are given as k/n with Wilson 95 % CIs.
Scripts and outputs are in `/scratch/sselvaraj/gaitnet-runs/idm/inv_fast/report/` (`tab.py`, `tab_all.csv`, `flat_gt_all.py`, `gt_all.csv/.txt`, `isaac_tab.py`).

## 1. Answer in one table

| Cause | Evidence | Share of the gap | Confidence |
|---|---|---|---|
| **A. SC is fragile to OCS2-like landing errors and lift-off delay (policy fragility)** | Isaac flat @0.2, controller matched to OCS2's best landing (along -1 cm, noise 4 cm, travel comp, 2-tick delay): SC@645 success **1.00 → 0.37 [0.30,0.45]**. WB500@1160 stays at 0.94, SCagn@1365 reaches 0.78 (n=150/cell). It is the **interaction**: delay alone 1.00, landing error alone 0.86. | Most of the collapse at 0.2 | high |
| **B. OCS2 execution/reference is unsafe at ≥ 0.15 m/s whatever the policy (platform)** | OCS2's own trot (absref15, no GaitNet) survives only **2/10 at 0.2** and 7/10 at 0.15, against 20/20 at 0.05-0.1. Its falls are rollovers (final roll ≈ -3 rad), not SC's nose-down. WB500 also survives 0/34 at 0.15-0.2 in Gazebo. | Every 0.15/0.2 cell is confounded | high (n=10/speed) |
| **C. Residual at 0.05-0.1 m/s that landing/delay does not explain** | Isaac-matched SC walks at 0.1 (success 0.82 [0.75,0.87], n=150), but every Gazebo SC arm survives only 0-3/10 at 0.1, while OCS2 trot survives 10/10 there. A gap of about 0.6-0.7 is left unexplained. Candidates: body-rate noise (yaw-rate sd 0.67 vs 0.07 rad/s), schedule→lift-off latency (45-56 ms, foot up about 75 ms after scheduling), +6 cm base_z, irregular 22 Hz history. | About 0.6-0.7 at 0.1 | medium (cause not isolated) |
| D. SC's concurrent and non-diagonal swings (gait OCS2 cannot execute) | Capping to 1 swing or to diagonal pairs only gives 8/40 for each, against 5/40 at baseline. That is +7.5 pp and not significant. All SC arms are still 0/10 at 0.2. | Secondary at most | medium |
| E. Input/decision mismatch | Analyst "obs": SC's dx/dy, duration, concurrency and no-op rate match Isaac within 10 %. The step rate is 8-14 % lower. Non-diagonal concurrent starts are 2x (18.6 % vs 9.2 % at 0.2). | Small, untested | medium (unverified) |

Bottom line:
- SC is **really fragile to landing error combined with lift-off delay**: in Isaac the same errors that WB500 shrugs off (0.94) drop SC to 0.37. Delay alone costs nothing (1.00) and landing error alone costs a little (0.86). This is the main SC-specific cause.
- On top of that, **the OCS2 setup itself fails at ≥ 0.15 m/s** (its own trot reaches 2/10 at 0.2).
- **Something else in OCS2's closed loop still kills SC at 0.1**, where Isaac-matched SC walks.

So "OCS2 cannot execute SC's gait" is not supported as the main cause: restricting the gait barely helped.

## 2. Quick experiments (launched 23:27 UTC)

### 2a. Gazebo sweep gzinv-fast (job 2366068, COMPLETED 23:30), flat, seeds 51-60, 20 s

Dir `/scratch/sselvaraj/gaitnet-runs/idm/inv_fast_1`. Survival = success for every arm. The feet and vx columns are ground truth from 1 s until termination (`flat_gt.py` with the no-response filter removed for OCS2 trot).

| Arm | n | Survival [95 % CI] | 0.05 / 0.1 / 0.15 / 0.2 | Median t_end (s) | Front feet vs hip x (m) | Rear (m) | vx (m/s) | Pitch (rad) | Steps/s |
|---|---|---|---|---|---|---|---|---|---|
| sc_ocs2match (= flat7 sc_t_lead04, reference) | 40 | 5/40 = 0.125 [0.055,0.261] | 4/1/0/0 | 10.3 | -0.046 | -0.040 | 0.119 | 0.022 | 7.75 |
| sc_ocs2match_single (max 1 swing) | 40 | 8/40 = 0.200 [0.105,0.348] | 6/2/0/0 | 10.9 | **-0.112** | -0.026 | **0.085** | 0.022 | **3.98** |
| sc_ocs2match_diag (diagonal pairs only) | 40 | 8/40 = 0.200 [0.105,0.348] | 7/1/0/0 | 10.7 | -0.047 | -0.040 | 0.125 | 0.023 | 7.47 |
| scagn1300_ocs2match (flat8, reference) | 40 | 5/40 = 0.125 [0.055,0.261] | 5/0/0/0 | 10.0 | +0.008 | -0.018 | 0.115 | 0.006 | 6.75 |
| scagn1300_ocs2match_single | 40 | 12/40 = 0.300 [0.181,0.454] | 7/3/2/0 | 11.8 | -0.013 | -0.027 | 0.117 | 0.012 | 3.96 |
| **ocs2_trot_absref15 (no GaitNet)** | 40 | **29/40 = 0.725 [0.572,0.839]** | **10/10/7/2** | 20.0 | -0.007 | -0.001 | 0.117 | 0.025 | n/a |

What the sweep shows:
- **Concurrency caps:** +7.5 pp for SC and +17.5 pp for SCagn. The CIs overlap at n=40, so this is suggestive but not significant, and every SC arm is still 0/10 at 0.2.
- **The single-swing cap halves the step rate** (7.75 → 3.98 /s). SC does not compensate: front feet trail the hips by 11.2 cm (4.6 cm at baseline) and vx falls to 0.085. SCagn copes better under the same cap (front -1.3 cm, vx 0.117), which fits its controller-randomised fine-tune.
- **OCS2 trot** walks 20/20 at 0.05-0.1 but only 9/20 at 0.15-0.2. Its 8 falls at 0.2 come at t ≈ 7.2-14.2 s (6 of 8 between 7.2 and 8.1 s) and end in a rollover: final roll -2.4 to -3.1 rad, pitch -0.04 to -0.17. **The absref15 reference/OCS2 setup is unsafe at ≥ 0.15 m/s, independent of GaitNet.**
- The steps/s and step_share columns for ocs2_trot read 0/NaN only because GaitNet does not schedule it.

### 2b. Isaac evals: training controller made to land like OCS2's best setup (150 envs/cell, seed 0, argmax)

Files: `training/gaitnet/evals/inv_fast_{matched,matched_nodelay,delay2}/summary.json` under `/scratch/sselvaraj/projects/gait-net-train/gaitnet/`. The baseline is `evals/step0_variant_gaitnet` (normal controller, 150 envs/cell). There is **no baseline for SCagn@1365** with the normal controller in this harness; the 0.91 in KNOWN is a different setup and checkpoint.

Success [95 % CI] on flat ground:

| Policy | Baseline 0.1 | Baseline 0.2 | matched 0.1 | matched 0.2 | matched_nodelay 0.2 | delay2 0.2 |
|---|---|---|---|---|---|---|
| SC@645 | 0.967 [0.92,0.99] | 1.000 [0.98,1.00] | **0.820 [0.75,0.87]** | **0.373 [0.30,0.45]** | 0.860 [0.80,0.91] | 1.000 [0.98,1.00] |
| WB500@1160 | 0.987 [0.95,1.00] | 1.000 [0.98,1.00] | 0.953 [0.91,0.98] | 0.940 [0.89,0.97] | 0.967 [0.92,0.99] | 1.000 [0.98,1.00] |
| SCagn@1365 | n/a | n/a | 0.867 [0.80,0.91] | 0.780 [0.71,0.84] | 0.980 [0.94,0.99] | 1.000 [0.98,1.00] |

The matched condition is `travel_compensation=1 touchdown_along=-0.01 touchdown_noise=0.04 liftoff_delay_ticks=2`:
- Landing error comes out at an rms of about 6.5 cm, with along-track mean -0.3..+0.9 cm.
- Steps per tick: SC 0.31 (0.39 at baseline), WB500 0.18-0.21, SCagn 0.18-0.24.
- SC@645 falls at 0.2: 81 bad_orientation and 13 bad_height out of 150.

Voids, success at 0.1 / 0.2 m/s (matched, against baseline):

| Voids | SC@645 matched | SC@645 base | WB500 matched | WB500 base | SCagn matched |
|---|---|---|---|---|---|
| 10 % | 0.087 / 0.000 | 0.993 / 0.627 | 0.327 / 0.067 | 0.980 / 0.640 | 0.293 / 0.040 |
| 20 % | 0.000 / 0.000 | 0.947 / 0.313 | 0.080 / 0.000 | 0.900 / 0.327 | 0.033 / 0.000 |
| 30 % | 0.000 / 0.000 | 0.940 / 0.200 | 0.033 / 0.000 | 0.867 / 0.167 | 0.007 / 0.000 |

In voids, 4 cm landing noise sends feet into holes: foot_below_ground accounts for 94-147 of the 150 falls per cell. This affects every policy and says nothing about the Gazebo flat-ground question.

### 2c. Lift-off delay vs landing error (jobs 2366070 matched_nodelay, 2366071 delay2)

All three jobs finished by 23:42; SCagn@1365 is included.

| SC@645 flat, success | 0.1 | 0.2 |
|---|---|---|
| baseline | 0.967 | 1.000 |
| delay2 only (`liftoff_delay_ticks=2`; along error +2.0 / +4.3 cm) | 0.967 [0.92,1.00] | **1.000 [0.98,1.00]** |
| nodelay (landing error only: along -1 cm, noise 4 cm, travel comp; along -0.9 cm) | 0.953 [0.91,0.98] | **0.860 [0.80,0.91]** (16 bad_orientation, 5 bad_height) |
| matched (both) | 0.820 [0.75,0.87] | **0.373 [0.30,0.45]** |

WB500@1160 at 0.2 for the same three rows: 1.000, 0.967 and 0.940. SCagn@1365: 1.000, 0.980 and 0.780. The randomised fine-tune roughly halves the interaction cost (0.22 vs 0.63) but does not remove it.

**Neither factor alone explains the collapse; the combination does.** Delay alone costs 0.00, and landing noise alone costs 0.14. Together they cost 0.63 at 0.2, against 0.06 for WB500. SC tolerates a late lift-off when its feet land accurately, and tolerates 4 cm scatter when lift-off is on time. It falls when both happen. This fits its short 0.14 s swings and its 40 % time on a 2-foot line: little margin to absorb a step that arrives both late and misplaced.

Mechanism (hypothesis, from the same tables):
- delay2 alone cuts SC's steps/tick from 0.39 to 0.30 (delay-pending legs are masked), but its feet land +4.3 cm ahead because that run has no travel compensation. SC survives (1.00).
- matched has the same reduced rate (0.31). Travel compensation removes the forward bias (along ≈ 0) and noise is added, so feet are late and no longer ahead: 0.37.
- This is the pattern seen in Gazebo sc_ocs2match_single: the rate is halved, SC does not raise it, and the front feet trail by 11 cm. It also fits KNOWN "5 cm lead → 25 %".
- **Cheap mitigation to test:** a forward landing lead of +4-8 cm (insert_lead 0.05-0.08) in Gazebo.

Voids at 10 % (0.1 / 0.2): nodelay gives SC 0.133 / 0.000 and WB500 0.467 / 0.113; delay2 gives SC 0.953 / 0.167 and WB500 0.947 / 0.220.

## 3. The decisive question

**Does SC fall on flat ground in Isaac when it lands like OCS2? Yes, partly.**

| Condition | SC flat 0.1 | SC flat 0.2 |
|---|---|---|
| Isaac, normal controller | 0.967 | 1.000 |
| Isaac, 2-tick lift-off delay only | 0.967 | 1.000 |
| Isaac, OCS2-like landing error only | 0.953 | 0.860 |
| Isaac, matched to OCS2's best landing + 2-tick delay (this run) | 0.820 | 0.373 |
| Isaac, harsher emu_ocs2 knob set (along -4 cm, outboard 4 cm, noise 5 cm, delay 1; `evals/emu_ocs2`, SC@645, n=150) | 0.500 [0.42,0.58] | 0.113 [0.07,0.17] |
| Gazebo, sc_ocs2match | 1/10 | 0/10 |
| Gazebo, OCS2's own trot | 10/10 | 2/10 |

How to read this:
- **At 0.2,** landing/timing alone takes SC from 1.00 to 0.37 (or 0.11 with the harsher emu_ocs2 knobs; 0.50 at 0.1), while WB500 stays at 0.94. That is SC-specific fragility, and it explains most of the collapse. The platform itself (OCS2 trot 2/10) makes 0.2 a poor test speed whatever the policy.
- **At 0.1,** landing/timing explains only 0.97 → 0.82 (0.50 with the harsher emu_ocs2 knobs), but Gazebo is 1/10 while OCS2 trot is 10/10. **The 0.1 gap is OCS2 closed-loop execution of SC's gait**, beyond footstep placement and lift-off delay. The analyst evidence is consistent with this:
  - the body is much noisier at command time (yaw-rate sd 0.67 vs 0.07 rad/s; wx/wy sd 0.8-0.9 vs 0.3-0.46);
  - planned-stance feet are still in the air (stance_air 0.22 for SC fallers vs 0.17 for survivors at 0.05);
  - SC spends about 40 % of its time on a 2-foot diagonal support line, against 13 % for WB500.

## 4. Root causes, ranked by evidence

1. **Policy fragility to landing error and lift-off delay together (SC-specific).**
   - Isaac flat 0.2: delay alone 1.00, landing error alone 0.86, both 0.37.
   - Isaac-matched SC flat 0.2 is 0.37 vs WB500 0.94: a difference of 0.57, with no CI overlap.
   - SCagn, fine-tuned with controller randomisation, recovers to 0.78.
   - The KNOWN results fit: SC does not raise its step rate when feet trail (WB500 does), and here the single-swing cap left SC's feet trailing by 11 cm.
2. **OCS2 setup unsafe at ≥ 0.15 m/s (platform, not SC).** OCS2 trot reaches 9/20 at 0.15-0.2 and ends in rollovers; WB500 is 0/34 in Gazebo at 0.15-0.2. Every Gazebo comparison at 0.15/0.2 is confounded by this. From KNOWN: velocity weight 150 lifted SCagn survival to 55 %.
3. **OCS2 closed-loop execution of SC's fast gait at low speed (cause not isolated).**
   - SC takes 7.5-8 steps/s with 0.14 s swings. OCS2 plans lift-off 45-56 ms after scheduling and the foot is up about 75 ms after, which is about half of SC's swing. OCS2's own gait uses 0.3 s phases.
   - The capping experiments say concurrency is not the main factor (+7.5 pp, not significant). The step rate and swing duration themselves are untested.
4. **Inputs / closed-loop mismatch (secondary, unverified).**
   - Analyst "obs": decisions match Isaac to first order.
   - Two input differences are untested: base_z at 0.323 m vs the Isaac target of 0.26 m (+6 cm, real posture), and history ticks at about 22 Hz with irregular gaps (3.7 % over 60 ms).
   - Analyst "dyn": rear-pair swings precede 38/79 SC nose-down falls, and rear-pair share climbs from 0.9 % to 16.6 % in the last 0.5 s. But they are native to the policy (4.8-8.7 % in Isaac vs 9.8 % in Gazebo), and the diag-only arm that removes them did not rescue SC (8/40). Treat them as a symptom of a degrading state, not a cause.

## 5. Fixes, ranked by expected gain and cost

| # | Fix | Expected gain | Cost |
|---|---|---|---|
| 1 | Fine-tune SC with landing-error **and** lift-off-delay randomisation **applied jointly** at the matched levels (along -1..-4 cm, noise 4-5 cm, delay 1-2 ticks), plus a reward/penalty on trailing feet. This is the SCagn direction pushed to the measured OCS2 levels. | Isaac-matched 0.37 → ≥ 0.78 (SCagn already); Gazebo 0.125 → 0.30 with single-swing (measured on SCagn) | medium (one fine-tune, ~hours of GPU) |
| 2 | Find why the OCS2 stack fails at ≥ 0.15 m/s before comparing policies there. **The reference mode is not it** (addendum: OCS2 trot 3/10 re-based, 2/10 velocity weight 400, 2/10 absref15 at 0.2). Next: OCS2 trot at 0.2 with oracle terrain instead of the camera chain, and with the repo's WBC changes (contact-aware loading, ground search) turned off. Until then, compare policies at 0.05-0.1 m/s only. | Removes the confound for all policies | low-medium (2-3 sweeps) |
| 3 | Make the footstep rate fit OCS2: cap concurrency to 1 swing **and** add a minimum swing duration of about 0.25 s, or a step-rate floor so feet keep up. | +7.5-17.5 pp measured for the cap alone (n.s.); the swing-duration effect is untested | low |
| 3b | Forward landing lead of +4-8 cm in Gazebo: shift every sent foothold forward (in the ROS service's command map, like the IDM does) to give back the overshoot that made delay alone harmless in Isaac. Hypothesis, untested. (Not `gait_insert_lead`, which only moves the swing in time; and not the 5 cm body-position lead that took SCagn 12.5 → 25 %.) | Unknown | low (one service flag + one arm) |
| 4 | Isolate cause 3 at 0.1 m/s. Run one Isaac eval with the matched controller plus body-rate noise injected at Gazebo's level, and with base_z offset +6 cm. Run one Gazebo arm with the planned-stance feet checked. | Diagnostic: tells you whether 0.6-0.7 of the 0.1 gap is state noise or execution | low |
| 5 | Cut schedule→lift-off latency in the OCS2 gait node (45-56 ms plan + about 75 ms to foot up) | Unknown; it is about half an SC swing | high |

## 6. Caveats

- Gazebo arms have n=40 (10 per speed), so per-speed rates have very wide CIs.
- The 0.15/0.2 cells are confounded by the platform's own failure (OCS2 trot 2/10 at 0.2).
- The analyst "obs"/"dyn" numbers in §4.4 were not re-derived here. dyn's swing types use ctrl.log schedules with an assumed 50 ms lift-off lag.
- Isaac-matched uses the training (GPU) MPC with injected errors. It reproduces landing statistics, not OCS2's WBC or its body-state noise.
- The Isaac baseline (step0_variant_gaitnet) and the matched eval share 150 envs/cell and the same voids grid, but the baseline's seed was not checked.

## Addendum (23:45 UTC): is it the NMPC + WBC stack? OCS2's own trot in all three reference modes

Sweep `inv_fast_2` (job 2366135), same flat worlds (seeds 51-60), 10 trials per speed, OCS2's own trot (no GaitNet,
`route_footholds:=false`), plus the absref15 arm from §2a. Survival = success.

| OCS2 trot, body reference | n | 0.05 | 0.10 | 0.15 | 0.20 | Overall |
|---|---|---|---|---|---|---|
| re-based (velocity from current pose; sim default) | 40 | 10/10 | 10/10 | 10/10 | **3/10** | 33/40 = 0.825 |
| velocity weight 400 (`task_vel400`) | 40 | 10/10 | 10/10 | 8/10 | **2/10** | 30/40 = 0.75 (success 0.725) |
| 15 cm position lead (absref15, §2a) | 40 | 10/10 | 10/10 | 7/10 | **2/10** | 29/40 = 0.725 |

- The 0.2 m/s failure of this OCS2 + WBC + perception stack is independent of the reference mode, so fix 2's original idea
  (switch to the velocity reference) does not remove the confound; corrected in §5.
- Whether the limit is NMPC + WBC as such, this repo's WBC modifications, or the camera -> elevation map -> plane
  decomposition chain is not yet separated (next: trot at 0.2 with oracle terrain / stock WBC settings).
- At 0.05-0.1 m/s the stack walks 60/60 with its own gait in every mode, so the SC failures there (0-7/10 per arm) are
  SC-specific, consistent with cause 1 (fragility to landing error + delay, reproduced in Isaac without OCS2).

Also corrected in §5: fix 3b's mechanism (a forward foothold offset, not the insert lead or the body-position lead).

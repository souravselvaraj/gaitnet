# Perception stack investigation: camera vs oracle terrain in Gazebo/OCS2

2026-10-13, written 04:40 UTC. Workflow `perception-stack-investigation` had 6 lenses (void, acc, lat, cspace, pose, compute), 2 skeptics per lens and a completeness critic. The closed-loop rounds loop_0 to loop_5 (`decisions.md` lines ~490-575) are included because they test most of the fixes the lenses proposed.

**Rules used in this report**
- A **root cause** is a claim that neither skeptic refuted and that a closed-loop or matched-geometry test supports.
- **Plausible** means the skeptics split, or the mechanism is untested.
- **Refuted** means both skeptics rejected it, or the data contradicts it.
- Rates are given as k/n with 95 % Wilson CIs.

## 0. Bottom line

| Setup (frozen benchmark: SC-idm@1590 + Isaac IDM, held-out seeds 1-20, voids 10/20/30 % @0.1 m/s n=60, flat @0.2 m/s n=20) | Voids | Flat 0.2 |
|---|---|---|
| Camera, default chain (loop_0) | 5/60 = 8 % [4, 18] | 0/20 [0, 16] |
| Camera, 2 cm + 1.5 m submap (loop_0 incumbent) | 10/60 = 17 % [9, 28] | 0/20 |
| + restamp, near range 0.40, prior 1.1 m, true 2 cm decomposition, gate 30 dps (loop_1 "combined") | 17 % | 7/20 [18, 57] |
| + pad-only prior, erodeUnobservedBorder 2, visibility cleanup 1 Hz (loop_4 `c4_stack`, **best complete camera-only arm**) | 17/60 = 28 % [19, 41] | 6/20 |
| + drift-free odometry (loop_4 `c4_stack_gtodom`, sim-only stand-in for a drift-free estimator) | 33/60 = 55 % [42, 67] | 7/20 |
| Oracle terrain, same benchmark | ~82 % | 17/20 |

- **The gap is perception, not GaitNet.**
  - OCS2's own trot on camera terrain shows the same degradation: in-void touchdowns 0.64 % [0.43, 0.95] vs 0.12 % with oracle terrain; success 6/30 vs 23/30.
  - The oracle arms run the full camera, mapping and decomposition chain at the same CPU load and reach 70-93 %.
- **Most of the camera's excess void falls happen after the foothold is chosen.** At the same true foothold margin, camera feet land over the void 5-14 % of the time vs 0.2-1.3 % for oracle.
  - The OCS2 foothold regions built from the perceived map do not stop the foot at the true edge. The oracle's exact regions, inset 2 cm, do.
  - Two independent decompositions put the share of the excess due to where GaitNet places candidates at ≤ 2 of 8.4 void landings per 1000 steps. The other ~6.4 come from what happens between placement and landing.
- **Six mechanisms are confirmed and each is a measured error.** Each one alone is worth ≤ +10 points. Stacked, they recover 8 % → 28 % with camera only, or → 55 % with drift-free odometry.
  - start prior over the first void row
  - perceived edges that do not hold the foot
  - rate gate starving the map
  - 0.6 m range cut (stale memory)
  - odometry drift after observation
  - one-frame depth stamp lag
- **What remains in the best arm is tip-overs: 32/80 failures** (flat 13/20, d30 12/20, d20 5/20). Void falls are down to 8/80.
  - In 25/32 tip-overs, a front leg had < 100 of 1369 C-space cells in the last 2.5 s; the same happens at the end of only 6/40 successful trials.
  - What empties the front-leg C-space is **not identified**. It is the main open item.
- **Ruled out:**
  - Compute: an 8x faster plane node gave no gain, and the oracle runs the same load.
  - The "7-9 % zero-candidate requests": these come after the fall or before walking starts, and the rate is ~0 while walking in every arm.
  - Estimator roll/pitch: 0.03 deg RMS.
  - Every inference drawn from `gt_pose_cloud.py`: the instrument is confounded.
- **loop_5 (job 2368391) is in flight; 107/197 trials were done at 04:40 UTC.** Partial result: gate off on the camera-only stack (`r5_gate0_noodom`) is at 13/22 = 59 % [39, 77] on voids, against 28 % for the same stack gated at 30 dps. Flat 0.2 is 2/6. Do not adopt it until the sweep finishes.

## 1. Gap accounting

| Quantity | Camera | Oracle | Source |
|---|---|---|---|
| Void landings per 1000 GaitNet steps | 8.84 [7.41, 10.56] (121/13682) | 0.47 [0.29, 0.75] (17/36519) | void lens, reproduced by both skeptics |
| Same 5 policies, same worlds | 8.91 (73/8191) | 0.25 (4/16209) | void |
| Landing − candidate, unconditional median / p90 | 3.9 / 7.6 cm | 3.8 / 7.4 cm | void |
| Over-void landing at true candidate margin 3-5 cm, straight far edge | 10.9 % (n=165) | 0.2 % (n=1316) | void sk2 |
| Same, late trial, outside the prior band, 3-4 cm | 14.4 % [8.8, 22.8] (14/97) | 1.3 % [0.4, 3.7] (3/233) | cspace sk2 |
| Gap split by per-bin decomposition (Kitagawa, per 1000 steps) | composition −0.85, hazard +9.23 of 8.38 | — | void sk2; sk1's swap test agrees (≤ 2/1000 from placement) |
| GaitNet C-space selection share of the per-step hazard gap | 17-48 % | — | cspace sk2 (matched edge distance × direction) |
| Success if every excess void fall were removed (baseline, independent competing risks) | ~50 % | 79 % | void sk2 |
| Tip-over hazard at baseline (per m past x = 0.38) | 0.20/m | 0.03/m | void sk2 |
| Matched policy pairs: share of the 140 excess failures | near-edge drops ~58 %, far-target drops ~14 %, tilt with no foot drop ~24 % | — | acc sk2 (209 vs 206 trials) |
| Front leg < 100 C-space cells before a tip, flat 0.2 | 31/52 (60 % [46, 72]) | 0/40 [0, 8.8] | cspace sk2 |
| Same, voids 0.1 | 41/71 (58 % [46, 69]) | 0/13 | cspace sk2 |
| Best arm (`c4_stack_gtodom`): ok / foot_in_void / bad_orientation | 40 / 8 / 32 of 80 | — | critic (loop_4) |

**How much is explained:**
- **Voids.** The excess void falls explain about 2/3 of the gap, and they come mainly through the map-edge → foothold-region channel. The 55 % arm sits at the ~50 % ceiling that removing those falls predicts. The remaining ~27 points against oracle's 82 % are tip-overs with no causal mechanism.
- **Flat 0.2.** From 0/20 to 7-12/20, the gain comes from gate, rate and drift fixes. The rest, 7/20 vs 17/20, is tip-overs; 9/13 of them have front-leg starvation.

## 2. Verdicts on every lens claim

Columns: sk1 / sk2 = each skeptic's verdict on the claim; "ok" means not refuted; "partial" means the skeptic upheld only part of the claim (the Status column gives which part).

| Lens | Claim | sk1 | sk2 | Status |
|---|---|---|---|---|
| void C1 | Rate is 19x per step; "not an execution problem" | refuted | refuted | Rate stands; **inference refuted** (the gap is post-placement) |
| void C2 | ~70 % of the excess comes from candidates in a void or < 3 cm from the edge | refuted | refuted | **Refuted** (accounting split, not causal) |
| void C3 | Edges not rigidly shifted: far ~0, near +6.6 cm | refuted | ok | Measurement reproduced; interpretation **plausible** |
| void C4 | Occlusion mechanism; "not lag, not odometry" | refuted | refuted | **Refuted as stated**; loop_3/4 partly rehabilitate never-observed cells (see §4) |
| void C5 | C-space probe and MPC regions give no real margin at far edges | ok | ok | **Root cause (RC1)** |
| void C6 | OCS2 trot degrades the same way → perception, not policy | ok | ok | **Stands** (the shadowed-side signature does not) |
| void C7 | Uniform margin cuts landings; falls become tips | ok | refuted | **Plausible**; "camera-only tips" and "stop at 1.2 m" refuted |
| acc C1 | Tile heights accurate (median −0.2 to −0.9 cm) | ok | ok | Stands (heights are not the problem) |
| acc C2 | 0.6 m range cut → tile tops seen only 0.77-1.21 m ahead; memory ~7 s | ok | ok | **Root cause (RC5)** |
| acc C3 | Drift −1.75 %/m x, −2.24 %/m y | ok | ok | Stands; drift is arm-dependent: camera y −3.3 to −4.3 %/m vs oracle −0.75 to −0.9 |
| acc C4 | Raw-map rims and left edges dilated into the void | ok | ok | Stands; the same in the oracle-walk arm (raw map alone does not separate the arms) |
| acc C5 | Drift explains 40 %, rest is an insertion offset | refuted | refuted | **Refuted** (time layer = last fusion, not insertion) |
| acc C6 | Init prior makes row-2 voids solid | ok | ok | **Root cause (RC4)**; the F1 "11 pp" gain is unsupported |
| acc C7 | Excess falls at rim and left edges, not wall | refuted | ok | **Plausible** (per-touchdown hazard is 5-12x at every orientation) |
| acc C8 | C-space margin thinner, same scatter | refuted | refuted | **Refuted** (the foot lands past the edge; post-placement) |
| acc C9 | Lost steppable area (flat C-space 85.6-93.1 % vs 100 %) | ok | ok | Stands; losses are in FL/FR |
| acc C10 | Void-floor regions; ConvexRegionSelector has no height filter (:115) | ok | ok | Code fact; never observed firing (not logged) |
| acc C11 / compute C1 | "2 cm" arms still decomposed at 1 cm | ok | ok (compute) | Stands |
| acc C12 | Rocking does not affect edge errors | refuted | refuted | **Refuted** (gate truncates the sample) |
| lat summary | "Most traces back to stamp timing" | — | refuted | **Refuted** |
| lat C1 / pose C1 | Depth frame shows the previous frame's pose: ~170-210 ms lag | ok | ok | **Root cause (RC6)**, small effect on success |
| pose C2 | Edges 3.0 cm (0.1 m/s) / 5.5 cm (0.2 m/s) forward at the stamp; 0.5 cm at the previous stamp | ok | ok | Stands (median; per-frame timing not pinned) |
| lat C2 / compute C4,C5 | Rate gate starves the map while walking; map stops re-centring | ok | ok / numbers corrected | **Root cause (RC3)** |
| lat C3 | r30/r60 alone does not fix the 1 cm chain (0/26) | ok | ok | Stands (r30 walks ~2x further on flat) |
| lat C4 | Flat falls come from running off stale terrain | refuted | refuted | **Refuted** (9/13 falls had maps ≤ 2.2 s old) |
| lat C5 / cspace C1(all-leg) / compute C8 | Zero-candidate requests are post-fall | ok | ok | Stands; **the "7-9 %" is a window artefact** |
| lat C6 | Candidate areas are never in live view; map at the feet is 3.4-9.5 s old | ok | ok | Stands (part of RC5) |
| lat C7 | Init prior (33 % vs 6 % of prior-void worlds) | ok | ok | RC4; gap contribution bounded at 3.5-11 points |
| lat C8 / compute C6 | Decomposition latency and CPU are not the cause | ok | ok | Stands |
| lat C9 | Replay: restamp does not clean the gate-off map | ok | ok | Stands (medium; tested at 1 cm only) |
| cspace C1 | Candidate set never degenerate before a fall | refuted | refuted | **Refuted** (single-leg starvation) |
| cspace C2 | Camera C-space has void/edge cells in 13.2 % of leg-requests | ok | ok | Stands |
| cspace C3 | Prior = 46 % of void cells | ok | ok | RC4; ~18-20 % of camera void falls, not 32 % |
| cspace C4 | Remaining void cells come from drift after observation | ok | ok | **Root cause (RC2)**, medium; plus a ~0.75 cm static part |
| cspace C5 | C-space errors are the main cause of void falls (59 %, "41x") | ok | refuted | **Plausible / overstated**; selection = 17-48 % of the hazard gap |
| cspace C6 | OCS2 region selection behaves the same in both modes | ok | refuted | **Contested**; the > 2 cm relocation metric is blind to the inset channel |
| cspace C7 | Prior cannot be shrunk alone; probe 0.06 trades falls for tips | ok | ok | Stands (n=12-16) |
| cspace C8 | Camera plans touchdowns above ground | refuted | refuted | **Refuted** (foot frame is 2 cm above contact; the oracle plans 2 cm low) |
| pose C3 | Estimator negligible; only drift matters | ok | refuted | Numbers stand; **the weight is refuted** (drift is camera-only, ~6x) |
| pose C4 | Range cut drops 64 % of ground pixels | ok | ok | Stands with a correction: 35 % of points, 26 % of area after the voxel grid |
| pose C5 | gt_pose_cloud confounded | ok | ok | Stands → **retire all p_gt* inferences** |
| pose C6 | 2 cm gt gains come from near field, not pose | refuted | refuted | **Refuted** (not identified) |
| pose C7 | Gate removes the tail; restamp removes the bulk | refuted | refuted | **Refuted** (restamp shows no gain in fused-map replay or closed loop) |
| compute C2 | applyKernelFunction is 82 % of post-process | ok | ok | Stands (reproduced 82.5-83.5 %) |
| compute C3 | r2 is 8x faster, no success change | ok | ok | Stands (underpowered null: 0/12, upper CI 24 %) |
| compute C7 / summary | Fresh, fast terrain still fails → "content, not compute/rate" | ok / refuted | refuted | **Plausible / refuted as stated** (submap and gate effects) |

## 3. Root causes (neither skeptic refuted, plus a closed-loop or matched-geometry test), ranked by evidence × share of gap

### RC1. Perceived foothold regions do not hold the foot at true void edges (post-placement channel)

| | |
|---|---|
| **Survived** | void C5 (both skeptics); corroborated by cspace sk2, acc sk2 and both void skeptics |
| **Share of gap** | the largest share of the void-fall excess (~2/3 of excess hazard at baseline) |

**Mechanism.**
- Oracle: regions = exact tile edge − 2 cm inset (`oracle_terrain.py` inset 0.02). The C-space probe = exact edge + 3 cm.
- Camera: the perceived far edge sits at −0.18 cm overall (positive = inside the tile) and −1.1 cm under the body, where an exact edge would read +2.0 cm after the marginSize-5 erosion.
- Camera inset ≈ 1 sub-pixel (≈ 0.33 cm; derived from ContourExtraction.cpp:22,38 and Upsampling.cpp:64; **not measured**).
- When the eroded region vanishes, ContourExtraction.cpp:42 retries with no margin.

**Evidence.**
- At straight far edges, oracle forward-landing error is clipped near the edge: p90 1.61 / 2.50 / 3.40 cm at 2-3 / 3-4 / 4-5 cm margin, reaching the free value of 5.66 cm only at 7-10 cm.
- Camera forward error is flat at 4.2-4.6 cm at every margin.
- Void landings in the 2-3 and 3-4 cm bins: oracle 0/154 and 0/474; camera 15/89 and 9/95.
- 43/128 camera over-void landings came from targets > 3 cm inside the tile (cspace sk2).
- Interventions:

| Change | What it moves | Void landings per 1000 | Success |
|---|---|---|---|
| marginSize 11, `pvm11` | regions | 15 → 5.3 [2.3, 12.3] | 0/20 |
| C-space margin 0.05, `pvcs05` | C-space only | 12.3 (no change) | 0/20 |
| cm06 | C-space only | — | 1/12 |
| probe 0.05 (loop_2) | C-space only | — | 15 % / 3/20 |
| erode-at-unobserved-border stack (loop_4) | regions | — | void falls 54 → 31 per 80 |

**Not yet measured.** The camera inset itself; projections < 2 cm in ConvexRegionSelector; a split-consumer test (oracle regions → OCS2 with camera C-space → GaitNet). See E3.

### RC2. Odometry drift between a cell's last observation and its use (camera-only by construction)

| | |
|---|---|
| **Survived** | cspace C4 (both, medium), acc C3 (both). pose C3's "estimator negligible" was refuted by pose sk2 on exactly this point. |
| **Share of gap** | closed loop: +10 points alone (combined 17 % → 27 % [17, 39], flat 7 → 12/20 [39, 78], loop_3); +27 points on the stack (28 % → 55 %, loop_4). Fixes interact: drift-free +10, stack +11, both +38. |

**Mechanism.** `oracle_terrain.py` re-expresses the world through the estimated-vs-true pose on every publish, so the oracle cannot drift. The camera map lives in drifting odom.

**Evidence.**
- Camera y drift −3.3 to −4.3 %/m vs oracle −0.75 %/m with the same estimator; x drift −0.7 to −1.5 vs −2.5 %/m. Flat 0.2 m/s survivors drift 4.8-6.8 %/m (n=7); voids n=1 at 2.9 %/m.
- Signed est y < true y in 14/14 px_lat_1 trials.
- The left/right half-difference in the fused map grows with cell age: 0.4 → 1.5 → 3.0 cm (fresh → 0.2-0.6 m ahead → under the body), Spearman 0.57 with |ey_end|.
- The C-space counterfactual (each cell read at its observation pose) removes 92-99 % of non-prior void cells and cuts leg-requests with a void cell from 13.2 % to 5.8 %. The sign-flip placebo makes it worse (5141 → 29844), which confirms the specificity. A static (−2, −3) cm shift removes nearly as much, so a ~0.75 cm static part exists.

**Caveats.**
- Loop_3 drift-free did **not** change per-step void landings (9.9 vs 9.7 per 1000). Its gain comes through other channels: fewer tips, flat walking, rear legs.
- Camera arms drift ~6x more than oracle arms with the same estimator, so drift is partly an effect of camera-driven slips (stance-foot y slip +1.8 vs +0.8 mm per stance). Cause vs effect is open.

### RC3. The rate gate starves the map while walking, and the map stops re-centring

| | |
|---|---|
| **Survived** | lat C2 (both), compute C5 (both); compute C4's mechanism stands but its numbers were corrected |
| **Share of gap** | dominant for flat 0.2: combined with gate 30, flat went 0/20 → 7/20 [18, 57] with CIs separate (loop_1). Gate 0 is being tested in loop_5. |

**Mechanism.**
- `cloud_gate.py` drops clouds while the body turns faster than 15 deg/s, using a ±60 ms window of /odom around the late stamp.
- While walking (strict window) it drops 55-74 % at 0.1 m/s and 83-94 % at 0.2 m/s. The quoted "16-36 %" counts 230-320 standing clouds.
- `updateMapLocation()` runs only in the cloud callback (ElevationMapping.cpp:390). The `min_update_rate` timer never fires because `(lastPointCloudUpdateTime_ - time) <= maxNoUpdateDuration_` at ElevationMapping.cpp:437 is always true.

**Evidence (flat 0.2, gated).**
- Fresh terrain arrives at 0.31-0.79 Hz (mean 0.51) with blackouts of 4.5-6.1 s.
- Map lag behind the base: mean 0.55 m, max 1.18 m. Terrain queries fall outside the map in 15/20 trials before the fall.
- 13/20 terminations come 0.2-1.5 s after the first refresh following a blackout, which arrives as a re-centre jump of 0.9-1.26 m. 2 more trials failed on a map 5-6 s stale.
- The front-leg C-space empties within ~2 s before the tip while tilt is < 5 deg (3/3 bagged trials).

**Limits.**
- 9/13 r15 flat falls had maps ≤ 2.2 s old, and r60's fresh maps did not survive longer (t_end median 7.5 s), so staleness is not the whole flat story.
- Restamping makes the ±60 ms gate drop **more** clouds: 78 % vs 61-64 % (pose sk1, loop_1).

### RC4. The elevation_mapping start prior turns the first void row into solid ground

| | |
|---|---|
| **Survived** | acc C6, lat C7 and cspace C3, both skeptics each |
| **Share of gap** | bounded at ~3.5-11 of ~60 points (lat sk2). The prior's excess is ~43 falls, about 18-20 % of camera void falls (cspace sk1). |

**Mechanism.** `elevation_mapping.yaml:146,172-176` sets `initialize_elevation_map: true`, a 1.5 × 1.5 m prior at z 0 with variance 0.05. `inpaint: false` (convex_plane_decomposition.yaml:57). The shadowed part of each first-row void (world x 0.62-0.86) is never fused, so it keeps tile height.

**Evidence.**
- 4304/9445 = 46 % of camera C-space void cells are exactly-0 prior cells.
- First void row in the prior: camera falls in it 87/102 = 85 % [77, 91], with 0/102 success. Oracle falls in it 9/54.
- 27-33 % of void falls happen at x < 0.9 m (loop_3).
- In the six affected cells (gridvoid24_{20,30}_s{52,53,60}), camera success is 0/54.

**Closed loop.**

| Arm | Voids | Flat 0.2 | Note |
|---|---|---|---|
| prior 1.1 m alone | 8 % | 0/20 | |
| pad-only prior alone (`perception_init_offset_x:=-0.25`) | 10 % | 6/20 | no gain alone; part of the 28 % / 55 % stacks |
| prior 0.6 m | 0/16 | — | all tipped within 2.3-4.7 s: the prior is load-bearing because of RC5 |

### RC5. The 0.6 m range cut means footholds come only from old memory

| | |
|---|---|
| **Survived** | acc C2, lat C6, pose C4, cspace C4 (all both) |
| **Share of gap** | an enabler of RC2 and RC4 rather than a direct share. `perception_cloud_min_range:=0.40` alone gave 15 % / 1/20; it is part of the combined arm. |

**Mechanism.** `cloud_gate.py:39` sets `always_min_range` 0.6, a Euclidean range in the optical frame.
- With the camera ~0.36 m above the tiles, the centreline ground starts at base x ≈ 0.77-0.79 m (0.57 m at the image sides).
- The lower FOV edge alone would allow base x ≈ 0.39-0.40, still beyond the front grid's far edge at 0.37 m.
- The comment "nearest ground 0.62 m" predates the 2026-09-12 mount change.

**Evidence.**
- Terrain age at the hips at 0.1 m/s: front 5.9 s (min 3.4), rear 9.5 s (min 7.3).
- 0 % of cells at odom x 0.00-0.55 are ever camera-updated.
- The cut removes 35 % of points and 26 % of ground area after the 5 mm voxel grid (60-64 % of raw pixels).

### RC6. Simulated depth frames are stamped one frame late

| | |
|---|---|
| **Survived** | lat C1, pose C1/C2 (all four skeptics; replicated with independent pose sources) |
| **Share of gap** | small. `perception_restamp_prev:=true` alone: 12 % / 0/20 vs 17 % / 0/20. The open-loop replay with exact restamp did not clean the gate-off map (flat missed-solid 0.30-0.38 with or without it). Its value is as a prerequisite for relaxing the gate. |

**Mechanism.** `gazebo_ros_realsense.cpp:212-221` stamps with `world->SimTime()` in OnNewDepthFrame; :238 copies the header to the cloud. The frame content is the scene at the previous frame's stamp + ~6 ms (best − dt_prev median −6 to −8 ms, 85-92 % within 15 ms). That is 170 ms at 5.7 Hz delivery (px_lat_2) and 205-210 ms at 4.7 Hz.

**Effect.**
- Tilt error at the stamp is 1.5-3.75 deg RMS, against 0.03-0.04 deg at the best offset.
- Void edges with true pose at the stamp come out 3.0 cm forward (0.1 m/s, n=820 frames) and 5.5 cm forward (0.2 m/s, n=184). At the previous stamp the residual is 0.5 cm.
- The geometric one-frame prediction at true speed is only 1.7 / 2.7 cm, and the shift search aliases with the 24 cm grid.

## 4. Plausible (mixed verdicts or untested mechanism)

| # | Hypothesis | For | Against / missing | Decisive test |
|---|---|---|---|---|
| P1 | **Front-leg C-space starvation causes the tip-overs** (the remaining gap) | Best arm: 25/32 tips with a front leg < 100 cells in the last 2.5 s (10 at 0) vs 6/40 successes; starvation before tilt > 8 deg in 20/32 (tilt first in 5, no starvation in 7). Baseline: 60 % of camera flat-0.2 tips vs 0/40 oracle. A ~0.3 m NaN band sits under the front hips (t52 bag). | CSpace log is 1 Hz, so the 0.44 s lead is inside the log resolution. What the starved cells are has not been identified: NaN or unobserved cells (inpaint off), fragmentation (7-25 regions on flat ground), erosion at unobserved borders (erode 2 may worsen it: flat 12/20 → 7/20, n.s.), or a planar_terrain stall. | E2 |
| P2 | **Never-observed cells, not lag or drift, set the far-edge overreach** | Restamp + drift-free (loop_3) left void landings at 9.9 → 9.7 per 1000; near-edge candidates 22 → 21 of 36. Only the occlusion-targeted stack cut void falls (54 → 31 → 8 per 80). Outer vs inner lateral edges differ (left +0.18 vs +4.78 cm). | void C4 was refuted by both skeptics as argued. A 4 cm erode would also mask a residual shift. The single-cloud / first-N-cloud test was never run. | E6 |
| P3 | **Submap 1.5 vs 2.8 m matters beyond compute** | p_gtfast_scahead 15/40 vs p_gt_scahead 0/40 (p = 1.2e-5), same sweep, args differ only in resolution and submap; flat 0/10 → 6/10. Flat camera maps break into 19-25 regions at 2.8 m vs 15 at 1.5 m. | gt arms are confounded (pose C5), and the 2 cm → 1 cm INTER_LINEAR upsampling grows NaN holes. The incumbent already uses 1.5 m with true 2 cm decomposition. | E5 |
| P4 | **The OCS2 consumer differs by terrain mode** (cspace C6) | Matched-geometry landing gap (RC1); logged command − intent p90 +2.6 cm (camera) vs −0.7 cm (oracle) in the isaacidm arms. | Relocations > 2 cm are similar (0.47 % vs 0.18 %); projections < 2 cm are not logged. | E3 |
| P5 | **Touchdown height offset** | The camera plane has +0.0198 (`extracted_planes_height_offset` 0.02); the oracle has 0. The foot frame sits 2 cm above contact, so the oracle aims 2 cm below contact ("press-in"). | Sign and effect unresolved; TDDBG is sampled 1/200. | E4 |
| P6 | **Lateral (left) bias beyond drift** | Trot near-void landings: left 155 / ahead 50 / behind 8 / right 2 (oracle 216/263/98/190). 68 % of FP void cells have the void to the left. | Confounded with drift (the camera trot drifts left ~2x). A principal-point or lateral offset in the plugin (~1.4 cm) has not been checked. | E7 |
| P7 | **"Second error source" in the replay** (gate-off map degrades even with 0.2 cm registration) | Flat missed-solid 0.30-0.38 in arms B/D/H. Candidate: ElevationMap.cpp:143-156 (same-scan higher outliers pull a cell up; `multi_height_noise` inflates variance). | Region z-spread of 0.2-0.55 m on flat ground is physically implausible (likely a metric or harness artefact); tested only at 1 cm / 2.8 m. | rerun the replay at 2 cm / 1.5 m with per-cell logging |
| P8 | **Excess falls concentrated at rim and left edges** (acc C7) | Matched px_acc rim falls 9/12 vs 0/12 (p = 3.4e-4); corrected pool rim 37.4 % vs 9.1 %. | Per near-edge touchdown, camera hazard is 5-12x oracle at **every** orientation (rim 16.1 vs 1.3 %, wall 8.5 vs 1.7 %, right 7.6 vs 0.7 %). The pattern is mostly exposure. | — |

## 5. Refuted

| Hypothesis | Why |
|---|---|
| Zero valid candidates (7-9 % at flat 0.2) cause falls | Before termination: 0/759, 0/1541, 0/421, 0/3937 camera; 0/7864 oracle (max 18/4253 in p_gt). All of the 7-9 % comes before walking (97-100 % of those requests are empty in every arm, oracle included) or after the fall. The metric sums min(16, cells) over legs and is blind to one starved leg (see P1). |
| Compute or latency of the plane decomposition is the gap | r2: node 144-163 → 16-21 ms, latency 0.21-0.24 → 0.08 s, success 0/8 → 0/12 on voids and 0/8 → 0/12 on flat. Oracle arms run the same plane node (146-225 ms) plus a 10 Hz SDF rebuild and reach 9-10/10. GaitNet service: median 7-9 ms in every arm. |
| Camera execution is "the same as oracle" | Equal unconditional scatter hides the matched-margin gap (RC1). |
| GaitNet C-space placement is the main lever | Composition ≤ 2 of 8.4 per 1000. C-space-only fixes are null: pvcs05 0/20, cm06 1/12, probe 0.05 15 %. Ceiling if every rejected fatal step were saved: 48.7 % [43.6, 53.7] vs oracle 74.1 %. |
| Estimator roll/pitch or z error | 0.016-0.05 deg RMS, z 1-3 mm, lag 0 to −2 ms (219 trials). |
| Flat 0.2 falls = running off stale terrain | 9/13 had maps ≤ 2.2 s old; the same collapse happens with fresh r30/r60 maps. |
| Raising cloud_max_rate_dps alone fixes the 1 cm chain | r30/r60 0/26 [0, 13]. |
| Restamping removes the bulk of the map error | Replay B/H (restamp, gate off): flat missed-solid unchanged (0.30-0.38); closed loop, restamp alone 12 % / 0/20. |
| "2 cm map" arms decomposed at 2 cm | preprocessing.resolution 0.01 resampled them to 1 cm; the elevation_mapping.launch:28-29 comment is wrong. |
| Uniform margins (marginSize 11, cspace 0.05/0.06) fix success | 0/20, 0/20, 1/12; failures shift to tips. |
| gt_pose_cloud results show that true pose does or does not help | Truth keyed by receipt time on 1-2 kHz /gazebo/link_states (gt_pose_cloud.py:79); odom-frame cloud disables cloud_gate's range cut; all p_gt arms ungated; output never bagged. Superseded by `gt_tracking_odom.py` (verified 0.1 cm rms). |
| "Tips are camera-only" / "every camera arm stops at 1.2 m" | Oracle GaitNet has bad_orientation in 84/420 trials. 1.2 m is just the median under a constant ~0.8/m hazard. |
| Camera touchdowns are planned above the ground | The foot frame is 2 cm above contact; the camera is ~1 cm low and the oracle 2 cm low. |

## 6. Fixes, ranked by expected gain and cost

### 6a. Config only (launch args / new yaml, no build)

| Rank | Fix | Args | Measured gain | Status |
|---|---|---|---|---|
| 1 | **Adopt the camera stack** (`c4_stack`) | `perception_map_resolution:=0.02 perception_submap_length:=1.5 perception_restamp_prev:=true perception_cloud_min_range:=0.40 cloud_max_rate_dps:=30 perception_init_offset_x:=-0.25 perception_elevation_params:=/scratch/sselvaraj/projects/gait-net/legged_perceptive/legged_perceptive_controllers/config/px_acc_elev_vis.yaml perception_plane_params:=/scratch/sselvaraj/projects/gait-net/legged_perceptive/legged_perceptive_controllers/config/loop_plane_r2_erode2.yaml` | Voids 8 % → 28 % [19, 41]; flat 0/20 → 6/20 (n=80) | Tested, loop_4 |
| 2 | **Rate gate off on the stack** | stack + `cloud_max_rate_dps:=0` | Partial: 13/22 = 59 % [39, 77] voids, 2/6 flat (`r5_gate0_noodom`). Compare with `c4_stack` (28 %), not the 55 % arm. If it holds, drift-free odometry was partly compensating for gate starvation (hypothesis). | loop_5 in flight |
| 3 | erodeUnobservedBorder 1 instead of 2 | `perception_plane_params:=.../loop_plane_r2_erode1.yaml` | Partial on gt-odom: 8/24 = 33 % voids, flat 6/7. Tests whether erode 2 starves the front legs on flat (P1). | loop_5 in flight |
| 4 | Touchdown offset parity | `gaitnet_touchdown_offset:=0.02` on oracle, or extracted_planes_height_offset 0 on camera | Unknown | Untested (E4) |
| — | Do **not** use | cspace margin 0.05/0.06, marginSize 11, prior 0.6 m (0/16 tips), prior 1.1 m alone (8 %), rear follow 0.06 (12 %), gate off at 1 cm without restamp (0/30), r30/r60 on the 1 cm chain (0/26) | Null or harmful | Tested |

Sim-only reference: `perception_gt_odom:=true` (`scripts/gt_tracking_odom.py`) on the stack → 55 % [42, 67] / 7/20. On hardware the equivalent is a drift-free pose: VIO into KalmanFilterEstimate's `/tracking_camera/odom/sample` input, or map re-registration (6b-1).

### 6b. Code changes (describe only; no edits made)

| Rank | Change | File:line | Expected gain | Cost |
|---|---|---|---|---|
| 1 | Drift-free map: a VIO / tracking-camera input to the estimator, or re-express map memory in the current pose (oracle_terrain-style) | legged_estimation KalmanFilterEstimate input; elevation_mapping pose input | +10 (alone) to +27 (on stack) points voids; flat 7 → 12/20 (from loop_3/4 with gt odom) | high (hardware) / medium |
| 2 | Keep the map centred when clouds are gated: fix the sign `(time - lastPointCloudUpdateTime_) <= maxNoUpdateDuration_` and call `updateMapLocation()` in `mapUpdateTimerCallback`; or publish an empty cloud with the same header when the gate rejects one | elevation_mapping/src/ElevationMapping.cpp:427-460 (bug at :437); or scripts/cloud_gate.py:84-89 | Removes 4.5-6 s blackouts and 0.9-1.26 m re-centre jumps (13/20 gated flat failures follow one); effect on success uncertain. Moot if gate 0 wins in loop_5. | low |
| 3 | Gate on the orientation change between stamp and stamp − 1 frame, not on the instantaneous rate in ±60 ms around the stamp | scripts/cloud_gate.py `turning_fast()` | Offline: a 1.5 deg threshold keeps 33 % of frames at 0.94 deg RMS vs 2.50 deg for the current gate. Restamp currently raises drops from 61-64 % to 78 %. | low |
| 4 | Separate inset for regions passed to OCS2: about 2 cm at least where the region borders unobserved or NaN cells; and no zero-margin retry for small fragments | convex_plane_decomposition ContourExtraction.cpp:22,38,42 (+ Upsampling.cpp:64); or legged_perceptive_interface ConvexRegionSelector (the authority box falls back to the whole grown region when the box is not inside it) | Targets RC1 directly (matched-margin landing gap 10.9 % vs 0.2 %); measure first (E3) | medium |
| 5 | Plugin-side stamp: keep `prev_depth_time_` and stamp each frame with the previous OnNewDepthFrame time (`LastMeasurementTime()` is unverified and may also be late) | realsense_gazebo_plugin/src/gazebo_ros_realsense.cpp:212-221 (and :65-81 for colour); needs a catkin build | Same as `restamp_prev_frame.py`, minus its mis-stamps after dropped frames (7 % of frames have dt > 270 ms). Small success effect. | low (sim only) |
| 6 | Hardware analogue: stamp clouds with capture time, not arrival time | scripts/restamp_pointcloud.py:12 (`rospy.Time.now()`) | Removes rotation smear and forward shift on the robot; size unmeasured | low |
| 7 | Height penalty for void-floor regions | legged_perceptive_interface/src/ConvexRegionSelector.cpp:115 (`penaltyFunction` returns 0) | Unknown; never observed (relocations are not logged) | low |
| 8 | Build NaN fill and SDF outside `mutex_`; swap under the lock | legged_perceptive_controllers/src/synchronized_module/PlanarTerrainReceiver.cpp:38-74 | Removes MPC stalls of 50-150 ms at 1 cm (10-20 ms at 2 cm); not the gap (the oracle has the same) | low |
| 9 | Pass `map_resolution` to the plane node; fix the "~4x less work" comment | launch/elevation_mapping.launch:28-29 and include at :112-116 | Prevents a silent 2 cm → 1 cm resample | trivial |
| 10 | Fix the stale comment ("nearest ground 0.62 m") | scripts/cloud_gate.py:39 | none (documentation) | trivial |
| 11 | Validate the anchored, commanded target (hip + IDM + v·(lead+T)) against the C-space, not only hip_now + intent | ocs2_legged_robot_ros/src/LeggedRobotGaitCommandNode.cpp ~2140-2155 | Unknown; the oracle works without it | low |

### 6c. GPU mapping

- `perception_gpu_mapping:=true` (elevation_mapping_cupy, `GAITNET_NV=1` for `--nv`) runs, but its terrain is **not fresher**: the gate, not the mapper, sets freshness.
- It reads void cells as walkable more often: 8.8 % vs 4.0 % (6 paired worlds, unexplained).
- The vendored plane_segmentation is the same CPU decomposition. Not adopted.
- Compute is not on the critical path:
  - the true 2 cm decomposition takes 16-21 ms (applyKernelFunction is 82 % of the post-process at 1 cm; cost ∝ cells × kernel side²);
  - the oracle arms carry the full load and succeed.
- **Recommendation:** do not pursue GPU mapping for this gap. Revisit only if gate-off at full cloud rate (~4-5 Hz; 10 Hz configured) shows mapping backlog in the loop_5 logs.

## 7. Next experiments (in order)

| # | Experiment | Setup | n | Decides | Needs |
|---|---|---|---|---|---|
| E1 | **Finish loop_5** (job 2368391) and score it | `python inv/loop/bench_score.py /scratch/sselvaraj/gaitnet-runs/idm/loop_5` | 4 arms × ~50 | Gate 0 vs 30; erode 1 vs 2; whether gate 0 without gt odom holds 59 % | nothing; wait. Launch nothing perception-related before it ends. |
| E2 | **Front-leg starvation diagnosis** (critic's top pick) | Best arm (gate setting per E1) on flat 0.2 and d30 @0.1. `bag_topics:=/convex_plane_decomposition_ros/planar_terrain,/elevation_mapping/elevation_map_raw,/gaitnet/foothold_command probe:=1`. Paired arm: plane yaml with `preprocessing.inpaint: true`, or erodeUnobservedBorder 0. | ~20 + 20 | Whether starved front cells are NaN/unobserved, eroded at unobserved borders, fragmented, or stalled, and whether filling them removes tips (32/40 of the best arm's failures) | yaml only |
| E3 | **Split-consumer test**: oracle regions → OCS2 with camera C-space → GaitNet, and the reverse. Cheaper partial: replay ConvexRegionSelector on bagged camera vs oracle terrain and log projections < 2 cm near true edges. | 2 arms × 60 voids | Whether OCS2 regions (RC1) or the GaitNet C-space carry the void-fall excess; sizes fix 6b-4 | launch change: route `planar_terrain_to_controller` and the gait node's terrain topic separately (load_controller.launch ~816-822) |
| E4 | Touchdown offset parity | best arm + `gaitnet_touchdown_offset:=0.02` (or plane offset 0) | 80 | P5; may affect tips and the near-edge execution gap | arg only |
| E5 | Submap length on the current stack | best arm with `perception_submap_length:=2.8` | 80 | P3 mechanism (fragmentation vs extent) | arg only |
| E6 | Far-edge mechanism | Map from the first N clouds only; far-edge position vs number of observations | offline + 10 bagged trials | P2 (never-observed vs accumulation vs residual shift) | bagged clouds |
| E7 | Lateral offset | Standing robot, known tile edge, both lateral sides; check the plugin principal point | 5 | P6 (~1.4 cm left bias) | bags only |
| E8 | Drift as cause or effect | Compare estimator drift and stance slip on matched footholds, camera vs oracle | existing data | RC2 caveat (6x drift in camera arms) | none |
| E9 | Hardware D435 latency | Rock the body on a flat floor; fit tilt vs time offset | 1 session | Size of fix 6b-6 | robot |

## 8. Every experiment run

Relative paths in this section are under /scratch/sselvaraj/gaitnet-runs/: `idm/...` as written, and `inv/...` means `idm/inv/...`. `scratchpad/` = /tmp/claude-901049456/-scratch-sselvaraj-projects-gaitnet/6258186b-6fd6-4ad8-8edd-d8a9e5b18296/scratchpad/.

### 8a. Closed-loop sweeps

Frozen benchmark unless noted (60 void trials + 20 flat-0.2 trials per arm).

| Job | What | n | Result (voids / flat 0.2) | Path |
|---|---|---|---|---|
| 2367930 | loop_0: camera default vs 2 cm + 1.5 m submap; oracle flat reference | 80 + 80 + 20 | 8 % [4, 18] / 0/20; 17 % [9, 28] / 0/20; oracle flat 17/20 | /scratch/sselvaraj/gaitnet-runs/idm/loop_0 |
| 2368171 | loop_1, one variable each on the 2 cm incumbent; plus all combined + gate 30 | 5 × 80 | restamp 12 % / 0/20; near 0.40 15 % / 1/20; prior 1.1 m 8 % / 0/20; true 2 cm decomposition 15 % / 1/20; **combined 17 % / 7/20** (tips 25 → 9, void falls 45 → 54 per 80) | idm/loop_1 |
| 2368235 | loop_2 on combined | 5 × 80 | erode 1: 13 % / 7/20; erode 2: 22 % [13, 34] / 5/20; visibility cleanup 15 % / 11/20 [34, 74]; rear follow 0.06: 12 % / 7/20; probe 0.05: 15 % / 3/20. None separates. | idm/loop_2 |
| 2368293 | loop_3: combined + drift-free odometry (`gt_tracking_odom`) | 80 | 27 % [17, 39] / 12/20 [39, 78]; void landings 9.7 vs 9.9 per 1000 | idm/loop_3 |
| (loop_4) | loop_4: pad-only prior; stack (pad prior + erode 2 + visibility cleanup); stack + drift-free | 3 × 80 | 10 % / 6/20; **28 % [19, 41] / 6/20** (void falls 54 → 31, tips 9 → 26); **55 % [42, 67] / 7/20** (40 ok / 8 void / 32 tips) | idm/loop_4 |
| 2368391 | loop_5 on the 55 % arm: gate 0; erode 1; gate 0 + erode 1; gate 0 without gt odom | 4 × ~50, **partial (107/197 at 04:40 UTC)** | gate0 9/19 = 47 % / 5/7; erode1 8/24 = 33 % / 6/7; gate0+erode1 7/15 = 47 % / 3/9; gate0_noodom 13/22 = 59 % [39, 77] / 2/6 | idm/loop_5 |

### 8b. Lens sweeps

Cancelled before any trial ran: 2367865, 2367885, 2367970, 2367971 (wrong task-file path).

| Job | What | n | Result | Path |
|---|---|---|---|---|
| 2367839 | px_acc_1: camera 1 cm and 2 cm, d20 + flat; maps bagged | 20 | d20 fell 5/6 in each arm; flat no falls; rim falls 9/12 | idm/px_acc_1 |
| 2367918 | px_acc_2: same worlds, oracle control, camera chain recorded | 14 | 0/12 rim falls; camera map errors match the camera arm's | idm/px_acc_2 |
| 2367852 | px_lat_1: camera; planar terrain / raw map bagged; probe; CPU sampler | 26 | 1/26; walking map rate 0.6-1.65 Hz; planar arrives 0.2-0.33 s after the cloud stamp | idm/px_lat_1 |
| 2368046 | px_lat_2: ungated cloud + truth bagged (6); r30 and r60 (13 each) | 32 | lag ~170 ms; r30/r60 0/26 [0, 13] | idm/px_lat_2 |
| 2368154, 2368216, 2368228 | lat open-loop replay through mapping + decomposition, arms A-I | 5 recorded trials | A reproduces live maps; restamp (B, H) leaves gate-off flat missed-solid at 0.30-0.38; min range 0.40 (C) lowers it to 0.15-0.31 | inv/px_lat/replay |
| 2367876 | px_cspace_1: camera, planar terrain + foothold commands bagged | 18 | 0/18; C-space replay matches logs (median per-leg error 1 cell) | idm/px_cspace_1 |
| 2368087 | px_cspace_2: prior 0.6 m; cspace 0.06; both | 12 + 4 / 12 / 12 | 0/16 (all tips at 2.3-4.7 s); 1/12; 0/12 | idm/px_cspace_2 |
| 2367900 | px_void_1: best arm, camera; planar terrain bagged | 30 | 1/30, 20 foot_in_void; perceived edges: far −0.18, near +6.60, left +1.66, right +3.42 cm (791 messages) | idm/px_void_1 |
| 2368108 | px_void_2: marginSize 11; cspace 0.05 (voids 20/30) | 2 × 20 | 0/20 (void landings 5.3 per 1000, 11 tips); 0/20 (12.3 per 1000) | idm/px_void_2 |
| 2368045 | px_pose_1: gt + gate 15; near field + gate 15; depth images bagged on flat | 40 | gt: voids 2/15, flat 2/5. near: voids 3/15, flat 2/5. | idm/px_pose_1 |
| 2368139 | px_pose_2: default chain, depth bagged | 20 | flat 0.1 5/5; d20/d30 @0.1 0/10; d20 @0.2 0/5 | idm/px_pose_2 |
| 2368001 | px_compute_1: default vs r2 (true 2 cm decomposition); probe | 40 | ctrl 0/8 + 0/8; r2 0/12 + 0/12; node 144-163 → 16-21 ms | idm/px_compute_1 |
| 2368002 | px_compute_2: gt pose, ungated; r2 vs e2h | 40 | 3/20 vs 0/20 (p = 0.23) | idm/px_compute_2 |
| — | GPU elevation mapping agent (`perception_gpu_mapping`) | 6 paired worlds | not fresher; void cells walkable 8.8 % vs 4.0 %; not adopted | decisions.md |

### 8c. Offline analyses (existing plus new data)

| Lens | Analysis | n | Key result | Path |
|---|---|---|---|---|
| void | Touchdown and termination classification | 13682 / 36519 steps | 8.84 vs 0.47 per 1000 | /scratch/sselvaraj/gaitnet-runs/idm/inv/px_void/ |
| void sk1/sk2 | Independent detector; swap / Kitagawa decomposition; init-prior check | same | Composition ≤ 2 per 1000; prior row 27 % walkable | inv/px_void_sk1/; scratchpad/skvoid2/ |
| acc | Map vs truth, edge profiles, drift, fall sites | 34 trials; 492/522 (corrected 462/552) | Heights OK; rim/left dilation; prior | inv/px_acc/ |
| acc sk1/sk2 | Exposure-adjusted hazard; matched pairs; target sd | ~89k touchdowns; 209 vs 206 trials | 5-12x per near-edge touchdown; tilt-without-drop 24 % | scratchpad/sk1/; inv/px_acc/skeptic2/ |
| lat | Fail locations, gate replay, footprint, zero-candidates, front collapse | 570 / 270 trials | Prior 33 % vs 6 %; gate passes 42 % / 14 % | inv/px_lat/ |
| cspace | C-space replay; drift counterfactual; void falls | 2130 requests; 220 falls; 18312 footholds | 13.2 % of leg-requests contain a void cell; 59 % of fatal footholds oracle-rejected | inv/px_cspace/ |
| cspace sk1/sk2 | Per-leg starvation; matched-bin hazard; sign placebo | 374 / 270 trials | Front-leg starvation before tips; selection 17-48 % | inv/px_cspace_sk1/, inv/px_cspace_sk2/ |
| pose | Depth lag, edge lag, pose error, gate eval | 2075 frames; 1004 frames; 219 trials | One-frame lag; 3.0 / 5.5 cm forward shift | inv/px_pose/ |
| pose sk1/sk2 | Independent lag fit; fused-map edges vs drift | 1354 frames; 14 bags | Lag replicated; left shift grows with cell age | inv/px_pose_sk1/, inv/px_pose_sk2/ |
| compute | bench_post micro-benchmark (container) | synthetic map | applyKernelFunction 82 %; SDF 92 ms at 1 cm | inv/px_compute/ (bench_post.txt missing; reproduced by both skeptics) |
| critic | Best-arm tip-over vs front C-space | 80 trials | 25/32 tips starved | scratchpad/crit_cs.py, crit_cs2.py |

## 9. Instrument and bookkeeping corrections

- **Gate drop rate.** "16-36 % dropped" counts the standing phase. While walking it is 55-74 % at 0.1 m/s and 83-94 % at 0.2 m/s. compute's `analyze.py` gate_window pads ±6 s, which inflates its numbers.
- **Zero-candidate column.** `candidates` sums min(16, cells) over legs, so it cannot show a single starved leg. Use ctrl.log "CSpace true counts" per leg instead. `bench_score.py` now cuts requests at termination.
- **`firstvoid.py`.** It hardcodes inv_perc_1 as camera, mislabelling 30 oracle trials (correct n is 462/552), and it counts 115 first-void events after a bad_orientation termination.
- **Per-step CIs.** They treat steps as independent, but void landings end trials. Use trial-level tests (Fisher) for the n=20 fix tests. Those tests also include 6 cells no camera arm wins (gridvoid24_{20,30}_s{52,53,60}).
- **Time layer.** The CPU map's `time` layer is the last fusion time, not the insertion time. Prior and single-point cells have NaN time, so "never scanned" means never fused.
- **Misc.**
  - TDDBG touchdown heights are sampled every 200th update.
  - `bench_post.txt` was cited but is absent; the numbers were reproduced.
  - `drift_cf_px_cspace_1_c_scidm.csv` holds the 0.9 m view run, not 0.72.
  - The px_void_2 zero-candidate column is a filter artefact.
- **Retire `gt_pose_cloud.py` as a diagnostic.** Use `gt_tracking_odom.py` (verified 0.1 cm rms) for any true-pose test.

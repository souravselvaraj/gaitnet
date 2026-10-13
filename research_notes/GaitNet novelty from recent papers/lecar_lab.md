# LeCAR Lab (Guanya Shi, CMU RI) 2024-2026: legged/humanoid locomotion, contact planning, sim-to-real, sampling-based MPC. Material for repositioning GaitNet

Scope and method: I started from the lab's own publication list ([LeCAR publications](https://lecar-lab.github.io/publications.html)), which is a curated subset, and checked it against the arXiv author listing for Guanya Shi ([arXiv author search](https://arxiv.org/search/?query=Guanya+Shi&searchtype=author)). That listing has 94 results, 68 of them since Jan 2024. Abstracts come from the arXiv abs pages. Numbers for the papers closest to GaitNet come from the arXiv HTML full text. Many papers are co-authored with Guanya Shi but led by another group (NVIDIA GEAR, Amazon FAR, ETH, MIT/DeepMind, UW), and I label these "co-authored, not LeCAR-led". "Venue" is taken from the lab page or the arXiv comment field. Where neither names one, I write "arXiv, venue not verified".

## Which LeCAR Lab papers (2024-2026) address legged/humanoid locomotion, contact/footstep planning, agile/terrain-aware locomotion, sim-to-real/sim-to-sim gaps, whole-body control, sampling/model-based control with learning, safe RL, or adaptive control?

### Takeaway
LeCAR's 2024-2026 legged work clusters into six lines:
1. Training-free sampling-based MPC on full-order dynamics (DIAL-MPC, MBD, CoVO-MPC, DIDA, Feedback-Sampling MPC).
2. Closing dynamics gaps with learned residuals or parameter search (ASAP delta-action, SPI-Active, FADA, RAMBO).
3. Hierarchical RL-over-model-based-controller quadruped agility on discontinuous terrain (Agile Continuous Jumping, and CAJun before it).
4. Depth-based perceptive humanoid locomotion (RPL, PHP, LadderMan).
5. Contact-sequence-conditioned humanoid whole-body control (WoCoCo, SceneBot, HOVER, FALCON, SoFTA).
6. Learned safety/adaptation (ABS, BAS, Safe-Stop).

None of these selects discrete footholds with a learned policy, so GaitNet's "learned discrete contact decision" niche is not occupied by this lab.

### Cited Findings

**A. Sampling-based / model-based control combined with learning**
- **DIAL-MPC: "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing"**
  - Who and where: Haoru Xue, Chaoyi Pan, Zeji Yi, Guannan Qu, Guanya Shi. ICRA 2025, Best Paper Award Finalist — [arXiv 2409.15610](https://arxiv.org/abs/2409.15610), [project](https://lecar-lab.github.io/dial-mpc/), [lab page](https://lecar-lab.github.io/publications.html).
  - Core idea: training-free sampling MPC over full-order quadruped dynamics. It uses diffusion-style annealing of the sampling distribution, which the authors motivate by a landscape analysis of MPPI and by MPPI's link to single-step diffusion, to get both global coverage and local convergence — [arXiv abs](https://arxiv.org/abs/2409.15610).
  - Setup: Unitree Go2 (real) and H1 (sim). The planner runs at 50 Hz over a 200 Hz low-level loop. Walking and jumping use 2048 samples with a 20-step (0.4 s) horizon. It is written in JAX/Brax and runs on an RTX 4090 desktop over Ethernet, with mocap state — [arXiv HTML](https://arxiv.org/html/2409.15610v1).
  - Results in the HTML text:
    - Walking tracking error: DIAL 0.024, MPPI-explore 0.190, MPPI-exploit 0.230, goal-conditioned RL (GCRL, PPO, 500M steps) 0.119.
    - Crate climbing success (0.6 m crate, 10 trials): 0.9 vs GCRL 0.6 vs best MPPI 0.3.
    - Sequential jumping contact reward: 0.885 vs GCRL 0.855 vs MPPI 0.44.
    - With a 10 kg payload, walking error was 0.036 vs GCRL 0.517 — [arXiv HTML](https://arxiv.org/html/2409.15610v1).
  - Inconsistency: the abstract claims a 13.4x reduction over MPPI, but the table values (0.024 vs 0.190/0.230) imply about 8-10x, and the text also cites 3.9x elsewhere — [arXiv abs](https://arxiv.org/abs/2409.15610) vs [arXiv HTML](https://arxiv.org/html/2409.15610v1).
  - Contacts and gaits:
    - Walking uses a *prescribed* trotting gait (a foot-height generator plus a gait reward).
    - Sequential jumping gives four foot placements plus a CoM target per stage, with a staged contact reward.
    - The crate-climb leap and edge catch emerge from the planner.
  - Stated limitations: it needs fast simulation for samples, perception is not integrated (climbing uses privileged environment information), and model mismatch is handled only by noise injection. Contact forces are disabled except at the feet — [arXiv HTML](https://arxiv.org/html/2409.15610v1).
- **Model-Based Diffusion for Trajectory Optimization (MBD).** Chaoyi Pan, Zeji Yi, Guanya Shi, Guannan Qu. NeurIPS 2024. It computes the diffusion score from the known model rather than from data, so it needs no training data (and can use imperfect data). It outperforms RL and sampling baselines on contact-rich tasks — [arXiv 2407.01573](https://arxiv.org/abs/2407.01573), [project](https://lecar-lab.github.io/mbd/), [lab page](https://lecar-lab.github.io/publications.html).
- **CoVO-MPC: Theoretical Analysis of Sampling-based MPC and Optimal Covariance Design.** Zeji Yi, Chaoyi Pan, Guanqi He, Guannan Qu, Guanya Shi. Jan 2024 (arXiv, venue not verified). It shows MPPI converges at least linearly on quadratic problems (including time-varying LQR), schedules the sampling covariance accordingly, and beats MPPI by 43-54% in simulation and quadrotor flight — [arXiv 2401.07369](https://arxiv.org/abs/2401.07369).
- **Global Convergence of Sampling-Based Nonconvex Optimization through Diffusion-Style Smoothing (DIDA).** Zeji Yi, Chaoyi Pan, Guanya Shi, Guannan Qu. TMLR 2026. It treats sampling-based optimization as gradient descent on a smoothed objective and exposes a coverage-vs-optimality trade-off. The annealed algorithm DIDA provably converges globally — [arXiv 2605.16520](https://arxiv.org/abs/2605.16520), [lab page](https://lecar-lab.github.io/publications.html).
- **Hybrid Feedback Sampling for Sample-Efficient Model Predictive Control.** Chaoyi Pan, Zeji Yi, John Zhang, Zachary Manchester, Guannan Qu, Guanya Shi. Aug 2026, arXiv, venue not verified. It samples from an optimized feedback policy to balance local and global search, converges faster than MPPI, and succeeds on unstable humanoid and dexterous tasks, with real-humanoid validation — [arXiv 2608.19443](https://arxiv.org/abs/2608.19443).
- **Whole-Body Model-Predictive Control of Legged Robots with MuJoCo.** John Z. Zhang, Taylor Howell, Zeji Yi, Chaoyi Pan, Guanya Shi, Guannan Qu, Tom Erez, Yuval Tassa, Zachary Manchester. arXiv comment says "to appear at ICRA 2026". Co-authored, not LeCAR-led (MIT/CMU + Google DeepMind).
  - Method: real-time iLQR on MuJoCo dynamics with finite-difference derivatives, on Go1, Go2 and H1 — [arXiv 2503.04613](https://arxiv.org/abs/2503.04613).
  - Rates: iLQR at about 50 Hz with TVLQR feedback at about 300 Hz. Horizon 0.35 s (quadruped).
  - Contacts: contact is *not* prescribed. The gait enters only as a cost residual, so the solver "can discover new contact modes", and foot placement emerges from the cost.
  - Results and limitations: TVLQR feedback improved H1 trotting cost by about 30%. The authors note iLQR struggles with contact-mode exploration compared with sampling methods, and that it relies on mocap — [arXiv HTML](https://arxiv.org/html/2503.04613).
- **TD-M(PC)²: Improving Temporal Difference MPC Through Policy Constraint.** Haotian Lin, Pengcheng Wang, Jeff Schneider, Guanya Shi. L4DC 2026. It finds that SAC-style value learning in TD-MPC2 persistently overestimates values because of a policy mismatch. A policy-regularization term fixes this at no extra compute, with large gains on 61-DoF humanoid tasks — [arXiv 2502.03550](https://arxiv.org/abs/2502.03550), [lab page](https://lecar-lab.github.io/publications.html).
- Wheeled-robot adaptive sampling MPC (not legged; included only as method precedent):
  - **AnyCar to Anywhere** (ICRA 2025): a transformer dynamics model with sampling MPC, beating specialist models by up to 54% — [arXiv 2409.15783](https://arxiv.org/abs/2409.15783).
  - **Agile Mobility with Rapid Online Adaptation via Meta-learning and Uncertainty-aware MPPI** (ICRA 2025 per the arXiv comment) — [arXiv 2410.06565](https://arxiv.org/abs/2410.06565).

**B. Sim-to-real / sim-to-sim dynamics-gap closure**
- **ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills.** Tairan He, Jiawei Gao, et al., Guanya Shi (with NVIDIA co-authors). RSS 2025 — [arXiv 2502.01143](https://arxiv.org/abs/2502.01143), [project](https://agile.human2humanoid.com/).
  - Delta-action model: Δa = π^Δ(s, a), trained with PPO starting from recorded real states. Its reward penalizes the gap between the simulated next state and the recorded real next state, plus an action-magnitude regularizer.
  - Use: the frozen delta model turns the simulator into f_sim(s, a + π^Δ(s, a)). The policy is fine-tuned there and deployed *without* the delta model — [arXiv HTML](https://arxiv.org/html/2502.01143v3).
  - Sim-to-sim results (open-loop, 1.0 s horizon, global MPJPE in mm):
    - IsaacSim: OpenLoop 80.8, SysID 77.6, DeltaDynamics 68.1, ASAP 37.9.
    - Genesis: 82.5, 76.5, 50.2, 36.9.
  - Closed loop, ASAP reaches 100% success in all cases shown; DeltaDynamics drops to 60-83% on some.
  - Sim-to-real on G1: MPJPE fell from 61.2 to 50.2 on an in-distribution kick and from 159 to 112 on an out-of-distribution motion — [arXiv HTML](https://arxiv.org/html/2502.01143v3).
  - Data cost: the full 23-DoF delta model needed more than 400 clips, so they trained a 4-DoF ankle model on 100 real clips. Two G1s were damaged during collection. MoCap is required — [arXiv HTML](https://arxiv.org/html/2502.01143v3).
- **SPI-Active: Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning.** Nikhil Sobanbabu, Guanqi He, Tairan He, Yuxiang Yang, Guanya Shi. CoRL 2025 Oral — [arXiv 2505.14266](https://arxiv.org/abs/2505.14266), [lab page](https://lecar-lab.github.io/publications.html).
  - Method: it identifies base mass/inertia (10 parameters, via log-Cholesky) and per-joint-type tanh actuator-saturation gains. It does this with CMA-ES over massively parallel Isaac Gym rollouts that replay real commands from real initial states.
  - Active exploration: Stage 2 picks command sequences, including a *discrete gait type* (pace, trot, bound, pronk), that minimize trace(FIM⁻¹).
  - Results on Go2 (normalized error vs vanilla = 1.00): forward jump 0.48, yaw jump 0.37, velocity tracking 0.58, attitude 0.73.
  - Limitations: offline, and it needs a multi-behavior policy — [arXiv HTML](https://arxiv.org/html/2505.14266v1).
- **FADA: Few-Shot Domain Adaptation via Dynamics Alignment for Humanoid Control.** Angchen Xie, Nikhil Sobanbabu, Ishayu Shikhare, Alan Wang, Max Simchowitz, Guanya Shi. CoRL 2026 per the lab page — [arXiv 2606.28476](https://arxiv.org/abs/2606.28476), [project](https://lecar-lab.github.io/FADA-humanoid/), [lab page](https://lecar-lab.github.io/publications.html).
  - Method: a transformer *planner* predicts a 6-step future of proprioceptive observations, and an *inverse-dynamics model* (IDM) maps history plus that predicted future to a 6-step action chunk. Only the IDM is adapted (LoRA) on about 2 min (about 6000 steps at 50 Hz) of target rollouts, with plain supervised learning and no rewards.
  - Sim-to-real (G1, T1): success rose from 20% zero-shot to 90% on completion tasks. Error fell 27.4% vs FADA-zs and 15.9% vs an RMA-style TF-DAgger baseline.
  - Sim-to-sim (IsaacSim→MuJoCo): normalized error fell 24.7% and 26.8% against the same two baselines.
  - Negative result: an end-to-end variant fine-tuned on future prediction got *worse*, so better prediction alone did not help execution.
  - Limitations: it needs non-trivial zero-shot performance, and the IDM may be task-specific — [arXiv HTML](https://arxiv.org/html/2606.28476v1).
- **RAMBO: RL-Augmented Model-Based Whole-Body Control for Loco-Manipulation.** Jin Cheng, Dongho Kang, Gabriele Fadini, Guanya Shi, Stelian Coros. RA-L per the arXiv comment. Co-authored, not LeCAR-led (ETH Zurich) — [arXiv 2504.06662](https://arxiv.org/abs/2504.06662).
  - Method: a single-rigid-body QP whole-body controller gives feedforward torques. A PPO policy (Isaac Lab, 4096 envs) outputs a base-acceleration correction and joint-position corrections, with gait phase and mode as inputs.
  - Go2 sim end-effector error: RAMBO 0.039 m vs pure RL 0.254, pure WBC 0.286, and RL-residual-without-WBC 0.077 (quadruped mode). Biped mode: 0.036 vs 0.431/0.447/0.347 — [arXiv HTML](https://arxiv.org/html/2504.06662v4).
- **Learning Sim-to-Real Humanoid Locomotion in 15 Minutes.** Younggyo Seo, Carmelo Sferrazza, Juyue Chen, Guanya Shi, Rocky Duan, Pieter Abbeel. arXiv Dec 2025. Co-authored, with code under amazon-far/holosoma. Off-policy FastSAC/FastTD3 trains G1 and T1 locomotion in about 15 min on one RTX 4090 under strong domain randomization — [arXiv 2512.01996](https://arxiv.org/abs/2512.01996), [lab page](https://lecar-lab.github.io/publications.html).
- **Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer** (Haoru Xue, Tairan He, et al., NVIDIA co-authored) — [arXiv 2512.01061](https://arxiv.org/abs/2512.01061). **VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation** (CVPR 2026) — [arXiv 2511.15200](https://arxiv.org/abs/2511.15200). Both are teacher-student visual sim-to-real for humanoid loco-manipulation, zero-shot on G1. Doorman is up to 31.7% faster than teleoperators, and VIRAL scales to 64 GPUs and 54 continuous cycles — [arXiv 2512.01061](https://arxiv.org/abs/2512.01061), [arXiv 2511.15200](https://arxiv.org/abs/2511.15200).

**C. Agile quadruped locomotion on discontinuous terrain (hierarchical RL over a model-based controller)**
- **Agile Continuous Jumping in Discontinuous Terrains.** Yuxiang Yang, Guanya Shi, et al., Byron Boots. ICRA 2025 per the lab page. The project page still says "Under Submission, ICRA 2024" (stale). Led by UW (Yang/Boots) with Guanya Shi as second author — [arXiv 2409.10923](https://arxiv.org/abs/2409.10923), [project](https://yxyang.github.io/jumping_cod/), [lab page](https://lecar-lab.github.io/publications.html).
  - Hierarchy:
    - Heightmap predictor at 100 Hz from D435i depth: 3-layer CNN + 1-layer GRU, trained DAgger-style in about 1 h.
    - RL motion policy at 100 Hz. It outputs stepping frequency (which advances a gait phase), swing-foot residuals on a Raibert heuristic, and body velocity/pitch-rate references.
    - Leg controller at 500 Hz: a *fixed bounding gait* with 4 contact modes; stance by a centroidal QP for ground reaction forces with friction cones; swing by IK + PD.
  - Training: the policy's reward includes the low-level controller's tracking cost, so it plans trackable motions. Camera latency (about 50 ms) and motor saturation (dynamometer-measured) are modeled — [arXiv HTML](https://arxiv.org/html/2409.10923v2).
  - Results on Go1:
    - 14-step staircase (3.5 m long, 2.8 m high) in 4.5 s.
    - Stepping stones up to 1.4 m/s; max step 0.6 m; max gap 0.8 m.
    - 100% success on 3 of 4 real terrains, but 20% on an unseen planked staircase.
    - Ablations: no heightmap randomization fails under small estimation errors; no system identification crashes into stair edges.
    - Limitation: sagittal-plane motion only — [arXiv HTML](https://arxiv.org/html/2409.10923v2).
- **Agile But Safe (ABS): Learning Collision-Free High-Speed Legged Locomotion.** Tairan He, Chong Zhang, Wenli Xiao, Guanqi He, Changliu Liu, Guanya Shi. RSS 2024, Outstanding Student Paper Finalist. An agile policy and a recovery policy are switched by a learned reach-avoid (RA) value network. It runs on a quadruped among static and dynamic obstacles, where prior conservative controllers were limited to under 1.0 m/s — [arXiv 2401.17583](https://arxiv.org/abs/2401.17583), [project](https://agile-but-safe.github.io/), [lab page](https://lecar-lab.github.io/publications.html).
- **Bridging Adaptivity and Safety (BAS): Learning Agile Collision-Free Locomotion Across Varied Physics.** Yichao Zhong, Chong Zhang, Tairan He, Guanya Shi. Jan 2025, arXiv, venue not verified. It extends ABS with a physical-parameter estimator and an adaptive RA value. Results: 50% better safety in sim; on hardware, a 19.8% speed gain and 2.36x lower collision rate than ABS, with payloads up to 8 kg — [arXiv 2501.04276](https://arxiv.org/abs/2501.04276).

**D. Perceptive humanoid locomotion (depth → policy, distillation)**
- **RPL: Learning Robust Humanoid Perceptive Locomotion on Challenging Terrains.** Yuanhang Zhang, Younggyo Seo, et al., Guanya Shi. CoRL 2026 per the lab page — [arXiv 2602.03002](https://arxiv.org/abs/2602.03002), [lab page](https://lecar-lab.github.io/publications.html).
  - Training: per-terrain PPO experts use a privileged 1.6×1.0 m heightmap at 0.1 m resolution, including a stepping-stone expert. They are distilled DAgger-style into a transformer on *raw depth*: 10 Hz in sim, 2 ZED 2i at 15 Hz on the real G1.
  - Stepping stones: 25 cm stones with 60 cm gaps on the real robot (sim: 0.25-0.40 m stones, 0.05-0.70 m gaps).
  - Results: the real whole course succeeded 6/10. Stepping-stone failures are attributed to landings near edges, because "small noise or drift pushes landings to the boundary". There is no explicit foothold selection, only a foot-edge penalty — [arXiv HTML](https://arxiv.org/html/2602.03002v3).
- **Perceptive Humanoid Parkour (PHP): Chaining Dynamic Human Skills via Motion Matching.** Zhen Wu, Xiaoyu Huang, et al. RSS 2026. Motion matching chains human skills into long-horizon parkour references, and tracking experts are distilled into one depth student. The G1 climbs obstacles up to 1.25 m (96% of its height) — [arXiv 2602.15827](https://arxiv.org/abs/2602.15827), [lab page](https://lecar-lab.github.io/publications.html).
- **LadderMan: Learning Humanoid Perceptive Ladder Climbing.** Siheng Zhao, Yuanhang Zhang, et al., Guanya Shi. Jun 2026, arXiv, venue not verified. Multiple climbing experts are distilled into a depth visuomotor policy, with vision foundation models bridging the depth gap, and transfer zero-shot to hardware — [arXiv 2606.05873](https://arxiv.org/abs/2606.05873).

**E. Humanoid whole-body control with contact conditioning**
- **WoCoCo: Learning Whole-Body Humanoid Control with Sequential Contacts.** Chong Zhang, Wenli Xiao, Tairan He, Guanya Shi. CoRL 2024 Oral per the lab page — [arXiv 2406.06005](https://arxiv.org/abs/2406.06005), [project](https://lecar-lab.github.io/wococo/), [lab page](https://lecar-lab.github.io/publications.html).
  - Task decomposition: each task is split into contact stages, each with a contact goal and a task goal.
  - Rewards: a task-agnostic dense contact reward (bonus per correct contact, penalty per wrong one), a stage-count reward, and a count-based curiosity reward. The curiosity reward hashes observations with a frozen random network.
  - What the policy sees: for parkour, the corner points of the next two stones per foot.
  - Contact sequences are *user-specified heuristically*. Integration with a high-level contact planner is named as future work.
  - Ablations: without stage-count reward, the robot "intentionally avoided fulfilling contact goals". Without curiosity, it under-explored and "did not move" — [arXiv HTML](https://arxiv.org/html/2406.06005v2).
- **SceneBot: Contact-Prompted General Humanoid Whole Body Tracking with Scene-Interaction.** Sirui Chen, Shibo Zhao, Zhen Wu, Jiaman Li, Guanya Shi, C. Karen Liu. Jun 2026, arXiv, venue not verified. Co-authored. A tracking policy is conditioned on reference motion plus *per-link contact labels*, trained on 7.5 h of reconstructed data, and handles terrain traversal and carrying a box upstairs — [arXiv 2606.27581](https://arxiv.org/abs/2606.27581).
- **HOVER** (ICRA 2025; NVIDIA-led, co-authored) distills several control modes into one whole-body policy via kinematic motion imitation — [arXiv 2410.21229](https://arxiv.org/abs/2410.21229).
- **FALCON** (L4DC 2026 Oral) uses a dual-agent lower-body/upper-body split for force-adaptive loco-manipulation, with 2x more accurate upper-body joint tracking — [arXiv 2505.06776](https://arxiv.org/abs/2505.06776), [lab page](https://lecar-lab.github.io/publications.html).
- **Hold My Beer / SoFTA** (arXiv May 2025) splits slow and fast agents (upper body 100 Hz, lower body 50 Hz) and gets 2-5x lower end-effector acceleration — [arXiv 2505.24198](https://arxiv.org/abs/2505.24198).
- **ResMimic** (arXiv Oct 2025, co-authored) puts a residual policy on a general motion tracker — [arXiv 2510.05070](https://arxiv.org/abs/2510.05070).
- **ResGAC: Precise SE(3) End-Effector Tracking in Whole-Body Humanoid Control** (arXiv Oct 2026, co-authored) adds residual RL on geometric admittance control: 90% peg-in-hole success vs 50% for SONIC — [arXiv 2610.09479](https://arxiv.org/abs/2610.09479).
- Teleoperation, behavior-model and data papers, lower relevance to footstep planning:
  - **H2O** (IROS 2024) — [arXiv 2403.04436](https://arxiv.org/abs/2403.04436).
  - **OmniH2O** (CoRL 2024) — [arXiv 2406.08858](https://arxiv.org/abs/2406.08858).
  - **BFM-Zero** (ICLR 2026) — [arXiv 2511.04131](https://arxiv.org/abs/2511.04131).
  - HDMI, OmniRetarget, TWIST2 — [lab page](https://lecar-lab.github.io/publications.html).

**F. Safe RL / safety filters**
- **Humanoid Safe Stop via Learned Stoppability Value.** Junfeng Long, Pieter Abbeel, Koushil Sreenath, Roberto Horowitz, Guanya Shi, C. Karen Liu. Sep 2026, arXiv, venue not verified. It frames emergency stopping as a reach-avoid problem. Two stoppability estimators (one outcome-supervised, one Hamilton-Jacobi backup) must agree before the robot commits to stopping; otherwise it falls back to damping — [arXiv 2609.02358](https://arxiv.org/abs/2609.02358).
- **Q-learning-based Model-free Safety Filter** (Nov 2024): a plug-and-play Q-learned filter that removes unsafe actions from a nominal policy. Tested on toy systems and a soft limb, not on legs — [arXiv 2411.19809](https://arxiv.org/abs/2411.19809).

**G. Surveys / reviews**
- "Evolution of Humanoid Locomotion Control" (Yan Gu, Guanya Shi, Fan Shi, et al.), Science Robotics 2026 — [Science](https://www.science.org/doi/abs/10.1126/scirobotics.aed3973), [lab page](https://lecar-lab.github.io/publications.html). I did not read its content (paywalled).
- "Humanoid Locomotion and Manipulation: Current Progress and Challenges in Control, Planning, and Learning" (Gu, Li, ... Guanya Shi, ... Ye Zhao), arXiv Jan 2025. It compares model-based and learning-based paradigms — [arXiv 2501.02116](https://arxiv.org/abs/2501.02116).

**H. Older foundational work (pre-2024)**
- **CAJun: Continuous Adaptive Jumping using a Learned Centroidal Controller.** Yuxiang Yang, Guanya Shi, Xiangyun Meng, Wenhao Yu, Tingnan Zhang, Jie Tan, Byron Boots. CoRL 2023. RL picks gait timing, base velocity and swing-foot targets over a 500 Hz QP centroidal leg controller. Reformulating the stance QP as clipped least squares sped training about 10x. Results on Go1: about 20 min of training on one GPU, and gaps up to 70 cm (over 40% wider than prior methods) — [arXiv 2306.09557](https://arxiv.org/abs/2306.09557), [PMLR](https://proceedings.mlr.press/v229/yang23b.html).
- **SafeDPA: Safe Deep Policy Adaptation.** Wenli Xiao, Tairan He, John Dolan, Guanya Shi. ICRA 2024 (arXiv Oct 2023). It jointly learns an adaptive policy and a dynamics model in sim, fine-tunes the model with few real samples, and adds a CBF safety filter. It reports a 300% higher safety rate under unseen disturbances on an RC car — [arXiv 2310.08602](https://arxiv.org/abs/2310.08602), [CMU RI](https://www.ri.cmu.edu/publications/safe-deep-policy-adaptation).
- Other foundational adaptive-control work: Neural-Fly (Science Robotics 2022) and Meta-Adaptive Nonlinear Control (NeurIPS 2021) — [lab page](https://lecar-lab.github.io/publications.html).

### Inferences
- LeCAR's locomotion stacks almost always keep a model-based layer under a learned or sampled decision layer: Agile Jumping, CAJun, RAMBO, and DIAL-MPC, which feeds a low-level loop. GaitNet's "policy picks footsteps, MPC makes torques" architecture is therefore in the same family. Its novelty must come from *what* the learned layer decides (discrete, gait-free contact events), not from the hierarchy itself.
- In the lab's quadruped locomotion work, the gait is still prescribed or phase-driven:
  - DIAL-MPC walking uses a trot foot-height generator.
  - Agile Jumping uses a fixed bounding gait whose frequency is learned.
  - RAMBO takes gait phase as input.
  - Exceptions where contacts emerge: whole-body iLQR MPC (gait only as a cost residual) and DIAL's crate climb.

  This supports positioning GaitNet as "learned *contact-event* selection" rather than "learned gait timing".
- The lab's strongest sim-to-sim evidence (ASAP open-loop MPJPE roughly halved; FADA 25-27% error reduction) is for *motion tracking* policies. No LeCAR paper closes a gap between two different *controllers* (convex MPC vs NMPC+WBC), so applying these tools to GaitNet's controller gap would be new.

### Gaps
- Several 2026 entries are arXiv-only and their venues are unverified: SceneBot, LadderMan, Safe-Stop, Hybrid Feedback Sampling, ResGAC, and BAS (2025). CoVO-MPC's venue is also not verified.
- I did not read the Science Robotics 2026 review (paywalled), so its content is unknown here.
- The ABS project page's specific peak-speed and success numbers were not fetched; only the abstract-level claims are used.
- The DIAL-MPC 13.4x claim could not be reconciled with the extracted table values; the paper appears internally inconsistent.
- "Flow Policy Gradients for Robot Control" (arXiv 2602.02481) appears in the author listing, but I did not read it and it is not assessed here.
- Flying Calligrapher (RA-L 2024, contact-aware motion/force planning) is excluded because it is aerial manipulation, not legged — [arXiv 2407.05587](https://arxiv.org/abs/2407.05587).

## Which of these ideas directly address GaitNet's specific weaknesses?

### Takeaway
Five LeCAR ideas map onto concrete GaitNet weaknesses:
- ASAP's delta-action model, FADA's planner/IDM split and SPI-Active's sampling-based SysID → the training-vs-deployment controller gap.
- WoCoCo's dense contact, stage-count and curiosity rewards → the no-op bias.
- DIAL-MPC/MBD-style annealed sampling → a model-based replacement or verifier for the ContactNet candidate generator.
- Agile Jumping's 100 Hz learned heightmap predictor (and RPL's raw-depth students) → the 3-4 Hz perception bottleneck.
- ABS/BAS reach-avoid switching → combining GaitNet with the fixed OCS2 trot that currently beats it.

### Cited Findings
- **Weakness: sim-to-sim gap between training controller (convex MPC, Isaac Lab) and deployment controller (OCS2 NMPC + WBC).**
  - ASAP learns a residual Δa = π^Δ(s, a) by PPO so that sim(s, a + Δa) reproduces target-domain transitions. It then fine-tunes the policy in that corrected sim and drops the delta at deployment — [arXiv HTML](https://arxiv.org/html/2502.01143v3).
  - ASAP was validated *sim-to-sim* (IsaacGym→IsaacSim, IsaacGym→Genesis). It beats SysID and DeltaDynamics there: 1.0 s open-loop MPJPE 37.9 vs 77.6/68.1 mm (IsaacSim) — [arXiv HTML](https://arxiv.org/html/2502.01143v3).
  - ASAP's stated cost is data volume: the 23-DoF delta needed more than 400 clips — [arXiv HTML](https://arxiv.org/html/2502.01143v3).
  - FADA freezes a high-level planner and LoRA-adapts only the inverse-dynamics layer on about 2 min of target rollouts. In sim-to-sim (IsaacSim→MuJoCo) that cut normalized error by 24.7%, and in sim-to-real it raised success from 20% to 90% — [arXiv HTML](https://arxiv.org/html/2606.28476v1).
  - SPI-Active fits physical parameters (inertia, actuator saturation) by CMA-ES over parallel GPU rollouts, cutting Go2 errors to 0.37-0.73 of vanilla — [arXiv HTML](https://arxiv.org/html/2505.14266v1). ASAP reports that SysID-style correction accumulates error over long horizons — [arXiv HTML](https://arxiv.org/html/2502.01143v3).
  - RAMBO shows that adding an RL correction (base-acceleration and joint-position residuals) on a QP whole-body controller beats both pure RL and the pure QP controller by large margins: end-effector error 0.039 vs 0.254/0.286 m — [arXiv HTML](https://arxiv.org/html/2504.06662v4).
- **Weakness: policy reluctant to step (85-95% no-ops) unless the body drifts ahead of the feet.**
  - WoCoCo's ablations show the same failure mode in a contact-sequence RL setting. Without a stage-count reward, the policy "intentionally avoided fulfilling contact goals". Without curiosity, it "did not move". The fix was dense per-contact rewards, stage-count rewards and count-based curiosity, the last active only in the first curriculum phase — [arXiv HTML](https://arxiv.org/html/2406.06005v2).
  - Agile Jumping and CAJun parameterize stepping as a continuous *stepping frequency*, or gait timing, that advances a phase, rather than a per-tick step/no-op choice. Agile Jumping's reward includes the low-level controller's tracking cost so that planned motions stay trackable — [arXiv HTML](https://arxiv.org/html/2409.10923v2), [arXiv 2306.09557](https://arxiv.org/abs/2306.09557).
- **Weakness: dependence on a candidate generator (ContactNet) that defines the action set.**
  - DIAL-MPC optimizes full-order dynamics without training, at 50 Hz with 2048 samples on one RTX 4090. It already handles stage-wise specified foot placements (sequential jumping onto plates) with a contact reward, and discovers a leap/edge-catch on crate climbing — [arXiv HTML](https://arxiv.org/html/2409.15610v1).
  - DIAL-MPC's crate-climbing success was 0.9 vs 0.6 for an RL policy and 0.3 for MPPI — [arXiv HTML](https://arxiv.org/html/2409.15610v1).
  - MBD derives the diffusion score from the model, needs no data, and beats RL and sampling baselines on contact-rich tasks — [arXiv 2407.01573](https://arxiv.org/abs/2407.01573).
  - DIDA formalizes a coverage-vs-optimality trade-off in sampling-based search and proves global convergence of annealed sampling — [arXiv 2605.16520](https://arxiv.org/abs/2605.16520).
  - CoVO-MPC shows that sampling-covariance design changes performance by 43-54% over MPPI — [arXiv 2401.07369](https://arxiv.org/abs/2401.07369).
- **Weakness: slow perception (depth → elevation map → plane decomposition at 3-4 Hz) causing falls.**
  - Agile Jumping replaces mapping with a learned heightmap predictor (CNN+GRU) from a D435i at 100 Hz, trained DAgger-style in about 1 h, with about 50 ms camera latency modeled in sim.
  - Its ablation: without heightmap randomization the policy fails under small estimation errors.
  - It reaches stepping stones at up to 1.4 m/s on Go1 — [arXiv HTML](https://arxiv.org/html/2409.10923v2).
  - RPL, PHP and LadderMan all distill privileged-heightmap experts into raw-depth students (RPL on the real G1: 2 cameras at 15 Hz). Domain randomization covers camera pose and FOV, depth noise of 0.1×depth, 5% dropout, and latency — [arXiv HTML](https://arxiv.org/html/2602.03002v3), [arXiv 2602.15827](https://arxiv.org/abs/2602.15827), [arXiv 2606.05873](https://arxiv.org/abs/2606.05873).
- **Weakness: a fixed OCS2 trot beats GaitNet on traversal success (45% vs at most 18%).**
  - ABS switches between an agile policy and a recovery policy with a learned reach-avoid value network — [arXiv 2401.17583](https://arxiv.org/abs/2401.17583).
  - BAS makes that switch adaptive to physics, giving 2.36x fewer collisions than ABS — [arXiv 2501.04276](https://arxiv.org/abs/2501.04276).
  - Safe-Stop commits to a learned action only when two value estimators agree, and otherwise falls back — [arXiv 2609.02358](https://arxiv.org/abs/2609.02358).
  - The residual-on-a-strong-base pattern recurs across the lab: RAMBO (RL on a QP WBC), ResMimic (residual on a general tracker), ResGAC (residual RL on admittance control, 90% vs 50% peg-in-hole) — [arXiv 2504.06662](https://arxiv.org/abs/2504.06662), [arXiv 2510.05070](https://arxiv.org/abs/2510.05070), [arXiv 2610.09479](https://arxiv.org/abs/2610.09479).
- **Weakness: two concurrent swings help only when the body is pulled along.**
  - Whole-body iLQR MPC lets contact modes emerge because the gait is only a cost residual — [arXiv HTML](https://arxiv.org/html/2503.04613).
  - DIAL-MPC still imposes a trot for walking — [arXiv HTML](https://arxiv.org/html/2409.15610v1).

  No LeCAR paper isolates when overlapping swings help on quadrupeds.

### Inferences
- **ASAP on the controller, not on the robot.** Collect paired rollouts: identical footstep sequences executed by (a) Isaac Lab + convex MPC and (b) Gazebo + OCS2 NMPC + WBC. Train a delta model on the *low-level command* (e.g. GRFs or torques), so that "convex MPC + Δ" reproduces OCS2+WBC transitions. Then fine-tune GaitNet in Isaac Lab with the frozen Δ.
  - This keeps GaitNet's discrete action untouched, which matters because ASAP's delta is additive on a continuous action.
  - Gazebo data is cheap, so the full-DoF delta model (ASAP's bottleneck, more than 400 clips) is affordable.
  - This would be a novel "controller-gap" use of delta-action learning; LeCAR applies it only to sim-vs-real *physics* gaps.
- **FADA-style split.** Treat GaitNet as FADA's frozen "planner" and insert a small learned adapter between footstep decisions and the deployment controller, for example foothold offsets, swing timing or reference smoothing for OCS2. LoRA-adapt only that adapter on minutes of Gazebo or hardware rollouts.
  - FADA's negative result suggests adapting the execution layer rather than retraining the decision layer on prediction losses.
- **SPI-Active alone will not close the convex-MPC-vs-OCS2 gap.** That gap is structural (different models and solvers), not a parameter error. SPI-Active is still useful for Go1 hardware mass, inertia and actuator saturation; ASAP's comparison shows SysID underperforms a learned residual on long horizons.
- **No-op bias.** Port WoCoCo's recipe into GaitNet's reward:
  - a per-correct-touchdown dense reward (a valid foothold reached);
  - a "contacts completed" stage-count term;
  - count-based curiosity on (base state, foot states) early in training.

  WoCoCo documents that without these an RL policy learns to avoid committing to contacts, which matches GaitNet's 85-95% no-op rate. Alternatively, add Agile Jumping's "low-level tracking cost in the planner reward" so that stepping early (keeping the support polygon under the CoM) is rewarded before drift forces it.
- **Replace or verify ContactNet with annealed sampling.** Use DIAL-MPC/MBD-style annealed sampling in a GPU (JAX/Brax or MJX) model to score each of the 65 discrete options (16 per leg × 4 + no-op) with short full-order rollouts. This can serve as an offline teacher that labels candidate quality, removing the hand-designed or learned ContactNet ranking. It could also run online as a verifier on GaitNet's top-k candidates.
  - DIDA's coverage-vs-optimality result gives a principled way to set the candidate count (16 per leg) and the annealing schedule.
  - Repositioning: "GaitNet amortizes a sampling-based contact planner into a single forward pass", the learning analogue of DIAL-MPC for discrete contacts. DIAL-MPC itself lists "learning a nominal policy/value" as future work.
- **Perception.** Swap the 3-4 Hz elevation-map + plane-decomposition stack for an Agile-Jumping-style learned predictor that outputs GaitNet's foothold-validity grid (the foothold C-space) directly from depth at camera rate. Train it DAgger-style with latency modeled and with randomized-heightmap training of the policy.
  - RPL's stepping-stone failures (edge landings from drift, with only an edge penalty) are a counter-example that favors GaitNet's explicit, margin-aware foothold choice. GaitNet can argue that explicit discrete footholds fix the failure end-to-end depth policies show on stepping stones.
- **Losing to the fixed trot.** Reposition GaitNet as a *residual contact-event layer on top of the OCS2 trot*: default to the trot, and let GaitNet override, delay or redirect individual steps when the trot's nominal foothold falls in a void. Gate the override with an ABS/BAS-style reach-avoid value, or a Safe-Stop-style two-estimator agreement, so the system is never worse than the trot.
  - This mirrors the lab's repeated "residual on a strong base" wins (RAMBO, ResMimic, ResGAC) and turns the 45% vs 18% comparison into a baseline-plus-improvement story.

### Gaps
- No LeCAR paper reports transferring a *footstep* or *contact-schedule* policy between two different low-level controllers. Whether ASAP or FADA residuals work when the upstream action is discrete is untested.
- DIAL-MPC compute numbers are for one MPC instance (2048 samples, 50 Hz, RTX 4090). Scoring 65 candidates online would cost about 65 such evaluations; no LeCAR source measures this, so feasibility is unverified.
- None of the perception papers report foothold-level accuracy or latency-induced fall rates in a form directly comparable to GaitNet's 3-4 Hz pipeline.

## Does LeCAR have work on discrete contact decisions, footstep/contact-sequence planning, stepping stones, or humanoid footstep planning?

### Takeaway
Partly. LeCAR treats contact *sequences as given goals* (WoCoCo, SceneBot, DIAL-MPC sequential jumping), crosses *discrete terrain* with continuous or phase-based outputs (Agile Continuous Jumping, CAJun, RPL, LadderMan), and chains *discrete skills* (PHP motion matching). I found no LeCAR paper in which a learned policy selects discrete footholds or contact events for a quadruped or humanoid. WoCoCo explicitly leaves "a high-level contact planner" as future work, which is an opening for GaitNet.

### Cited Findings
- WoCoCo: contact stages are "heuristically designed" by the user. The policy observes the next two stones' corner points per foot. The authors say the framework "can seamlessly be integrated with high-level contact planners" but did not do so. Stage switching is done externally by sensors or humans — [arXiv HTML](https://arxiv.org/html/2406.06005v2).
- SceneBot conditions whole-body tracking on per-link contact labels, so contact is a *prompt*, not a decision — [arXiv 2606.27581](https://arxiv.org/abs/2606.27581).
- DIAL-MPC sequential jumping: "each stage specifies four foot placements and a center-of-mass target", with plates about 10 cm and a new target every 1 s. Footholds are inputs, not decisions — [arXiv HTML](https://arxiv.org/html/2409.15610v1).
- Agile Continuous Jumping crosses stepping stones and stairs with a *fixed* bounding contact pattern. Foot placement is expressed only through residuals on a Raibert heuristic; there are no explicit landing-position outputs — [arXiv HTML](https://arxiv.org/html/2409.10923v2).
- CAJun: RL picks gait timing, base velocity and swing-foot targets over a QP leg controller (CoRL 2023, pre-2024) — [arXiv 2306.09557](https://arxiv.org/abs/2306.09557).
- RPL humanoid stepping stones (real: 25 cm stones, 60 cm gaps, 21 cm foot) are handled end-to-end from depth with a foot-edge penalty. Failures come from edge landings — [arXiv HTML](https://arxiv.org/html/2602.03002v3).
- PHP selects and chains discrete human skills via motion matching into long-horizon parkour references (G1, obstacles up to 1.25 m) — [arXiv 2602.15827](https://arxiv.org/abs/2602.15827).
- LadderMan climbs ladders, an inherently discrete-rung contact problem, with distilled depth policies. The abstract does not describe explicit rung or contact selection — [arXiv 2606.05873](https://arxiv.org/abs/2606.05873).
- Whole-body iLQR MPC (co-authored) lets contact modes emerge because the gait is only a cost residual, but foot placement is not planned explicitly — [arXiv HTML](https://arxiv.org/html/2503.04613).
- SPI-Active's active exploration makes a *discrete gait-type* choice (pace, trot, bound, pronk) inside its CMA-ES command optimization. This is the only discrete locomotion-mode decision I found, and it serves system identification, not locomotion — [arXiv HTML](https://arxiv.org/html/2505.14266v1).

### Inferences
- GaitNet can be pitched as the missing "high-level contact planner" that WoCoCo and DIAL-MPC assume: a learned, perception-conditioned, per-tick *discrete contact-event selector* that outputs exactly the foothold and contact goals these controllers consume.
- A concrete combination: a GaitNet output (leg, foothold) is a WoCoCo/SceneBot-style contact prompt or a DIAL-MPC stage target, and the downstream controller (MPC, or a learned contact-conditioned tracker) executes it. This moves GaitNet from "a planner that needs a convex MPC" to "a contact-decision layer that is controller-agnostic". That directly addresses the controller-gap weakness, provided the interface is footholds and timings.
- LeCAR's humanoid stepping-stone results (RPL) and quadruped jumping results show end-to-end or phase-based methods failing at edges or on unseen discontinuities: 20% success on an unseen planked staircase in Agile Jumping, and edge landings in RPL. Explicit foothold C-space reasoning is a defensible differentiator if GaitNet can show lower edge-landing rates.
- Humanoid footstep planning is an open extension: none of the LeCAR humanoid papers selects footholds explicitly, so a humanoid GaitNet variant (2 legs, at most 1 swing) on RPL-style stepping stones would be novel relative to this lab.

### Gaps
- I found no LeCAR paper that learns discrete footstep selection, contact-sequence planning, or humanoid footstep planning. This is based on the 68 post-2024 titles in the arXiv author listing plus the lab page; unlisted workshop papers could exist.
- Whether LadderMan or SceneBot internally use explicit contact targets beyond what their abstracts state was not checked in full text.

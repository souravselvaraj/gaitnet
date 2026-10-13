# Pieter Abbeel group (UC Berkeley / Amazon FAR), 2024-2026: legged and humanoid locomotion, contact planning and robot learning, read for GaitNet repositioning

Scope and method. I enumerated Abbeel's arXiv output from Jan 2024 to 11 Oct 2026 with an arXiv API author query (`au:"Pieter Abbeel"`, 400 most recent entries, which covers that whole window). I then pulled abstracts for every locomotion, humanoid, RL, world-model or generative-policy paper. For six key papers (RPL, PHP, FastSAC, FPO++, Workhorse, CQN-AS) I ran targeted extraction prompts against the arXiv HTML full text. Those details are model-extracted, not read line by line, so check any load-bearing number against the paper. PAWS details come from a search-result summary of its PMLR page, not a direct fetch. Every paper below has Abbeel as a co-author unless it is marked otherwise. The arXiv ID is the link. Venues are given (in bold) only where I saw them in the paper's own arXiv comment field or a proceedings page. Every other paper is an arXiv preprint as far as I checked, whether or not it is tagged "venue not verified".

GaitNet weaknesses referred to below as W1-W6: (W1) reluctant to step, 85-95% no-op; (W2) sim-to-sim gap between the convex MPC used in training and OCS2 NMPC+WBC used in deployment; (W3) the action set depends on the ContactNet candidate generator; (W4) slow perception, 3-4 Hz depth → elevation map → planes; (W5) two concurrent swings help only when the body is pulled along; (W6) a fixed OCS2 trot beats GaitNet on traversal (45% vs at most 18%).

## Q1. Which Abbeel-group papers (2024-2026) address legged or humanoid locomotion, and what does each do?

### Takeaway
From 2025 the group's locomotion output is almost entirely **humanoid**, coming out of an Amazon FAR cluster (Rocky Duan, Carmelo Sferrazza, Guanya Shi, C. Karen Liu, Younggyo Seo, with Abbeel as an Amazon Scholar). It has three strands: (a) perceptive humanoid locomotion and parkour learned end-to-end from depth (RPL, PHP, LadderMan); (b) fast and expressive RL for locomotion (FastTD3/FastSAC, FPO++, QF3, CQN-AS, BRC); (c) real-to-sim and human-data pipelines (GaussGym, PAWS, VideoMimic, OmniRetarget, Workhorse, VLK, PRISM). The only quadruped-specific paper is PAWS (CoRL 2024). FPO++ (Go2, Spot) and MuJoCo Playground include quadruped tasks. GaussGym's abstract does not name its sim-to-real robot platform. Several of the "Berkeley humanoid" examples in the brief are **not** Abbeel papers (see Gaps).

### Cited Findings

**Organizational context**
- Amazon hired Covariant co-founders Pieter Abbeel, Peter Chen and Rocky Duan and took a non-exclusive license to Covariant's models. About a quarter of Covariant's staff was slated to join Amazon — [About Amazon](https://aboutamazon.com/news/company-news/amazon-covariant-ai-robots); [The Decoder](https://the-decoder.com/amazon-acqui-hires-covariant-for-foundational-robotics-models-and-key-ai-talent/)
- Papers from 2024 to 2026 state that "Pieter Abbeel holds a concurrent appointment as an Amazon Scholar", and some say the work "is not associated with Amazon" (CQN-AS, Workhorse) — [CQN-AS](https://arxiv.org/abs/2411.12155); [Workhorse](https://arxiv.org/abs/2610.09117)
- Several 2025-2026 locomotion papers list "Amazon FAR (Frontier AI & Robotics)" as an affiliation: the FastSAC humanoid paper (its code is in FAR's "Holosoma" repo), RPL, PHP and FPO++ — [FastSAC](https://arxiv.org/abs/2512.01996); [RPL](https://arxiv.org/abs/2602.03002); [PHP](https://arxiv.org/abs/2602.15827); [FPO++](https://arxiv.org/abs/2602.02481)

**A. Perceptive and terrain locomotion (the most relevant to GaitNet)**
- **RPL: Learning Robust Humanoid Perceptive Locomotion on Challenging Terrains.** Y. Zhang, Y. Seo, J. Chen, Y. Yuan, K. Sreenath, P. Abbeel, C. Sferrazza, K. Liu, R. Duan, G. Shi. arXiv Feb 2026, venue not verified. — [arXiv 2602.03002](https://arxiv.org/abs/2602.03002)
  - Two stages. Stage 1 trains terrain-specific PPO experts (slopes, stairs, stepping stones) on a privileged 1.6 m × 1.0 m height map at 0.1 m resolution. Stage 2 distills them into a transformer student that fuses CNN features from several depth cameras — [arXiv 2602.03002](https://arxiv.org/html/2602.03002)
  - **Stepping stones are solved end-to-end with no footstep planner or explicit foothold selection.** The experts get a "foot-edge penalty" for stepping stones and a foothold penalty for stairs — [arXiv HTML](https://arxiv.org/html/2602.03002)
  - Real-world terrain: 20° slopes, stairs with 22/25/30 cm step lengths, and "25 cm by 25 cm stepping stones separated by 60 cm gaps", carrying a 2 kg payload — [arXiv 2602.03002](https://arxiv.org/abs/2602.03002)
  - Whole-course real success was 6/10 (3/5 without payload, 3/5 with 2 kg). Two of the four failures were on the stepping stones — [arXiv HTML](https://arxiv.org/html/2602.03002)
  - Sensing: ZED 2i depth at 15 Hz on the robot, downsampled to 48×27. The policy runs at 50 Hz on a Jetson Orin NX. Training models control delay U(0,20) ms, pixel dropout 0.05 and depth noise σ = 0.1·depth — [arXiv HTML](https://arxiv.org/html/2602.03002)
  - A custom ray-caster against dynamic robot meshes and static terrain meshes gives a claimed "5× speedup". Table 2 iteration times at 1024 envs on one L40S, for 1/2/4 cameras, are RPL 1.3/1.5/1.9 s vs IsaacSim Warp 3.5/5.9/9.1 s. The 5× figure matches only the 4-camera case — [arXiv HTML](https://arxiv.org/html/2602.03002)
  - Stepping-stone terrain levels for the bidirectional policy: expert 6.0; student with 1 camera 5.1, with 2-4 cameras 6.0. Omnidirectional: expert 5.6; student 3.0/4.5/4.6 with 1/2/4 cameras — [arXiv HTML](https://arxiv.org/html/2602.03002)
- **Perceptive Humanoid Parkour (PHP): Chaining Dynamic Human Skills via Motion Matching.** Z. Wu, X. Huang, L. Yang, Y. Zhang, X. Chen, P. Abbeel, R. Duan, A. Kanazawa, C. Sferrazza, G. Shi, C. K. Liu. arXiv Feb 2026, venue not verified. — [arXiv 2602.15827](https://arxiv.org/abs/2602.15827)
  - Motion matching is nearest-neighbor search (argmin squared L2) over a 27-D feature: future root trajectory at 0.33/0.67/1 s, local foot state, and root velocity. It runs over a locomotion database plus one database per parkour skill. When a skill is needed, the search is restricted to that skill's annotated "entry window". The clip then plays to its end and locomotion search resumes — [arXiv HTML](https://arxiv.org/html/2602.15827)
  - The composed motions are tracked by RL experts, which are distilled with DAgger+RL into one depth student: a 3-layer CNN on 58×87 depth at 30 Hz, observation delay randomized 60-80 ms, trained with 16,384 envs. **The choice of skill (step over, climb, vault, roll) is implicit in the student policy; there is no discrete selection head** — [arXiv HTML](https://arxiv.org/html/2602.15827)
  - Results: climbs a 1.25 m wall (96% of robot height); cat-vaults a 0.40 m × 0.50 m obstacle at up to 3.41 m/s; sim success 1.00/0.99/0.95 on 36/58/76 cm obstacles; multi-obstacle traversal of 48-60 s adapting to obstacles moved about 0.5 m — [arXiv HTML](https://arxiv.org/html/2602.15827)
- **LadderMan: Learning Humanoid Perceptive Ladder Climbing.** S. Zhao, Y. Zhang, Z. Lu, P. Abbeel, R. Duan, K. Sreenath, Y. Wang, C. K. Liu, G. Shi. arXiv Jun 2026, venue not verified. It targets "sparse footholds and handholds". Hybrid motion tracking learns several climbing experts from a single reference motion, and these are distilled into a depth-based visuomotor policy by hybrid IL+RL. Vision foundation models close the depth sim-to-real gap, and the policy transfers zero-shot to hardware — [arXiv 2606.05873](https://arxiv.org/abs/2606.05873)
- **GaussGym: An open-source real-to-sim framework for learning locomotion from pixels.** A. Escontrela, J. Kerr, A. Allshire, J. Frey, R. Duan, C. Sferrazza, P. Abbeel. arXiv Oct 2025, venue not verified. 3D Gaussian Splatting is a drop-in renderer inside IsaacGym, running at more than 100,000 steps/s on consumer GPUs. Scenes come from iPhone scans, GrandTour, ARKit and Veo-generated video. Sim-to-real is demonstrated (the robot platform is not named in the abstract), and RGB semantics help avoid "undesirable regions" — [arXiv 2510.15352](https://arxiv.org/abs/2510.15352)
- **PAWS: Learning Robotic Locomotion Affordances and Photorealistic Simulators from Human-Captured Data.** A. Escontrela, J. Kerr, K. Stachowicz, P. Abbeel. **CoRL 2024 (PMLR v270)**. A portable human-carried rig records camera poses, foot poses, terrain meshes and radiance fields, released as PAWS-Data (more than 10 h). A navigation affordance model is fine-tuned on traversability labels rendered from the radiance fields. PAWS-Sim is an IsaacSim photoreal simulator. On a **quadruped** the model follows paths and sidewalks and avoids people — [PMLR](https://proceedings.mlr.press/v270/escontrela25a.html); [ML Anthology](https://mlanthology.org/corl/2024/escontrela2024corl-learning)
- **VideoMimic: Visual Imitation Enables Contextual Humanoid Control.** A. Allshire, H. Choi, J. Zhang, D. McAllister, A. Zhang, C. M. Kim, T. Darrell, P. Abbeel, J. Malik, A. Kanazawa. arXiv May 2025, venue not verified. A real-to-sim-to-real pipeline reconstructs humans and the environment from everyday videos. A single policy, conditioned on the environment and global root commands, does stair ascent and descent and sits on chairs and benches on a real humanoid — [arXiv 2505.03729](https://arxiv.org/abs/2505.03729)
- **Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control.** Z. Li, X. B. Peng, P. Abbeel, S. Levine, G. Berseth, K. Sreenath. **IJRR 2024**. A dual-history architecture combines long-term and short-term I/O history. It adapts to time-invariant dynamics shifts and to time-variant changes "such as contact events". Task randomization adds robustness. On Cassie it walks, runs (a 400 m dash) and jumps — [arXiv 2401.16889](https://arxiv.org/abs/2401.16889)
- **Humanoid Safe Stop via Learned Stoppability Value.** J. Long, P. Abbeel, K. Sreenath, R. Horowitz, G. Shi, C. K. Liu. arXiv Sep 2026, venue not verified. Emergency stopping is cast as reach-avoid. A stop-probability estimator (supervised by "the actual outcomes of the fixed stop policy") is paired with a Hamilton-Jacobi reach-avoid estimator. The robot commits to the stop only when both agree, otherwise it falls back to a damping fall policy — [arXiv 2609.02358](https://arxiv.org/abs/2609.02358)

**B. Fast and expressive RL algorithms used for locomotion**
- **Learning Sim-to-Real Humanoid Locomotion in 15 Minutes (FastSAC/FastTD3).** Y. Seo, C. Sferrazza, J. Chen, G. Shi, R. Duan, P. Abbeel. arXiv Dec 2025, venue not verified. Humanoid locomotion trains in 15 min on a single RTX 4090 and transfers to Unitree G1 and Booster T1 under strong domain randomization (dynamics, rough terrain, pushes) — [arXiv 2512.01996](https://arxiv.org/abs/2512.01996)
  - Recipe: a C51 distributional critic; averaging the two Qs instead of clipped double-Q; observation normalization plus LayerNorm; joint-limit-aware action bounds; γ = 0.97 for locomotion; batches up to 8K; fewer than 10 reward terms; 16,384 envs for whole-body tracking. No perception or footstep planner — [arXiv HTML](https://arxiv.org/html/2512.01996)
- **FastTD3: Simple, Fast, and Capable RL for Humanoid Control.** Y. Seo, C. Sferrazza, H. Geng, M. Nauman, Z.-H. Yin, P. Abbeel. arXiv May 2025, venue not verified. Off-policy TD3 with parallel simulation, large-batch updates and a distributional critic solves a range of HumanoidBench tasks in under 3 h on one A100. It also runs in IsaacLab and MuJoCo Playground — [arXiv 2505.22642](https://arxiv.org/abs/2505.22642)
- **Flow Policy Gradients for Robot Control (FPO++).** B. Yi, H. Choi, H. G. Singh, X. Huang, T. E. Truong, C. Sferrazza, Y. Ma, R. Duan, P. Abbeel, G. Shi, K. Liu, A. Kanazawa. arXiv Feb 2026. The figure paths suggest an ICLR 2026 template, but the venue is not confirmed — [arXiv 2602.02481](https://arxiv.org/abs/2602.02481)
  - It trains flow-matching policies with a policy gradient that needs no likelihood. Changes from FPO: a per-sample ratio, and asymmetric SPO (PPO clipping for positive advantages, an SPO quadratic penalty for negative ones). At test time it uses zero-noise sampling — [arXiv HTML](https://arxiv.org/html/2602.02481)
  - Locomotion was trained in IsaacLab on Go2, Spot, H1 and G1. Sim-to-real was shown for Booster T1 locomotion and G1 motion tracking, qualitatively only. **On Spot, FPO++ produced more consistent trot gaits where Gaussian PPO tended to pronk, under identical rewards**, because flow policies capture action-dimension correlations that a diagonal Gaussian cannot. Costs: G1 locomotion reached return 25 in 23 min vs 19 min for PPO, and motion tracking was up to 3× slower in wall-clock — [arXiv HTML](https://arxiv.org/html/2602.02481)
- **QF3: Fast Flow RL with Filtered Q-Gradients.** C. M. Kim, B. Yi, D. McAllister, H. Choi, H. G. Singh, J. Cao, K. Goldberg, P. Abbeel, C. Sferrazza, A. Kanazawa. arXiv Oct 2026, venue not verified. An off-policy flow RL method: flow matching plus the critic's action gradient through a one-step flow prediction, applied only to action dimensions near the replay action. It claims to be the first off-policy flow RL to train humanoid locomotion from scratch with zero-shot hardware transfer, at 10× the wall-clock speed of FPO++ — [arXiv 2610.08789](https://arxiv.org/abs/2610.08789)
- **CQN-AS: Coarse-to-fine Q-Network with Action Sequence.** Y. Seo, P. Abbeel. arXiv Nov 2024 (v6 Nov 2025), venue not verified. A value-based method over discretized actions: 3 levels × 5 bins per dimension, with argmax per level, after which the interval zooms into the chosen bin. The critic outputs Q over a K-step action sequence through a GRU — [arXiv 2411.12155](https://arxiv.org/abs/2411.12155); [arXiv HTML](https://arxiv.org/html/2411.12155)
  - Results: it beats baselines on BiGym (25 sparse-reward humanoid mobile-bimanual tasks) and RLBench, and on HumanoidBench locomotion the largest gains are on Walk and Run. When action sequences were added, SAC and TD3 over-estimated values and failed on HumanoidBench Stand, while **CQN stayed robust to added no-op actions**. It underperforms CQN on DMC torque-controlled locomotion — [arXiv HTML](https://arxiv.org/html/2411.12155)
- **BRC: Bigger, Regularized, Categorical.** M. Nauman, M. Cygan, C. Sferrazza, A. Kumar, P. Abbeel. arXiv May 2025. High-capacity value models trained with cross-entropy and conditioned on learnable task embeddings, tested on 7 benchmarks and more than 280 tasks including humanoid control — [arXiv 2505.23150](https://arxiv.org/abs/2505.23150)
- Related scaling and replay studies (general RL, not locomotion-specific):
  - Value-Based Deep RL Scales Predictably (ICML 2025) — [arXiv 2502.04327](https://arxiv.org/abs/2502.04327)
  - Compute-Optimal Scaling for Value-Based Deep RL, which identifies "TD-overfitting" — [arXiv 2508.14881](https://arxiv.org/abs/2508.14881)
  - When Does Non-Uniform Replay Matter in RL?, which proposes truncated-geometric replay — [arXiv 2605.10236](https://arxiv.org/abs/2605.10236)
  - Prioritized Generative Replay, where a diffusion model densifies replay guided by curiosity or value "relevance functions" — [arXiv 2410.18082](https://arxiv.org/abs/2410.18082)
  - REPPO, on-policy pathwise policy gradients — [arXiv 2507.11019](https://arxiv.org/abs/2507.11019)
- Exploration and steerability:
  - MaxInfoRL (info-gain intrinsic reward plus Boltzmann exploration) — [arXiv 2412.12098](https://arxiv.org/abs/2412.12098)
  - SOMBRL, optimistic model-based RL with hardware results on an RC car — [arXiv 2511.20066](https://arxiv.org/abs/2511.20066)
  - Reward-Conditioned RL (RCRL), which conditions the agent on reward parameterizations, recomputes counterfactual rewards from shared replay, and "supports zero-shot behavioral adjustment at deployment" — [arXiv 2603.05066](https://arxiv.org/abs/2603.05066)
- Generative-policy improvement:
  - CFGRL: classifier-free guidance as a policy-improvement operator — [arXiv 2505.23458](https://arxiv.org/abs/2505.23458)
  - DiPOD: stable diffusion-policy RL via an on-policy ELBO regularizer — [arXiv 2606.13795](https://arxiv.org/abs/2606.13795)
  - Shortcut models: one-step diffusion/flow sampling with a single network — [arXiv 2410.12557](https://arxiv.org/abs/2410.12557)

**C. Simulators, benchmarks and architectures**
- **HumanoidBench.** C. Sferrazza, D.-M. Huang, X. Lin, Y. Lee, P. Abbeel. arXiv Mar 2024, venue not verified. Flat state-of-the-art RL "struggle[s] with most tasks", whereas a **hierarchical** approach on top of robust low-level policies (walking, reaching) does better — [arXiv 2403.10506](https://arxiv.org/abs/2403.10506)
- **MuJoCo Playground.** K. Zakka, …, Q. Liao, A. Allshire, …, K. Sreenath, …, C. Sferrazza, Y. Tassa, P. Abbeel. arXiv Feb 2025. An MJX framework covering quadrupeds, humanoids, hands and arms that trains "in minutes on a single GPU", with zero-shot sim-to-real from state and pixels — [arXiv 2502.08844](https://arxiv.org/abs/2502.08844)
- **mjlab.** K. Zakka, Q. Liao, B. Yi, L. Le Lay, K. Sreenath, P. Abbeel. arXiv Jan 2026. Isaac Lab's manager-based API on MuJoCo Warp, with velocity-tracking, motion-imitation and manipulation reference tasks — [arXiv 2601.22074](https://arxiv.org/abs/2601.22074)
- **Body Transformer (BoT).** C. Sferrazza, D.-M. Huang, F. Liu, J. Lee, P. Abbeel. arXiv Aug 2024, venue not verified. The robot body is a graph of sensors and actuators, with masked attention following that graph. It outperforms a vanilla transformer and an MLP for both IL and RL in task completion, scaling and compute — [arXiv 2408.06316](https://arxiv.org/abs/2408.06316)
- **Rodrigues Network (RodriNet).** **ICLR 2026**. Kinematics-aware neural operator for action learning — [arXiv 2506.02618](https://arxiv.org/abs/2506.02618)

**D. Humanoid whole-body control and loco-manipulation from human data (planner/tracker splits)**
- **Workhorse: Learning Robust Whole-Body Humanoid Loco-Manipulation from Human Data.** S. Hu, Q. Liao, Y. Chi, K. Zakka, Y. S. Shao, P. Abbeel, K. Sreenath. arXiv Oct 2026, venue not verified — [arXiv 2610.09117](https://arxiv.org/abs/2610.09117)
  - A visual planner predicts five link targets (torso, both wrists, **both feet**) as 58-step chunks (1.16 s), replanned at 5 Hz. An RL tracker at 50 Hz follows a six-sample, 0.2 s command window — [arXiv HTML](https://arxiv.org/html/2610.09117)
  - **Each policy is trained against synthetic models of the other's deployment errors.** The planner's input history gets random-walk rigid drift. The tracker's command gets a drift that grows linearly between random replans (0.2-1 s), up to 10 cm and 10° — [arXiv HTML](https://arxiv.org/html/2610.09117)
  - Box-sorting ablation over 100 episodes (success / fall):

    | Variant | No pushes | With pushes |
    | --- | --- | --- |
    | Full system | 77% / 1% | 64% / 15% |
    | Without command augmentation | 76% / 4% | 52% / 23% |
    | Without planner augmentation | 20% / 79% | 0% / 95% |

    — [arXiv HTML](https://arxiv.org/html/2610.09117)
- **VLK.** Y.-J. Wang, …, P. Abbeel, R. Duan, K. Sreenath, A. Kanazawa, C. Sferrazza, G. Shi, K. Liu. arXiv Jun 2026. It synthesizes 48,000 paired egocentric-image / language / kinematic trajectories in 3DGS-reconstructed scenes. A VLK policy predicts short-horizon whole-body kinematics, which a whole-body tracker executes on a G1 — [arXiv 2606.30645](https://arxiv.org/abs/2606.30645)
- **OmniRetarget.** L. Yang, X. Huang, Z. Wu, A. Kanazawa, P. Abbeel, C. Sferrazza, C. K. Liu, R. Duan, G. Shi. arXiv Sep 2025. Interaction-mesh retargeting minimizes Laplacian deformation under kinematic constraints and preserves agent-terrain and agent-object contacts. It augments one demonstration across embodiments, terrains and objects, producing more than 8 h of trajectories. Proprioceptive RL trained with only 5 reward terms and no curriculum performs parkour and loco-manipulation for up to 30 s on a G1 — [arXiv 2509.26633](https://arxiv.org/abs/2509.26633)
- **ResMimic.** A two-stage residual policy on top of a general motion-tracking base. Contact rewards and a curriculum virtual-object controller, tested on a real G1 — [arXiv 2510.05070](https://arxiv.org/abs/2510.05070)
- **PRISM: Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation.** **CoRL 2026**. Video-to-video generation turns a few real clips into hundreds of counterfactual ones. Reconstruction is contact-anchored real-to-sim, and the resulting depth-only policy is deployed zero-shot — [arXiv 2609.38172](https://arxiv.org/abs/2609.38172)
- Other related humanoid papers:
  - TWIST2, VR teleoperation data collection: 100 demos in 15 min — [arXiv 2511.02832](https://arxiv.org/abs/2511.02832)
  - SkillBlender, hierarchical RL that blends pretrained primitive skills — [arXiv 2506.09366](https://arxiv.org/abs/2506.09366)

**E. World models, video models and learned dynamics**
- **Contact-Aware Neural Dynamics.** C. Jing, J. K. Bandi, J. Ye, Y. Duan, P. Abbeel, X. Wang, S. Yi. arXiv Jan 2026. Implicit sim-to-real alignment: the off-the-shelf simulator is a base prior, and a learned contact-aware neural dynamics model refines simulated states from real observations. It uses tactile contact on robot **hands**, not legs. The model is used to predict policy performance and to refine sim-trained policies — [arXiv 2601.12796](https://arxiv.org/abs/2601.12796)
- **Residual Off-Policy RL for Finetuning BC Policies.** L. Ankile, Z. Jiang, R. Duan, G. Shi, P. Abbeel, A. Nagabandi. arXiv Sep 2025. The base policy is a black box. Lightweight per-step residuals are learned with sample-efficient off-policy RL from sparse binary rewards, including the claimed first real-world RL on a humanoid with dexterous hands — [arXiv 2509.19301](https://arxiv.org/abs/2509.19301)
- **Large Video Planner** (Dec 2025): an internet-scale video foundation model that produces zero-shot video plans, which are post-processed into robot actions — [arXiv 2512.15840](https://arxiv.org/abs/2512.15840)
- **DreamDojo** (Feb 2026): a world model pretrained on 44k h of egocentric human video with continuous latent actions, distilled to run in real time at 10.81 FPS. Abbeel is one of 30 authors; he is not the lead — [arXiv 2602.06949](https://arxiv.org/abs/2602.06949)
- **World Model for Robot Learning: A Comprehensive Survey** (Apr 2026) — [arXiv 2605.00080](https://arxiv.org/abs/2605.00080)
- **Video as the New Language for Real-World Decision Making** (Feb 2024, position paper) — [arXiv 2402.17139](https://arxiv.org/abs/2402.17139)
- **Foundational, older: DayDreamer: World Models for Physical Robot Learning.** P. Wu, A. Escontrela, D. Hafner, K. Goldberg, P. Abbeel (2022). Dreamer trains a quadruped to roll off its back, stand up and walk from scratch, with no resets, "in only 1 hour" in the real world, and adapts to pushes within 10 minutes — [arXiv 2206.14176](https://arxiv.org/abs/2206.14176)

### Inferences
- The group's locomotion frontier has moved to **humanoids, end-to-end from depth, distilled from privileged experts**. A quadruped footstep planner is therefore a niche the group is not working in. It also means GaitNet will be judged against "why not end-to-end depth → joints?" (RPL does 25×25 cm stones with 60 cm gaps end-to-end).
- The recurring recipe across RPL, PHP and LadderMan is privileged height-map experts → multi-view depth student with modeled latency and noise. That is the obvious answer to W4, and also the obvious competitor architecture.
- The group never plans contacts explicitly. Even Workhorse, which outputs foot poses, does so as continuous kinematic targets for a tracker. This is a positioning opening for GaitNet's explicit, inspectable contact interface (see Q3).

### Gaps
- **Examples in the brief that are Berkeley but not Abbeel papers.** None of these lists Abbeel as an author:
  - "Humanoid Locomotion as Next Token Prediction" (NeurIPS 2024): Radosavovic, B. Zhang, Shi, Rajasegaran, Kamat, Darrell, Sreenath, Malik — [arXiv 2402.19469](https://arxiv.org/abs/2402.19469); [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2024/hash/90afd20dc776bc8849c31d61a0763a0b-Abstract.html)
  - "Real-World Humanoid Locomotion with Reinforcement Learning": Radosavovic, Xiao, B. Zhang, Darrell, Malik, Sreenath. arXiv 2023; journal venue not verified here — [arXiv 2303.03381](https://arxiv.org/abs/2303.03381)
  - "Berkeley Humanoid": Liao, B. Zhang, Xuanyu Huang, Xiaoyu Huang, Z. Li, Sreenath — [arXiv 2407.21781](https://arxiv.org/abs/2407.21781)

  They belong to the Malik, Darrell and Sreenath groups and should be attributed to them, not to Abbeel. The next-token paper (a causal transformer over sensorimotor tokens, with zero-shot walking in San Francisco) is still a relevant BAIR idea for Q2.
- I did not search for Covariant technical publications such as RFM-1 (2024), so none are verified here. The arXiv author query turned up no Covariant-era locomotion paper.
- Venues are unverified for most 2025-2026 items, which are arXiv-only as far as I checked: RPL, PHP, LadderMan, GaussGym, FastSAC, FastTD3, CQN-AS, BRC, QF3, Workhorse, VLK, OmniRetarget, Body Transformer, HumanoidBench, MuJoCo Playground and VideoMimic. I did not fill them from memory.
- No quantitative real-world success numbers were reported for FastSAC, FPO++ or PHP hardware runs. PHP's real-world results are capability demos.
- Out of scope and not detailed here: dexterous manipulation (GOTT, Mana, T-Rex, DexterityGen, ViTaMIn, etc.), LLM/agent and OS work (RPG, OSGym, Agentic Web), and chip placement and materials work. Also not detailed: EgoMI, the ABC behavior-cloning stack, and GSR (RSS 2024, offline imitation via graph search, [arXiv 2407.15403](https://arxiv.org/abs/2407.15403)). The author query could also miss papers that list Abbeel under a different name string.

## Q2. Which ideas could address GaitNet's weaknesses?

### Takeaway
The most transferable pieces are:
- Workhorse's **cross-policy error augmentation** between a planner and a tracker (W2).
- CQN-AS's **value learning over discretized and multi-step action sequences**, which stayed robust to no-op actions (W1, W5).
- Flow-policy RL (FPO++/QF3) as a **generative foothold proposer** trained by RL, which removes ContactNet (W3). It also gives in-group evidence that expressive policies yield cleaner emergent trots.
- RPL- and PHP-style **direct multi-view depth students** with modeled latency (W4).
- **Residual RL** on a strong base, i.e. "learn when to deviate from the OCS2 trot" (W6).

### Cited Findings
- Workhorse trains its planner on drifted input histories and its tracker on drifting commands. Removing the planner-side augmentation drops box-sorting success from 77% to 20% (falls rise from 1% to 79%). Removing the command augmentation drops success under pushes from 64% to 52% — [Workhorse arXiv HTML](https://arxiv.org/html/2610.09117)
- Contact-Aware Neural Dynamics keeps the simulator as a prior and learns a residual neural dynamics model from real data. The model "can be effectively used to predict policy performance and refine policies trained purely in standard simulators" — [arXiv 2601.12796](https://arxiv.org/abs/2601.12796)
- The dual-history (long + short I/O history) bipedal controller adapts to dynamics shifts and to "time-variant changes, such as contact events" — [arXiv 2401.16889](https://arxiv.org/abs/2401.16889)
- Residual off-policy RL treats the base policy as a black box and learns per-step residual corrections from sparse binary rewards — [arXiv 2509.19301](https://arxiv.org/abs/2509.19301). ResMimic and OmniRetarget-based policies likewise build on a general tracking base — [ResMimic](https://arxiv.org/abs/2510.05070)
- When action sequences are added, SAC and TD3 over-estimate values and fail on HumanoidBench Stand, while CQN "stayed robust to added no-op actions". CQN-AS's critic scores whole K-step action sequences — [CQN-AS arXiv HTML](https://arxiv.org/html/2411.12155)
- FPO++ policies learn cross-action-dimension correlations, and on Spot they produce consistent trots where Gaussian PPO pronks under the same reward. QF3 makes flow RL off-policy and about 10× faster than FPO++ — [FPO++ HTML](https://arxiv.org/html/2602.02481); [QF3](https://arxiv.org/abs/2610.08789)
- Shortcut models give one-step or few-step generation from a single network, and CFGRL turns guidance weight into a policy-improvement knob — [Shortcut](https://arxiv.org/abs/2410.12557); [CFGRL](https://arxiv.org/abs/2505.23458)
- RPL runs a 50 Hz policy directly on 15 Hz multi-view depth (48×27). In training it models delay U(0,20) ms, noise σ = 0.1·depth and dropout 0.05, with no elevation mapping. PHP uses 58×87 depth at 30 Hz with a 60-80 ms randomized delay — [RPL HTML](https://arxiv.org/html/2602.03002); [PHP HTML](https://arxiv.org/html/2602.15827)
- GaussGym renders photoreal RGB at more than 100k steps/s inside IsaacGym and imports iPhone scans, GrandTour, ARKit and Veo scenes — [arXiv 2510.15352](https://arxiv.org/abs/2510.15352)
- RCRL conditions a policy on reward parameterizations using counterfactual rewards from shared replay, enabling "zero-shot behavioral adjustment at deployment" — [arXiv 2603.05066](https://arxiv.org/abs/2603.05066)
- MaxInfoRL combines information-gain intrinsic rewards with Boltzmann exploration. Prioritized Generative Replay uses a conditional diffusion model, guided by curiosity or value, to densify useful transitions — [MaxInfoRL](https://arxiv.org/abs/2412.12098); [PGR](https://arxiv.org/abs/2410.18082)
- FastSAC/FastTD3 train sim-to-real humanoid locomotion in 15 min on one RTX 4090 with off-policy RL at massive parallelism — [arXiv 2512.01996](https://arxiv.org/abs/2512.01996)
- The Humanoid Safe Stop paper pairs an outcome-supervised feasibility estimator with a Hamilton-Jacobi reach-avoid estimator and commits only when both agree — [arXiv 2609.02358](https://arxiv.org/abs/2609.02358)
- Body Transformer encodes the robot as a sensor/actuator graph with masked attention and beats an MLP and a vanilla transformer — [arXiv 2408.06316](https://arxiv.org/abs/2408.06316)
- HumanoidBench finds that a hierarchical policy on top of robust low-level skills beats flat RL — [arXiv 2403.10506](https://arxiv.org/abs/2403.10506)
- DayDreamer learned quadruped walking in the real world in 1 h with a world model — [arXiv 2206.14176](https://arxiv.org/abs/2206.14176)

### Inferences
These are my proposals for GaitNet, not claims made by the papers.

- **W2, sim-to-sim gap → Workhorse-style two-sided error augmentation.**
  - During GaitNet training, perturb the planner's observations with drift and error models fitted to how OCS2+WBC tracks footsteps differently from the convex MPC: swing timing, touchdown error, base drift. Also train or finetune the low-level side against noisy and jumpy footstep commands.
  - This is cheap: no NMPC in the training loop. The Workhorse ablation (77% → 20%) is concrete evidence that interface-error modeling matters more than either module alone.
  - A stronger variant is a learned residual dynamics model of "OCS2 response minus convex-MPC response". It would be fitted on Gazebo rollouts, in the spirit of Contact-Aware Neural Dynamics, and used as a correction inside Isaac Lab training.
  - A dual-history encoder (Li et al. 2024) could let the planner identify the controller in-context.
- **W6, fixed trot wins → residual or "deviation-from-gait" formulation.**
  - Make the OCS2 trot (or any periodic gait) the base policy. GaitNet then learns only when to advance, delay or redirect a footstep away from the trot schedule, as in Residual Off-Policy RL and ResMimic.
  - This reframes GaitNet's novelty as "non-gaited corrections on top of a gaited prior" and should inherit the trot's 45% floor instead of starting from 18%.
- **W1, no-op reluctance → value over contact sequences.**
  - CQN-AS's critic over K-step action sequences suggests scoring short sequences of contact events (which leg, which foothold, when) rather than one tick's argmax. That makes the payoff of stepping now visible to the critic.
  - The finding that CQN stayed robust to added no-ops while SAC/TD3 over-estimated is directly relevant to a 25 Hz policy where most ticks are no-ops.
  - RCRL-style conditioning on the step-cost or no-op-penalty weight would let operators dial step frequency at deployment without retraining.
  - MaxInfoRL or Prioritized Generative Replay could oversample or densify the rare stepping transitions.
- **W3, ContactNet dependence → generative foothold proposals trained by RL.**
  - A flow policy, trained with FPO++ or QF3 and sampled in one to few steps with shortcut-model distillation for 25 Hz, would output a foothold (and leg) directly. The current samplers would become a validity projection or filter rather than the action-set definition.
  - FPO++'s Spot result (consistent trot vs pronk under identical rewards) is in-group evidence that expressive policies support cleaner emergent gait structure, which fits GaitNet's "gaits emerge" premise.
  - Alternatively, CQN-style coarse-to-fine discretization of the foothold C-space (coarse cell, then finer cell, argmax per level) gives a learned candidate hierarchy without a separate cost network.
- **W4, slow perception → drop the elevation-map → plane-decomposition chain at deployment.**
  - Distill the height-map-privileged GaitNet into a depth-student scorer as RPL and PHP do, with modeled 60-80 ms delay and noise, and run the expensive plane segmentation only for the validity mask if at all.
  - GaussGym- or PAWS-style real-to-sim scenes from phone scans would let the student train on realistic stepping-stone appearance.
- **W5, concurrent swings matter only when pulled along.** A Body-Transformer-style per-leg token architecture, with candidate tokens attached to leg nodes, could model inter-leg coordination explicitly rather than through a flat CNN. This is speculative; no group paper tests it on contact selection.
- **Safety and feasibility gating.** A learned "steppability" or recoverability value, in the Safe-Stop style of two agreeing estimators, could gate no-op vs. step and filter candidates. It is a principled replacement for heuristic foothold costs.
- **Training-throughput angle.** GaitNet's training is dominated by the MPC (CPU pool), so the FastSAC/FastTD3 off-policy recipe, which reuses data, plausibly reduces the number of MPC-stepped env transitions needed. This must be benchmarked; it is not claimed by the papers.

### Gaps
- No Abbeel-group paper applies world models (DayDreamer-style or video world models) to legged locomotion in 2024-2026. DayDreamer (2022) is the last one I found. Learning the deployment controller's dynamics with a world model is therefore my extrapolation, not a group result.
- The next-token/causal-transformer idea for discrete contact events is from the Malik/Sreenath line, not Abbeel. No Abbeel paper tokenizes contact events.
- Contact-Aware Neural Dynamics is shown only for tactile hand contact. Transfer to foot contacts and to a controller-to-controller gap (rather than sim-to-real) is untested.
- No group paper reports large-scale pretraining on diverse quadruped gaits.

## Q3. Is there work on discrete action selection over candidate sets, contact sequences, or footstep planning?

### Takeaway
Mostly negative. **No Abbeel-group paper from 2024-2026 does footstep or foothold selection over a candidate set, or explicit contact-sequence planning.** The closest analogs are:
- CQN-AS: discrete bins with per-level argmax, and Q over action sequences.
- PHP's motion matching: argmin over a discrete motion library with skill entry windows. The final skill choice is implicit in the student.
- Workhorse: explicit foot-pose targets passed from planner to tracker.
- RPL: solves stepping stones **end-to-end without a footstep planner**.

The absence of explicit contact planning in this group is itself a positioning point for GaitNet.

### Cited Findings
- CQN-AS discretizes each action dimension into 5 bins over 3 coarse-to-fine levels, takes the argmax bin per level, and zooms into it. Its critic scores K-step action sequences — [arXiv HTML](https://arxiv.org/html/2411.12155)
- PHP selects motion frames by argmin squared-L2 nearest-neighbor search over a 27-D feature (future root trajectory, foot state, root velocity). Searches are restricted to per-skill entry windows when a skill is triggered. The deployed student "selects and executes" step-over, climb, vault or roll from depth and a discrete 2D velocity command, without an explicit selection module — [arXiv HTML](https://arxiv.org/html/2602.15827); [arXiv abs](https://arxiv.org/abs/2602.15827)
- RPL handles stepping stones with a foot-edge penalty during expert training and an end-to-end depth student, with no footstep planner. Two of the four real-world failures in its 10-trial course were on stepping stones — [arXiv HTML](https://arxiv.org/html/2602.03002)
- Workhorse's planner outputs both feet's 6-D poses (plus torso and wrists) as a 1.16 s chunk at 5 Hz for a 50 Hz tracker. It has no separate contact-state output; contacts are implied by the foot trajectories — [arXiv HTML](https://arxiv.org/html/2610.09117)
- OmniRetarget explicitly preserves agent-terrain and agent-object contact relationships through an interaction mesh when generating reference motions — [arXiv 2509.26633](https://arxiv.org/abs/2509.26633)
- LadderMan targets a sparse foothold and handhold problem (ladder rungs) with a learned end-to-end depth policy rather than a contact planner — [arXiv 2606.05873](https://arxiv.org/abs/2606.05873)
- GSR (RSS 2024) organizes experience into a graph, searches it for behavior values, and retrieves the best action per state. It is a discrete-search-then-clone approach, applied to manipulation — [arXiv 2407.15403](https://arxiv.org/abs/2407.15403)
- PAWS learns a traversability affordance for a quadruped, a navigation-level choice of where to walk, not of footholds — [PMLR](https://proceedings.mlr.press/v270/escontrela25a.html)

### Inferences
- **Repositioning angle.** The group's state of the art reaches sparse footholds (25×25 cm stones, ladder rungs) end-to-end, which makes a raw traversal-success claim for GaitNet weak, especially given W6. Three claims are more defensible:
  - an **explicit, inspectable contact interface**: discrete per-leg foothold choices that can be validated, filtered and constrained at deployment;
  - **controller-agnostic** footstep decisions that can drive any MPC or WBC, now paired with interface-error augmentation;
  - **non-gaited decisions as a residual over a gaited prior**.
- PHP's entry-window search maps neatly onto GaitNet. "Restrict the nearest-neighbor or argmax search to an admissible window" is the same pattern as GaitNet's terrain-valid C-space plus the at-most-two-swings rule. PHP shows it working for long-horizon skill chaining with a learned student. A GaitNet variant could restrict per-leg candidate sets by a learned admissibility value rather than by ContactNet ranking.
- CQN-AS shows value-based discrete selection scales to high-DoF humanoid control. A Q(s, leg, foothold-cell) critic with coarse-to-fine cells, extended to short contact sequences, is a natural, group-adjacent upgrade of GaitNet's "score candidates and take the argmax" head.

### Gaps
- I found no Abbeel-group paper on quadruped footstep planning, foothold scoring, gait-free contact scheduling, or stepping-stone-specific planners in 2024-2026.
- PHP's real-world skill-selection accuracy is not reported, so the reliability of its implicit discrete choice cannot be compared with GaitNet's explicit argmax.
- I did not search other groups' footstep-planning work (e.g. ETH or CMU); that is outside this assignment's scope.

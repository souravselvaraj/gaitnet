# C. Karen Liu group (Stanford, The Movement Lab), 2024-2026: papers relevant to repositioning GaitNet

How this list was built: the paper list comes from the arXiv API author queries for "C. Karen Liu" (129 entries) and "Karen Liu" (136 entries, which picks up a few papers where her name is printed without the "C."), queried on 2026-10-11 ([arXiv API, "C. Karen Liu"](http://export.arxiv.org/api/query?search_query=au:%22C.%20Karen%20Liu%22&start=0&max_results=300&sortBy=submittedDate&sortOrder=descending); [arXiv author search](https://arxiv.org/search/?query=C.+Karen+Liu&searchtype=author&abstracts=show&order=-announced_date_first&size=200)). The core idea of each paper comes from its arXiv abstract. Key results for the most relevant papers come from the arXiv HTML full text. Venues are given only when the arXiv comment or journal-ref field, or a proceedings page, states them; otherwise the entry says "arXiv preprint, venue not stated".

Many 2025-2026 humanoid papers are multi-institution collaborations. Their author lists include Pieter Abbeel, Rocky Duan, Guanya Shi, Carmelo Sferrazza and Angjoo Kanazawa, with Liu as a co-author or senior author. Those are flagged as "collaboration": the lead lab may not be TML, and this note does not verify it.

---

## Q1. Which 2024-2026 papers from the group address contact-rich motion, footstep or contact planning, emergent gaits, generative models of motion with contacts, humanoid or quadruped control, sim-to-real, or learning from human or animal motion?

### Takeaway
From 2024 to 2026 the group has published a lot of humanoid-control and physics-based-animation work. Four lines matter most for GaitNet:
- **Contact-conditioned control.** SceneBot (2026) conditions a whole-body tracker on per-link binary contact labels and calls contact "a powerful interface for humanoid control". It explicitly leaves the high-level policy that would issue those contact prompts to future work.
- **Diffusion policies and guided state-action diffusion for physics-based control.** PDP (SIGGRAPH Asia 2024), BeyondMimic (2025), FPO++ (2026), and the adjacent Diffuse-CLoC (2025).
- **Discrete skill or contact selection from perception.** Locomotion Beyond Feet (2026) and Perceptive Humanoid Parkour (2026).
- **Learning locomotion and navigation priors from human data.** TWIST, GMR, OmniRetarget, EgoNav and HEAD.

The group has **no 2024-2026 paper on quadruped footstep planning**. The quadruped results are limited to FPO++ (Go2 and Spot locomotion in simulation) and the 2018 "Learning Symmetric and Low-energy Locomotion" paper, in which quadruped and hexapod gaits emerge. Explicit learned foothold selection for quadrupeds is therefore a niche the group does not occupy.

### Cited Findings

#### A. Contact as an explicit command, and contact-rich or sparse-foothold locomotion

- **SceneBot: Contact-Prompted General Humanoid Whole Body Tracking with Scene-Interaction.** Sirui Chen, Shibo Zhao, Zhen Wu, Jiaman Li, Guanya Shi, C. Karen Liu. Submitted 25 Jun 2026; arXiv preprint, venue not stated. [arXiv 2606.27581](https://arxiv.org/abs/2606.27581)
  - **Core idea:** a single tracking policy conditioned on both reference motions and per-link contact labels, which "explicitly defin[e] expected environmental interactions". It covers free-space locomotion, terrain traversal and whole-body manipulation. Interaction data comes from a "hindsight scene reconstruction approach that infers scene-interaction graphs from retargeted human motion", and the policy is trained on 7.5 hours of such data. — [arXiv abstract](https://arxiv.org/abs/2606.27581)
  - **Representation:** the labels are a binary vector c_t ∈ {0,1}^n over key links (left/right wrist, left/right foot, pelvis) × scene type (terrain or object). The policy is π(c_t, g_t, s_t, p_root,t), where g_t is a 93-dimensional kinematic reference. A "contact-mismatch termination" requires the commanded contact to be established within a time window. Candidate contact edges are added where a link has low relative velocity and acceleration against a scene node, and are pruned if they cause collisions or violate force closure. Terrain is reconstructed as a 2.5D elevation map of plateaus. — [arXiv HTML](https://arxiv.org/html/2606.27581)
  - **Results (MuJoCo sim-to-sim, 20 unseen motions per category; success on free / object / terrain / sit):**
    - SceneBot: 100 / 95 / 100 / 100 %.
    - SONIC baseline: 100 / 5 / 15 / 0 %.
    - No contact labels: 100 / 0 / 60 / 95 %.
    - No foot label: 100 / 70 / 60 / 100 %.
    - Terrain root-position error: 0.147 m versus 0.808 m for SONIC.
    - On Unitree G1 hardware with LiDAR-based odometry, task success was 80% over 10 trials.
    — [arXiv HTML](https://arxiv.org/html/2606.27581)
  - **Positioning statement:** SceneBot asks the high-level decision maker for one extra command, namely which links should make contact. It describes itself as a low-level tracker, and lists "an autonomous high-level controller for perceptive, goal-driven scene interactions" as future work. — [arXiv HTML](https://arxiv.org/html/2606.27581)

- **Locomotion Beyond Feet.** Tae Hoon Yang, Haochen Shi, Jiacheng Hu, Zhicong Zhang, Daniel Jiang, Weizhuo Wang, Yao He, Zhen Wu, Yuming Chen, Yifan Hou, Monroe Kennedy, Shuran Song, C. Karen Liu. Submitted 7 Jan 2026; arXiv preprint, venue not stated. [arXiv 2601.03607](https://arxiv.org/abs/2601.03607); [project](https://locomotion-beyond-feet.github.io/)
  - **Core idea:** whole-body humanoid locomotion that uses hands, knees and elbows as contacts on terrain such as low-clearance spaces under chairs, knee-high walls and platforms, and steep stairs. It combines "physics-grounded keyframe animation with reinforcement learning". A hierarchy of terrain-specific tracking policies, failure recovery and a vision-based skill planner sits on top. — [arXiv abstract](https://arxiv.org/abs/2601.03607)
  - **Details:** runs on ToddlerBot (30 DoF). The skill planner is an explicit **discrete skill classifier**: a ResNet over depth that Foundation Stereo estimates from dual fisheye cameras, running at **3.1 Hz**. Its output is smoothed with an exponential moving average, and the planner switches skill only above a confidence threshold. Low-level tracking runs at 50 Hz.
  - **Results:** classifier accuracy 93.9% on 3,394 real held-out samples. TensorRT downsampling raised depth inference from 0.5 Hz to 3.1 Hz. A wall policy designed for 12 cm generalized to 9-14 cm. All five tested obstacle orders were completed zero-shot. No locomotion success rates are reported. — [arXiv HTML](https://arxiv.org/html/2601.03607)

- **Perceptive Humanoid Parkour (PHP): Chaining Dynamic Human Skills via Motion Matching.** Zhen Wu, Xiaoyu Huang, Lujie Yang, Yuanhang Zhang, Xi Chen, Pieter Abbeel, Rocky Duan, Angjoo Kanazawa, Carmelo Sferrazza, Guanya Shi, C. Karen Liu. 2026; arXiv preprint, venue not stated; collaboration. [arXiv 2602.15827](https://arxiv.org/abs/2602.15827)
  - **Core idea:** motion matching (nearest-neighbour search in a feature space) composes retargeted human skills into long-horizon kinematic references. Motion-tracking RL experts are trained on these references and then distilled with DAgger plus RL into one depth-based student. Using only depth and a discrete 2D velocity command, the robot decides whether to step over, climb, vault or roll off. On a Unitree G1 it climbed 1.25 m, 96% of its own height. — [arXiv abstract](https://arxiv.org/abs/2602.15827)
  - **Details:** the matching feature is 27-dimensional: future root trajectory at 0.33, 0.67 and 1 s, local foot positions and velocities, and root velocity. Skill selection by the student is implicit; no discrete selection head is described. Depth runs at 30 Hz, with observation delay randomized between 60 and 80 ms.
  - **Results (simulation success on 36 / 58 / 76 cm obstacles at 1 m/s):**
    - PHP: 1.00 / 0.99 / 0.95.
    - End-to-end depth baseline: 0.95 / 0.07 / 0.08.
    - Uncomposed data: 0.06 / 0.02 / 0.00.
    — [arXiv HTML](https://arxiv.org/html/2602.15827)

- **LadderMan: Learning Humanoid Perceptive Ladder Climbing.** Siheng Zhao, Yuanhang Zhang, Ziqi Lu, Pieter Abbeel, Rocky Duan, Koushil Sreenath, Yue Wang, C. Karen Liu, Guanya Shi. 2026; arXiv preprint; collaboration. [arXiv 2606.05873](https://arxiv.org/abs/2606.05873)
  - Ladder climbing is hard because of "sparse footholds and handholds". The method uses hybrid motion tracking to learn several climbing experts from a single reference motion, then distills them into one depth-based visuomotor policy with hybrid imitation and RL. Vision foundation models close the depth sim-to-real gap. The policy transfers zero-shot to hardware. — [arXiv abstract](https://arxiv.org/abs/2606.05873)

- **RPL: Learning Robust Humanoid Perceptive Locomotion on Challenging Terrains.** Yuanhang Zhang, Younggyo Seo, Juyue Chen, Yifu Yuan, Koushil Sreenath, Pieter Abbeel, Carmelo Sferrazza, Karen Liu, Rocky Duan, Guanya Shi. 2026; arXiv preprint; collaboration. [arXiv 2602.03002](https://arxiv.org/abs/2602.03002)
  - Terrain-specific experts that use privileged height maps are distilled into a transformer policy over multiple depth cameras. The paper adds a multi-depth ray-caster that is 5× faster than existing simulator depth rendering and models "realistic sensor latency, noise, and dropout".
  - Real-world results: multi-directional locomotion with a 2 kg payload on 20° slopes, stairs, and "25 cm by 25 cm stepping stones separated by 60 cm gaps". There is no explicit footstep planner. — [arXiv abstract](https://arxiv.org/abs/2602.03002)

- **OmniRetarget: Interaction-Preserving Data Generation for Humanoid Whole-Body Loco-Manipulation and Scene Interaction.** Lujie Yang, Xiaoyu Huang, Zhen Wu, Angjoo Kanazawa, Pieter Abbeel, Carmelo Sferrazza, C. Karen Liu, Rocky Duan, Guanya Shi. 2025; arXiv preprint; collaboration. [arXiv 2509.26633](https://arxiv.org/abs/2509.26633)
  - An interaction mesh explicitly preserves the spatial and contact relationships between agent, terrain and objects, by minimizing Laplacian deformation under kinematic constraints. One demonstration can be augmented across embodiments, terrains and object layouts. The paper generates more than 8 hours of trajectories with better contact preservation than baselines.
  - Proprioceptive RL then performs parkour and loco-manipulation lasting up to 30 s on a G1, using 5 reward terms and no curriculum. — [arXiv abstract](https://arxiv.org/abs/2509.26633)

#### B. Diffusion and generative models for physics-based control

- **PDP: Physics-Based Character Animation via Diffusion Policy.** Takara E. Truong, Michael Piseno, Zhaoming Xie, C. Karen Liu. SIGGRAPH Asia 2024 Conference Papers, Article 86. [arXiv 2406.00960](https://arxiv.org/abs/2406.00960)
  - **Core idea:** naive behaviour cloning into a diffusion policy is unstable for under-actuated, high-frequency biped control because errors compound. PDP uses RL experts as *corrective* teachers in sub-optimal states, combining RL and BC. It is shown on perturbation recovery, universal motion tracking and text-to-motion. — [arXiv abstract](https://arxiv.org/abs/2406.00960)
  - **Recipe:** roll experts out with noisy actions, and store the noisy state paired with the *clean* expert action.
  - **Effect of the sampling strategy on recovery success:** clean-state/clean-action 3.36%, noisy-state/noisy-action 66.9%, noisy-state/clean-action 100.0%.
  - **Perturbation recovery (Bump'em):**
    - PDP: 100% in-distribution, 96.3% out-of-distribution.
    - C-VAE: 91.3% out-of-distribution.
    - MLP: 81.0% out-of-distribution.
  - **Multimodality:** PDP produces distinct recovery modes for the same push, while the C-VAE collapses to one mode. The authors attribute the MLP's weaker OOD score to it "averaging over multimodal responses".
  - **Motion tracking (AMASS test):** 97.1% success versus PHC's 96.4%. The paper's text and its table disagree on this number.
  - **Text-to-motion:** 57.1% versus 11.9% for the MLP.
  - **Limitations:** inference is K times slower than an MLP, and an action horizon of 12 drops success to 6.5%. — [arXiv HTML](https://arxiv.org/html/2406.00960)

- **BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion.** Qiayuan Liao, Takara E. Truong, Xiaoyu Huang, Yuman Gao, Guy Tevet, Koushil Sreenath, C. Karen Liu. 2025; arXiv preprint, no venue stated in v4. [arXiv 2508.08241](https://arxiv.org/abs/2508.08241); [project](https://beyondmimic.github.io/)
  - **Core idea:** a compact motion-tracking formulation learns agile skills such as cartwheels, flips and sprinting with shared hyperparameters. A unified latent diffusion model, steered by classifier guidance, then solves unseen tasks: motion inpainting, joystick control and obstacle avoidance, zero-shot on real hardware. — [arXiv abstract](https://arxiv.org/abs/2508.08241)
  - **Details:** the model diffuses state-latent trajectories (past N steps, current step and H future steps), where actions are encoded as CVAE latents. It uses per-element noise levels, which enable inpainting. Guidance takes the gradient of a differentiable cost; obstacle avoidance uses a signed-distance-field cost. The horizon is 0.64 s.
  - **Results:**
    - In sim-to-sim tests, cartwheel success was 95% with the latent diffusion versus 5% without latent encoding.
    - The robot ran more than 50 m continuously on a track.
    - Users preferred its motion over the baseline 70.8% to 29.2% (N=77).
  - **Stated limitations:** the horizon is too short for "early obstacle anticipation", and guidance works "less well for fine-grained" objectives. — [arXiv HTML](https://arxiv.org/html/2508.08241)

- **Diffuse-CLoC: Guided Diffusion for Physics-based Character Look-ahead Control** (adjacent work: Liu is *not* an author, but co-first author Takara Truong is her student). Xiaoyu Huang, Takara Truong, Yunbo Zhang, Fangzhou Yu, Jean Pierre Sleiman, Jessica Hodgins, Koushil Sreenath, Farbod Farshidian. 2025; ACM DOI 10.1145/3731206. [arXiv 2503.11801](https://arxiv.org/abs/2503.11801)
  - The paper models the joint distribution of states and actions in one diffusion model, so that "action generation [is] steerable" by conditioning on predicted states. No separate high-level planner is needed. It reports beating the high-level-diffusion-plus-low-level-tracker hierarchy on obstacle avoidance, in-betweening and task-space control. — [arXiv abs](https://arxiv.org/abs/2503.11801)
  - A secondary summary quotes a 16% fall rate in perturbed walking and 96% success in forest navigation. I did not verify these against the paper, so treat them as unconfirmed. — [themoonlight summary](https://www.themoonlight.io/en/review/diffuse-cloc-guided-diffusion-for-physics-based-character-look-ahead-control)

- **Constrained Diffusion with Trust Sampling.** William Huang, Yifeng Jiang, Tom Van Wouwe, C. Karen Liu. NeurIPS 2024. [arXiv 2411.10932](https://arxiv.org/abs/2411.10932)
  - The paper recasts training-free loss-guided diffusion as a sequence of constrained optimizations. At each diffusion level it takes several gradient steps on the proxy constraint "until we can no longer trust the proxy", judged by the variance at that level. It also estimates the state manifold so it can stop early when a sample drifts off it. Results show better constrained generation for both images and 3D motion. — [arXiv abstract](https://arxiv.org/abs/2411.10932)

- **Flow Policy Gradients for Robot Control (FPO++).** Brent Yi, Hongsuk Choi, Himanshu Gaurav Singh, Xiaoyu Huang, Takara E. Truong, Carmelo Sferrazza, Yi Ma, Rocky Duan, Pieter Abbeel, Guanya Shi, Karen Liu, Angjoo Kanazawa. 2026; arXiv preprint; collaboration. [arXiv 2602.02481](https://arxiv.org/abs/2602.02481)
  - Flow-matching policy gradients avoid computing likelihoods, which lets RL train expressive, non-Gaussian policies. The method is shown on legged locomotion, humanoid motion tracking and manipulation, with sim-to-real on two humanoids. — [arXiv abstract](https://arxiv.org/abs/2602.02481)
  - **Gait finding:** locomotion benchmarks ran in Isaac Lab on **Unitree Go2, Boston Dynamics Spot**, H1 and G1. "Spot trained with the same default rewards" gave **more consistent trotting with FPO++, whereas Gaussian PPO tended toward symmetric pronking**. The authors attribute this to Gaussian policies sampling action dimensions independently, which makes correlated behaviours hard to explore.
  - Correlations between action dimensions emerge during training, for example negatively correlated left and right hips. FPO++ "almost always" converges to higher returns than Gaussian PPO, with lower variance across seeds.
  - Hardware: Booster T1 locomotion and G1 tracking. — [arXiv HTML](https://arxiv.org/html/2602.02481)

- **Hierarchical Planning and Control for Box Loco-Manipulation** (borderline: posted June 2023). Zhaoming Xie, Jonathan Tseng, Sebastian Starke, Michiel van de Panne, C. Karen Liu. [arXiv 2306.09532](https://arxiv.org/abs/2306.09532)
  - A physics-simulated human rearranges boxes using a hierarchy of a planner, diffusion models, and RL imitation of sparse motion clips. — [arXiv abstract](https://arxiv.org/abs/2306.09532)

- **DROP: Dynamics Responses from Human Motion Prior and Projective Dynamics** (2023). Yifeng Jiang, Jungdam Won, Yuting Ye, C. Karen Liu. SIGGRAPH Asia 2023. [arXiv 2309.13742](https://arxiv.org/abs/2309.13742)
  - A kinematic generative motion prior is plugged in as one projective-dynamics energy, which yields a minimalist physics-based human simulator. — [arXiv abstract](https://arxiv.org/abs/2309.13742)

#### C. Humanoid control learned from human motion: tracking, retargeting, teleoperation, hierarchies

- **TWIST: Teleoperated Whole-Body Imitation System.** Yanjie Ze, Zixuan Chen, João Pedro Araújo, Zi-ang Cao, Xue Bin Peng, Jiajun Wu, C. Karen Liu. CoRL 2025, PMLR 305:2143-2154. [arXiv 2505.02833](https://arxiv.org/abs/2505.02833); [PMLR](https://proceedings.mlr.press/v305/ze25a.html)
  - Human mocap is retargeted to the humanoid, and an RL+BC whole-body controller is trained on it. Privileged future frames and real mocap data improve tracking. One network covers whole-body manipulation, legged manipulation, locomotion and expressive motion. — [arXiv abstract](https://arxiv.org/abs/2505.02833)

- **TWIST2: Scalable, Portable, and Holistic Humanoid Data Collection System.** Yanjie Ze, Siheng Zhao, Weizhuo Wang, Angjoo Kanazawa, Rocky Duan, Pieter Abbeel, Guanya Shi, Jiajun Wu, C. Karen Liu. 2025; arXiv preprint. [arXiv 2511.02832](https://arxiv.org/abs/2511.02832)
  - A mocap-free system using PICO4U VR and a $250 2-DoF robot neck for whole-body teleoperation. It collects 100 demonstrations in 15 minutes at almost 100% success. A hierarchical egocentric visuomotor policy performs whole-body dexterous tasks and dynamic kicking. — [arXiv abstract](https://arxiv.org/abs/2511.02832)

- **Retargeting Matters: General Motion Retargeting (GMR) for Humanoid Motion Tracking.** Joao Pedro Araujo, Yanjie Ze, Pei Xu, Jiajun Wu, C. Karen Liu. 2025; arXiv preprint. [arXiv 2510.02252](https://arxiv.org/abs/2510.02252)
  - Retargeting artifacts "such as foot sliding, self-penetration, and physically infeasible motion" significantly reduce policy robustness, especially for dynamic or long sequences.
  - GMR beats the PHC and ProtoMotions retargeters and comes close to Unitree's closed-source data. The comparison holds reward tuning fixed by training every policy with BeyondMimic. — [arXiv abstract](https://arxiv.org/abs/2510.02252)

- **ResMimic: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation via Residual Learning.** Siheng Zhao, Yanjie Ze, Yue Wang, C. Karen Liu, Pieter Abbeel, Guanya Shi, Rocky Duan. 2025; arXiv preprint; collaboration. [arXiv 2510.05070](https://arxiv.org/abs/2510.05070)
  - A general-motion-tracking base policy trained on human-only motion supplies task-agnostic movement. A residual policy refines it for precision and object interaction, with a contact reward and a curriculum virtual-object controller. The paper reports substantial gains in success, efficiency and robustness on a real G1. — [arXiv abstract](https://arxiv.org/abs/2510.05070)

- **VisualMimic: Visual Humanoid Loco-Manipulation via Motion Tracking and Generation.** Shaofeng Yin, Yanjie Ze, Hong-Xing Yu, C. Karen Liu, Jiajun Wu. 2025; arXiv preprint. [arXiv 2509.20322](https://arxiv.org/abs/2509.20322)
  - A low-level keypoint tracker learned from human motion is paired with a high-level visual policy that outputs keypoint commands.
  - Two measures keep training stable: noise is injected into the low-level policy, and high-level actions are clipped "using human motion statistics".
  - Zero-shot sim-to-real for box lifting, pushing, dribbling and kicking. — [arXiv abstract](https://arxiv.org/abs/2509.20322)

- **Hand-Eye Autonomous Delivery (HEAD): Learning Humanoid Navigation, Locomotion and Reaching.** Sirui Chen, Yufei Ye, Zi-Ang Cao, Jennifer Lew, Pei Xu, C. Karen Liu. CoRL 2025. [arXiv 2508.03068](https://arxiv.org/abs/2508.03068)
  - A high-level planner commands three points (eyes and both hands). A whole-body controller learned from large-scale mocap tracks them. The high-level policy is learned from human data recorded with Aria glasses. — [arXiv abstract](https://arxiv.org/abs/2508.03068)

- **VLK: Learning Humanoid Loco-Manipulation from Synthetic Interactions in Reconstructed Scenes.** Yen-Jen Wang, Jiaman Li, Sirui Chen, Takara E. Truong, Pei Xu, Pieter Abbeel, Rocky Duan, Koushil Sreenath, Angjoo Kanazawa, Carmelo Sferrazza, Guanya Shi, Karen Liu. 2026; arXiv preprint; collaboration. [arXiv 2606.30645](https://arxiv.org/abs/2606.30645)
  - 48,000 paired vision-language-kinematics trajectories are synthesized in 3D Gaussian Splatting scenes. A policy predicts short-horizon whole-body kinematics, and a tracker executes them on a G1. — [arXiv abstract](https://arxiv.org/abs/2606.30645)

- **Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation (PRISM).** Zihan Wang, Zhen Wu, Pieter Abbeel, Rocky Duan, Jitendra Malik, Carmelo Sferrazza, C. Karen Liu, Guanya Shi, Angjoo Kanazawa. CoRL 2026. [arXiv 2609.38172](https://arxiv.org/abs/2609.38172)
  - Video-to-video generation amplifies a few real videos into hundreds of counterfactual interaction videos. A "contact-anchored real-to-sim pipeline" then reconstructs physically plausible trajectories. The resulting depth-only policy works on real hardware without real fine-tuning. — [arXiv abstract](https://arxiv.org/abs/2609.38172)

- **Learning to Ball: Composing Policies for Long-Horizon Basketball Moves.** Pei Xu, Zhen Wu, Ruocheng Wang, Vishnu Sarukkai, Kayvon Fatahalian, Ioannis Karamouzas, Victor Zordan, C. Karen Liu. SIGGRAPH Asia 2025 (ACM TOG, Dec 2025). [arXiv 2509.22442](https://arxiv.org/abs/2509.22442)
  - A policy-integration framework plus a high-level soft router composes very different skills across multi-phase tasks whose intermediate states are ill-defined. — [arXiv abstract](https://arxiv.org/abs/2509.22442)

- **Humanoid Locomotion and Manipulation: Current Progress and Challenges in Control, Planning, and Learning** (survey). Zhaoyuan Gu et al., with C. Karen Liu among the co-authors. 2025. [arXiv 2501.02116](https://arxiv.org/abs/2501.02116) — existence and author list from the [arXiv API listing](http://export.arxiv.org/api/query?search_query=au:%22C.%20Karen%20Liu%22&start=0&max_results=300&sortBy=submittedDate&sortOrder=descending). I did not review its contents.

#### D. Perception and navigation learned from human data

- **Learning Humanoid Navigation from Human Data (EgoNav, 2026).** Weizhuo Wang, Yanjie Ze, C. Karen Liu, Monroe Kennedy. 2026; arXiv preprint. [arXiv 2604.00416](https://arxiv.org/abs/2604.00416)
  - **Core idea:** a diffusion model trained only on 5 hours of human walking data predicts distributions of future trajectories. It is conditioned on past trajectory, a 360° visual memory fusing colour, depth and semantics, and DINOv3 features. A receding-horizon controller selects a path from the predicted distribution. Deployment on a Unitree G1 is zero-shot. — [arXiv abstract](https://arxiv.org/abs/2604.00416)
  - **Inference:** 5 DDIM steps plus 5 DDPM refinement steps produce 110 trajectories per second on a Jetson Thor, so the loop runs at about 1.7-2 Hz.
  - **Selection:** sample 64 candidates over 5 s; discard those that collide with the visual-memory point cloud; cluster the rest with K-means (k=3); score clusters by popularity and a momentum penalty; execute the medoid of the best cluster.
  - **Results:** collision-free score 91.4 versus 84.5 for VAE-LSTM. Real-world autonomy was 96.0-99.3% over 1,137 m.
  - **Ablation:** without the momentum penalty the robot "oscillat[es] between modes".
  - **Emergent behaviours:** waiting at doors, threading through pedestrians. — [arXiv HTML](https://arxiv.org/html/2604.00416)

- **EgoNav: Egocentric Scene-aware Human Trajectory Prediction (2024).** Weizhuo Wang, C. Karen Liu, Monroe Kennedy. [arXiv 2403.19026](https://arxiv.org/abs/2403.19026)
  - A diffusion model predicts the wearer's future trajectories from a compact visual-memory representation, with a fast sampling scheme for real time. The motivation is exoskeletons and fall prevention. — [arXiv abstract](https://arxiv.org/abs/2403.19026)

- **LookOut: Real-World Humanoid Egocentric Navigation.** Boxiao Pan, Adam W. Harley, C. Karen Liu, Leonidas J. Guibas. 2025. [arXiv 2508.14466](https://arxiv.org/abs/2508.14466)
  - Predicts future 6D head poses from egocentric video, using a 4-hour Aria Navigation Dataset. — [arXiv abstract](https://arxiv.org/abs/2508.14466)

- **HMD^2: Environment-aware Motion Generation from Single Egocentric Head-Mounted Device.** Vladimir Guzov, Yifeng Jiang, Fangzhou Hong, Gerard Pons-Moll, Richard Newcombe, C. Karen Liu, Yuting Ye, Lingni Ma. 3DV 2025. [arXiv 2409.13426](https://arxiv.org/abs/2409.13426)
  - A conditional motion diffusion model with autoregressive inpainting gives online inference at 0.17 s latency. — [arXiv abstract](https://arxiv.org/abs/2409.13426)

#### E. Sim-to-real, real-world adaptation, safety, platforms

- **Robot Trains Robot (RTR): Automatic Real-World Policy Adaptation and Learning for Humanoids.** Kaizhe Hu, Haochen Shi, Yao He, Weizhuo Wang, C. Karen Liu, Shuran Song. CoRL 2025. [arXiv 2508.12252](https://arxiv.org/abs/2508.12252)
  - A robot arm acts as teacher, providing protection, schedule, reward, perturbation and resets. Sim-to-real transfer is stabilized by "optimizing a single dynamics-encoded latent variable in the real world". Demonstrated on walking-speed fine-tuning and on learning swing-up from scratch. — [arXiv abstract](https://arxiv.org/abs/2508.12252)

- **Humanoid Safe Stop via Learned Stoppability Value.** Junfeng Long, Pieter Abbeel, Koushil Sreenath, Roberto Horowitz, Guanya Shi, C. Karen Liu. 2026; arXiv preprint. [arXiv 2609.02358](https://arxiv.org/abs/2609.02358)
  - Emergency stopping is cast as reach-avoid. Two learned estimators judge feasibility: an outcome-supervised stop probability, and a Hamilton-Jacobi reach-avoid value. The robot commits to the stop only when both agree, and otherwise hands off to a fall/damping fallback. The estimators transfer across upstream tasks without retraining. — [arXiv abstract](https://arxiv.org/abs/2609.02358)

- **ToddlerBot: Open-Source ML-Compatible Humanoid Platform for Loco-Manipulation.** Haochen Shi, Weizhuo Wang, Shuran Song, C. Karen Liu. 2025. [arXiv 2502.00893](https://arxiv.org/abs/2502.00893)
  - Zero-point calibration and transferable motor system identification give a "high-fidelity digital twin" with zero-shot sim-to-real. The robot is 0.56 m tall, weighs 3.4 kg and costs under $6,000. — [arXiv abstract](https://arxiv.org/abs/2502.00893)

- **CHIP: Adaptive Compliance for Humanoid Control through Hindsight Perturbation** (Sirui Chen, Zi-ang Cao et al., with Yuke Zhu and NVIDIA co-authors; 2025; [arXiv 2512.14689](https://arxiv.org/abs/2512.14689)). **GentleHumanoid** (Qingzhou Lu et al., 2025; [arXiv 2511.04679](https://arxiv.org/abs/2511.04679)). **Minimalist Compliance Control** (Haochen Shi et al., 2026; [arXiv 2603.00913](https://arxiv.org/abs/2603.00913)).
  - These three cover compliance and forceful contact for humanoid tracking controllers or across embodiments. They are of low relevance to footstep planning. — arXiv abstracts as linked.

#### F. Emergent locomotion without demonstrations, and biomechanics

- **Learning Realistic Athletic Sprinting Without Demonstrations.** William Wang, Nicholas Bianco, Guy Tevet, Jennifer Hicks, C. Karen Liu, Scott Delp, Kayvon Fatahalian. 2026; arXiv preprint. [arXiv 2609.11083](https://arxiv.org/abs/2609.11083)
  - Muscle-driven GPU simulation runs at 1000× real time. Large-batch RL in muscle-excitation space is guided "only by task-specific episode termination conditions and a reward that encourages maximizing speed while reducing forces needed to respect joint limits".
  - Result: near-realistic 100 m sprints and drills such as side-shuffle and carioca, in "strong agreement with experimental data captured from sprinters". — [arXiv abstract](https://arxiv.org/abs/2609.11083)

- **AddBiomechanics Dataset: Capturing the Physics of Human Motion at Scale.** Keenon Werling et al., C. Karen Liu. 2024. [arXiv 2406.18537](https://arxiv.org/abs/2406.18537)
  - 273 subjects, more than 70 hours of motion with force-plate data, and more than 24 million frames. — [arXiv abstract](https://arxiv.org/abs/2406.18537)

#### G. Kinematic motion generation and reconstruction with contact or timing structure

- **Controllable Human-Object Interaction Synthesis (CHOIS).** Jiaman Li, Alexander Clegg, Roozbeh Mottaghi, Jiajun Wu, Xavier Puig, C. Karen Liu. ECCV 2024. [arXiv 2312.03913](https://arxiv.org/abs/2312.03913)
  - A conditional diffusion model is conditioned on language and sparse waypoints. Guidance terms "enforce contact constraints during the sampling process", covering hand-object contact and human-floor contact. — [arXiv abstract](https://arxiv.org/abs/2312.03913)

- **Lifting Motion to the 3D World via 2D Diffusion (MVLift).** Jiaman Li, C. Karen Liu, Jiajun Wu. CVPR 2025 Highlight. [arXiv 2411.18808](https://arxiv.org/abs/2411.18808)
  - Global 3D motion is recovered while training only on 2D pose sequences. The method generalizes to "human poses, human-object interactions, and animal poses", and outperforms prior work, including 3D-supervised methods, on five datasets. — [arXiv abstract](https://arxiv.org/abs/2411.18808)

- **AnyLift: Scaling Motion Reconstruction from Internet Videos via 2D Diffusion.** Hongjie Li, Heng Yu, Jiaman Li, Hong-Xing Yu, Ehsan Adeli, C. Karen Liu, Jiajun Wu. CVPR 2026. [arXiv 2604.17818](https://arxiv.org/abs/2604.17818) — follow-up to MVLift for internet video. Existence and venue are from the [arXiv API listing](http://export.arxiv.org/api/query?search_query=au:%22C.%20Karen%20Liu%22&start=0&max_results=300&sortBy=submittedDate&sortOrder=descending); I did not read the details.

- **Generative Motion Infilling From Imprecisely Timed Keyframes.** Purvi Goel, Haotian Zhang, C. Karen Liu, Kayvon Fatahalian. Eurographics 2025. [arXiv 2503.01016](https://arxiv.org/abs/2503.01016)
  - The architecture "explicitly outputs a time-warping function to correct mistimed keyframes", plus spatial residuals. Constraints are retimed as part of generation. — [arXiv abstract](https://arxiv.org/abs/2503.01016)

- **Human-Object Interaction from Human-Level Instructions.** Zhen Wu, Jiaman Li, Pei Xu, C. Karen Liu. ICCV 2025. [arXiv 2406.17840](https://arxiv.org/abs/2406.17840)
  - An LLM produces the plan, kinematic generation follows, and an RL tracker in physics simulation ensures plausibility. — [arXiv abstract](https://arxiv.org/abs/2406.17840)

- **DiffusionPoser.** Tom Van Wouwe, Seunghwan Lee, Antoine Falisse, Scott Delp, C. Karen Liu. CVPR 2024. [arXiv 2308.16682](https://arxiv.org/abs/2308.16682)
  - One diffusion model reconstructs motion in real time from an arbitrary combination of IMUs and pressure insoles, using autoregressive inference. — [arXiv abstract](https://arxiv.org/abs/2308.16682)

- **RoMo** (CVPR 2026; large curated motion dataset; [arXiv 2605.26241](https://arxiv.org/abs/2605.26241)). **FürElise** (SIGGRAPH Asia 2024; piano hand motion; [arXiv 2410.05791](https://arxiv.org/abs/2410.05791)). **MUSIC** (2026; muscle-driven hand control; [arXiv 2604.23886](https://arxiv.org/abs/2604.23886)).
  - These are group outputs with low direct relevance to GaitNet. — arXiv listing as linked.

#### H. Foundational work from the group (pre-2024, marked as older)

- **Learning a Unified Control Policy for Safe Falling.** Visak C.V. Kumar, Sehoon Ha, C. Karen Liu. 2017. [arXiv 1703.02905](https://arxiv.org/abs/1703.02905)
  - The optimization "solves for both a discrete contact planning problem and a continuous optimal control problem". The policy outputs "the optimal next contacting body part..., contact location and timing, and the required joint actuation".
  - The policy is a **mixture of actor-critics**, one per possible contacting body part. At run time "the policy corresponding to the highest value function will be executed while the associated body part will be the next contact".
  - It matches or beats dynamic-programming search at 50-400× speed. — [arXiv abstract](https://arxiv.org/abs/1703.02905)

- **Footstep and Motion Planning in Semi-unstructured Environments Using Randomized Possibility Graphs.** Michael X. Grey, Aaron D. Ames, C. Karen Liu. ICRA 2017. [arXiv 1610.00700](https://arxiv.org/abs/1610.00700)
  - The paper criticizes footstep planners that "can only select from a discrete set of predetermined footstep actions". It uses high-level approximations of constraint manifolds to explore the "possibility" of actions before running lower-level planners.
  - A follow-up proves probabilistic completeness for quasi-static biped walking ([arXiv 1702.00425](https://arxiv.org/abs/1702.00425)). — [arXiv abstract](https://arxiv.org/abs/1610.00700)

- **Learning Symmetric and Low-energy Locomotion.** Wenhao Yu, Greg Turk, C. Karen Liu. SIGGRAPH 2018. [arXiv 1801.08093](https://arxiv.org/abs/1801.08093)
  - Two changes, used together: a symmetry term in the loss (not the reward), and a curriculum of "modulated physical assistance" that is gradually relaxed.
  - Shown on a biped, humanoid, **quadruped** and hexapod. "Speed-appropriate gait patterns emerge without any guidance from motion examples or contact planning." — [arXiv abstract](https://arxiv.org/abs/1801.08093)

- **Sim-to-Real Transfer for Biped Locomotion.** Wenhao Yu, Visak CV Kumar, Greg Turk, C. Karen Liu. IROS 2019. [arXiv 1903.01390](https://arxiv.org/abs/1903.01390)
  - A Projected Universal Policy is conditioned on a latent η of the dynamics parameters. Bayesian optimization over η on hardware (post-sysID) finds the best setting. — [arXiv abstract](https://arxiv.org/abs/1903.01390)
  - Related earlier work: **Preparing for the Unknown: UP-OSI** (Wenhao Yu, Jie Tan, C. Karen Liu, Greg Turk; RSS 2017; [arXiv 1702.02453](https://arxiv.org/abs/1702.02453)).

- **SimGAN: Hybrid Simulator Identification for Domain Adaptation via Adversarial RL.** Yifeng Jiang, Tingnan Zhang, Daniel Ho, Yunfei Bai, C. Karen Liu, Sergey Levine, Jie Tan. ICRA 2021. [arXiv 2101.06005](https://arxiv.org/abs/2101.06005)
  - A neural-plus-physics hybrid simulator is identified adversarially to match target-domain trajectories. — [arXiv abstract](https://arxiv.org/abs/2101.06005)

- **Transformer Inertial Poser.** Yifeng Jiang et al., C. Karen Liu. SIGGRAPH Asia 2022. [arXiv 2203.15720](https://arxiv.org/abs/2203.15720)
  - Predicts "stationary body points" (contact-like anchors) from 6 IMUs. These correct drift and also generate terrain height maps. — [arXiv abstract](https://arxiv.org/abs/2203.15720)

- **EDGE: Editable Dance Generation From Music.** Jonathan Tseng, Rodrigo Castellon, C. Karen Liu. 2022 (CVPR 2023 per common citation; that venue is not verified here). [arXiv 2211.10658](https://arxiv.org/abs/2211.10658)
  - A diffusion transformer with joint-wise conditioning and in-betweening. Introduces "a new metric for physical plausibility". — [arXiv abstract](https://arxiv.org/abs/2211.10658)

- **Real-time MPC and System Identification Using Differentiable Physics Simulation.** Sirui Chen, Keenon Werling, Albert Wu, C. Karen Liu. 2022. [arXiv 2202.09834](https://arxiv.org/abs/2202.09834)
  - Online system identification and MPC run together in a differentiable simulator. — [arXiv abstract](https://arxiv.org/abs/2202.09834)

### Inferences
- The group's dominant 2025-2026 paradigm is "retarget human motion → RL tracker → high-level selector or generator": TWIST, GMR, OmniRetarget, BeyondMimic, PHP, LadderMan, SceneBot. **Footholds are never an explicit learned decision for legged robots.** The closest analogues are discrete *skill* selection (Locomotion Beyond Feet, PHP) and *contact labels as commands* (SceneBot). GaitNet's explicit per-foot, per-tick foothold choice is therefore distinct from this body of work. The flip side is that the group's results suggest that, on humanoids, strong references plus tracking beat explicit contact decisions, which echoes GaitNet's finding that a fixed OCS2 trot outperforms it.
- RPL shows humanoids crossing stepping stones (25 cm × 25 cm, 60 cm gaps) with an end-to-end distilled depth policy and no footstep planner ([arXiv 2602.03002](https://arxiv.org/abs/2602.03002)). Reviewers will likely cite this as evidence that explicit foothold selection is unnecessary. GaitNet's novelty case therefore has to rest on something end-to-end policies lack, such as interpretability or verifiability of footholds, a controller-agnostic interface, or emergent gait structure.
- The quadruped gap is real: in the 2024-2026 list, quadrupeds appear only in FPO++ benchmarks (Go2, Spot) and as an embodiment class in "Transformer Transformer" co-design ([arXiv 2607.25798](https://arxiv.org/abs/2607.25798)).

### Gaps
- The TML website could not be used as a cross-check: `tml.stanford.edu/publications` returned 404 and the homepage rendered only its title ([tml.stanford.edu](https://tml.stanford.edu/)). SIGGRAPH or other papers by the group that are not on arXiv could therefore be missing.
- Venues for many 2025-2026 preprints (BeyondMimic, GMR, OmniRetarget, ResMimic, SceneBot, PHP, RPL, LadderMan, FPO++, Locomotion Beyond Feet, EgoNav-2026) were not stated on arXiv, and I did not verify them elsewhere.
- I did not determine which institution leads the collaboration papers (Abbeel, Duan, Shi, Sferrazza, Kanazawa co-authors).
- Diffuse-CLoC's quantitative results were seen only in a third-party summary and are unverified.
- Detailed results for BeyondMimic's hardware guidance tasks (obstacle-avoidance success rates) and the diffusion inference rate were not in the portion of the paper I read.

---

## Q2. Which ideas could address GaitNet's weaknesses?

The weaknesses are: reluctance to step (85-95% no-op), the sim-to-sim gap between controllers, dependence on ContactNet candidates, slow perception, limited benefit from concurrent swings, and a fixed trot beating GaitNet.

### Takeaway
Five transferable ideas stand out:
1. **Corrective-expert distillation and multimodal policies.** PDP's noisy-state/clean-action data, and diffusion or flow policies that avoid mode-averaging, could counter a collapse to the no-op.
2. **Generative candidate proposal with constraint guidance and filter-cluster-select.** EgoNav-2026, BeyondMimic and Diffuse-CLoC guidance, CHOIS contact guidance and Trust Sampling could replace or augment ContactNet.
3. **A contact-conditioned learned tracker as the shared low-level controller.** SceneBot's interface, with a learned dynamics latent as in RTR and PUP, could close the training-versus-deployment controller gap.
4. **Assistance curricula and symmetry losses.** "Learning Symmetric and Low-energy Locomotion" got emergent quadruped gaits without contact planning this way.
5. **Gating non-gaited decisions behind a default gait with learned feasibility values.** ResMimic-style residual over a base policy, and Safe-Stop-style dual estimators with a fallback, could let GaitNet keep the trot's success rate while adding non-gaited steps.

### Cited Findings
- **Multimodality and averaging.**
  - PDP attributes the MLP's lower OOD success (81.0% versus 96.3% for PDP) to "averaging over multimodal responses". The diffusion policy produced distinct recovery modes from the same push. — [PDP arXiv HTML](https://arxiv.org/html/2406.00960)
  - FPO++: with "the same default rewards", Spot trotted more consistently than with Gaussian PPO, which "tended toward symmetric pronking". The cause given is that independent per-dimension Gaussian sampling makes correlated behaviours hard to explore. — [FPO++ arXiv HTML](https://arxiv.org/html/2602.02481)
- **Corrective data.** Recovery success was 3.36% with clean-state/clean-action data, 66.9% with noisy/noisy, and 100% with noisy-state/clean-action. — [PDP arXiv HTML](https://arxiv.org/html/2406.00960)
- **Emergent gaits from curriculum and loss design.**
  - A symmetry loss plus an annealed "modulated physical assistance" curriculum makes "speed-appropriate gait patterns emerge without any guidance from motion examples or contact planning", including on a quadruped. — [Yu, Turk, Liu 2018](https://arxiv.org/abs/1801.08093)
  - Sprinting emerges from a speed-maximizing reward plus termination conditions, with no demonstrations. — [arXiv 2609.11083](https://arxiv.org/abs/2609.11083)
- **Generative propose, filter, select.** EgoNav-2026 samples 64 diffusion trajectories, removes colliding ones, clusters them, and scores clusters by popularity and a momentum penalty. Without the momentum term the robot "oscillat[es] between modes". — [arXiv HTML 2604.00416](https://arxiv.org/html/2604.00416)
- **Guidance as test-time constraint satisfaction.**
  - BeyondMimic adds a cost gradient to the score, using signed-distance-field costs for obstacles, with no retraining. It "works well for coarse objectives, less well for fine-grained ones". — [arXiv HTML 2508.08241](https://arxiv.org/html/2508.08241)
  - CHOIS enforces hand-object and human-floor contact through guidance terms. — [arXiv 2312.03913](https://arxiv.org/abs/2312.03913)
  - Trust Sampling takes multiple guided steps per diffusion level, bounded by trust in the proxy, and terminates when a sample leaves the manifold. — [arXiv 2411.10932](https://arxiv.org/abs/2411.10932)
  - Diffuse-CLoC co-diffuses states and actions so that actions become steerable through predicted states. — [arXiv 2503.11801](https://arxiv.org/abs/2503.11801)
- **Contact-conditioned tracker as the interface.**
  - SceneBot's binary per-link contact command lifts terrain success from 60% (no labels) to 100%, and object success from 0% to 95%. — [arXiv HTML 2606.27581](https://arxiv.org/html/2606.27581)
  - HEAD (three-point commands), VisualMimic (keypoint commands, high-level actions clipped by human motion statistics) and VLK (short-horizon kinematics) all decouple a learned high-level policy from a general tracker. — [HEAD](https://arxiv.org/abs/2508.03068); [VisualMimic](https://arxiv.org/abs/2509.20322); [VLK](https://arxiv.org/abs/2606.30645)
- **Closing a controller or dynamics gap with a latent.**
  - RTR stabilizes sim-to-real by "optimizing a single dynamics-encoded latent variable in the real world". — [arXiv 2508.12252](https://arxiv.org/abs/2508.12252)
  - PUP conditions a policy on a latent η of the dynamics and searches η on hardware with Bayesian optimization. — [arXiv 1903.01390](https://arxiv.org/abs/1903.01390)
  - SimGAN learns a hybrid simulator adversarially. — [arXiv 2101.06005](https://arxiv.org/abs/2101.06005)
- **Base policy plus residual, and gated fallback.**
  - ResMimic: a task-agnostic tracking base plus a learned residual for precision. — [arXiv 2510.05070](https://arxiv.org/abs/2510.05070)
  - Safe-Stop commits to an action only when two learned feasibility estimators agree, and otherwise hands off to a fallback. — [arXiv 2609.02358](https://arxiv.org/abs/2609.02358)
- **Slow perception.**
  - Locomotion Beyond Feet faces the same bottleneck, 3.1 Hz stereo-to-depth. Its mitigations: decide only *discrete* skills, smooth them with an exponential moving average, switch above a confidence threshold, and ignore the classifier during transitions. The 50 Hz trackers carry the motion in between. — [arXiv HTML 2601.03607](https://arxiv.org/html/2601.03607)
  - RPL trains with simulated "sensor latency, noise, and dropout" through a ray-caster that is 5× faster. — [arXiv 2602.03002](https://arxiv.org/abs/2602.03002)
  - PHP randomizes a 60-80 ms observation delay. — [arXiv HTML 2602.15827](https://arxiv.org/html/2602.15827)
  - EgoNav keeps a 360° visual memory rather than relying on the current frame. — [arXiv 2604.00416](https://arxiv.org/abs/2604.00416)
- **Data from human and animal motion.**
  - MVLift lifts 2D pose sequences, including **animal poses**, to global 3D motion without 3D supervision. — [arXiv 2411.18808](https://arxiv.org/abs/2411.18808)
  - SceneBot infers contact edges from retargeted motion by low relative velocity and acceleration. — [arXiv HTML 2606.27581](https://arxiv.org/html/2606.27581)
  - GMR shows that foot sliding and other retargeting artifacts "significantly reduce policy robustness". — [arXiv 2510.02252](https://arxiv.org/abs/2510.02252)
  - PHP's motion-matching features include local foot positions and velocities. — [arXiv HTML 2602.15827](https://arxiv.org/html/2602.15827)

### Inferences
These are my suggestions for GaitNet, not claims made by the papers.

- **Reluctance to step (no-op dominance).**
  - GaitNet takes the argmax over per-candidate scores plus a no-op. This has the same structure as the safe-falling mixture-of-critics, where you pick the body part with the highest value. Value-based argmax over options with very different variance tends to favour the "safe" option. PDP and FPO++ both show unimodal or factorized policies collapsing to one mode.
  - Remedy 1: train the step-versus-no-op decision with a richer distribution, such as a flow or diffusion scorer over joint (leg, foothold, timing) options, instead of factorized per-leg logits.
  - Remedy 2: distill from a *proactive* expert (an OCS2 trot, or a MIP footstep planner) using PDP's noisy-state/clean-action recipe. This gives labels that step *before* the body drifts ahead.
  - Remedy 3: borrow the 2018 "physical assistance" curriculum and anneal the MPC's ability to hold the body without stepping. Today the MPC's competence may make the no-op locally optimal.
  - Remedy 4: use a speed or progress reward in the style of the sprinting paper.
- **ContactNet dependence.** Reposition the candidate generator as a **conditional generative model over foothold sequences**, rather than a cost map plus top-16. It would be conditioned on the terrain grid and body state. Constraint guidance would keep every sample on valid cells, with CHOIS-style contact guidance and Trust Sampling for hard constraints. EgoNav's filter, cluster and momentum-scored selection could pick among the samples. The momentum penalty specifically targets dithering between modes. Training data could come from hindsight-labelled foothold sequences in GaitNet's own successful rollouts, using SceneBot's low-velocity contact detection, or from dog videos lifted with MVLift. This makes the action set learned rather than hand-specified.
- **Sim-to-sim gap (convex MPC versus OCS2 and WBC).** Two options:
  - Train a contact-conditioned *learned* tracker that takes GaitNet's foothold commands as "contact prompts", in the style of SceneBot, and deploy the same tracker in Isaac Lab and on the Go1. This removes the controller swap.
  - Alternatively, condition GaitNet on a controller-identity or dynamics latent, as in PUP or RTR, and search or fine-tune that latent against OCS2.
- **Slow perception.** Adopt Locomotion Beyond Feet's pattern: let the slow perception channel make coarse discrete decisions, with confidence gating and hysteresis, while the fast loop executes. Train with simulated latency and dropout as in RPL and PHP, and keep a persistent terrain memory as in EgoNav.
- **The trot beats GaitNet.**
  - Reposition GaitNet as a **residual or gated decision layer over a default gait**, ResMimic style: the base is the OCS2 trot, and GaitNet decides deviations such as delaying, advancing or re-targeting a step.
  - Pair it with Safe-Stop-style feasibility estimators that let GaitNet override the gait only when a learned "step-feasibility" value is confident, and otherwise fall back to the trot.
  - This reframes the contribution from "non-gaited beats gaited", which the data do not support, to "learned contact decisions extend a gait where the gait fails", which is testable on grid voids.
- **Novelty framing.**
  - SceneBot explicitly lacks "an autonomous high-level controller for perceptive, goal-driven scene interactions" and argues that contact is the right interface. GaitNet can be pitched as exactly that missing layer, for quadrupeds: a perceptive policy that emits contact commands.
  - In addition, the safe-falling precedent (discrete next-contact selection via the best critic) offers a principled lineage, with GaitNet extending it to cyclic locomotion with emergent concurrency.

### Gaps
- None of the group's papers report a quantitative "no-op or hesitation" failure analogous to GaitNet's 85-95% no-op rate, so the mapping to PDP and FPO++ mode-collapse is my inference.
- I found no group paper that generates **quadruped** foothold sequences generatively, or that uses animal motion for quadruped control in 2024-2026. MVLift handles animal poses only for reconstruction.
- I could not verify inference rates for BeyondMimic or Diffuse-CLoC. These matter for whether a diffusion candidate generator could fit GaitNet's 25 Hz tick; EgoNav's generator runs at only about 1.7-2 Hz.

---

## Q3. Does any of the group's work treat contact timing or placement as an explicit discrete or latent decision?

### Takeaway
Yes, but mostly outside quadruped locomotion:
- **Explicit discrete next-contact selection with continuous location and timing:** safe falling, 2017.
- **Explicit binary per-link contact commands to a tracker:** SceneBot, 2026.
- **Explicit discrete skill selection that implies contact patterns:** Locomotion Beyond Feet, 2026. PHP makes the same kind of choice implicitly.
- **Explicit retiming of constraint timing as a learned output:** Generative Motion Infilling, 2025.
- **Latent action representations where contacts are implicit:** BeyondMimic and Diffuse-CLoC.

### Cited Findings
- **Safe falling.** The policy computes "the optimal next contacting body part..., contact location and timing". The discrete contact sequence is solved "through the selection of the best critics" in a mixture of actor-critics, at 50-400× the speed of dynamic programming. — [arXiv 1703.02905](https://arxiv.org/abs/1703.02905)
- **SceneBot.** Uses c_t ∈ {0,1}^n over links × {terrain, object} as a policy input, with contact-mismatch termination if a commanded contact is not made within a time window. Removing the foot labels drops object success from 95% to 70%. — [arXiv HTML 2606.27581](https://arxiv.org/html/2606.27581)
- **Locomotion Beyond Feet.** A discrete skill classifier (93.9% accuracy) selects among keyframe-derived tracking skills that encode "critical contact states", including hands, knees and elbows. — [arXiv HTML 2601.03607](https://arxiv.org/html/2601.03607)
- **Perceptive Humanoid Parkour.** The depth student chooses step over, climb, vault or roll. No explicit selection head is described, so the choice appears implicit. — [arXiv HTML 2602.15827](https://arxiv.org/html/2602.15827)
- **Generative Motion Infilling.** "Explicitly outputs a time-warping function to correct mistimed keyframes." — [arXiv 2503.01016](https://arxiv.org/abs/2503.01016)
- **Possibility graphs.** Footstep and whole-body planning that explores a continuum of actions beyond "a discrete set of predetermined footstep actions". — [arXiv 1610.00700](https://arxiv.org/abs/1610.00700)
- **Contact retargeting for extrinsic manipulation (2024).** Composes primitive policies through state constraints derived from each primitive's contact specification. It reached 80.5% hardware success over 4 long-horizon tasks. — [arXiv 2404.07468](https://arxiv.org/abs/2404.07468)
- **BeyondMimic.** Diffuses over state-latent trajectories in which actions are CVAE latents, so contacts are implicit. — [arXiv HTML 2508.08241](https://arxiv.org/html/2508.08241)
- **Transformer Inertial Poser.** Predicts "stationary body points" as a learned target and uses them to correct drift and build terrain height maps. These are contact-like latent anchors. — [arXiv 2203.15720](https://arxiv.org/abs/2203.15720)
- **OmniRetarget and PRISM.** Both preserve or anchor contacts during data generation: OmniRetarget through an interaction mesh, PRISM through a contact-anchored real-to-sim pipeline. — [arXiv 2509.26633](https://arxiv.org/abs/2509.26633); [arXiv 2609.38172](https://arxiv.org/abs/2609.38172)

### Inferences
- GaitNet's formulation (per-tick discrete choice among terrain-valid footholds plus a no-op, under a concurrency rule) is closest to safe falling's critic-argmax over contact options. The differences are that it is cyclic, sits in a quadruped multi-leg setting, and uses a perception-derived candidate set. Citing safe falling as lineage, together with SceneBot's "contact as interface", would frame GaitNet's novelty as "**learned discrete contact decision-making for multi-legged locomotion, where gaits are an emergent consequence**". That framing is distinct from the group's dominant reference-tracking paradigm.
- One idea combines the two lines: train a SceneBot-like contact-prompted quadruped tracker, and let GaitNet emit the prompts. This would turn GaitNet's MPC dependence into a learned, contact-conditioned low-level policy and make the foothold or contact command the clean interface. SceneBot itself names that missing high-level layer as future work.
- The time-warping idea (Generative Motion Infilling) suggests letting GaitNet output *when* a chosen foothold is touched down (a timing latent) rather than only whether to lift off now. This could address the "steps only when the body drifts ahead" behaviour by allowing anticipatory, scheduled steps.

### Gaps
- I found no group paper that learns contact *timing* as a discrete or latent variable for legged *locomotion*. Safe falling covers a single fall, not cyclic gait. Locomotion Beyond Feet and PHP select skills, not individual contacts.
- SceneBot's paper does not spell out how scene-interaction graph edges become per-timestep label vectors, or whether labels include a future horizon ([arXiv HTML 2606.27581](https://arxiv.org/html/2606.27581)).

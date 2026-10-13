# Deepak Pathak lab (CMU) and Skild AI: legged and humanoid locomotion, 2024-2026, as material for repositioning GaitNet

Scope note for the report writer. Pathak is listed as faculty at CMU and Co-Founder/CEO of Skild AI on his lab page ([lab page](https://www.cs.cmu.edu/~dpathak/)). The lab's peer-reviewed legged-locomotion output in 2024-2026 is thin. There are two core papers (Extreme Parkour, ICRA 2024, and LocoFormer, CoRL 2025), a few adjacent ones (Meta-Evolve, SAPG, EPO, BRIDGE, a Spot mobile-manipulation RL paper), and several Skild AI blog posts that have no methods or metrics. Most of the transferable ideas come from foundational papers from 2021 to 2023, which are labelled "pre-2024" below. GaitNet's own numbers quoted here (25 Hz tick, 16 candidates per leg, 85-95% no-op, 3-4 Hz perception, 45% vs at most 18% traversal) come from the task brief, not from any external source.

Paper list sources: the lab publication page ([lab page](https://www.cs.cmu.edu/~dpathak/)) and an arXiv API author query for Pathak papers whose abstracts mention locomotion, legged, humanoid, quadruped or walking ([arXiv query](https://export.arxiv.org/api/query?search_query=au:Pathak_Deepak%20AND%20%28abs:locomotion%20OR%20abs:legged%20OR%20abs:humanoid%20OR%20abs:quadruped%20OR%20abs:walking%29&sortBy=submittedDate&sortOrder=descending&max_results=40)). That query returned 16 entries; the newest is BRIDGE (Sep 2026).

## Q1. Which Pathak-lab papers (2024-2026) address legged or humanoid locomotion from egocentric vision, cross-embodiment or foundation locomotion, world models, learning from video, or adaptation?

### Takeaway
Only two 2024-2026 lab papers are about legged locomotion itself:
- **Extreme Parkour** (ICRA 2024; arXiv Sep 2023): end-to-end depth-to-joint control on a Unitree A1.
- **LocoFormer** (CoRL 2025 Best Paper finalist): one "omni-bodied" Transformer-XL policy that adapts in context, over long horizons, to unseen legged and wheeled robots, including Unitree G1, H1, Go2 and Go2-W.

Around them sit:
- RL-scaling methods (SAPG, EPO);
- a cross-embodiment policy-transfer paper (Meta-Evolve);
- a humanoid hardware/control co-design paper (BRIDGE, arXiv Sep 2026);
- Skild AI blog posts claiming end-to-end humanoid locomotion from vision with "no planning or mapping", with no paper behind them.

The lab's locomotion line is end-to-end. No paper found in this search outputs footholds.

### Cited Findings

**2024-2026 peer-reviewed or arXiv papers (Pathak co-author)**

- **Extreme Parkour with Legged Robots.** Xuxin Cheng, Kexin Shi, Ananye Agarwal, Deepak Pathak. Venue ICRA 2024 per the lab page; arXiv v1 posted 25 Sep 2023. Links: [arXiv 2309.14341](https://arxiv.org/abs/2309.14341), [project](https://extreme-parkour.github.io/), code at github.com/chengxuxin/extreme-parkour ([lab page](https://www.cs.cmu.edu/~dpathak/)).
  - Core idea: a single neural network goes from a noisy front depth camera plus proprioception to joint targets. It is trained with large-scale RL in simulation instead of separately engineered perception, planning and control ([arXiv abs](https://arxiv.org/abs/2309.14341)).
  - Hardware and rates ([arXiv html](https://arxiv.org/html/2309.14341)):
    - Unitree A1 with a front Intel RealSense D435 at about 10 ± 2 Hz; depth is cropped and downsampled to 58×87.
    - A fixed 0.08 s depth latency and 0.016 s proprioception latency are enforced.
    - The depth backbone runs at 10 Hz and the base policy at 50 Hz, both on a Jetson NX.
    - The deployable policy trains on one RTX 3090 in under 20 h.
  - Training recipe ([arXiv html](https://arxiv.org/html/2309.14341)):
    - Phase 1: an RL teacher with privileged "scandots" (terrain heights sampled under the robot), a waypoint heading and a commanded speed. Environment properties are estimated with regularized online adaptation (ROA). The paper says ROA lets the two phases collapse into one but does not benchmark ROA against RMA.
    - Phase 2: DAgger distillation into a convnet-GRU depth student.
    - The student also predicts heading from depth. Its predicted yaw is used only when it is within 0.6 of the oracle heading.
  - Rewards ([arXiv html](https://arxiv.org/html/2309.14341)):
    - An inner-product velocity reward in the world frame, so the robot cannot exploit base-frame tracking by turning around obstacles.
    - A foot-edge clearance penalty for any foot contacting the ground within 5 cm of a terrain edge.
  - Results ([arXiv html](https://arxiv.org/html/2309.14341)):
    - High jump onto a 0.5 m box (about 2× hip height), long jump across a 0.8 m gap (about 2× front-to-rear foot separation), and tilted ramps (Table 1 lists 37°).
    - Simulation ablation, mean x-displacement: 0.98 full vs 0.75 without the inner-product reward vs 0.82 with a noisy elevation map.
    - Simulation ablation, edge violations: 0.03 full vs 0.08 without the clearance penalty vs 0.20 with a noisy elevation map.
    - Real world: 20-80% higher success than baselines on the hardest instance of each terrain.

- **LocoFormer: Generalist Locomotion via Long-context Adaptation.** Min Liu, Deepak Pathak, Ananye Agarwal. CoRL 2025, Best Paper Award Finalist. Links: [arXiv 2509.23745](https://arxiv.org/abs/2509.23745), [project](https://generalist-locomotion.github.io/), [lab page](https://www.cs.cmu.edu/~dpathak/).
  - Core idea: an "omni-bodied" policy that controls unseen legged and wheeled robots without precise kinematic knowledge and adapts at test time to morphology and dynamics changes ([arXiv abs](https://arxiv.org/abs/2509.23745)). It rests on two choices:
    - large-scale RL on procedurally generated robots with aggressive domain randomization;
    - a context extended "by orders of magnitude" so that it spans episode boundaries.
  - Architecture ([arXiv html](https://arxiv.org/html/2509.23745v1)):
    - Transformer-XL with 6 layers and segment length 128; maximum memory 896 timesteps, about 18 s.
    - At deployment the KV cache keeps 255 steps, about 5 s at 50 Hz.
    - Actions are joint position targets in a unified joint space.
  - Training ([arXiv html](https://arxiv.org/html/2509.23745v1)):
    - PPO; the appendix mentions "100k robots".
    - Each episode contains several trials, so the policy is rewarded for improving across trials.
    - Randomized: mass, CoM, inertia, control gains, joint limits and kinematics.
    - A 6-layer model takes 28 h for 4K iterations.
  - Results ([arXiv html](https://arxiv.org/html/2509.23745v1)):
    - Simulation, normalized displacement on 10 unseen robots on rough terrain: 0.96 zero-shot and 0.98 with 5 s adaptation, vs 0.37 for a GRU baseline, 0.78 for a conditioning baseline and 0.99 for per-robot experts.
    - Real robots: Unitree G1, H1, Go2 and Go2-W, plus Go2/Go2-W walking bipedally on their rear legs.
    - Qualitative: walks with a locked knee after 2-3 s of adaptation, and on its knees "after legs are cut" after 7-8 s (the Skild blog describes this as losing its calves).
    - No real-world success rates are reported.
  - Inputs: no exteroception or vision input is described in the paper HTML, the project page or the Skild blog ([arXiv html](https://arxiv.org/html/2509.23745v1); [project](https://generalist-locomotion.github.io/)).
  - Stated limitations ([arXiv html](https://arxiv.org/html/2509.23745v1); [project](https://generalist-locomotion.github.io/)):
    - Training is very resource-intensive.
    - The procedural task space is hand-crafted.
    - Some real morphologies, such as anhedral hips, fall outside its parameterization.
    - It may not adapt fast enough to sudden disturbances on hard morphologies.
  - The project page links to Skild AI and to the Skild "omni-bodied" blog ([project](https://generalist-locomotion.github.io/)).

- **Meta-Evolve: Continuous Robot Evolution for One-to-many Policy Transfer.** Xingyu Liu, Deepak Pathak, Ding Zhao. ICLR 2024. Link: [arXiv 2405.03534](https://arxiv.org/abs/2405.03534).
  - Transfers one expert policy to several target robots through a tree of continuously "evolving" intermediate robots.
  - Reports 2.4× better simulation-cost efficiency for one-to-six agile-locomotion transfer and up to 3.2× for one-to-three manipulation transfer, against independent one-to-one transfers ([arXiv abs](https://arxiv.org/abs/2405.03534)).

- **SAPG: Split and Aggregate Policy Gradients.** Jayesh Singla, Ananye Agarwal, Deepak Pathak. ICML 2024 (Oral). Link: [arXiv 2407.20230](https://arxiv.org/abs/2407.20230).
  - PPO "fails to ingest the benefit of parallelized environments beyond a certain point".
  - SAPG splits the environments into chunks and fuses them with importance sampling. It beats vanilla PPO on "challenging environments" ([arXiv abs](https://arxiv.org/abs/2407.20230)).

- **Evolutionary Policy Optimization (EPO).** Jianren Wang, Yifan Su, Abhinav Gupta, Deepak Pathak. arXiv preprint, Mar 2025, revised Nov 2025; no venue listed. Link: [arXiv 2503.19037](https://arxiv.org/abs/2503.19037).
  - A latent-conditioned population of agents shares actor-critic parameters and aggregates experience into a master agent, targeting on-policy RL's poor scaling with batch size.
  - Evaluated on dexterous manipulation, legged locomotion and classic control ([arXiv abs](https://arxiv.org/abs/2503.19037)).

- **BRIDGE: An Open-Source Humanoid Platform via Morphology-Control Co-Design for Physical AI.** Jianren Wang, Letian Qian, Zikai Wang, Weiwei Wu, Junjie Zong, Abhinav Gupta, Deepak Pathak. arXiv, 3 Sep 2026. Link: [arXiv 2609.03497](https://arxiv.org/abs/2609.03497).
  - Co-designs humanoid morphology and control, scored by a metric that combines human-motion retargetability with dynamic tracking.
  - Result is an 88 cm open-source humanoid, compared against Bumi, K1 and Toddlerbot. The abstract makes only general locomotion and balance claims ([arXiv abs](https://arxiv.org/abs/2609.03497)).

- **Continuously Improving Mobile Manipulation with Autonomous Real-World RL.** Russell Mendonca, Emmanuel Panov, Bernadette Bucher, Jiuguang Wang, Deepak Pathak. CoRL 2024. Link: [arXiv 2409.20568](https://arxiv.org/abs/2409.20568).
  - Fully autonomous real-world RL on Boston Dynamics Spot, using task-relevant autonomy, behaviour priors and generic rewards.
  - Reaches an average 80% success across four mobile-manipulation tasks ([arXiv abs](https://arxiv.org/abs/2409.20568)).
  - This is legged hardware, but the paper is about manipulation, not locomotion.

- **SPIN: Simultaneous Perception, Interaction and Navigation.** Uppal, Agarwal, Xiong, Shaw, Pathak. CVPR 2024 (Oral). Link: [arXiv 2405.07991](https://arxiv.org/abs/2405.07991).
  - The robot is wheeled (a Hello Robot base, "wheel odometry"), not legged ([project](https://spin-robot.github.io/)).
  - Relevant only for its active ego-vision idea: a reactive policy that "moves to see and sees to move", using only ego-vision "without needing to create environment maps" ([arXiv abs](https://arxiv.org/abs/2405.07991); [project](https://spin-robot.github.io/)).

- **Learning from video and world models (2026, not locomotion).**
  - ViPRA: Video Prediction for Robot Actions (Routray, Pan, Jain, Bahl, Pathak; ICLR 2026) learns generalist policies from human and robot videos without action labels ([arXiv 2511.07732](https://arxiv.org/abs/2511.07732); [lab page](https://www.cs.cmu.edu/~dpathak/)).
  - Latent Particle World Models (Daniel, Qi, Haramati, Zadeh, Li, Tamar, Pathak, Held; ICLR 2026 Oral) is a self-supervised, object-centric, stochastic video world model ([arXiv 2603.04553](https://arxiv.org/abs/2603.04553)).
  - Neither touches legged locomotion.

- **Open X-Embodiment (ICRA 2024 Best Paper).** It appears on the lab page as a cross-embodiment dataset and models, but the lab-page fetch did not show Pathak among the listed authors ([lab page](https://www.cs.cmu.edu/~dpathak/)). It is a manipulation dataset.

**Skild AI blog posts (company material, no named authors, no linked papers, no methods; not peer-reviewed)**

- **"One Model, Any Scenario: End-to-end Locomotion from Vision"** (6 Aug 2025) ([Skild blog](https://www.skild.ai/blogs/one-policy-all-scenarios)).
  - A humanoid low-level policy goes from "raw images and joint feedback" directly to motor commands, "without any planning, mapping, or manual switching between behaviors".
  - It sits under a low-frequency high-level policy.
  - Scenes shown: stairs, unstable pallets, gaps, uneven steps, clutter, fire escapes, box carrying on stairs.
  - The only number is precise footwork on stairs 3 cm deeper than the foot.
  - It names stepping stones only as a problem robots struggle with. It does **not** claim to traverse them.
  - It does not say whether the images are RGB or depth.
- **"The case for an omni-bodied robot brain"** (24 Sep 2025) ([Skild blog](https://www.skild.ai/blogs/omni-bodied)).
  - The blog counterpart of LocoFormer: about 100,000 simulated robots, "a millennium" of simulated time, and zero-shot tests on robots held out of training.
  - Demos: a quadruped that loses its calves adapts in 7-8 s; locked knees adapt in 2-3 s; jammed wheels make a wheeled robot switch to walking.
  - On stilts the robot "adjusts step timing and foot placement".
- **"Learning by watching human videos"** (12 Jan 2026) ([Skild blog](https://www.skild.ai/blogs/learning-by-watching)): manipulation skills learned from human video plus under 1 h of robot data. Locomotion is not mentioned.
- **"Introducing S1"** (18 Aug 2026) ([Skild blog](https://www.skild.ai/blogs/s1)).
  - A manipulation foundation model that learns in context from one video demonstration, with about 66% vs 9% success on unseen tasks against a language-prompted VLA.
  - It cites LocoFormer as Skild's earlier in-context locomotion learner.
- **"Physical Self-Play"** (23 Sep 2026) ([Skild blog](https://www.skild.ai/blogs/physical-self-play)).
  - Humanoid soccer trained by self-play in Isaac Sim for about 140 simulated years, then transferred to a physical humanoid.
  - Locomotion and get-up skills emerge. No metrics are given.
- Full post index: [Skild blog index](https://www.skild.ai/blogs).

**Foundational, pre-2024 (for context)**

- **RMA: Rapid Motor Adaptation for Legged Robots.** Kumar, Fu, Pathak, Malik. RSS 2021. Link: [arXiv 2107.04034](https://arxiv.org/abs/2107.04034).
  - A base policy plus an adaptation module adapt "in fractions of a second" to terrain, payload and wear.
  - Trained entirely in simulation with no reference trajectories or foot-trajectory generators, and deployed on A1 without fine-tuning ([arXiv abs](https://arxiv.org/abs/2107.04034)).
- **A-RMA: Adapting Rapid Motor Adaptation for Bipedal Robots.** Kumar, Li, Zeng, Pathak, Sreenath, Malik. IROS 2022. Link: [arXiv 2205.15299](https://arxiv.org/abs/2205.15299); venue per [lab page](https://www.cs.cmu.edu/~dpathak/).
- **Legged Locomotion in Challenging Terrains using Egocentric Vision.** Agarwal, Kumar, Malik, Pathak. CoRL 2022 (Oral; Best System Paper Award per the lab page). Links: [arXiv 2211.07638](https://arxiv.org/abs/2211.07638), [lab page](https://www.cs.cmu.edu/~dpathak/). Details in Q3.
- **Minimizing Energy Consumption Leads to the Emergence of Gaits in Legged Robots.** Fu, Kumar, Malik, Pathak. CoRL 2021. Link: [arXiv 2111.01674](https://arxiv.org/abs/2111.01674). Details in Q3.
- **Coupling Vision and Proprioception for Navigation of Legged Robots.** Fu, Kumar, Agarwal, Qi, Malik, Pathak. CVPR 2022. Link: [arXiv 2112.02094](https://arxiv.org/abs/2112.02094); venue per [lab page](https://www.cs.cmu.edu/~dpathak/).
- **Deep Whole-Body Control: Learning a Unified Policy for Manipulation and Locomotion.** Fu, Cheng, Pathak. CoRL 2022 (Oral). Link: [arXiv 2210.10044](https://arxiv.org/abs/2210.10044); venue per [lab page](https://www.cs.cmu.edu/~dpathak/).
- **Legs as Manipulator: Pushing Quadrupedal Agility Beyond Locomotion.** Cheng, Kumar, Pathak. ICRA 2023. Link: [arXiv 2303.11330](https://arxiv.org/abs/2303.11330). Quadrupeds climb, press buttons and manipulate with a leg ([lab page](https://www.cs.cmu.edu/~dpathak/)).
- **Planning in Learned Latent Action Spaces for Generalizable Legged Locomotion.** Li, Calandra, Pathak, Tian, Meier, Rai. RA-L 2021. Link: [arXiv 2008.11867](https://arxiv.org/abs/2008.11867); venue per [lab page](https://www.cs.cmu.edu/~dpathak/).
- **One Policy to Control Them All: Shared Modular Policies for Agent-Agnostic Control.** Huang, Mordatch, Pathak. ICML 2020. Link: [arXiv 2007.04976](https://arxiv.org/abs/2007.04976). An early cross-embodiment controller ([lab page](https://www.cs.cmu.edu/~dpathak/)).
- **Structured World Models from Human Videos.** Mendonca, Bahl, Pathak. RSS 2023. Link: [arXiv 2308.10901](https://arxiv.org/abs/2308.10901). A world model over affordances learned from human video ([lab page](https://www.cs.cmu.edu/~dpathak/)).

### Inferences
- There is no 2024-2026 Pathak-lab academic paper on perceptive humanoid locomotion. The only evidence is the Skild blog, which has no methods or metrics. A reviewer citing "Skild does vision humanoid locomotion" has nothing quantitative to compare against.
- The lab's research direction has moved from task-specific perceptive skills (2022-2023) to generalist and adaptive policies (LocoFormer) and RL scaling (SAPG, EPO). Positioning GaitNet against the 2022-2023 depth line, where the numbers are, is more concrete than positioning it against LocoFormer.

### Gaps
- The lab page is about 207 KB and was read in two chunks; the last ~7 KB (oldest entries) were not read. The arXiv API query filters on abstract keywords and missed Extreme Parkour, so the list may not be exhaustive. dblp blocked automated access, and Google Scholar was not queried.
- LocoFormer's observation vector is not fully specified in the sources fetched. The real-world terrains are not named.
- Meta-Evolve's locomotion robots are unnamed in the abstract, so it is unconfirmed that they are legged.
- SAPG's benchmark environments and EPO's legged-locomotion environments and numbers were not verified.
- Open X-Embodiment authorship by Pathak was not verified.
- Xuxin Cheng's later humanoid papers (for example ExBody or humanoid parkour) are believed to be with other groups. Pathak co-authorship was not verified, so they are excluded.
- Planning in Learned Latent Action Spaces (RA-L 2021): only the title, authors and venue were verified, not its method.

## Q2. Which of these ideas could address GaitNet's weaknesses?

### Takeaway
Four transferable ideas come out of the lab's work:
1. Teacher-student distillation from privileged height samples into a depth-plus-memory student running at 10 Hz depth and a 50 Hz policy, instead of GaitNet's 3-4 Hz depth → elevation map → plane-decomposition pipeline.
2. RMA-style or LocoFormer-style history and in-context adaptation, to absorb the training-vs-deployment controller gap by treating "which low-level controller executes the footstep" as a randomized dynamics factor.
3. Massive procedural cross-embodiment training. GaitNet's hip-frame foothold action space is naturally morphology-agnostic, which joint-space LocoFormer is not.
4. Reward and evaluation details from Extreme Parkour: world-frame inner-product progress reward, foot-edge clearance penalty and an edge-violation metric.

### Cited Findings
- **Perception latency.**
  - Agarwal et al. 2022 argue that elevation mapping followed by foothold planning is "fragile", needs specialized hardware and is biologically implausible. They replace it with a policy driven directly by egocentric depth ([arXiv abs](https://arxiv.org/abs/2211.07638)).
  - Their system ([arXiv html](https://arxiv.org/html/2211.07638v1)):
    - Depth frames arrive every ~100 ms ±20 ms and are downsampled to 58×87.
    - The policy runs at 50 Hz with a 400 Hz PD loop (Kp = 40, Kd = 0.5).
    - The depth convnet runs on a Jetson NX and the base policy on an UPboard Atom CPU.
    - The policy "needs only a single feedforward pass and no state tracking or localization".
  - Extreme Parkour uses a 10 Hz depth backbone with a 50 Hz base policy on a Jetson NX and enforces depth latency in training ([arXiv html](https://arxiv.org/html/2309.14341)).
  - In Extreme Parkour's simulation ablation, a noisy elevation-map variant raised edge violations from 0.03 to 0.20 and cut x-displacement from 0.98 to 0.82 ([arXiv html](https://arxiv.org/html/2309.14341)).
- **Memory for the unseen hind-foot terrain.** A front camera cannot see under the hind feet, so the GRU integrates depth and proprioception to "implicitly track and estimate the terrain under the robot" ([arXiv html](https://arxiv.org/html/2211.07638v1)).
- **Teacher-student recipe.** Phase 1 is RL with cheap "scandots" (sampled terrain heights). Phase 2 is DAgger distillation into a depth student. In the RMA-style variant, phase 2 trains only the estimators: a GRU for terrain geometry and an MLP for extrinsics ([arXiv html](https://arxiv.org/html/2211.07638v1)).
- **Adaptation to dynamics mismatch.**
  - RMA's adaptation module infers environment factors online "in fractions of a second", without fine-tuning on the real robot ([arXiv abs](https://arxiv.org/abs/2107.04034)).
  - LocoFormer's long context (896-step memory) and multi-trial episodes let it learn from earlier falls within one deployment. It improves from 0.96 zero-shot to 0.98 with 5 s of adaptation, against a GRU baseline at 0.37 ([arXiv html](https://arxiv.org/html/2509.23745v1)).
  - LocoFormer randomizes control gains and joint limits in addition to mass, CoM and inertia ([arXiv html](https://arxiv.org/html/2509.23745v1)).
- **Cross-embodiment scale.** About 100k procedurally generated robots across bipeds, quadrupeds and their wheeled versions, with test robots held out ([arXiv html](https://arxiv.org/html/2509.23745v1); [Skild blog](https://www.skild.ai/blogs/omni-bodied)).
  - Meta-Evolve transfers policies across robots through an evolution tree, 2.4× more efficient for one-to-six locomotion transfer ([arXiv abs](https://arxiv.org/abs/2405.03534)).
- **Reward shaping against exploitation.** Extreme Parkour measures progress as the inner product of world-frame velocity with the direction to the next waypoint, capped at the commanded speed. This stops the robot exploiting base-frame tracking by turning around obstacles; removing it cut displacement from 0.98 to 0.75 ([arXiv html](https://arxiv.org/html/2309.14341)).
- **Heading from vision.** The Extreme Parkour student predicts its own heading from depth, mixing teacher and student yaw during distillation ([arXiv html](https://arxiv.org/html/2309.14341)).
- **RL scaling.** SAPG targets PPO's saturation under massively parallel simulation ([arXiv abs](https://arxiv.org/abs/2407.20230)). EPO targets the same issue with a latent-conditioned population ([arXiv abs](https://arxiv.org/abs/2503.19037)).
- **Stated limitation of the end-to-end line.** Agarwal et al. note that failures from sim-to-real visual or terrain mismatch are currently fixed by recreating the case in simulation and retraining. They call this a fundamental limitation ([arXiv html](https://arxiv.org/html/2211.07638v1)).
- **Strong baseline evidence.** On a real A1 the 2022 depth policy achieves 100% on gaps up to 26 cm and 94% on stepping stones, against 0% for a blind baseline ([arXiv html](https://arxiv.org/html/2211.07638v1)). Extreme Parkour code is public ([lab page](https://www.cs.cmu.edu/~dpathak/)).

### Inferences
- **Slow perception (3-4 Hz, falls).** Distil GaitNet's foothold scorer, and ContactNet, from a privileged height-scan teacher into a depth-plus-GRU student, following Agarwal 2022 and Extreme Parkour. Candidate validity and the cost map would then come from a learned latent updated at camera rate (about 10 Hz or more), not from plane decomposition. GaitNet keeps its discrete candidate interface, but the C-space or cost map becomes a predicted quantity.
  - The training-time latency randomization in Extreme Parkour (fixed 0.08 s depth latency) is a cheap first step even if the map pipeline stays.
- **Training controller vs deployment controller gap.** Treat the low-level controller as a randomized dynamics factor:
  - randomize convex MPC parameters, tracking delays and swing-trajectory errors in training, or mix several controllers;
  - give the footstep policy an RMA-style extrinsics estimator, or a LocoFormer-style long-context history over (commanded foothold, achieved foothold, base state), so it infers the active controller's characteristics online.
  - This supports a novel framing: a "controller-agnostic footstep policy that adapts in context to whichever MPC or WBC executes it", the footstep-level analogue of LocoFormer's omni-bodied claim. LocoFormer adapts across bodies at joint level and does not address mismatch between the training and deployment controllers.
- **Cross-embodiment.** GaitNet's action is a foothold in the leg's hip-yaw frame and torques come from a model-based controller. A footstep policy trained over procedurally varied leg lengths, hip offsets and masses, with the MPC adjusted per robot, is a plausible "omni-bodied footstep planner". Its claim would differ from LocoFormer's: interpretable contact decisions plus model-based safety, rather than joint-space generality.
- **Reluctance to step (85-95% no-op, per the brief).** Extreme Parkour's ablation suggests that how progress is rewarded strongly changes behaviour (0.75 vs 0.98 displacement). Switching GaitNet's progress term to world-frame inner-product progress toward waypoints, rather than a body-frame velocity, is a cheap experiment, not a predicted fix.
  - LocoFormer's multi-trial episodes, which reward improvement across falls, are another option if early-termination penalties are driving the conservative no-op behaviour. This is a hypothesis.
- **Candidate-generator dependence (ContactNet defines the action set).** The lab's stance is that the policy should learn terrain affordances implicitly ([arXiv abs](https://arxiv.org/abs/2211.07638)). GaitNet can answer in either of two ways:
  - train ContactNet jointly, or distil it into the depth encoder, so the action set is learned end-to-end;
  - present the explicit candidate set as a feature: auditable, constraint-respecting foothold choice that a model-based controller can verify. The end-to-end line has no equivalent.
- **Traversal deficit (OCS2 trot 45% vs GaitNet at most 18%, per the brief).** Reviewers aware of this lab will expect an end-to-end depth baseline on GaitNet's grid-void terrain. Extreme Parkour's public code makes one feasible, and its 94% stepping-stone result on A1 (2022) sets a high bar.
- **Training scale with `gpu_mpc`.** SAPG and EPO are relevant if GaitNet's PPO stops improving as the number of environments grows. Only worth trying if that saturation is actually measured.

### Gaps
- None of these papers combines a learned high-level policy with a model-based MPC or WBC low-level controller. So there is no direct evidence that RMA or LocoFormer adaptation transfers to footstep-level actions executed by different controllers. This is untested.
- No Pathak-lab paper measures the effect of perception rate (Hz) on falls in isolation. The noisy-elevation-map ablation is the closest comparison.
- Skild's "end-to-end locomotion from vision" has no published method, sensor type or metrics, so it cannot be used as a quantitative baseline.

## Q3. Any work on explicit foothold or contact selection, stepping-stone terrain, or gait emergence?

### Takeaway
The lab has published no explicit foothold or contact-selection method, in 2024-2026 or earlier. Its 2022 flagship explicitly rejects the elevation-map → foothold-planning pipeline in favour of implicit foothold choice inside an end-to-end depth policy.
- **Stepping stones:** handled in 2022 (94% on a real A1) through rewards and implicit learning only.
- **Foothold constraints:** appear only as reward terms (Extreme Parkour's 5 cm foot-edge penalty and edge-violation metric).
- **Gait emergence:** prior art in this lab is strong (energy minimization, CoRL 2021; emergent hip-abduction climbing, 2022; Skild's stilts demo adjusting "step timing and foot placement").

So GaitNet must claim emergence at the level of discrete, explicit footstep decisions under a contact rule, not gait emergence in general.

### Cited Findings
- **Agarwal et al., CoRL 2022 (pre-2024).**
  - Stairs, curbs, stepping stones and gaps from a single front depth camera ([arXiv abs](https://arxiv.org/abs/2211.07638)).
  - Real A1: 100% success on gaps up to 26 cm and 94% on stepping stones; the blind baseline scores 0% on both ([arXiv html](https://arxiv.org/html/2211.07638v1)).
  - Simulated stepping-stone displacement: 20.72 monolithic and 18.83 RMA-style, vs 1.09 noisy and 1.02 blind ([arXiv html](https://arxiv.org/html/2211.07638v1)).
- **Emergent behaviour without gait priors.** A hip-abduction climbing motion emerges for curbs and obstacles up to 26 cm because the small robot has no room to step up ([arXiv html](https://arxiv.org/html/2211.07638v1)). The abstract says the robot develops gait patterns not seen elsewhere ([arXiv abs](https://arxiv.org/abs/2211.07638)).
- **Fu et al., CoRL 2021 (pre-2024).** Minimizing mechanical energy produces natural gaits at different speeds on real quadrupeds. On flat terrain they are structured and horse- or sheep-like; on rough terrain they become unstructured, matching animal motor-control findings ([arXiv abs](https://arxiv.org/abs/2111.01674)).
- **Implicit foothold constraint, Extreme Parkour (ICRA 2024).** A clearance penalty punishes any foot touching within 5 cm of a terrain edge, and "edge violation" is reported as a metric. Without it the robot places feet too close to edges and sometimes falls ([arXiv html](https://arxiv.org/html/2309.14341)).
- **RMA (RSS 2021).** Trained with no reference trajectories or predefined foot-trajectory generators ([arXiv abs](https://arxiv.org/abs/2107.04034)).
- **Skild omni-bodied blog (2025).** A robot fitted with stilts "adjusts step timing and foot placement". A wheeled robot with jammed wheels switches to walking and back ([Skild blog](https://www.skild.ai/blogs/omni-bodied)). These are company claims without metrics.
- **Skild end-to-end vision blog (2025).** Stepping stones are named only as a hard case. Traversing them is not claimed ([Skild blog](https://www.skild.ai/blogs/one-policy-all-scenarios)).

### Inferences
- **Defensible niche.** Every Pathak-lab system chooses footholds implicitly inside joint-space actions. GaitNet's explicit, discrete foothold-and-timing decision, with a hard contact rule and a model-based executor, sits between that line and classical footstep planners. The novelty claim should be "learned, inspectable, non-gaited contact sequencing compatible with MPC or WBC", not "end-to-end" or "gaits emerge".
- **Tightening the emergence claim.** Given Fu et al. 2021, GaitNet should show what is new in its emergence: for example, how often concurrent swings occur and how they depend on terrain void density or body-foot offset. Comparing with an energy-regularized end-to-end baseline would make the claim specific.
- **Report Extreme Parkour's metric.** Its edge-violation metric maps directly onto GaitNet's grid voids. Reporting the share of footholds within X cm of a void edge, for GaitNet vs the OCS2 trot vs an end-to-end depth policy, would give a comparison the lab's own papers use.
- **Stepping stones as unoccupied ground.** On hard-void (stepping-stone) terrain, the 2022 result is the closest lab precedent (94% on A1), and no 2024-2026 lab paper or Skild post claims stepping stones. GaitNet can occupy sparse-foothold terrain with explicit foothold reasoning as a niche the end-to-end line has not revisited since 2022, if GaitNet's traversal rate can be raised.

### Gaps
- The 2022 paper's stepping-stone geometry (stone size and spacing) and number of trials were not extracted, so direct comparison with GaitNet's grid voids is not yet possible.
- Fu et al. 2021: which gait appears at which speed, and the robot model, were not verified from the abstract page.
- No Pathak-lab work on legged contact-sequence or footstep selection as an explicit action space was found. Planning in Learned Latent Action Spaces (RA-L 2021) is the only planning-flavoured locomotion paper, and its content was not verified.

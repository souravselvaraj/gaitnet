# Ye Zhao's LIDAR Lab (Georgia Tech), 2024-2026: contact/footstep planning, TAMP, safety and learning-for-planning, as material for repositioning GaitNet

Scope note: the lab is the Laboratory for Intelligent Decision and Autonomous Robots (LIDAR), Georgia Tech, directed by Ye Zhao. Its publication list is the primary source for authorship and venues ([lab publications page](https://lab-idar.gatech.edu/publications/)). Where a venue (e.g. "ICRA 2026") appears only on the lab page and not on arXiv, the lab page is cited for it. Pre-2024 work is marked **[foundational, pre-2024]**.

## Q1. Which LIDAR Lab papers (2024-2026) address footstep/contact-sequence planning, learning-guided MPC/TAMP, safety, temporal logic, and diffusion/learning-based planning?

### Takeaway
The lab has three quadruped foothold planners for sparse terrain that are directly comparable to GaitNet: MICP-based reactive synthesis with symbolic repair (IROS 2025, plus an extended version on Go2 and a rebar robot), experience-informed A* contact planning on rebar grids (ICRA 2024), and QuadPiPS, perception-space A* over superpixel footholds (Go2). Around these sit a large body of temporal-logic MIP/TAMP work (Logic Network Flow, Benders decomposition, STL-guided MPC), probabilistic-safety work for bipeds (GP terrain, conformal prediction, contraction tubes, learned traversability), and learning-augmented MPC (RL that parametrizes MPC, imitation of model-based controllers). The lab has diffusion-based planning papers, but none plans contacts.

### Cited Findings

#### A. Quadruped foothold / contact-sequence planning on sparse terrain (most relevant to GaitNet)

**A1. Physically-Feasible Reactive Synthesis for Terrain-Adaptive Locomotion via Trajectory Optimization and Symbolic Repair.** Ziyi Zhou, Qian Meng, Hadas Kress-Gazit, Ye Zhao. IROS 2025. arXiv 2503.03071.
- The lab page lists it as IROS 2025 — [lab page](https://lab-idar.gatech.edu/publications/); [arXiv 2503.03071](https://arxiv.org/abs/2503.03071)
- Core idea: reactive synthesis builds correct-by-construction symbolic controllers. Mixed-integer convex programming (MICP) plans dynamically feasible footsteps for each symbolic transition. When the specification is unrealizable because of dynamic infeasibility, symbolic repair adds only the transitions it needs, which limits costly MICP solves. MICP is re-run online on real terrain data. — [arXiv 2503.03071](https://arxiv.org/abs/2503.03071)
- The paper motivates itself against two alternatives: heuristic "instantaneous foothold selection," which it says compromises safety and versatility, and expensive long-horizon trajectory optimization. Its demonstrations include scattered stepping stones and rebars. — [arXiv 2503.03071](https://arxiv.org/abs/2503.03071)

**A1-ext. Physically-Feasible Reactive Synthesis for Terrain-Adaptive Locomotion (extended version).** Same authors. Under review; v1 Sep 2025, v2 Jul 2026. arXiv 2509.23185. Project page: synthesis-micp.github.io. All numbers below come from this extended version, not the IROS paper.
- The lab page lists it as a manuscript under review — [lab page](https://lab-idar.gatech.edu/publications/); [arXiv 2509.23185](https://arxiv.org/abs/2509.23185)
- MICP formulation: a single-rigid-body model in which an Euler-angle double integrator replaces the rotational dynamics to keep the problem convex. Each convex terrain polygon r has a binary H_{r,k,j} for footstep k and foot j, and exactly one region is selected per footstep. It adds friction cones, zero velocity in stance, zero force out of contact, torque limits from a fixed Jacobian, and a box around a nominal foot position for kinematics. Because the problem is convex, a feasible solution certifies that the skill is physically feasible. — [arXiv 2509.23185 (HTML)](https://arxiv.org/html/2509.23185)
- Symbolic layer: the environment is an n×m grid with a terrain type per cell. "Skills" are locomotion gaits (e.g. a one-second trot) with pre- and postconditions under a GR(1) LTL specification. Online, a **gait-fixed** MICP is re-solved on perceived terrain, and the automaton advances only when the MICP finds a solution. The **gait-free** MICP validates repair candidates: offline during synthesis, and online only as a runtime-repair step. — [arXiv 2509.23185 (HTML)](https://arxiv.org/html/2509.23185)
- Runtime repair, unforeseen gap (Unstructured-4, 3×3): 12.36 s in total, of which 0.18 s is symbolic repair and 12.18 s is the gait-free MICP. Re-synthesis after a solve failure, with no full repair needed, takes 0.07-0.11 s. On hardware (Chotu), an obstacle detour was re-synthesized within 0.05 s. — [arXiv 2509.23185 (HTML)](https://arxiv.org/html/2509.23185)
- Results: repair resolves 93.4% of terrain/request pairs. It needs 71.6-97.6% fewer new skill checks than exhaustive enumeration. A gait-free MICP takes 27.23 s on average (max 145.66 s). Online gait-fixed MICP solves take 0.37-2.65 s in simulation, about 430 ms on unstructured terrain and 1.1 s on rebar without collision avoidance. Collision avoidance raises binaries by 271% and solve time by 441%. — [arXiv 2509.23185 (HTML)](https://arxiv.org/html/2509.23185)
- Against a pure-MIP baseline on dense rebar (10 seeded waypoints): pure MIP with a 2 s horizon succeeds 60%, and with a 3 s horizon 90% (MIP solve 11.54 s). The proposed planner succeeds 90% with a 5.26 s MIP solve, at about 75% of the H=3 s traversal time. Without the feasibility check, pure MIP drops to 20% and 40%. — [arXiv 2509.23185 (HTML)](https://arxiv.org/html/2509.23185)
- Hardware: Unitree Go2 and SkyMul's "Chotu" rebar-tying quadruped. Mean online MICP solve times were 0.53-9.13 s (max 19.23 s). Gaits were conservative: a 3 s trot, a 1.5 s leap over a ~30 cm gap, and a 2.25 s static walk. On an extreme sparse rebar layout (max spacing 0.48 m), a heuristic nearest-feasible-foothold planner at 0.2 m/s failed through foot trapping, while the proposed planner completed the traversal. — [arXiv 2509.23185 (HTML)](https://arxiv.org/html/2509.23185)
- Stated limitations: terrain abstraction is manual, and hardware perception was motion-capture-aided manual polygon labeling. Feasible gaits are chosen by "first success," with no optimality criterion. Angular-momentum coupling is neglected. No RL or TOWR-style baseline was run. — [arXiv 2509.23185 (HTML)](https://arxiv.org/html/2509.23185)

**A2. Hierarchical Experience-informed Navigation for Multi-modal Quadrupedal Rebar Grid Traversal.** Max Asselmeier, Jane Ivanova, Ziyi Zhou, Patricio A. Vela, Ye Zhao. ICRA 2024 (Yokohama, pp. 8065-8072). arXiv 2311.08354.
- Venue and pages — [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10610248); [NSF PAR](https://par.nsf.gov/biblio/10538080); [arXiv](https://arxiv.org/abs/2311.08354)
- Hierarchy: an optional global torso-path A* (edges under τ = 0.15 m, weighted by obstacle proximity) feeds a discrete contact planner, A* over a **mode-transition graph**, which outputs a "lead," a mode sequence from start to goal. Whole-body trajectory optimization (an OCS2 SQP with a 0.5 s horizon and 50 knots) connects consecutive modes. An MPC (100 Hz) and WBC (500 Hz) track the result. — [arXiv HTML](https://arxiv.org/html/2311.08354v3)
- A mode is "three stance legs in contact with three separate rebars," written as a tuple of mode family and continuous foothold "coparameters." Edges exist only if swing legs alternate in a **user-defined contact sequence** and the transition is kinematically plausible, which it checks with superquadric reachability regions. — [arXiv HTML](https://arxiv.org/html/2311.08354v3)
- **Experience as a learned candidate prior:** each attempted transition yields a cost J, with J = ∞ on failure, mapped to a penalty δ = w₁·tanh(w₂·J + w₃). Experience is stored as a weighted sum of radial basis functions centred at attempted footholds, so untried nearby footholds inherit information. The edge weight combines this experience distribution with CoM-distance and torso-path terms, and the admissible heuristic is a weighted CoM distance to the goal. — [arXiv HTML](https://arxiv.org/html/2311.08354v3)
- Results: with uniform initialization (500 trials), the planner never reached the goal in the footstep-adjustment case. With obstacle-proximity priors it found collision-free plans after roughly 30 trials, and the cluttered case needed roughly 80 trials. The extracted text reads "3030" and "8080," which looks like a rendering duplication; see Gaps. The torso-path heuristic cut offline trials "significantly" but raised graph-search time. Hardware was a single traversal on Chotu, reported qualitatively. — [arXiv HTML](https://arxiv.org/html/2311.08354v3)

**A3. QuadPiPS: A Perception-informed Footstep Planner for Quadrupeds With Semantic Affordance Prediction.** Max Asselmeier, Ye Zhao, Patricio A. Vela. Under review; arXiv v1 Dec 2024, v3 May 2026. arXiv 2501.00112.
- Listed as under review — [arXiv](https://arxiv.org/abs/2501.00112); [lab page](https://lab-idar.gatech.edu/publications/)
- Core idea: the planner splits foothold search into a discrete subspace and a continuous one. Candidate footholds are convexified planar regions obtained by superpixel (modified SLIC) over-segmentation of an ego-centric "legged egocan" floor image. A* runs over a lazily built graph whose nodes are partial stances (contact mode families). Candidates must satisfy a **gait-based contact sequence**, superquadric reachability, stance-width/length thresholds and a maximum swing distance. OCS2 trajectory optimization (0.5 s, 50 knots per swing) produces references, tracked by MPC (15 ms step, 1.0 s horizon, one warm-started SQP iteration) and a hierarchical-QP WBC. — [arXiv HTML](https://arxiv.org/html/2501.00112)
- Semantic affordance: a DeepLabV3+ network labels terrain as steppable, passable or non-passable. It is trained on 600 synthetic scenes of primitive shapes in about 65 minutes, with depth-noise augmentation for transfer. Steppability updates at 5 Hz, normals at 30 Hz, and old information is propagated forward at 150 Hz. — [arXiv HTML](https://arxiv.org/html/2501.00112)
- Simulation results (ANYmal C, 10 trials per environment, 10 environments, 5 methods), successes out of 10:

  | Environment | QuadPiPS | S-MPC (superpixels + heuristic footholds) | EM-MPC / EM-RL / EM-DTC (elevation-map baselines) |
  |---|---|---|---|
  | Sparse stepping stones | 8 | 3 | 0 / 0 / 0 |
  | Ramped stepping stones | 9 | 9 | 0 / 0 / 0 |
  | Ramped balance beam | 8 | 9 | 1 / 0 / 1 |

  EM-DTC was the only baseline that succeeded on every trial across the first five, easier, environments. — [arXiv HTML](https://arxiv.org/html/2501.00112)
- Perception timing: superpixel over-segmentation runs at about 20 Hz on hardware. In the authors' setup the elevation-mapping pipeline is limited to 25 Hz, with planar-region decomposition as the bottleneck. The mapping step itself takes under 1 ms. — [arXiv HTML](https://arxiv.org/html/2501.00112). An earlier read of the same paper also contained the phrase that plane decomposition runs "roughly four times the speed" of superpixel segmentation, which conflicts with the 20 Hz vs 25 Hz figures; see Gaps.
- Hardware: Unitree Go2 on flat ground, single, double and triple platforms, a ramped platform and ramped stepping stones, reported qualitatively; foot slips were recovered by MPC/WBC. Stated limitations: superpixel jitter, contact sequences assumed to be followed exactly, kinematic but not dynamic feasibility guaranteed, and breadth limited by MPC/WBC robustness. — [arXiv HTML](https://arxiv.org/html/2501.00112)

#### B. Temporal-logic / mixed-integer TAMP and its acceleration

**B1. Optimization-based Task and Motion Planning under Signal Temporal Logic Specifications using Logic Network Flow.** Xuan Lin, Jiming Ren, Samuel Coogan, Ye Zhao. ICRA 2025. arXiv 2409.19168.
- Encodes STL into mixed-binary linear programs built on dynamic network flows. Temporal predicates are polyhedral constraints on network **edges**, where earlier "Logic Tree" encodings put them between nodes. This gives a tighter convex relaxation and explores fewer branch-and-bound nodes, at a higher cost per node. Evaluated on multi-robot planning case studies. — [arXiv 2409.19168](https://arxiv.org/abs/2409.19168)

**B2. Towards Tighter Convex Relaxation of Mixed-integer Programs: Leveraging Logic Network Flow for Task and Motion Planning.** Xuan Lin, Jiming Ren, Yandong Luo, Weijun Xie, Ye Zhao. Under review; arXiv v1 Sep 2025, v2 Aug 2026. arXiv 2509.24235.
- Extends LNF in a Graph-of-Convex-Sets style and adds a network-flow Fourier-Motzkin elimination of the continuous flow variables. It claims "provably tighter convex relaxations and fewer constraints" than Logic Tree, and speedups "up to several orders of magnitude." Dynamics are point-mass and linear-inverted-pendulum models, and quadruped hardware demonstrations show real-time replanning. — [arXiv 2509.24235](https://arxiv.org/abs/2509.24235)
- Vehicle-routing numbers (3 robots / 9 tasks): 5× to 31× faster with Gurobi and 344× with CPLEX. Root-relaxation gap is 6.7% for LNF vs 31.2% for LT (3 robots) and 8.1% vs 38.0% (6 robots). The quadruped demo, in which the robot is dragged off course and replans to an alternative target, has no reported solve time. — [project page](https://logicnetworkflow.github.io)

**B3. Accelerating Signal-Temporal-Logic-Based Task and Motion Planning of Bipedal Navigation using Benders Decomposition.** Jiming Ren, Xuan Lin, Roman Mineyev, Karen M. Feigh, Samuel Coogan, Ye Zhao. Under review; arXiv Aug 2025. arXiv 2508.13407.
- Bipedal STL TAMP is a MIP made harder by nonconvex kinematic reachability and footstep-rotation constraints. Benders decomposition splits it into a **master problem that "prototypes a plan that meets the task specification"** and **subproblems that check kinematic and dynamic feasibility**, iterating with cutting planes. It reports faster planning than alternatives that solve the full nonlinear program, with no numbers in the abstract. — [arXiv 2508.13407](https://arxiv.org/abs/2508.13407)

**B4. LTL-D\*: Incrementally Optimal Replanning for Feasible and Infeasible Tasks in Linear Temporal Logic Specifications.** Jiming Ren, Haris Miller, Karen M. Feigh, Samuel Coogan, Ye Zhao. IROS 2024. — [IEEE Xplore](https://ieeexplore.ieee.org/document/10802218/); venue from the [lab page](https://lab-idar.gatech.edu/publications/). Only the title and venue were verified.

**B5. A Unified Approach to Multi-task Legged Navigation: Temporal Logic Meets Reinforcement Learning.** Jesse Jiang, Samuel Coogan, Ye Zhao. arXiv Jul 2024 (2407.06931).
- Builds a Multi-task Product Interval MDP over an abstraction of 3D hopping-robot dynamics. One policy-synthesis algorithm combines LTL goal-reaching with task-agnostic RL exploration, which learns the motion perturbations, and the authors prove the trade-off between prioritizing LTL actions and RL actions. Validated only in 2D simulation case studies. — [arXiv 2407.06931](https://arxiv.org/abs/2407.06931)

**B6. Robust-Locomotion-by-Logic: Perturbation-Resilient Bipedal Locomotion via Signal Temporal Logic Guided Model Predictive Control.** Zhaoyuan Gu, Yuntian Zhao, Yipu Chen, Rongming Guo, Jennifer K. Leestma, Gregory S. Sawicki, Ye Zhao. IEEE T-RO 2025 (DOI 10.1109/TRO.2025.3582820). The conference precursor is **Walking-by-Logic** (ICRA 2024).
- Venue — [IEEE Xplore T-RO](https://ieeexplore.ieee.org/document/11049016); [ICRA 2024 precursor](https://ieeexplore.ieee.org/document/10610811/)
- Defines an analytical stability metric for bipedal locomotion, written as STL specifications. The MPC generates recovery trajectories that maximize the STL **robustness degree**. Data-driven self-leg-collision constraints are about 1000× faster than IK-based checks and enable crossed-leg recovery steps. Tested on Cassie in simulation and hardware, including stepping stones, inclines and ship-motion-like disturbances, against a state-of-the-art controller, MPC without STL, and an LTL planner. — [arXiv 2403.15993](https://arxiv.org/abs/2403.15993); [project page](https://bipedal-stl-mpc.github.io/)

#### C. Safety and uncertainty-aware legged navigation (probabilistic, reachability, learned traversability)

**C1. Bipedal Safe Navigation over Uncertain Rough Terrain: Unifying Terrain Mapping and Locomotion Stability.** Kasidit Muenprasitivej, Jesse Jiang, Abdulaziz Shamsah, Samuel Coogan, Ye Zhao. IROS 2024. arXiv 2403.16356.
- GP regression of terrain elevation, comparing three kernels. A learned model of motion deviation captures the mismatch between the reduced-order Prismatic Inverted Pendulum model and full-order dynamics on terrain. A hierarchical sampling-based planner pairs a global planner of stability-respecting waypoints with a local planner of dynamically feasible footsteps, and the trajectory metric penalizes deviation and rewards information gain. Digit in MuJoCo, with no numbers in the abstract. — [arXiv 2403.16356](https://arxiv.org/abs/2403.16356); venue from the [lab page](https://lab-idar.gatech.edu/publications/)

**C2. Probabilistically-Safe Bipedal Navigation over Uncertain Terrain via Conformal Prediction and Contraction Analysis.** Kasidit Muenprasitivej, Ye Zhao, Glen Chou. ICRA 2026 per the lab page; arXiv Oct 2025 (2510.07725).
- GP elevation maps plus **conformal prediction**, which gives calibrated confidence intervals meant to contain the true terrain elevation. A high-level MPC navigation framework drives toward the goal at a specified confidence level of safety. **Contraction-based reachable tubes** account for terrain uncertainty, with state convergence and tube invariance. A contraction-based flywheel torque controller on a reduced-order LIPM stabilizes angular momentum about the CoM. The authors claim forward invariance at the given confidence level, which yields probabilistic safety and goal reachability. Evaluated on Digit in MuJoCo; the abstract gives no numbers. — [arXiv 2510.07725](https://arxiv.org/abs/2510.07725); venue from the [lab page](https://lab-idar.gatech.edu/publications/)

**C3. STATE-NAV: Stability-Aware Traversability Estimation for Bipedal Navigation on Rough Terrain.** Ziwon Yoon, Lawrence Y. Zhu, Jingxi Lu, Lu Gan, Ye Zhao. IEEE RA-L 2025. arXiv 2506.01046.
- Defines traversability as "the fastest command velocity that keeps instability below a user-defined limit." A transformer, TravFormer, predicts bipedal instability together with its uncertainty. TravRRT\* uses that prediction for risk-aware planning, and MPC executes. Validated in MuJoCo and the real world. — [arXiv 2506.01046](https://arxiv.org/abs/2506.01046)

**C4. Zonotope-based safety for bipedal social navigation.** "Real-time Model Predictive Control with Zonotope-Based Neural Networks for Bipedal Social Navigation" (Shamsah, Agarwal, Kousik, Zhao; IROS 2024) — [IEEE Xplore](https://ieeexplore.ieee.org/document/10801435/). "Socially Acceptable Bipedal Robot Navigation via Social Zonotope Network Model Predictive Control" (Shamsah, Agarwal, Katta, Raju, Kousik, Zhao; IEEE T-ASE 2024) — [IEEE Xplore](https://ieeexplore.ieee.org/document/10810741/); [project page](https://szn-mpc.github.io/). Only titles and venues were verified, from the [lab page](https://lab-idar.gatech.edu/publications/).

**C5. Terrain-Aware Model Predictive Control of Heterogeneous Bipedal and Aerial Robot Coordination for Search and Rescue Tasks.** Shamsah, Jiang, Yoon, Coogan, Zhao. ICRA 2025 — [arXiv 2409.15174](http://arxiv.org/abs/2409.15174); [lab page](https://lab-idar.gatech.edu/publications/). Title and venue only; marginal relevance.

#### D. Learning-augmented MPC and model-based-to-learned transfer

**D1. RL-augmented Adaptive Model Predictive Control for Bipedal Locomotion over Challenging Terrain.** Junnosuke Kamohara, Feiyang Wu, Chinmayee Wamorkar, Seth Hutchinson, Ye Zhao. ICRA 2026 per the lab page; arXiv Sep 2025 (2509.18466). Project page: rl-augmented-mpc.github.io.
- The venue appears only on the lab page — [lab page](https://lab-idar.gatech.edu/publications/); [arXiv 2509.18466](https://arxiv.org/abs/2509.18466)
- An RL policy (SAC, at 100 Hz) parametrizes a single-rigid-body convex MPC (H = 10, dt 0.025 s) on the HECTOR biped in Isaac Lab. It outputs 15 values: 12 dynamics residuals (linear and angular accelerations, diagonal force and moment actuation), 2 swing-trajectory parameters (apex-height residual and a Bézier control-point shift), and 1 **gait-frequency / MPC sampling-time coefficient**. **Foot placement is not output; a Raibert heuristic sets it.** — [arXiv HTML](https://arxiv.org/html/2509.18466)
- Success, vanilla MPC vs RL-augmented: 10 cm pyramid stairs 1% vs 86%; ±7 cm stepping stones 68% vs 89%; μ = 0.05 slippery ground 5% vs 74%. The vanilla RL baseline reached 15% on 10 cm stairs. Ablation on 10 cm stairs: swing only 74%, swing + gait 67%, dyn + swing 51%, dyn + gait 2%, gait only 1%. — [arXiv HTML](https://arxiv.org/html/2509.18466)
- A preliminary batched GPU MPC (CusADi sparse QP, batch 4096) is reported more than 20× faster than a CPU baseline and reaches 85% on 10 cm stairs. There are **no hardware, sim-to-sim or sim-to-real experiments**; the authors leave these to future work. — [arXiv HTML](https://arxiv.org/html/2509.18466)

**D2. PPF: Pre-training and Preservative Fine-tuning of Humanoid Locomotion via Model-Assumption-based Regularization.** Hyunyoung Jung\*, Zhaoyuan Gu\*, Ye Zhao, Hae-Won Park, Sehoon Ha. IEEE RA-L 10(11), Nov 2025. arXiv 2504.09833.
- A policy is pre-trained by imitating a model-based controller, then fine-tuned with RL. During fine-tuning, **model-assumption-based regularization (MAR)** keeps it close to the controller's actions only in states where the controller's model assumptions hold, which prevents catastrophic forgetting. Digit reaches 1.5 m/s and walks on slippery, sloped, uneven and sandy terrain in simulation and on hardware. — [arXiv 2504.09833](https://arxiv.org/abs/2504.09833)

**D3. Opt2Skill: Imitating Dynamically-feasible Whole-Body Trajectories for Versatile Humanoid Loco-Manipulation.** Fukang Liu, Zhaoyuan Gu, Yilin Cai, Ziyi Zhou, Hyunyoung Jung, Jaehwi Jang, Shijie Zhao, Sehoon Ha, Yue Chen, Danfei Xu, Ye Zhao. IEEE RA-L 2025. arXiv 2409.20514.
- DDP generates dynamically feasible, contact-consistent references for Digit, and RL policies are trained to track them. Including torque in the references improves force tracking. It outperforms baselines that use human demonstrations or IK references on tracking and task success, and transfers to hardware. — [arXiv 2409.20514](https://arxiv.org/abs/2409.20514); venue from the [lab page](https://lab-idar.gatech.edu/publications/)

**D4. Learn to Teach: Sample-Efficient Privileged Learning for Humanoid Locomotion over Diverse Terrains.** Feiyang Wu, Xavier Nal, Jaehwi Jang, Wei Zhu, Zhaoyuan Gu, Anqi Wu, Ye Zhao. IEEE RA-L 2025. arXiv 2402.06783.
- One-stage teacher-student training: the student reuses the teacher's simulator samples, which cuts sample complexity. Transfer to Digit is zero-shot sim-to-real, and the policy works on more than 12 terrains without depth estimation. Code is in a GTLIDAR Isaac Lab fork. — [arXiv 2402.06783](https://arxiv.org/abs/2402.06783); [lab page](https://lab-idar.gatech.edu/publications/)

**D5. Further lab entries in this category, verified by title and venue only on the [lab page](https://lab-idar.gatech.edu/publications/):**
- "SEEC: Stable End-Effector Control with Model-Enhanced Residual Learning for Humanoid Loco-Manipulation" (ICRA 2026, [arXiv 2509.21231](https://arxiv.org/pdf/2509.21231))
- "ACLM: ADMM-Based Distributed Model Predictive Control for Collaborative Loco-Manipulation" (IROS 2026, [arXiv 2603.07095](https://arxiv.org/pdf/2603.07095))
- "Infer and Adapt: Bipedal Locomotion Reward Learning from Demonstrations via Inverse Reinforcement Learning" (ICRA 2024, [IEEE](https://ieeexplore.ieee.org/document/10611685/))

#### E. Diffusion / generative planners

**E1. Structured-Diffuser: Diffusion with Task-Conditioned Structured Priors for Motion Planning.** Amelie Minji Kim, Junling Mei, Anqi Wu, Ye Zhao. arXiv 2509.25685, v1 Sep 2025 to v3 Sep 2026, under review.
- Replaces isotropic Gaussian corruption with task-conditioned, non-isotropic Gaussian priors whose means and covariances come from Gaussian Process Motion Planning. An upper level predicts sparse key states and timings that set the prior, and a lower level denoises the full trajectory under it. Evaluated on three motion-planning tasks and deployed on a physical G1 humanoid; the abstract gives no numbers. — [arXiv 2509.25685](https://arxiv.org/abs/2509.25685)
- A search snippet reported that another version is titled "Hierarchical Diffusion Motion Planning with Task-Conditioned Uncertainty-Aware Priors," with a different author list. The current abstract page shows the Structured-Diffuser title. — [arXiv 2509.25685](https://arxiv.org/abs/2509.25685); see Gaps.

**E2. REFINE-DP: Diffusion Policy Fine-tuning for Humanoid Loco-manipulation via Reinforcement Learning.** Zhaoyuan Gu\*, Yipu Chen\*, Zimeng Chai\*, et al., Yongxin Chen, Ye Zhao. IEEE RA-L, in press 2026. — [IEEE Xplore](https://ieeexplore.ieee.org/document/11654533); [lab page](https://lab-idar.gatech.edu/publications/). Only the title and venue were verified.

#### F. Surveys (useful for positioning text)
- "Humanoid Locomotion and Manipulation: Current Progress and Challenges in Control, Planning, and Learning." Gu, Li, Shen, Yu, Xie, McCrory, Cheng, Shamsah, Griffin, Liu, Kheddar, Peng, Zhu, Shi, Nguyen, Cheng, Gao, Zhao. IEEE/ASME T-Mech 2026 — [arXiv 2501.02116](https://arxiv.org/abs/2501.02116); [lab page](https://lab-idar.gatech.edu/publications/)
- "A Survey of Optimization-Based Task and Motion Planning: From Classical to Learning Approaches." Zhigen Zhao, Shuo Cheng, Yan Ding, Ziyi Zhou, Shiqi Zhang, Danfei Xu, Ye Zhao. IEEE/ASME T-Mech 30(4), 2024 — [IEEE Xplore](https://ieeexplore.ieee.org/document/10705419/)

#### G. [foundational, pre-2024] Earlier lab work these papers build on
- "Integrated Task and Motion Planning for Safe Legged Navigation in Partially Observable Environments." Shamsah, Gu, Warnke, Hutchinson, Zhao. IEEE T-RO 39(6), 2023 — [IEEE Xplore](https://ieeexplore.ieee.org/document/10215076/); code at GTLIDAR/safe-nav-locomotion per the [lab page](https://lab-idar.gatech.edu/publications/)
- "GPF-BG: A Hierarchical Vision-Based Planning Framework for Safe Quadrupedal Navigation." Feng, Zhou, Smith, Asselmeier, Zhao, Vela. ICRA 2023 — [IEEE Xplore](https://ieeexplore.ieee.org/document/10160804/)
- "Abstraction-Based Planning for Uncertainty-Aware Legged Navigation." Jiang, Coogan, Zhao. IEEE OJ-CSYS 2023 — [IEEE Xplore](https://ieeexplore.ieee.org/document/10184473/)

### Inferences
- The lab's quadruped contact-planning line (A1, A2 and A3; first authors Ziyi Zhou and Max Asselmeier) runs on the same kind of terrain GaitNet targets: rebar grids, sparse stepping stones and balance beams. It also uses hardware of the same class (Unitree Go2; GaitNet deploys on a Go1) and the same OCS2-based optimization and tracking that GaitNet uses at deployment. Of all the lab's work, these are the closest comparators and the strongest baselines for a GaitNet paper.
- The lab has no 2024-2026 paper that learns a contact/footstep *selection policy* by RL. Its RL work learns joint-level policies or MPC parameters, with foot placement left to Raibert in D1. Its learning-for-planning work learns priors (experience RBFs, steppability segmentation, traversability, structured diffusion priors) that feed search or optimization. GaitNet, an RL policy over discrete footholds, therefore sits in a gap of this lab's portfolio.
- The lab's safety guarantees (C1-C3, B6) are all built for bipeds on LIP-type models, typically for navigation-level footstep plans. None certifies a single foothold choice for a quadruped at control rate.

### Gaps
- No 2024-2026 LIDAR Lab paper was found that applies control barrier functions to footstep or foothold selection. A possible "linear discrete CBF" bipedal navigation paper by Shamsah et al. could not be verified and is omitted.
- No lab paper was found on learned value functions or heuristics over contact sequences, or on diffusion models that generate contact plans.
- Xuan Lin, first author of B1 and B2, has related sole-authored work on warm-started hybrid MPC with Generalized Benders Decomposition. It stores cuts across MPC iterations and is reported 2-3× faster than Gurobi, often above 1000 Hz, with a humanoid multi-contact balance example ([arXiv 2406.00780](https://arxiv.org/abs/2406.00780)). It does **not** list Ye Zhao as co-author and is not on the lab page, so its affiliation with the lab is unverified. Treat it as adjacent, not as a lab paper.
- Rebar paper (A2): the extracted HTML prints trial counts as "3030" and "8080." These are most likely 30 and 80 duplicated by rendering, but that was not confirmed against the PDF.
- QuadPiPS (A3) timing conflict: one read says plane decomposition for elevation mapping runs "roughly four times the speed" of superpixel segmentation. Another says the elevation pipeline is capped at 25 Hz, bottlenecked by plane decomposition, against about 20 Hz for superpixels. The exact claim could not be reconciled from the extracted text, and graph-search and TO timings were not reported.
- No quantitative results were retrieved for B3 (Benders speedups), B6 (STL-MPC success metrics), C1-C3, D2-D4 or E1; their abstracts give none.
- Several venues (A1 IROS 2025, C2 and D1 ICRA 2026, D5 entries, E2 RA-L) rest only on the lab's publication page.
- E1's title and author list differ between versions according to a search snippet; only the current abstract page was checked.

## Q2. Which of the lab's ideas could address GaitNet's weaknesses?

### Takeaway
Each GaitNet weakness has a concrete counterpart in the lab's work.
- **Candidate generator (ContactNet):** experience-weighted RBF foothold costs (A2) or convex-region decomposition with one-of-R binaries (A1) and semantic steppability (A3).
- **No feasibility certificate:** a feasible convex MICP certifies the skill (A1), and Benders master/subproblem splitting (B3) is a template for learned proposals plus feasibility cuts.
- **Reluctance to step:** STL robustness of a stability metric (B6) is a dense, interpretable stepping trigger.
- **Sim-to-sim gap:** RL that parametrizes rather than replaces the MPC (D1), imitation of a model-based controller with regularization where its assumptions hold (D2), and tracking of dynamically feasible TO references (D3).
- **Slow perception:** in A3 the lab's pipelines run at 20 Hz on hardware (superpixels) and at up to 25 Hz with plane decomposition (elevation-map baseline, in the authors' setup), so GaitNet's 3-4 Hz is probably not inherent to plane decomposition.
- **Repositioning:** the lab's gait-free MICP takes about 27 s per solve, so online the lab still fixes gaits. GaitNet can be framed as an amortized, control-rate approximation of gait-free contact planning.

### Cited Findings

#### Weakness: dependence on ContactNet to define the action set
- A2 learns a foothold-quality field from attempted transitions: RBFs centred at tried footholds, weighted by normalized cost, with failures mapped to bounded penalties through tanh. That field becomes an A* edge cost, so footholds near successful experience are preferred and untried neighbours inherit information. — [arXiv 2311.08354 HTML](https://arxiv.org/html/2311.08354v3)
- A1-ext encodes stepping regions as convex polygons with one binary per region per footstep, and the foothold is continuous inside the chosen region. No candidate sampler is needed beyond terrain segmentation. — [arXiv 2509.23185 HTML](https://arxiv.org/html/2509.23185)
- A3 generates candidates directly from perception: superpixels are convexified into planar regions and filtered by a learned steppable/passable/non-passable segmentation and by reachability and stance constraints. Against a heuristic-foothold baseline on the same perception (S-MPC), it scores 8/10 vs 3/10 on sparse stepping stones. — [arXiv 2501.00112 HTML](https://arxiv.org/html/2501.00112)
- C3 learns a terrain score with uncertainty (TravFormer predicts instability and its uncertainty) for risk-aware planning. C2 calibrates a learned terrain model with conformal prediction so that a planner can meet a user-specified probability of safety. — [arXiv 2506.01046](https://arxiv.org/abs/2506.01046); [arXiv 2510.07725](https://arxiv.org/abs/2510.07725)

#### Weakness: no feasibility or safety certificate for the chosen foothold; myopic choice loses to a fixed trot
- In A1-ext, a feasible solution of the convex MICP certifies the skill's physical feasibility, and the automaton advances only when the MICP finds a solution. Removing the feasibility check drops pure-MIP success from 60% to 20% (H = 2 s) and from 90% to 40% (H = 3 s). — [arXiv 2509.23185 HTML](https://arxiv.org/html/2509.23185)
- In A1-ext, horizon length matters: pure MIP succeeds 60% at H = 2 s and 90% at H = 3 s. A nearest-feasible-foothold heuristic fails on 0.48 m rebar spacing at 0.2 m/s, through foot trapping, and needed retuning to 0.1 m/s. — [arXiv 2509.23185 HTML](https://arxiv.org/html/2509.23185)
- B3 splits planning into a master problem that "prototypes a plan that meets the task specification" and subproblems that check kinematics and dynamics feasibility, which return cuts. — [arXiv 2508.13407](https://arxiv.org/abs/2508.13407)
- B1 and B2 make the MIPs behind such plans much cheaper: tighter relaxations (root gap 6.7% vs 31.2%) and speedups of 5×-344× on routing benchmarks, with quadruped real-time replanning demos. — [arXiv 2509.24235](https://arxiv.org/abs/2509.24235); [project page](https://logicnetworkflow.github.io)
- C2 provides probabilistic safety for footstep plans: conformal-prediction-calibrated terrain plus contraction-based reachable tubes, meeting a user-specified probability of safety, for a LIP biped. — [arXiv 2510.07725](https://arxiv.org/abs/2510.07725)

#### Weakness: reluctance to step (85-95% no-ops until the body drifts ahead)
- B6 writes an analytical bipedal stability metric as STL specifications. The MPC maximizes the STL robustness degree to generate recovery steps, including crossed-leg steps, under perturbations. — [arXiv 2403.15993](https://arxiv.org/abs/2403.15993)
- In D1, the RL policy modulates gait frequency continuously, through an MPC sampling-time coefficient, rather than triggering steps. In the ablation, gait timing alone barely helps (1% on 10 cm stairs). Swing-trajectory shaping is the critical module (74% alone), and all three modules together reach 86%. — [arXiv 2509.18466 HTML](https://arxiv.org/html/2509.18466)
- D2's MAR keeps a fine-tuned policy close to a model-based controller's actions only where that controller's assumptions hold. — [arXiv 2504.09833](https://arxiv.org/abs/2504.09833)

#### Weakness: sim-to-sim gap (convex MPC in training vs OCS2 NMPC + WBC at deployment)
- D1 keeps the model-based controller in the loop and lets RL output dynamics residuals for the SRBD MPC (12 values), compensating unmodeled leg inertia and impact. It also reports a batched GPU MPC (CusADi, batch 4096, more than 20× faster than CPU) for training. It has no transfer experiments. — [arXiv 2509.18466 HTML](https://arxiv.org/html/2509.18466)
- D3 trains RL to track dynamically feasible, contact-consistent DDP references, including torques, and reports better tracking and task success than IK or human-demo references, with hardware transfer. — [arXiv 2409.20514](https://arxiv.org/abs/2409.20514)
- D4 trains teacher and student in one stage with shared samples and transfers zero-shot to Digit. — [arXiv 2402.06783](https://arxiv.org/abs/2402.06783)

#### Weakness: slow perception (depth to elevation map to plane decomposition at 3-4 Hz)
- In A3, superpixel over-segmentation of an ego-centric "legged egocan" runs at about 20 Hz on Go2 hardware, and the authors' elevation-mapping baseline pipeline is capped at 25 Hz by plane decomposition. Steppability inference runs in its own 5 Hz thread, and old information is propagated forward at 150 Hz. — [arXiv 2501.00112 HTML](https://arxiv.org/html/2501.00112)
- In A3's benchmark, the elevation-map baselines (EM-MPC, EM-RL, EM-DTC) fail completely on sparse and ramped stepping stones (0/10). Causes cited include elevation-map merging of nearby stones and the lack of explicit foothold planning. — [arXiv 2501.00112 HTML](https://arxiv.org/html/2501.00112)
- In A1-ext's hardware experiments, perception was replaced by motion-capture-aided manual polygon labeling. — [arXiv 2509.23185 HTML](https://arxiv.org/html/2509.23185)

### Inferences
- **Repositioning: "amortized gait-free contact planning at control rate."** The lab's own state of the art cannot run gait-free contact planning at control rate. Gait-free MICP averages 27 s; it runs offline for synthesis and online only as a repair step, which took 12.18 s in the unforeseen-gap case while execution waits. Control-rate replanning uses gait-fixed MICP (0.4-9 s) with hand-chosen trot, leap and static-walk gaits. GaitNet chooses order-free swings at 25 Hz. Its novelty can be argued as amortizing what A1's offline gait-free MICP does. That framing needs evidence, though: GaitNet still loses to a fixed trot (≤18% vs 45%), so the claim has to come with a fix for myopia.
- **Replace ContactNet with an optimization- or experience-derived candidate model.** One option is to train ContactNet's replacement from MICP feasibility labels (A1), learning a per-foothold feasibility or value map. Another is an A2-style experience field updated from GaitNet's own rollouts. Either turns the candidate set from a separately trained heuristic into a quantity tied to downstream feasibility. Alternatively, act on convex regions, choosing a region and then a continuous point within it as in A1 and A3, which removes the fixed 16-per-leg discretization.
- **Add a certificate or lookahead layer in Benders form (B3).** GaitNet's argmax acts as the "master" proposer. A short-horizon convex check (an A1-style one-region-per-step MICP over 1-2 future steps, or the MPC's own QP feasibility) acts as the "subproblem," and infeasible proposals are masked or turned into training penalties. A1-ext's horizon ablation (60% vs 90%) and heuristic-baseline failure suggest that some lookahead is what separates success from failure on sparse footholds.
- **Fix no-op reluctance with an STL-robustness stepping signal (B6).** Write "always (support/capture margin ≥ ε)" or "eventually swing within T when the margin shrinks" as STL, and use its robustness degree as a dense reward or as a feature in the scoring network. Stepping would then be driven before the body drifts ahead, rather than after. D2's MAR is a second option: pull the policy toward a trot prior only in states where a trot is valid, and leave it free elsewhere.
- **Close the sim-to-sim gap the D1/D3 way.** Keep the training controller in the loop, but let the policy, or a residual head, adapt MPC parameters such as swing apex and timing, which D1 finds most important on rough terrain. Alternatively, generate training references with the deployment controller's model (OCS2-like) and imitate them (D3). D1's GPU-batched MPC is also a direct peer to GaitNet's `gpu_mpc` preset.
- **Perception.** A3 reports 20 Hz superpixel perception on Go2 hardware, and a 25 Hz cap for an elevation-map pipeline with plane decomposition in the authors' setup; whether that cap was measured on hardware or in simulation is not stated. GaitNet's 3-4 Hz therefore looks like a pipeline engineering problem, not a fundamental cost. A3's egocentric image representation is one way to raise the rate. C2's conformal intervals are a way to plan safely on stale maps.

### Gaps
- None of the lab papers trains a contact-selection policy, so there is no direct evidence that their learned priors (A2, A3, C3) would improve an RL foothold scorer.
- D1's mechanisms are demonstrated only in simulation on a biped (HECTOR), with Raibert foot placement and no transfer. Whether residual-dynamics adaptation shrinks a convex-MPC-to-OCS2 gap is untested.
- No quantitative comparison of STL-robustness-driven stepping against learned stepping was found.

## Q3. How do the lab's discrete contact-planning formulations compare to GaitNet's per-tick greedy argmax over candidates?

### Takeaway
All three of the lab's quadruped contact planners optimize a **sequence over a horizon**, with an **explicit feasibility check per candidate transition**:
- A1: MICP with one-region binaries per footstep and SRB dynamics.
- A2: A\* over a mode-transition graph with superquadric reachability and experience costs.
- A3: A\* over superpixel contact families with reachability, stance and swing-distance constraints.

All three also **fix the contact order online**: a gait-fixed MICP in A1 (gait-free MICP, about 27 s per solve on average, runs only offline or as an online repair step), a user-defined alternating sequence in A2, and a "gait-based contact sequence" in A3. None performs per-tick, order-free swing selection. GaitNet makes exactly that order-free choice in milliseconds at 25 Hz, which is its novelty, but without lookahead or a certificate. The lab's results suggest that missing lookahead and certificate is why heuristic or one-step foothold choosers lose on sparse terrain.

### Cited Findings
- **A1-ext (MICP):**
  - Decision variables are binaries H_{r,k,j} (one region per footstep k per foot j) over a horizon of N steps, with a single-rigid-body model, friction cones and kinematic boxes.
  - The online gait-fixed solve takes about 0.4-2.7 s in simulation and 0.53-9.13 s mean on hardware. The gait-free solve takes 27.23 s on average.
  - Symbolic skills are gaits; the planner picks feasible gaits by first success, with no optimality criterion.
  - Source: [arXiv 2509.23185 HTML](https://arxiv.org/html/2509.23185)
- **A2 (graph search):**
  - A\* over modes, where a mode is three stance legs on three rebars plus continuous foothold coparameters.
  - Edges require alternating swing legs in a user-defined sequence and superquadric reachability. Edge costs combine learned experience, CoM distance and torso-path terms.
  - Each edge is realized by a 0.5 s OCS2 trajectory optimization.
  - Source: [arXiv 2311.08354 HTML](https://arxiv.org/html/2311.08354v3)
- **A3 (graph search in perception space):**
  - A\* over partial stances (contact mode families) built lazily.
  - Candidates are filtered by gait-based contact sequence, reachability, stance-stability thresholds and maximum swing distance. Edge cost is torso-position distance, and the heuristic is the distance to a waypoint from a coarse torso A\*.
  - The authors state that the graph ensures kinematic but not dynamic feasibility, and that planned sequences are assumed to be followed exactly.
  - Source: [arXiv 2501.00112 HTML](https://arxiv.org/html/2501.00112)
- **Evidence that one-step or heuristic foothold choice underperforms on sparse terrain:**
  - Sparse stepping stones (A3): QuadPiPS 8/10 vs the heuristic-foothold S-MPC 3/10 — [arXiv 2501.00112 HTML](https://arxiv.org/html/2501.00112)
  - A nearest-feasible-foothold heuristic fails at 0.48 m rebar spacing (A1-ext) — [arXiv 2509.23185 HTML](https://arxiv.org/html/2509.23185)
  - Pure MIP improves from 60% to 90% as the horizon grows from 2 s to 3 s (A1-ext) — [arXiv 2509.23185 HTML](https://arxiv.org/html/2509.23185)
- **Cost of the lab's sequence planners:** QuadPiPS took longer to find paths on sparse stepping stones, and those paths were often more circuitous — [arXiv 2501.00112 HTML](https://arxiv.org/html/2501.00112). A1-ext's hardware gaits were slow and conservative (3 s trot, 2.25 s static walk), and MICP solves reached 19 s in the worst case — [arXiv 2509.23185 HTML](https://arxiv.org/html/2509.23185).
- **Decomposition patterns that map onto GaitNet's structure:**
  - Reactive synthesis (A1) separates a symbolic decision ("which skill next") from a continuous feasibility certificate (MICP) and advances only on success — [arXiv 2503.03071](https://arxiv.org/abs/2503.03071)
  - Benders (B3) separates a task-level master proposal from feasibility subproblems — [arXiv 2508.13407](https://arxiv.org/abs/2508.13407)
  - LNF (B1, B2) tightens the MIP relaxation so these searches scale — [arXiv 2409.19168](https://arxiv.org/abs/2409.19168); [arXiv 2509.24235](https://arxiv.org/abs/2509.24235)

### Inferences
- **Side-by-side comparison:**

  | | Lab planners (A1-A3) | GaitNet |
  |---|---|---|
  | Horizon | Multi-step | One step per tick |
  | Contact order | Fixed or prescribed | Emergent |
  | Feasibility | Explicit kinematic and/or dynamic checks per candidate | Validity of the C-space sample only |
  | Speed | Seconds per plan, often replanned slowly | Milliseconds at 25 Hz |
  | Action space | Continuous footholds within regions, or discretized coparameters | Fixed 16 candidates per leg plus no-op |

- GaitNet's per-tick argmax is, in effect, a learned A\* that keeps only the first edge and has no heuristic-to-go. The policy's value function is the only lookahead. On sparse terrain the lab's data shows that lookahead and feasibility checking drive success, which matches GaitNet's loss to a fixed trot (≤18% vs 45%).
- **Most defensible novelty claim against this lab:** "contact order is not prescribed online," together with "reactivity at control rate." Neither A1, A2 nor A3 offers both. The weakest flank is the absence of any guarantee, where A1 and C2 set the bar.
- **Hybrid positioning that would be novel relative to the lab:**
  - Use GaitNet's scores as the A\*/MICP heuristic or as a branching prior: a learned heuristic for contact search, which none of A1-A3 has; A2's experience field is the closest.
  - Or invert the roles: run GaitNet at 25 Hz and run a 1-2-step MICP or Benders feasibility subproblem asynchronously, as A1's delay-aware coordination already does for slow MICP solves, to veto or re-rank.
- **Benchmark opportunity:** A1-ext's scenario families (Unstructured-4/8, Rebar-7/14) and A3's ten environments (sparse and ramped stepping stones, balance beams, pegboard) are published, robot-agnostic layouts. Evaluating GaitNet on comparable layouts would allow a direct comparison against MICP and A\* planners.

### Gaps
- No lab paper reports a direct comparison against an RL footstep policy. A1-ext lists RL and TOWR baselines as future work.
- QuadPiPS's graph-search and TO times were not reported in the extracted text, so a wall-clock comparison against GaitNet's per-tick inference cannot be made precisely.
- A1-ext's numbers come from only 10 waypoints per method, and A3's from 10 trials per environment, so differences such as 8/10 vs 3/10 carry wide uncertainty.

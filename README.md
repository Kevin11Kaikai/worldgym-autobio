# WorldGym–AutoBio

A compact research project for **understanding, evaluating and debugging world-model-based robot policy learning** on the AutoBio MuJoCo tube-pickup task. This is an independent project using WorldGym, not an official WorldGym release.

**System:** demonstrations → supervised policy training → policy evaluation in MuJoCo / native WorldGym → VLM-scored imagined rollouts → GRPO policy updates → separate physical-simulator evaluation.

Native world-model RL has been executed. A reliable policy improvement attributable to WorldGym has **not** been established. All results concern one simulated task; there are no real-robot, wet-lab deployment or surgical results.

## Three stages, three guided notebooks

Start with notebook 01, even if your main interest is RL. Each notebook opens with a **new panoramic teaching image**, followed by **five sections matching its numbered panels**. The lessons explain the input-to-output process before results, then discuss failures, repairs and what the next stage needs. They assume basic familiarity with training a model, without requiring robotics or RL background.

| Notebook | Follow the process | Reading time |
|---|---|---:|
| [01 · ACT imitation learning](01_act_imitation_learning.ipynb) | Demonstrations → ACT action chunks → closed-loop MuJoCo testing | 15–20 min |
| [02 · WorldGym policy evaluation](02_worldgym_policy_evaluation.ipynb) | Fixed ACT → predicted futures → scores and rankings → MuJoCo comparison | 15–20 min |
| [03 · World-Gymnast-inspired RL](03_worldgymnast_reinforcement_learning.ipynb) | Native imagined trajectories → VLM rewards → actual policy updates → physical evaluation | ~20 min |

Read the saved notebook outputs directly on GitHub, or run the short CPU cells. Code is optional for understanding. Stage 3 describes **completed RL experiments**; only its fifth section proposes a **DRAFT / NOT EXECUTED** next experiment.

## Results, attached to the correct stage

| Stage | Recorded evidence | What it establishes |
|---|---|---|
| 1 · ACT training | Repaired 10-demo ACT: 31/50; 80-demo ACT: 50/50 on shared test scenes | Usable simulated pickup policies under a matched training/execution setup; not real-robot reliability |
| 2 · Policy evaluation | Hybrid evaluator: 18/20 non-tied policy pairs correctly ordered; 125/140 episode judgments correct | Useful task-specific hybrid evaluation; these are evaluator metrics, not a robot's pickup rate or native WorldGym fidelity |
| 3 · Native model RL | SFT 19; native GRPO 13; matched MuJoCo GRPO 13; native continuation 23; Dyna 18, each /100 | Actual generated rollouts and policy updates ran; reliable RL gains were not established |

These are different evaluation batches. The notebooks show the denominators, comparators and limitations before interpreting any improvement.

<details><summary>Later selection and repair experiments</summary>

In a separate 100-scene ACT selection experiment, original ACT scored 78, the WorldGym-selected candidate 94 and the strong no-WorldGym selection 93. The fixed SFT candidate itself also scored 94, so a unique WorldGym gain remains unestablished.

Another 100-scene SFT repair experiment improved SmolVLA from 19 to 53, with 44 improved scenes and ten new failures; strong ACT40 scored 91. This was additional SFT, not RL. World-model repair reduced stationary prediction MAE by about 33% versus matched replay on four new initial states, but worsened movement and reversed-gripper predictions. Reward repair reduced false positives while increasing false negatives. The later diagnostics are explained in Stage 3. Scene-paired intervals do not capture variation across training seeds.

</details>

## Reproduce at the right level

**CPU: verify the published evidence and execute all three notebooks.** No GPU, original project directory or model downloads are needed.

```bash
python -m pip install -r requirements.txt
python tools/verify_notebooks.py
```

**GPU: run the actual SFT, evaluation and native RL code in `Scripts/`.** These are executable modules with their dependency closure, not `.py.txt` excerpts or wrappers that call an unavailable original project.

| Entry | What it actually runs |
|---|---|
| `Scripts/prepare.py` | Pinned upstreams, public model files, dataset decoding, asset checks / import |
| `Scripts/sft_act.py` | ACT fine-tuning from the frozen 10-demo ACT baseline |
| `Scripts/sft.py` | Initial SmolVLA SFT, or matched precision / learning-rate continuation |
| `Scripts/evaluate.py` | MuJoCo official success; native WorldGym rollouts; optional VLM scoring; SFT development selection |
| `Scripts/train_native_rl.py` | Actual Gaussian GRPO in native WorldGym, or a matched MuJoCo training arm |
| `Scripts/audit.py` | CPU numerical checks of saved likelihood / policy-gradient / KL / Adam probes and official labels |

Start with **[Scripts/README.md](Scripts/README.md)** for setup, exact commands, inputs and outputs. CUDA / Linux (or WSL2) is required for GPU experiments; the original runs used one RTX 4080 SUPER with 16 GB.

**Asset availability matters:** public base-model and dataset sources are pinned. Locally trained ACT, WorldGym, proprioception and reward adapters are external assets. Their filenames, sizes and SHA256 values are included, but their downloadable bundle has **not yet been published**. A fresh clone can run the CPU notebooks immediately after dependency installation; native GPU reproduction requires that bundle or matching locally available assets. The scripts fail clearly on missing or mismatched assets. No nonexistent download URL is implied.

The release ports preserve the core calculations while replacing historical host paths, scheduler queues and expired budgets with explicit configuration and fresh output directories. Short smoke runs test the portable entries; they do not repeat the historical experiment matrix or guarantee identical full-run success counts. See the recorded validation scope in [Scripts/VALIDATION.json](Scripts/VALIDATION.json).

## Repository map

```text
README.md + 3 notebooks            # concise learning and research narrative
Scripts/                           # executable reproduction core
  configs/                         # fixed settings, splits and asset checksums
  runtime/                         # shared policy, environment, reward and training code
evidence/                          # compact full outcome tables and archived diagnostics
media/figures/                     # one teaching image per notebook
media/demos/                       # four clips, including regressions
tools/verify_notebooks.py          # CPU evidence / notebook checks
```

Data, model weights, downloaded upstream repositories, run logs and personal outreach materials are excluded from Git. [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json) identifies the bundled evidence; [Scripts/SOURCE_PROVENANCE.json](Scripts/SOURCE_PROVENANCE.json) records the original implementation sources and portability changes.

The compact native implementation uses SmolVLA, a Gaussian policy over flow-derived action means, and a locally adapted Qwen VLM reward. It is an explicit resource adaptation of the World-Gymnast idea, not the original large-scale OpenVLA/token-GRPO reproduction. Full historical Dyna/TTT orchestration and the old hybrid candidate-bank campaign are not included in this focused script release; their saved evidence is explained in notebooks 02 and 03.

Upstream projects and teaching-image attribution are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

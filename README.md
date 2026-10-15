# WorldGym–AutoBio

A compact research project for **understanding, evaluating and debugging world-model-based robot policy learning** on the AutoBio MuJoCo tube-pickup task. This is an independent project using WorldGym, not an official WorldGym release.

**System:** demonstrations → supervised policy training → policy evaluation in MuJoCo / native WorldGym → VLM-scored imagined rollouts → GRPO policy updates → separate physical-simulator evaluation.

Native world-model RL has been executed. A reliable policy improvement attributable to WorldGym has **not** been established. All results concern one simulated task; there are no real-robot, wet-lab deployment or surgical results.

## Read the project in 25 minutes

| Notebook | Core question | Time |
|---|---|---:|
| [01 · Understand the MVP](01_understand_the_mvp.ipynb) | How do images, language and joint states produce actions? | 6 min |
| [02 · Results and findings](02_results_and_findings.ipynb) | What improved, what failed, and which component deserves credit? | 10 min |
| [03 · Next WorldGym RL](03_next_worldgym_rl.ipynb) | What should be fixed before another matched RL experiment? | 9 min |

Each notebook includes a new teaching illustration, saved outputs and short runnable analysis. The next experiment in notebook 03 remains **DRAFT / NOT EXECUTED**.

## Findings worth inspecting

| Experiment | Result on its own 100-scene evaluation set | Interpretation |
|---|---|---|
| ACT expansion and policy selection | Original ACT 78; WorldGym-selected candidate 94; strong no-WorldGym selector 93 | +16 points over the original, but the same fixed SFT candidate also scored 94. Unique WorldGym gain is unestablished. |
| Native WorldGym RL | SmolVLA SFT 19; native GRPO 13; matched MuJoCo GRPO 13 | The native loop and actual updates worked; improved policy success was not demonstrated. |
| SFT diagnosis and repair | Original SmolVLA SFT 19; development-selected continuation 53; strong ACT40 91 | +34 points [21, 47], with 44 improvements and 10 new failures. This is additional SFT, not an RL or WorldGym gain. |

The world-model repair reduced static-frame prediction MAE by about 33% against matched replay training on four new initial states, but worsened movement and reversed-gripper predictions. Reward repair reduced false positives while increasing false negatives. The notebooks preserve these negative results and their small-sample limitations. Intervals describe scene-paired uncertainty, not variation across training seeds.

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

The compact native implementation uses SmolVLA, a Gaussian policy over flow-derived action means, and a locally adapted Qwen VLM reward. It is an explicit resource adaptation of the World-Gymnast idea, not the original large-scale OpenVLA/token-GRPO reproduction. Full historical Dyna/TTT orchestration and the old hybrid candidate-bank campaign are not included in this focused script release; their saved evidence remains in notebook 02.

Upstream projects and teaching-image attribution are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

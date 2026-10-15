# Running the reproduction core

The loop is: **SFT checkpoint + image/state/language → action chunk → MuJoCo or native WorldGym → stored trajectory → real VLM reward → GRPO update → separate MuJoCo comparison**. Generated views never receive official success labels. Native rollouts use only simulator reset observations and legal actuator bounds; future images come from the WorldGym/VAE and future proprioception from the learned transition model.

Commands below run from the repository root. `--help` is safe without CUDA. Actual compute commands require CUDA and will never silently switch to CPU.

## 1. Environment and assets

Use Linux or WSL2 with an NVIDIA GPU. AutoBio includes a Linux MuJoCo plugin. The supplied Dockerfile describes the CUDA 12.4 / Python 3.11 environment; its fresh build is distinct from the smoke checks using the existing tested environment.

```bash
docker build -f Scripts/Dockerfile -t worldgym-autobio .
docker run --gpus all --shm-size=8g --rm -it -v "$PWD:/repo" worldgym-autobio
```

Inside that environment:

```bash
python Scripts/prepare.py upstreams
python Scripts/prepare.py public-models --profile sft
python Scripts/prepare.py data --download
python Scripts/prepare.py check --profile sft --full-hash
```

`upstreams` checks out exact AutoBio / LeRobot / WorldGym commits, and applies only the two documented AutoBio import / plugin-path edits. `data` downloads the 40 training and 10 validation demonstrations at the pinned dataset revision, checks frame alignment, and decodes the two RGB streams. It needs substantial disk space; model and dataset files are intentionally outside Git. Hash failures stop execution.

For native rollouts and VLM rewards, additionally prepare the public assets and import the task bundle:

```bash
python Scripts/prepare.py public-models --profile rl
python Scripts/prepare.py import-task /path/to/worldgym-autobio-task-assets
python Scripts/prepare.py check --profile rl --full-hash
```

**The task bundle is prepared locally, but is not published yet.** The command above specifies an input directory, not an existing public download. Its manifest is `configs/assets.json`. It contains the adapted WorldGym parent/endpoint, learned proprioception, action normalization, calibrated Qwen adapter, selected SmolVLA checkpoint/normalization and original ACT checkpoint/statistics. Public WorldGym, VAE, SmolVLA and Qwen base weights are downloaded from their original sources, never repackaged into this Git repository. Access to the SD3 VAE may require the reader's own authorized account; no terms are accepted automatically.

Existing assets can be reused without copying or downloading. Create an ignored `assets.local.json` mapping selected asset names to absolute paths, for example:

```json
{"smolvla": "/models/smolvla_base", "decoded": "/data/decoded"}
```

Pass `--assets assets.local.json` to each command. Every unspecified key uses the relative path in `configs/assets.json`; overriding a path does not override its expected hash. Training and inference are offline after preparation.

## 2. Supervised training

The early MVP uses ACT. The native language-conditioned RL branch uses SmolVLA. They are separate policies, not sequential names for the same model.

```bash
# ACT: one of the original matched fine-tuning arms; requires the task ACT baseline.
python Scripts/sft_act.py --run runs/act40 --condition 40 --seed 5101

# SmolVLA: public pretrained action expert, 40 demos, 15,000 SFT steps.
python Scripts/sft.py --run runs/sft

# Select among 3,000 / 7,500 / 15,000 using the eight fixed development scenes.
# Writes runs/sft/sft/selected.pt; no final labels are read.
python Scripts/evaluate.py --run runs/sft_selection --sft-run runs/sft
```

ACT training keeps the frozen baseline's 10-demo normalization and trains 10,000 additional steps. SmolVLA trains its full pretrained Transformer action expert and projections with a frozen vision/language backbone, train-only statistics, masked 7D flow matching and next-step action alignment. Data, checkpoints and receipts remain in the chosen run directory.

The third-round continuation can also be run explicitly:

```bash
python Scripts/sft.py --run runs/sft_continue_mixed --continue-from task_sft --precision mixed_original --learning-rate scaled
python Scripts/sft.py --run runs/sft_continue_fp32 --continue-from task_sft --precision fp32 --learning-rate scaled
```

Both continuations use 3,000 steps, the same seed/data/start and the predeclared scaled learning rate. `original` selects `1e-5`; `scaled` selects `4.601757083066472e-6`. FP32 did not establish a physical performance improvement in the archived experiments. New runs must preserve all arms and use development-only selection; do not tune using the archived final labels.

## 3. Evaluation

```bash
# ACT physical evaluation (full saved ACT pretrained directory).
python Scripts/evaluate.py --run runs/act_eval --policy act --checkpoint runs/act40/sft_act/last/pretrained

# SmolVLA official physical-simulator success.
python Scripts/evaluate.py --run runs/sft_eval --checkpoint runs/sft/sft/selected.pt --stats runs/sft/DATA.json --environment mujoco

# Native action-conditioned generation plus optional real VLM scores.
python Scripts/evaluate.py --run runs/sft_imagined --checkpoint runs/sft/sft/selected.pt --stats runs/sft/DATA.json --environment worldgym --vlm-reward
```

Each full trajectory contains 114 frames and 1,130 controls, executing 50-control policy chunks. The MuJoCo official check is evaluated every 0.2 seconds: tube body height above 0.925 m and left-finger/cap contact. The reported primary outcome is **ever-success**, not sustained stable grasping. No future MuJoCo state/image is fed into the native rollout.

Native output is a prediction and VLM judgment, not physical success. The task-adapted reward has documented language / out-of-domain failures. This reproduction entry retains that original reward; it does not silently substitute the later repaired reward or call it reliable.

## 4. Native RL and the matched baseline

```bash
python Scripts/train_native_rl.py --run runs/native_rl --checkpoint runs/sft/sft/selected.pt --stats runs/sft/DATA.json --environment worldgym
python Scripts/train_native_rl.py --run runs/mujoco_rl --checkpoint runs/sft/sft/selected.pt --stats runs/sft/DATA.json --environment mujoco
```

Both arms use the same starting SFT, 16 groups × 4 sampled trajectories, task-language VLM reward, sampling seeds, clipping/KL and optimizer settings. The native arm generates future observations with WorldGym; the matched arm obtains them from MuJoCo. Models are loaded/offloaded in phases to fit 16 GB, with one GPU process at a time.

The deterministic 10-step flow output is the Gaussian policy mean. GRPO uses actual sampled actions and matching scalar Gaussian likelihoods, per-coordinate clipped ratios, group-relative advantages and KL to the frozen SFT reference. It updates FP32 trainable weights under BF16 autocast. Flow MSE is used only for SFT; it is not substituted for RL likelihood. Zero-variance reward groups are retained as zero-advantage outcomes.

Evaluate a fixed complete paired matrix after training:

```bash
python Scripts/evaluate.py --run runs/final_comparison --split final --environment mujoco --stats runs/sft/DATA.json --checkpoint runs/sft/sft/selected.pt --checkpoint runs/native_rl/rl/native_GRPO/endpoint.pt --checkpoint runs/mujoco_rl/rl/MuJoCo_GRPO/endpoint.pt
python Scripts/audit.py --run runs/native_rl
python Scripts/audit.py --run runs/mujoco_rl
python Scripts/audit.py --run runs/final_comparison
```

The evaluator prints coverage while running and aggregates only after all requested policies/scenes complete. `RESULT.json` contains all outcomes, paired improvements/new failures and scene-bootstrap intervals. Historical seed lists reproduce the old setting; they are **not** new unseen evidence after the archived labels have been read. For a genuinely new claim, freeze a new config with disjoint training/development/final seeds before running it.

## 5. Small checks, resume and scope

Before a long run, add `--smoke` and use a separate output directory. SFT smoke runs take two real optimizer steps. Evaluation smoke runs one full development trajectory. RL smoke runs four full native trajectories, actual VLM scoring and one optimizer epoch on two sampled observation chunks per trajectory. These test engineering behavior, not effectiveness.

```bash
python Scripts/sft.py --run runs/sft_smoke --smoke
python Scripts/train_native_rl.py --run runs/rl_smoke --checkpoint task_sft --stats normalization --smoke
python Scripts/audit.py --run runs/rl_smoke
```

An existing run requires explicit `--resume`, identical code/config/checkpoint hashes and the same arguments. Training resumes completed optimizer checkpoints; incomplete trajectories are recomputed only within the interrupted group. Optional `--max-hours` establishes one absolute wall-clock limit on first start; resuming does not reset it. Do not run two experiments on the same GPU.

`VALIDATION.json` records which portable paths actually ran and which remain untested. No fresh full 10,000/15,000-step experiment or full final matrix was run merely to package these scripts. The new Dockerfile's complete from-scratch build / public download path is not implied by tests using cached assets. Historical Dyna/TTT and the hybrid selection campaign remain archived evidence; they are outside these core entries.

The external task assets currently allow checkpoint-based reproduction. Training every world-model/reward ancestor from scratch, releasing assets, and demonstrating a reliable WorldGym-specific policy gain are separate work. No private original checkout is imported by the portable runtime; only pinned upstream projects and explicitly configured model/data assets are required.

<div align="center">
  <h1>Neural Introspection Gating for Adaptive KV-Cache Reuse in Vision-Language-Action Models</h1>
  <p>
    <a href="https://zjw4321.github.io/neural-introspection-gating-page/">
      <img alt="Project Page" src="https://img.shields.io/badge/Project-Page-Green">
    </a>
    <a href="LICENSE">
      <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue">
    </a>
    <a href="https://arxiv.org/pdf/2608.10824">
      <img alt="Paper URL" src="https://img.shields.io/badge/arXiv-2608.10824-red?logo=arxiv">
    </a>
  </p>
</div>

Neural introspection gating uses a VLA model's own confidence to decide when cached key-value (KV) states can be reused. After each step it computes the logit margin, the mean gap between the top-1 and top-2 token probabilities over the action tokens. If the margin is below a threshold θ_m, the next step drops the cache and runs full inference. Otherwise the caching method runs as usual. The gate is training-free and only reads the model's output logits, so it does not depend on a particular caching scheme.

This repository implements the gate on top of [VLA-Cache](https://github.com/siyuhsu/vla-cache), which reuses the KV states of image patches that stay static between frames. We call this implementation Gated VLA-Cache and provide it for OpenVLA and OpenVLA-OFT.

## Repository layout

| Folder | Model | VLA-Cache transformers branch |
|---|---|---|
| `openvla/` | OpenVLA | `vla-cache-openvla` |
| `openvla-oft/` | OpenVLA-OFT | `vla-cache-openvla-oft` |

The code is VLA-Cache plus a small set of changes (files listed in [NOTICE](NOTICE)):

- `prismatic/extern/hf/modeling_prismatic.py` computes the logit margin.
- `experiments/robot/openvla_utils.py` implements the gate.
- `experiments/robot/libero/run_libero_eval.py` adds the `--action_margin_threshold` flag.
- OpenVLA only: the static-patch top-k is 120 instead of 130 (the setting used in the paper), and a step with no reusable patches no longer crashes.

## Setup

The two models use different branches of the VLA-Cache transformers fork, so each folder needs its own conda environment. For OpenVLA, from the repository root:

```bash
conda create -n gated-vla-cache python=3.10 -y
conda activate gated-vla-cache
pip install -e openvla

git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git ../LIBERO
pip install -e ../LIBERO --config-settings editable_mode=compat
pip install -r openvla/experiments/robot/libero/libero_requirements.txt
pip install numpy==1.26.4 opencv-python==4.11.0.86 mujoco==3.2.3 tensorflow-metadata==1.17.1
```

The last line pins packages whose newer releases break the setup: numpy 2 (pulled in by OpenCV 4.12+) and tensorflow-metadata 1.18+ break TensorFlow 2.15, and newer MuJoCo breaks robosuite 1.4.1. LIBERO needs `editable_mode=compat` because its top-level `libero/` folder is not a regular Python package.

For OpenVLA-OFT, repeat these steps in a second environment (for example `gated-vla-cache-oft`) with `openvla-oft` in place of `openvla`, then run `pip install diffusers==0.32.2`: newer diffusers releases require a newer peft than the pinned 0.11.1. The LIBERO clone can be shared. Flash-attention is not needed for evaluation.

### PyTorch on newer GPUs

`pyproject.toml` pins PyTorch 2.2.0, as VLA-Cache does. RTX 50-series (Blackwell) GPUs need PyTorch 2.7 or newer. On those GPUs, run this after `pip install -e` (pip will warn about the pin):

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

PyTorch 2.6 and newer also changed the default of `torch.load`, which LIBERO uses to load its initial states. Set `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in that environment, for example with `conda env config vars set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`.

### LIBERO

On first import, LIBERO asks whether to use a custom dataset folder; answer `N`. It then writes its paths to `~/.libero/config.yaml`. If that file already exists from another LIBERO install, point LIBERO to a new folder first, for example `export LIBERO_CONFIG_PATH=~/.libero-gated-vla-cache`. On a machine without a display, also set `export MUJOCO_GL=egl`.

## Checkpoints

Download checkpoints into the local `checkpoints/` folder and pass that local path to the eval script. The eval script copies this repository's `modeling_prismatic.py` into the checkpoint folder at startup. A Hugging Face Hub ID would load the original model code instead, without caching or the gate. From `openvla/`:

```bash
python vla_cache_scripts/download_model_local.py --model_id openvla/openvla-7b-finetuned-libero-spatial
```

For OpenVLA-OFT, run the same script from `openvla-oft/` with `--model_id moojink/openvla-7b-oft-finetuned-libero-spatial`. Each LIBERO suite has its own checkpoint: replace `spatial` with `object`, `goal` or `10`.

## Evaluation

| Method | Flags |
|---|---|
| Full inference | `--use_vla_cache False` |
| VLA-Cache | `--use_vla_cache True` |
| Gated VLA-Cache | `--use_vla_cache True --action_margin_threshold 0.65` (OpenVLA) or `0.50` (OpenVLA-OFT) |

For example, Gated VLA-Cache with OpenVLA on LIBERO-Spatial, from `openvla/` (the same command is in `openvla/run.sh`):

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_vla_cache True \
  --action_margin_threshold 0.65
```

The task suites are `libero_spatial`, `libero_object`, `libero_goal` and `libero_10`, each with its matching checkpoint. A run evaluates 50 episodes per task (`--num_trials_per_task`), prints the success rate, writes a log to `experiments/logs/` and saves rollout videos to `rollouts/`. The transformers fork prints the average TFLOPs per step. Success rates on other GPUs or PyTorch versions can differ from the paper's.

## Changes since the paper

**OpenVLA-OFT margin indexing.** In the code used for the paper, the action-token logits for the margin were indexed from the start of the sequence. On steps where VLA-Cache reuses tokens, the sequence is shorter, so the margin was read from shifted positions, or was empty and the gate did not fire. `main` indexes from the end, which reads the intended positions on every step (`_regression_or_discrete_prediction` in `openvla-oft/prismatic/extern/hf/modeling_prismatic.py`):

```python
action_token_logits = language_model_output.logits[:, -(ACTION_DIM * NUM_ACTIONS_CHUNK) - 2 : -2, :]
```

The OpenVLA-OFT results in the paper were produced with the earlier indexing, and θ_m = 0.50 was tuned with it. To reproduce the paper exactly, use the `iros2026` tag (`git checkout iros2026`). OpenVLA is not affected.

We measured both versions on LIBERO-Spatial with 500 episodes per setting on one machine (RTX 5090, PyTorch 2.7):

| OpenVLA-OFT, LIBERO-Spatial | Success | TFLOPs / step | Gate fired (share of queries) |
|---|---|---|---|
| Full inference | 98.4% | 4.00 | – |
| VLA-Cache | 97.2% | 3.12 | – |
| Gated VLA-Cache, paper code (`iros2026`) | 98.4% | 3.25 | 14.5% |
| Gated VLA-Cache, `main` | 98.0% | 3.44 | 33.7% |

With the corrected indexing the gate fires more than twice as often, which raises compute, while success stays within run-to-run noise (about ±0.9 points at this level). On this machine every method scores higher than in the paper's Table II, while the TFLOPs match it closely, so compare settings run on the same machine.

## Citation

```bibtex
@article{wu2026gated,
  title     = {Neural Introspection Gating for Adaptive KV-Cache Reuse in Vision-Language-Action Models},
  author    = {Wu, Zhijie and Kawaharazuka, Kento and Okada, Kei},
  journal   = {arXiv preprint arXiv:2608.10824},
  year      = {2026}
}
```

## Acknowledgements and license

This code builds on [VLA-Cache](https://github.com/siyuhsu/vla-cache), [OpenVLA](https://github.com/openvla/openvla), [OpenVLA-OFT](https://github.com/moojink/openvla-oft) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO). See [NOTICE](NOTICE) for attribution and the list of changed files.

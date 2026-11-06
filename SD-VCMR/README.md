# MTVR / MML Anonymous Research Artifact

This repository is the feature-level reference implementation of **Multi-modal
Moment Localization (MML)** for MTVR. It contains the model, the paper's
training-only 100-to-10 spatial candidate-pruning path, full-corpus inference,
TVR-style VR/SVMR/VCMR metrics, strict data validation, deterministic run
logging, tests, and a one-command end-to-end smoke test.

The online model consumes frozen `.npy` features. BLIP2 frame selection,
stylization, MiniGPT-v2 captioning, ResNet/SlowFast extraction, and RoBERTa
encoding are offline preprocessing stages; no large vision-language model is
called during MML training or inference.

## Artifact at a glance

```text
configs/tvr_mml.json       complete reference experiment configuration
docs/DATA_FORMAT.md        manifest and feature schemas
docs/IMPLEMENTATION.md     paper-equation-to-code map
docs/REPRODUCIBILITY.md    full replication protocol and scope
mtvr/                      config, data, model, loss, inference, metrics
scripts/train.py           deterministic training and epoch checkpoints
scripts/infer.py           full-corpus VR/SVMR/VCMR inference
scripts/evaluate.py        single-run and multi-seed evaluation
scripts/validate_data.py   eager manifest/feature integrity checks
scripts/run_smoke_test.py  data -> train -> infer -> evaluate smoke test
tests/                     unit, failure-path, and CLI integration tests
CHECKSUMS.sha256           release inventory and SHA-256 digests
```

## 1. Install

The reference environment is Python 3.10.13, PyTorch 2.1.2, NumPy 1.26.4,
CUDA 11.8. The Conda specification is the preferred GPU installation:

```bash
conda env create -f environment.yml
conda activate SD-VCMR
python -m pip install -e .
```

For a CPU diagnostic environment:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
```

## 2. Verify the release

```bash
python scripts/verify_package.py
python -m pytest -q
python scripts/run_smoke_test.py --output runs/smoke --device cpu
```

The smoke run generates deterministic synthetic features and checks data
loading, candidate sampling/pruning, forward/backward, checkpoint loading,
full-corpus ranking, span decoding, metadata capture, and all three metrics.
`runs/smoke/SMOKE_TEST_REPORT.json` is the machine-readable receipt. **Smoke
metrics are functional-test outputs, not benchmark results.**

## 3. Prepare and validate MTVR features

Create video and query JSONL manifests exactly as described in
[`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md). Keep the official TVR splits and
use the same frozen extractors for training and evaluation. Then edit only the
paths and feature widths in `configs/tvr_mml.json` and run:

```bash
python scripts/validate_data.py \
  --config configs/tvr_mml.json \
  --output runs/data_validation.json
```

Validation eagerly reads every array with `allow_pickle=False` and checks
dimensions, finite values, duplicate IDs, cross-manifest references, proxy
style count, clip alignment, and labeled boundaries. Training repeats this
preflight by default.

## 4. Train the three reference seeds

The release declares seeds `42`, `43`, and `44`. Run each seed in an isolated
directory (example in Bash):

```bash
for seed in 42 43 44; do
  python scripts/train.py \
    --config configs/tvr_mml.json \
    --output "runs/tvr_seed${seed}" \
    --seed "${seed}" \
    --device cuda
done
```

Each run records `config.resolved.json`, `run_metadata.json`, `metrics.jsonl`,
epoch checkpoints, `checkpoint_last.pt`, and `train_summary.json`. The metadata
includes the command, seed, determinism flag, resolved data paths, manifest
hashes, package versions, OS/GPU information, parameter count, and pruning
protocol. Resume only from an epoch checkpoint with `--resume PATH`.

Training implements the protocol stated in the supplement: each query samples
100 in-batch candidate videos, the cheap pooled spatial score retains the top
10 while forcibly preserving the paired video, and cross-attention is applied
only to the retained candidates. Duplicate positive video IDs are handled by
multi-positive InfoNCE rather than counted as false negatives.

## 5. Infer and evaluate

```bash
python scripts/infer.py \
  --config configs/tvr_mml.json \
  --checkpoint runs/tvr_seed42/checkpoint_last.pt \
  --split val \
  --output runs/tvr_seed42/val_predictions.jsonl \
  --device cuda

python scripts/evaluate.py \
  --config configs/tvr_mml.json \
  --predictions runs/tvr_seed42/val_predictions.jsonl \
  --output runs/tvr_seed42/val_metrics.json
```

Inference disables training-time pruning: every corpus video receives the
Eq. (13) VR score, after which boundary heads are applied to the configured
top-ranked videos. A `.meta.json` sidecar records checkpoint/config/manifest
and prediction hashes.

Aggregate the three seeded prediction files with mean and sample standard
deviation:

```bash
python scripts/evaluate.py \
  --config configs/tvr_mml.json \
  --predictions \
    runs/tvr_seed42/val_predictions.jsonl \
    runs/tvr_seed43/val_predictions.jsonl \
    runs/tvr_seed44/val_predictions.jsonl \
  --output runs/tvr_three_seed_metrics.json
```

TVR test annotations are held out; final test numbers must be produced through
the official TVR evaluation server. Local evaluation is intended for labeled
train/validation manifests.

## Reproducibility boundary

The manuscript explicitly states `alpha=0.3`, `lambda_svmr=25`, batch size
128, AdamW, 100 sampled candidates, top-10 retention, nine epochs, and one A100
(40 GB). The manuscript does not uniquely specify `tau`, `beta`, projection
width, convolution kernel, exact learning-rate schedule, OS/CPU, or exact
seeds. This repository makes every such choice explicit in the versioned
configuration; those release choices must not be presented as independently
verified values from an undisclosed original run.

Matching paper tables additionally requires the MTVR proxy bundle/captions,
official splits, exact frozen feature caches, and official test evaluation.
See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the complete
claim-to-artifact matrix and expected output contract.

## Integrity and license

`CHECKSUMS.sha256` covers every released file. After an intentional release
change, regenerate and immediately verify it:

```bash
python scripts/verify_package.py --write-checksums
python scripts/verify_package.py
```

The code is released under the MIT License. Dataset, pretrained model, and
third-party feature licenses remain with their respective owners.

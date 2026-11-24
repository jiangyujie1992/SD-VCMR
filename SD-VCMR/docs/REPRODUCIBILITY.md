# Reproducibility protocol

## Implementation checks

| Component | Verification |
|---|---|
| The MML equations are executable | method map, model/loss modules, gradient test |
| The 100-to-10 training protocol is active | `forward_pruned`, strict config, positive-retention tests |
| Full-corpus ranking at inference | `score_corpus` scans every manifest video |
| VR/SVMR/VCMR use inclusive clip boundaries | metric implementation and exact IoU tests |
| Runs are traceable | resolved config, environment, commands, hashes, checkpoints |
| Randomness is controlled | declared seeds; Python/NumPy/PyTorch/DataLoader/candidate RNGs |
| Feature loading, training and evaluation | eager validator and end-to-end synthetic CLI test |

## Required external assets

For a TVR benchmark run, prepare the following inputs.

1. the TVR 21,800-video corpus and official split definitions;
2. MTVR's three cached proxy hints and cached captions for each query;
3. frozen ResNet+SlowFast video, RoBERTa subtitle/enhanced-query, and ResNet
   proxy features with the exact dimensions in the configuration;
4. official TVR test-server access for held-out test annotations.

The [visual query builder](VISUAL_QUERIES.md) generates the three proxy images and their ResNet features from annotated clips.

Place manifests at the paths declared in `configs/tvr_mml.json`, or update only
those path fields. `scripts/validate_data.py` records manifest digests so that
all seeded runs can be shown to use identical inputs.

## Final settings

| Setting | Value | Provenance |
|---|---:|---|
| GPU used in paper efficiency experiment | 1 x NVIDIA A100 40 GB | manuscript/supplement |
| batch size | 128 | manuscript/supplement |
| optimizer | AdamW | manuscript/supplement |
| epochs | 9 | manuscript/supplement |
| candidates sampled / retained | 100 / 10 | manuscript/supplement |
| `alpha` | 0.3 | manuscript/supplement |
| `lambda_svmr` | 25 | manuscript/supplement |
| seeds | 42, 43, 44 | release configuration |
| learning rate / weight decay | 1e-4 / 0.01 | release configuration |
| schedule / warmup | constant / 0 | release configuration |
| `tau` / `beta` | 0.07 / 1.0 | release configuration |
| hidden width / heads | 512 / 8 | release configuration |
| dropout / boundary kernel | 0.1 / 3 | release configuration |
| deterministic kernels | enabled | release configuration |

## Software environment

`environment.yml` pins Python 3.10.13, PyTorch 2.1.2, NumPy 1.26.4 and CUDA 11.8. The core pip versions are in `requirements.txt`. Each training run records its software and hardware environment in `run_metadata.json`.

## Execution and artifacts

For each declared seed:

```bash
python scripts/train.py --config configs/tvr_mml.json \
  --output runs/tvr_seed42 --seed 42 --device cuda
python scripts/infer.py --config configs/tvr_mml.json \
  --checkpoint runs/tvr_seed42/checkpoint_last.pt \
  --split val --output runs/tvr_seed42/val_predictions.jsonl --device cuda
python scripts/evaluate.py --config configs/tvr_mml.json \
  --predictions runs/tvr_seed42/val_predictions.jsonl \
  --output runs/tvr_seed42/val_metrics.json
```

Repeat for seeds 43 and 44, then pass all prediction files to one evaluation
command. The aggregate output reports per-seed values, arithmetic mean, and
sample standard deviation computed from per-run predictions.

Training artifacts:

- `config.resolved.json`: immutable effective hyperparameters;
- `run_metadata.json`: command, environment, inputs, hashes, protocol, status;
- `metrics.jsonl`: append-only epoch metrics and learning rate;
- `checkpoints/checkpoint_epoch_NNN.pt`: epoch-resumable state;
- `checkpoint_last.pt`: final/most recent epoch state;
- `train_summary.json`: final status and diagnostic losses.

Inference artifacts:

- `*.jsonl`: ordered per-query rankings and spans;
- `*.jsonl.meta.json`: input/checkpoint/output hashes and environment;
- evaluation JSON: protocol, query count, metrics, and prediction hashes.

## Determinism expectations

The trainer seeds Python, NumPy, CPU/CUDA PyTorch, DataLoader workers, shuffling,
and candidate sampling. cuDNN benchmarking is disabled and deterministic
algorithms are requested with warnings for unavailable deterministic kernels.
Compare checkpoint hashes for runs on the same software and hardware stack. Compare retrieval metrics across GPU architectures and library builds.

## Evaluation protocol

- VR: a query is correct at K if its labeled video appears in the first K video
  entries.
- SVMR: among the first K spans predicted inside the labeled video, at least one
  must meet the temporal-IoU threshold.
- VCMR: among the first K corpus spans, both video ID and temporal IoU must match.
- Clip boundaries are inclusive and zero-based; `[s,e]` is evaluated as
  `[s,e+1)`.
- The default cutoffs are K in `{1,10,100}` and IoU in `{0.5,0.7}`.

Submit held-out test predictions to the official TVR evaluator. Record the prediction metadata hashes alongside the three-seed aggregate.

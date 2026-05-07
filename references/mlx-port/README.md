# Original MLX Port (karpathy/autoresearch-mlx)

These files are the original Karpathy `autoresearch-mlx` reference implementation,
preserved for understanding the outer loop pattern.

The active codebase adapts this pattern for **TimesFM 2.5** (PyTorch-based
time-series foundation model) instead of MLX-based language model training.

## Files

- `train.py.ref` — Original MLX training script (language model pretraining)
- `prepare.py.ref` — Original MLX data preparation (FineWeb-Edu dataset)
- `program.md.orig` — Original research program for LLM pretraining
- `results.tsv.orig` — Original experiment log from MLX runs

## Adaptation Notes

| Original (MLX) | Adapted (TimesFM) |
|---|---|
| `import mlx.core as mx` | `import torch` via TimesFM |
| Training GPT-like model | Frozen TimesFM 2.5 inference |
| `val_bpb` metric | Cost-asymmetric loss |
| 5-min training budget | 200ms inference budget |
| Single GPU (M-series) | Any PyTorch GPU/CPU |
| `train.py` (modifiable) | `experiments/*.py` + config search |
| Language modeling | Multi-tenant resource forecasting |

The outer loop architecture (program.md → run experiment → log → repeat) is
preserved exactly. The inner loop was redesigned for the TimesFM thesis.

# Upstream attribution

This directory follows the **autoresearch** pattern: a `program.md` that
defines the research protocol, a fixed `prepare.py` that owns data
loading and evaluation, a mutable `train.py` that the autonomous agent
edits, and a `results.tsv` ledger tracking every trial.

The pattern originates with **Andrej Karpathy** in
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) and was
ported to Apple Silicon / MLX by **trevin-creator** in
[trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx).
Both upstream repositories are MIT-licensed. Full credit to @karpathy
for the original conception and to @trevin-creator for the MLX port.

## What we vendored

- **File layout**: `program.md`, `prepare.py`, `train.py`, `results.tsv`
  — the same canonical four-file shape used by upstream. Renaming our
  earlier `infra.py` / `run_experiment.py` was a deliberate choice so
  that a reader familiar with autoresearch / autoresearch-mlx can
  navigate this directory without translation.
- **Conventions**: one mutable training script, one immutable data /
  eval module, a `program.md`-driven protocol, and keep-or-revert via
  git.

## What we did NOT vendor

- **MLX itself**: the inference backend remains frozen TimesFM on
  PyTorch (see `src/tsfm_autoresearch/tsfm_client.py`). The 200ms
  latency budget claim and the GCP / Cloud Run scale-out path in
  `deploy/` both depend on the PyTorch backend.
- **Upstream `train.py` / `prepare.py` bodies**: those are
  nanochat-training scripts. Our `train.py` and `prepare.py` are
  forecasting-domain code authored for this project. Only the file
  names and the protocol are shared.

## License

Both upstream repositories are MIT-licensed (compatible with this
project's MIT license, see the top-level `pyproject.toml`).

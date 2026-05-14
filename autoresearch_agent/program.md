# autoresearch: TSFM Multi-Tenant Forecasting

This is an adaptation of the
[karpathy/autoresearch](https://github.com/karpathy/autoresearch) pattern —
and its Apple Silicon port
[trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) —
applied to multi-tenant time-series forecasting with a frozen TimesFM
foundation model.

The canonical file layout (`program.md`, `prepare.py`, `train.py`,
`results.tsv`) is preserved so that anyone familiar with upstream
autoresearch / autoresearch-mlx can navigate this directory without
translation. See `NOTICE.md` for attribution.

The core thesis: **a per-request autoresearch loop over a frozen TimesFM
achieves better cost-asymmetric performance than any single fixed
configuration, within a 200ms inference latency budget.**

This directory implements the OUTER LOOP — autonomous experiment management
that iterates on experiment configurations to empirically validate the thesis.
The inner loop (per-request config search) is in `src/tsfm_autoresearch/`.

## Architecture

```
autoresearch_agent/                   ← THIS DIRECTORY (outer loop)
├── program.md                        ← Research program (you are here)
├── train.py                          ← Experiment runner (MODIFIABLE)
├── prepare.py                        ← Fixed infrastructure (READ-ONLY)
├── results.tsv                       ← Experiment results ledger
└── NOTICE.md                         ← Upstream attribution (karpathy + trevin-creator)

src/tsfm_autoresearch/                ← Inner loop (per-request optimization)
├── autoresearch.py                   ← AutoresearchHarness
├── tsfm_client.py                    ← Frozen TimesFM wrapper
├── losses.py                         ← Cost-asymmetric loss functions
├── archetype_store.py                ← FAISS retrieval for cold start
└── workload_gen.py                   ← Synthetic workload generator

experiments/                          ← Experiment definitions
├── 01_workload_characterization.py   ← Sanity-check the generator
├── 02_tsfm_wrapper_validation.py     ← Sanity-check the frozen model
├── m4_retrieval_accuracy.py          ← Archetype retrieval accuracy (M4)
├── m6_headline.py                    ← Headline experiment (M6)
├── m7_latency_sweep.py               ← Latency sweep (M7)
├── m8_cold_start.py                  ← Cold-start experiment (M8)
└── m9_sla_asymmetry.py               ← SLA tier asymmetry (M9)
```

## Setup

Work with the user to:

1. **Agree on a run tag**: e.g. `m6-v1`, `latency-sweep-1`. The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current branch.
3. **Read the in-scope files**: These are the files you need context on:
   - `autoresearch_agent/prepare.py` — fixed constants, data loading, evaluation harness. **Do not modify.**
   - `autoresearch_agent/train.py` — the file you modify. Experiment config, what to run, what metrics to track.
   - `experiments/*.py` — the experiment scripts themselves. You can change WHICH experiments run and with what parameters.
   - `src/tsfm_autoresearch/` — the inner loop. You can change config search space parameters.
4. **Verify data exists**: Run `uv run python -m tsfm_autoresearch.workload_gen --tenants 1000 --days 30 --seed 42` if not already generated.
5. **Verify model loads**: `uv run python -c "from tsfm_autoresearch.tsfm_client import TSFMClient; TSFMClient()"` — should complete without error.
6. **Initialize results.tsv**: Create with header row. Run `uv run python autoresearch_agent/train.py` once to establish a baseline.
7. **Confirm and go**: Once setup looks good, begin experimentation.

## Experimentation

Each experiment runs for a **fixed time budget** (configurable in prepare.py, default: no hard limit — runs until completion). You launch it as: `uv run python autoresearch_agent/train.py`.

**What you CAN modify:**
- `train.py` — which experiment to run, experiment parameters, config search space
- `src/tsfm_autoresearch/autoresearch.py` — the inner loop config sampling strategy
- `src/tsfm_autoresearch/losses.py` — SLA tier parameters (α values)
- Experiment scripts under `experiments/` — what baselines to compare, sample sizes, horizon

**What you CANNOT modify:**
- `autoresearch_agent/prepare.py` — data loading, evaluation metrics, fixed constants
- `src/tsfm_autoresearch/tsfm_client.py` — the model is FROZEN
- `src/tsfm_autoresearch/workload_gen.py` — synthetic data is fixed for reproducibility

## Acceptance criteria

The thesis is considered empirically validated if the following hold on the
1,000-tenant synthetic fleet. Each criterion is concrete and gateable.

| ID | Criterion | Target | Source |
|----|-----------|--------|--------|
| C1 | Paired win-rate of autoresearch vs FixedConfigTSFM on cost-asymmetric loss (standard SLA, α=0.75) | win_rate ≥ 0.55, two-sided binomial p < 0.05, 95% CI lower bound > 0.50 | M6 |
| C2 | Mean cost-asymmetric loss improvement | ≥ 5% relative reduction vs FixedConfigTSFM | M6 |
| C3 | Per-archetype win-rate on the three hardest archetypes (wp-cron-heavy, cache-driven, idle-ish) | win_rate ≥ 0.60 each | M6 |
| C4 | End-to-end latency (autoresearch with K=8) | p95 ≤ 200ms on the target inference host | M7 |
| C5 | Max K that respects budget | max_k_within_200ms ≥ 8 | M7 |
| C6 | Archetype retrieval accuracy from full history (held-out tenants) | top-1 accuracy ≥ 0.90 | M4 |
| C7 | Cold-start retrieval accuracy at 120 min history (held-out tenants) | top-1 accuracy ≥ 0.70 | M4 |
| C8 | Archetype-guided cold-start closes ≥ 50% of the cold-vs-oracle gap by 120 min history | gap_closure ≥ 0.50 | M8 |
| C9 | SLA-tier monotonicity (mean forecast values) | premium > standard > basic, all checks pass | M9 |
| C10 | Direct loss-asymmetry: premium forecast wins at α=0.90; basic forecast wins at α=0.65 | both checks pass | M9 |

A criterion is reported as **PASS** only when the experiment script that owns
it prints the corresponding metric AND the metric meets the target. Anything
below target is **FAIL** — investigate; do not paper over.

**Discipline:**
- Never tune the FixedConfigTSFM grid search on tenants that appear in the
  evaluation set (m6 passes `exclude_tenant_ids` automatically; do not bypass).
- Never build the archetype store on tenants that will be queried in the
  cold-start experiment (m8 passes `exclude_tenants` automatically).
- Always use the paired evaluation harness (`evaluate_paired` in
  `experiments/m6_headline.py`) when comparing two forecasters — never zip
  two independent loss lists.
- Report binomial CI + p-value for any win-rate claim. Do not report the bare
  percentage without uncertainty.

**Simplicity criterion**: All else being equal, simpler is better. An approach that achieves similar results with less complexity is preferred.

**The first run**: Always run the experiment as-is to establish a baseline before making changes.

## Output format

The experiment runner prints a summary:

```
---
experiment:        m6_headline
n_tenants:         200
cost_asym_loss:    0.0423
mae:               0.0156
p50_latency_ms:    1450
p95_latency_ms:    2100
win_rate:          0.78  [0.71, 0.84]  p=2.1e-08
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

Columns:
```
timestamp	commit	experiment	cost_asym_loss	mae	p50_latency_ms	status	description
```

1. ISO-8601 UTC timestamp (auto-populated by `log_result`)
2. git commit hash (short, 7 chars)
3. experiment name (e.g. "m6_headline")
4. cost-asymmetric loss (lower is better)
5. MAE (sanity check)
6. p50 latency in ms
7. status: `keep`, `discard`, or `crash`
8. short description of what this experiment tried

The writer takes an OS-level exclusive lock on the ledger, so concurrent
runs are serialized rather than racing.

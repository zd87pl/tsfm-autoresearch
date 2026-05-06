# autoresearch: TSFM Multi-Tenant Forecasting

This is an adaptation of Karpathy's autoresearch pattern applied to
multi-tenant time-series forecasting with a frozen TimesFM foundation model.

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
├── run_experiment.py                 ← Experiment runner (MODIFIABLE)
├── infra.py                          ← Fixed infrastructure (READ-ONLY)
└── results.tsv                       ← Experiment results ledger

src/tsfm_autoresearch/                ← Inner loop (per-request optimization)
├── autoresearch.py                   ← AutoresearchHarness
├── tsfm_client.py                    ← Frozen TimesFM wrapper
├── losses.py                         ← Cost-asymmetric loss functions
├── workload_gen.py                   ← Synthetic workload generator

experiments/                          ← Experiment definitions
├── 02_fixed_vs_autoresearch.py       ← Headline experiment (M6)
├── 03_latency_budget_sweep.py        ← Latency sweep (M7)
├── 04_cold_start_archetype.py        ← Cold-start experiment (M8)
└── 05_sla_tier_asymmetry.py          ← SLA tier experiment (M9)
```

## Setup

Work with the user to:

1. **Agree on a run tag**: e.g. `m6-v1`, `latency-sweep-1`. The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current branch.
3. **Read the in-scope files**: These are the files you need context on:
   - `autoresearch_agent/infra.py` — fixed constants, data loading, evaluation harness. **Do not modify.**
   - `autoresearch_agent/run_experiment.py` — the file you modify. Experiment config, what to run, what metrics to track.
   - `experiments/*.py` — the experiment scripts themselves. You can change WHICH experiments run and with what parameters.
   - `src/tsfm_autoresearch/` — the inner loop. You can change config search space parameters.
4. **Verify data exists**: Run `uv run python -m tsfm_autoresearch.workload_gen --tenants 1000 --days 30 --seed 42` if not already generated.
5. **Verify model loads**: `uv run python -c "from tsfm_autoresearch.tsfm_client import TSFMClient; TSFMClient()"` — should complete without error.
6. **Initialize results.tsv**: Create with header row. Run `uv run python autoresearch_agent/run_experiment.py` once to establish a baseline.
7. **Confirm and go**: Once setup looks good, begin experimentation.

## Experimentation

Each experiment runs for a **fixed time budget** (configurable in infra.py, default: no hard limit — runs until completion). You launch it as: `uv run python autoresearch_agent/run_experiment.py`.

**What you CAN modify:**
- `run_experiment.py` — which experiment to run, experiment parameters, config search space
- `src/tsfm_autoresearch/autoresearch.py` — the inner loop config sampling strategy
- `src/tsfm_autoresearch/losses.py` — SLA tier parameters (α values)
- Experiment scripts under `experiments/` — what baselines to compare, sample sizes, horizon

**What you CANNOT modify:**
- `autoresearch_agent/infra.py` — data loading, evaluation metrics, fixed constants
- `src/tsfm_autoresearch/tsfm_client.py` — the model is FROZEN
- `src/tsfm_autoresearch/workload_gen.py` — synthetic data is fixed for reproducibility

**The goal: empirically validate the thesis.**
Specifically:
1. Autoresearch beats fixed-config TimesFM on cost-asymmetric loss (M6)
2. The gap widens on archetypes farthest from the global optimum (wp-cron-heavy, cache-driven, idle-ish)
3. The latency budget (200ms) is achievable at scale (M7, M10)
4. Cold-start with archetype retrieval closes the gap to oracle (M8)
5. Different α values produce measurably different allocation behavior (M9)

**Simplicity criterion**: All else being equal, simpler is better. An approach that achieves similar results with less complexity is preferred.

**The first run**: Always run the experiment as-is to establish a baseline before making changes.

## Output format

The experiment runner prints a summary:

```
---
experiment:        02_fixed_vs_autoresearch
n_tenants:         200
cost_asym_loss:    0.0423
mae:               0.0156
p50_latency_ms:    1450
p95_latency_ms:    2100
win_rate:          0.78
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated).

Columns:
```
commit	experiment	cost_asym_loss	mae	p50_latency_ms	status	description
```

1. git commit hash (short, 7 chars)
2. experiment name (e.g. "02_fixed_vs_autoresearch")
3. cost-asymmetric loss (lower is better)
4. MAE (sanity check)
5. p50 latency in ms
6. status: `keep`, `discard`, or `crash`
7. short description of what this experiment tried

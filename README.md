# TSFM-Autoresearch: Empirical Validation

**Per-request autoresearch over frozen TimesFM for multi-tenant resource forecasting.**

Proof-of-concept empirically validating the thesis:

> For multi-tenant resource forecasting, a per-request autoresearch loop over a frozen time-series foundation model achieves better cost-asymmetric performance than the same foundation model deployed with any single fixed configuration, while staying within a 200ms inference latency budget.

## Project Status

| Milestone | Status | Branch |
|-----------|--------|--------|
| M1: Synthetic Workload Generator | ✅ Complete | (merged) |
| M2: Frozen TimesFM Wrapper | ✅ Complete | (merged) |
| M3: Autoresearch Harness + Loss Functions | ✅ Complete | (merged) |
| M4: Archetype Store (FAISS) | ✅ Complete | (merged) |
| M5: Forecast Baselines | ✅ Complete | (merged) |
| M6: Headline Experiment | ✅ Complete | (merged) |
| M7: Latency Budget Sweep | ✅ Complete | (merged) |
| M8: Cold-Start Experiment | ✅ Complete | (merged) |
| M9: SLA Tier Asymmetry | ✅ Complete | (merged) |
| M10: GCP Scale-Out | ✅ Complete | (merged) |

**Tests:** 116 pass, 1 skip (non-TimesFM fast suite)
**Setup:** `bash setup.sh` — zero to results in one command

## Architecture

```
┌──────────────────────────────────────────────────┐
│ OUTER LOOP (karpathy/autoresearch-mlx pattern)   │
│ autoresearch_agent/                              │
│ ├── program.md      ← Research goals for AI agent│
│ ├── run_experiment.py  ← Modifiable experiment   │
│ ├── infra.py        ← Fixed infrastructure       │
│ └── results.tsv     ← Experiment ledger          │
│                                                   │
│   Agent: modify → run → evaluate → log → repeat  │
└──────────────────────┬───────────────────────────┘
                       │ calls per tenant/request
┌──────────────────────▼───────────────────────────┐
│ INNER LOOP (per-request autoresearch)            │
│ src/tsfm_autoresearch/                           │
│ ├── autoresearch.py    ← K-config search loop    │
│ ├── losses.py          ← Cost-asymmetric loss (α)│
│ ├── tsfm_client.py     ← Frozen TimesFM wrapper  │
│ ├── archetype_store.py ← FAISS retrieval         │
│ └── workload_gen.py    ← Synthetic data generator│
│                                                   │
│   Per-request: split history → sample K configs  │
│   → batch forecast → score → select winner       │
└──────────────────────────────────────────────────┘
```

## Repository Structure

```
tsfm-autoresearch/
├── CLAUDE.md
├── README.md
├── pyproject.toml                     # uv-managed
├── autoresearch_agent/               # karpathy-style outer loop
│   ├── program.md                    # Research program
│   ├── infra.py                      # Fixed infrastructure (read-only)
│   ├── run_experiment.py             # Modifiable experiment runner
│   └── results.tsv                   # Experiment results ledger
├── src/
│   ├── tsfm_autoresearch/
│   │   ├── workload_gen.py           # M1: Synthetic workload generator ✓
│   │   ├── tsfm_client.py            # M2: Frozen TimesFM wrapper ✓
│   │   ├── autoresearch.py           # M3: Autoresearch harness ✓
│   │   ├── losses.py                 # M3: Cost-asymmetric loss ✓
│   │   └── archetype_store.py        # M4: FAISS archetype store ✓
│   └── baselines/
│       ├── protocol.py               # M5: Forecaster protocol
│       ├── naive_last.py             # M5: Last-value baseline ✓
│       ├── naive_seasonal.py         # M5: Seasonal baseline ✓
│       ├── fixed_config.py           # M5: Fixed TimesFM baseline ✓
│       └── per_tenant_arima.py       # M5: Per-tenant ARIMA ✓
├── references/
│   └── mlx-port/                     # Original karpathy/autoresearch-mlx files
├── experiments/
│   ├── 01_workload_characterization.py  ✓
│   ├── 02_tsfm_wrapper_validation.py    ✓
│   ├── m6_headline.py                   ✓ (M6)
│   ├── m7_latency_sweep.py              ✓ (M7)
│   ├── m8_cold_start.py                 ✓ (M8)
│   ├── 09_sla_tier_asymmetry.py         (M9)
│   └── 10_gcp_scale_out.py              (M10)
├── data/
│   ├── synthetic/                    # Generated workloads (gitignored)
│   └── boom/                         # Datadog BOOM benchmark
├── results/
├── deploy/
│   ├── gcp/
│   └── docker/
└── tests/
    ├── test_workload_gen.py          ✓
    ├── test_tsfm_client.py           ✓
    ├── test_losses.py                ✓
    ├── test_autoresearch.py          ✓
    ├── test_archetype_store.py       ✓
    ├── test_baselines.py             ✓
    ├── test_headline_experiment.py   ✓ (M6)
    ├── test_latency_sweep.py         ✓ (M7)
    └── test_cold_start.py            ✓ (M8)
```

## Quick Start

```bash
# One-command setup (recommended)
bash setup.sh

# Or manual:
# Install Python 3.12 + uv
uv python install 3.12

# Clone and set up
git clone https://github.com/zd87pl/tsfm-autoresearch.git
cd tsfm-autoresearch
uv sync --extra dev

# Install TimesFM (from source, not PyPI)
git clone https://github.com/google-research/timesfm.git /tmp/timesfm
uv pip install -e "/tmp/timesfm[torch]"

# Generate synthetic workloads (M1)
uv run python -m tsfm_autoresearch.workload_gen --tenants 1000 --days 30 --seed 42

# Run all fast tests (no GPU/TimesFM required)
uv run pytest tests/ --ignore=tests/test_autoresearch.py --ignore=tests/test_tsfm_client.py

# Run all tests including TimesFM (requires model download, ~15GB RAM)
uv run pytest tests/
```

## Milestones

### M1: Synthetic Workload Generator
1,000 tenants across 8 WordPress-hosting archetypes, 30 days of 1-min resolution.

| Archetype | Signature |
|-----------|-----------|
| `low-traffic-blog` | Low baseline, weak diurnal, occasional comment spikes |
| `ecommerce-retail` | Strong diurnal+weekly, sharp campaign spikes |
| `news-publisher` | Diurnal + rare extreme breaking-news bursts |
| `b2b-saas` | Binary on/off business hours, weekend trough |
| `wp-cron-heavy` | Flat traffic, frequent uncorrelated CPU spikes |
| `cache-driven` | High CPU–network correlation, cache-miss cascades |
| `compute-heavy` | High constant baseline, low variance |
| `idle-ish` | Near-zero, rare crawler spikes |

```bash
uv run python -m tsfm_autoresearch.workload_gen --tenants 1000 --days 30 --seed 42
```

### M2: Frozen TimesFM Wrapper
Thin wrapper around `google/timesfm-2.5-200m-pytorch`. Model loaded once, never updated.
- `forecast(history, config, horizon)` → multivariate → (horizon, D, quantiles)
- `forecast_batch(histories, configs, horizon)` → single forward pass for K configs

### M3: Autoresearch Harness + Cost-Asymmetric Loss
Per-request 6-stage loop: split history → sample K configs → batch forecast → score with α-weighted loss → select winner → final forecast.

**Cost-asymmetric loss:** L = α·max(0, y-ŷ) + (1-α)·max(0, ŷ-y)
- Premium (α=0.90): 90% weight on under-prediction
- Standard (α=0.75)
- Basic (α=0.65)

**karpathy-style outer loop:** `autoresearch_agent/` — autonomous experiment management. Agent reads `program.md`, modifies `run_experiment.py`, runs experiments, logs to `results.tsv`, analyzes, repeats.

### M4: Archetype Store
FAISS-backed archetype embeddings. 28 statistical features per tenant, StandardScaler normalization, cosine similarity retrieval. **>90% retrieval accuracy** on held-out tenants with full history.

### M5: Forecast Baselines
Four baselines conforming to `Forecaster` protocol:
- **NaiveLast**: predict last observed value
- **NaiveSeasonal**: 24h seasonal lag (1440 steps at 1-min resolution)
- **PerTenantARIMA**: ARIMA(1,0,1) per tenant per dimension (slow by design)
- **FixedConfigTSFM**: TimesFM with grid-searched best context_len (strongest baseline)

### M6: Headline Experiment
Compares FixedConfigTSFM vs AutoresearchHarness on cost-asymmetric loss across the synthetic tenant fleet. This is the central empirical validation of the thesis.

```bash
uv run python experiments/m6_headline.py --tenants 200 --horizon 60 --timestamps 50
uv run python autoresearch_agent/run_experiment.py  # via outer loop
```

### M7: Latency Budget Sweep
Sweeps K ∈ {1, 2, 4, 8, 16, 32} to measure the latency/loss trade-off curve and validate the 200ms budget claim.

```bash
uv run python experiments/m7_latency_sweep.py --tenants 50 --timestamps 5
```

### M8: Cold-Start Experiment
Tests archetype retrieval from minimal history (30-480 min) to close the gap between cold autoresearch and the oracle. Wires the `archetype_embedding` parameter in `AutoresearchHarness.forecast()` for the first time.

```bash
uv run python experiments/m8_cold_start.py --tenants 50 --lengths 30,60,120,240,480
```

### M9: SLA Tier Asymmetry
Validates that different α values (premium=0.90, standard=0.75, basic=0.65) produce measurably different forecast behavior — premium forecasts are systematically higher (protective) than basic ones (cost-efficient). Confirms monotonicity: premium > standard > basic.

```bash
uv run python experiments/m9_sla_asymmetry.py --tenants 100
```

### M10: GCP Scale-Out
Production deployment scaffold: Dockerfile for Cloud Run, FastAPI forecast service with `/forecast` and `/health` endpoints, Cloud Build pipeline. Qdrant migration path documented for horizontal scaling.

```bash
gcloud builds submit --tag gcr.io/PROJECT/tsfm-autoresearch -f deploy/docker/Dockerfile
gcloud run deploy tsfm-autoresearch --image gcr.io/PROJECT/tsfm-autoresearch --gpu 1
```

## Tech Stack

- **Python 3.12**, `uv` for environment management
- **PyTorch** for TimesFM inference (frozen, never fine-tuned)
- **FAISS** for archetype embeddings (PoC; Qdrant for GCP)
- **Polars** for time-series data manipulation
- **Pydantic v2** for all config and request/response schemas
- **pytest** + **hypothesis** for testing (7 property-based loss invariants)
- **Ruff** for lint/format
- **scikit-learn** for feature standardization
- **statsmodels** for ARIMA baselines

## License

MIT

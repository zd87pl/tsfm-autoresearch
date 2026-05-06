# TSFM-Autoresearch: Empirical Validation

**Per-request autoresearch over frozen TimesFM for multi-tenant resource forecasting.**

This is a proof-of-concept that empirically validates the thesis:

> For multi-tenant resource forecasting, a per-request autoresearch loop over a frozen time-series foundation model achieves better cost-asymmetric performance than the same foundation model deployed with any single fixed configuration, while staying within a 200ms inference latency budget.

## Project Status

**M1 (Synthetic Workload Generator)** — in progress.

## Repository Structure

```
tsfm-autoresearch/
├── CLAUDE.md                         # Persistent context for AI agents
├── README.md
├── pyproject.toml                    # uv-managed dependencies
├── src/
│   ├── tsfm_autoresearch/
│   │   ├── __init__.py
│   │   ├── workload_gen.py           # Synthetic tenant workload generator ✓
│   │   ├── archetype_store.py        # FAISS-backed archetype embeddings
│   │   ├── tsfm_client.py            # Frozen TimesFM wrapper
│   │   ├── autoresearch.py           # The inventive core
│   │   ├── losses.py                 # Cost-asymmetric scoring
│   │   ├── translator.py             # Forecast → orchestrator directive
│   │   └── service.py                # Stateless worker entrypoint
│   └── baselines/
│       ├── fixed_config.py
│       ├── naive_last.py
│       ├── naive_seasonal.py
│       └── per_tenant_arima.py
├── experiments/
│   ├── 01_workload_characterization.py  ✓
│   ├── 02_fixed_vs_autoresearch.py
│   ├── 03_latency_budget_sweep.py
│   ├── 04_cold_start_archetype.py
│   └── 05_sla_tier_asymmetry.py
├── notebooks/
├── data/
│   ├── synthetic/                    # Generated workloads (gitignored)
│   └── boom/                         # Datadog BOOM benchmark
├── results/                          # Experiment outputs
├── deploy/
│   ├── gcp/
│   └── docker/
└── tests/
    └── test_workload_gen.py          ✓
```

## Quick Start

```bash
# Install Python 3.12 + uv
uv python install 3.12

# Clone and set up
git clone https://github.com/zd87pl/tsfm-autoresearch.git
cd tsfm-autoresearch
uv sync

# Generate synthetic workloads (M1)
uv run python -m tsfm_autoresearch.workload_gen --tenants 1000 --days 30 --seed 42

# Generate characterization plots
uv run python experiments/01_workload_characterization.py

# Run tests
uv run pytest
```

## M1: Synthetic Workload Generator

Generates 1,000 synthetic tenants spanning 8 archetypes that model WordPress-hosting fleet characteristics:

| Archetype | Description |
|-----------|-------------|
| `low-traffic-blog` | Low baseline, weak diurnal, occasional comment spikes |
| `ecommerce-retail` | Moderate baseline, strong diurnal+weekly, campaign spikes |
| `news-publisher` | Diurnal + stochastic breaking-news bursts |
| `b2b-saas` | Weekday business hours, weekend trough |
| `wp-cron-heavy` | Flat traffic, periodic CPU spikes from cron jobs |
| `cache-driven` | Spiky CPU on cache misses, high CPU-network correlation |
| `compute-heavy` | High baseline, low variance, constant work |
| `idle-ish` | Mostly flat, near-zero load, rare crawler hits |

Each tenant produces 30 days of 1-minute resolution data across:
- `cpu_util` — CPU utilization (fraction)
- `mem_util` — Memory utilization (fraction)
- `net_bytes` — Network bytes/sec
- `disk_iops` — Disk I/O operations/sec

**Usage:**
```bash
uv run python -m tsfm_autoresearch.workload_gen --tenants 1000 --days 30 --seed 42
```

Output: `data/synthetic/<tenant_id>.parquet` + `manifest.csv`

## Tech Stack

- **Python 3.12**, `uv` for environment management
- **PyTorch** for TimesFM inference
- **FAISS** for archetype embeddings (PoC; Qdrant for GCP)
- **Optuna** for Bayesian optimization in autoresearch
- **Polars** for time-series data (not pandas)
- **Pydantic v2** for all schemas
- **pytest** + **hypothesis** for testing
- **Ruff** for lint/format

## License

MIT — see LICENSE file.

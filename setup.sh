#!/usr/bin/env bash
# tsfm-autoresearch: one-command setup
#
# Usage:
#   curl -sSL https://raw.githubusercontent.com/zd87pl/tsfm-autoresearch/main/setup.sh | bash
#   # or:
#   bash setup.sh [--full] [--tenants 1000] [--skip-tests]
#
# What it does:
#   1. Installs Python 3.12 + uv (if missing)
#   2. Clones the repo (if not already in it)
#   3. Installs all dependencies
#   4. Installs TimesFM 2.5 from source
#   5. Generates synthetic tenant data
#   6. Runs all fast tests
#   7. (--full) Runs the headline experiment (requires GPU for reasonable time)
#
# Flags:
#   --full       Run M6 headline experiment after setup
#   --tenants N  Number of tenants for data generation (default: 1000)
#   --days N     Days of data (default: 30)
#   --skip-tests Skip test suite
#   --gpu        GPU-optimized TimesFM config (per_core_batch_size=8, torch_compile)

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────

REPO_URL="https://github.com/zd87pl/tsfm-autoresearch.git"
REPO_DIR="tsfm-autoresearch"
PYTHON_VERSION="3.12"
TIMESFM_REPO="https://github.com/google-research/timesfm.git"
TIMESFM_TAG="v2.5.0"

# Defaults
N_TENANTS=1000
DAYS=30
SKIP_TESTS=false
RUN_FULL=false
GPU_MODE=false

# ── Parse args ────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --full)    RUN_FULL=true ;;
        --tenants) N_TENANTS="$2"; shift ;;
        --days)    DAYS="$2"; shift ;;
        --skip-tests) SKIP_TESTS=true ;;
        --gpu)     GPU_MODE=true ;;
        *)         echo "Unknown flag: $1"; exit 1 ;;
    esac
    shift
done

# ── Colors ────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

section()  { echo -e "\n${BLUE}═══ $1 ═══${NC}"; }
success() { echo -e "  ${GREEN}✓${NC} $1"; }
warn()    { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()    { echo -e "  ${RED}✗${NC} $1"; exit 1; }

# ── 1. Python + uv ────────────────────────────────────────────────────

section "1/7: Checking Python $PYTHON_VERSION + uv"

if command -v uv &>/dev/null; then
    success "uv found: $(uv --version)"
else
    echo "  Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    success "uv installed: $(uv --version)"
fi

if uv python find "$PYTHON_VERSION" &>/dev/null; then
    success "Python $PYTHON_VERSION found"
else
    echo "  Installing Python $PYTHON_VERSION..."
    uv python install "$PYTHON_VERSION"
    success "Python $PYTHON_VERSION installed"
fi

# ── 2. Clone repo ─────────────────────────────────────────────────────

section "2/7: Repository"

if [[ -f "pyproject.toml" ]] && grep -q "tsfm-autoresearch" pyproject.toml 2>/dev/null; then
    success "Already in tsfm-autoresearch directory"
    REPO_DIR="."
else
    if [[ -d "$REPO_DIR" ]]; then
        success "Repository exists, pulling latest..."
        cd "$REPO_DIR"
        git pull origin main
    else
        echo "  Cloning $REPO_URL..."
        git clone "$REPO_URL"
        cd "$REPO_DIR"
        success "Repository cloned"
    fi
fi

# ── 3. Install dependencies ───────────────────────────────────────────

section "3/7: Installing dependencies"

uv sync --extra dev
success "Dependencies installed ($(uv pip list 2>/dev/null | wc -l) packages)"

# ── 4. TimesFM 2.5 ────────────────────────────────────────────────────

section "4/7: TimesFM 2.5"

if uv run python -c "from tsfm_autoresearch.tsfm_client import TSFMClient; print('OK')" 2>/dev/null; then
    success "TimesFM already importable"
else
    TIMESFM_DIR="/tmp/timesfm-install"
    if [[ ! -d "$TIMESFM_DIR" ]]; then
        echo "  Cloning timesfm..."
        git clone --depth 1 "$TIMESFM_REPO" "$TIMESFM_DIR"
    fi
    echo "  Installing TimesFM[torch]..."
    uv pip install -e "$TIMESFM_DIR[torch]"
    success "TimesFM 2.5 installed"
fi

# Try loading the model (downloads ~800MB from HuggingFace Hub)
echo "  Checking if model can load (may download ~800MB)..."
if uv run python -c "
from tsfm_autoresearch.tsfm_client import TSFMClient
c = TSFMClient(max_context=512, max_horizon=128, per_core_batch_size=1, torch_compile=False)
print('Model loaded successfully')
" 2>/dev/null; then
    success "TimesFM model loaded and ready"
else
    warn "TimesFM model download skipped (will download on first forecast)"
    warn "Set HF_TOKEN env var if HuggingFace Hub requires auth"
fi

# ── 5. Generate synthetic data ────────────────────────────────────────

section "5/7: Synthetic data ($N_TENANTS tenants, $DAYS days)"

if [[ -f "data/synthetic/manifest.csv" ]]; then
    existing=$(wc -l < data/synthetic/manifest.csv)
    success "Data exists ($existing tenants in manifest)"
    echo "  To regenerate: rm -rf data/synthetic/ && re-run"
else
    echo "  Generating $N_TENANTS tenants × $DAYS days (seed=42)..."
    uv run python -m tsfm_autoresearch.workload_gen \
        --tenants "$N_TENANTS" --days "$DAYS" --seed 42 --output data/synthetic
    success "Data generated → data/synthetic/"
fi

# ── 6. Run tests ──────────────────────────────────────────────────────

section "6/7: Tests"

if $SKIP_TESTS; then
    warn "Tests skipped (--skip-tests)"
else
    echo "  Running fast tests (no GPU/TimesFM required)..."
    if uv run pytest tests/ \
        --ignore=tests/test_autoresearch.py \
        --ignore=tests/test_tsfm_client.py \
        -q 2>&1; then
        success "All fast tests pass"
    else
        fail "Tests failed — check output above"
    fi

    # Count test modules
    N_TESTS=$(uv run pytest tests/ \
        --ignore=tests/test_autoresearch.py \
        --ignore=tests/test_tsfm_client.py \
        --collect-only -q 2>&1 | tail -1 | grep -oP '\d+(?= tests)')
    success "Test count: ${N_TESTS:-?} tests"
fi

# ── 7. Experiment (optional) ──────────────────────────────────────────

section "7/7: Experiment"

if $RUN_FULL; then
    echo "  Running M6 headline experiment (100 tenants, 10 timestamps)..."
    echo "  ⚠ This requires TimesFM model loaded and may take minutes/hours on CPU."
    echo ""
    if $GPU_MODE; then
        export TSFM_PER_CORE_BATCH_SIZE=8
        echo "  GPU mode: batch_size=8, torch_compile=True"
    fi
    uv run python experiments/m6_headline.py \
        --tenants 100 --horizon 60 --timestamps 10
    success "M6 headline experiment complete → results/"
else
    echo "  Skipping experiment (use --full to run M6 headline)"
    echo ""
    echo "  Manual experiments:"
    echo "    uv run python experiments/m6_headline.py --tenants 200 --horizon 60 --timestamps 50"
    echo "    uv run python experiments/m7_latency_sweep.py --tenants 50 --timestamps 5"
    echo "    uv run python experiments/m8_cold_start.py --tenants 50"
    echo "    uv run python experiments/m9_sla_asymmetry.py --tenants 100"
fi

# ── Done ───────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  tsfm-autoresearch setup complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "  Next steps:"
echo "    cd $REPO_DIR"
echo "    uv run pytest tests/ -q              # re-run tests"
echo "    uv run python experiments/m6_headline.py --tenants 100"
echo ""
echo "  Full experiment suite (GPU recommended):"
echo "    bash setup.sh --full --gpu --tenants 1000"

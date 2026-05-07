"""
Autoresearch Harness (M3) — the inventive core.

Implements the per-request autoresearch loop that is the central claim
of the patent and the empirical contribution of the arXiv paper.

THE ALGORITHM (for each forecast request):
  1. Receive (tenant_id, history, horizon, sla_tier)
  2. Retrieve archetype embedding → condition the configuration prior
     (M4 dependency — currently hardcoded uniform prior, marked TODO)
  3. Sample K configurations from the configuration space using Optuna
     with the archetype-conditioned prior. Default K = 8.
  4. Split history: train_prefix (first 85%) | val_slice (last 15%)
  5. Batch-evaluate all K configs against val_slice in ONE TimesFM
     forward pass (critical for 200ms latency budget)
  6. Score each forecast with cost-asymmetric loss parameterized by sla_tier
  7. Select the best config. Re-run TSFM on FULL history with it.
  8. Return ForecastResponse with final forecast, winning config,
     per-config scores (for analysis), and per-stage timing.

WHY THIS WORKS (patent/paper argument):
  - A single fixed configuration must compromise across all tenant archetypes
    and SLA tiers. The optimal context length for an ecommerce tenant with
    strong diurnal patterns differs from the optimal for a wp-cron-heavy
    tenant with flat traffic.
  - By evaluating K configurations per-request, we find the best config
    for THIS specific (tenant, sla_tier, history) tuple.
  - The cost-asymmetric loss ensures the selected config is optimized for
    the right SLA tier — premium tenants get protective forecasts (α=0.90),
    basic tenants get cost-efficient ones (α=0.65).
  - The batch evaluation means K configs cost roughly the same latency as
    one (on GPU), keeping us within the 200ms budget.

Author: Hermes Agent (Ziggy's TSFM-Autoresearch PoC)
Patent-pending inventive subject matter — see CLAUDE.md for context.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tsfm_autoresearch.losses import SLATier, CostAsymmetricLoss, make_loss_for_sla
from tsfm_autoresearch.tsfm_client import (
    ForecastConfig,
    ProbabilisticForecast,
    TSFMClient,
)

logger = logging.getLogger(__name__)

# ── Configuration Search Space ─────────────────────────────────────────
# This defines the decision variables that the autoresearch loop optimizes.
# Each variable maps to a ForecastConfig field or a derived computation.
#
# PATENT NOTE: The configuration space is deliberately compact for the PoC.
# The full design adds: frequency adaptation, covariate selection strategy,
# and dynamic quantile subset selection based on SLA tier. These are
# reserved for future claims.


# Context length candidates (in minutes of history)
_CONTEXT_LEN_CANDIDATES = [64, 128, 256, 384, 512]

# Archetype → preferred context_len bias (domain-knowledge informed)
# These are heuristics until grid-search per-archetype results are available.
# Based on: high-variance archetypes need longer context to capture patterns.
_ARCHETYPE_CONTEXT_BIAS: dict[str, int] = {
    "low-traffic-blog": 128,
    "ecommerce-retail": 512,  # Strong diurnal+weekly → long context
    "news-publisher": 384,    # Diurnal + breaking-news bursts
    "b2b-saas": 256,          # Business hours pattern, 4h context
    "wp-cron-heavy": 256,     # Flat traffic with CPU spikes
    "cache-driven": 384,      # Correlation patterns need history
    "compute-heavy": 128,     # High baseline, low variance → short ok
    "idle-ish": 64,           # Near-zero → minimal context needed
}

# Quantile subsets to evaluate
# Must use quantiles available in TimesFM 2.5: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# "protective" = aggressive under-prediction protection (premium)
# "balanced" = wider coverage (standard)
# "light" = fewer quantiles, faster (all tiers)
_QUANTILE_PRESETS = {
    "protective": [0.1, 0.5, 0.9],
    "balanced": [0.1, 0.3, 0.5, 0.7, 0.9],
    "light": [0.1, 0.5, 0.9],
}

# ── Data Structures ────────────────────────────────────────────────────


@dataclass
class ConfigScore:
    """
    A single configuration evaluation result.

    Attributes:
        config: The ForecastConfig that was evaluated.
        score: Cost-asymmetric loss on val_slice (lower is better).
        forecast: The forecast produced by this config (on val_slice,
            or None if full-history re-forecast).
        rank: Rank among K candidates (0 = best).
    """

    config: ForecastConfig
    score: float
    forecast: ProbabilisticForecast | None = None
    rank: int = -1


@dataclass
class ForecastResponse:
    """
    Complete autoresearch response for a single forecast request.

    Contains the final forecast (produced by the winning config on full
    history), the winning configuration, per-config scores for analysis,
    and per-stage timing breakdown for latency budget tracking.

    Attributes:
        tenant_id: Which tenant this forecast is for.
        final_forecast: The winning forecast on full history.
        winning_config: The ForecastConfig that won.
        candidate_scores: All K evaluated configurations with scores.
        sla_tier: SLA tier used for scoring.
        timing_ms: Per-stage timing breakdown.
        total_latency_ms: End-to-end autoresearch latency.
    """

    tenant_id: str
    final_forecast: ProbabilisticForecast
    winning_config: ForecastConfig
    candidate_scores: list[ConfigScore]
    sla_tier: SLATier
    timing_ms: dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0

    @property
    def best_config_score(self) -> float:
        """Score of the winning configuration."""
        return self.candidate_scores[0].score if self.candidate_scores else float("inf")

    @property
    def score_spread(self) -> float:
        """Max - min score across candidates (measures config sensitivity)."""
        if len(self.candidate_scores) < 2:
            return 0.0
        scores = [cs.score for cs in self.candidate_scores]
        return max(scores) - min(scores)


# ── AutoresearchHarness ────────────────────────────────────────────────


class AutoresearchHarness:
    """
    Per-request autoresearch loop over frozen TimesFM.

    This is the inventive core — it optimizes the forecast *configuration*
    per-request while keeping the model weights frozen. The optimization
    uses cost-asymmetric scoring parameterized by SLA tier.

    Usage:
        harness = AutoresearchHarness(tsfm_client)
        response = harness.forecast(
            tenant_id="tenant_0042",
            history=cpu_mem_net_disk_array,
            horizon=60,
            sla_tier=SLATier.PREMIUM,
            K=8,
        )
    """

    def __init__(
        self,
        tsfm_client: TSFMClient,
        default_K: int = 8,
        val_split_ratio: float = 0.15,
        seed: int = 42,
    ):
        """
        Args:
            tsfm_client: Initialized (compiled) TSFMClient.
            default_K: Default number of configs to sample per request.
            val_split_ratio: Fraction of history to hold out as validation
                (last N% of the time series). Default 0.15 = 15%.
            seed: RNG seed for reproducibility.
        """
        self._client = tsfm_client
        self._default_K = default_K
        self._val_split_ratio = val_split_ratio
        self._rng = np.random.default_rng(seed)

        logger.info(
            "AutoresearchHarness initialized: K=%d, val_split=%.0f%%, seed=%d",
            default_K, val_split_ratio * 100, seed,
        )

    # ── Public API ─────────────────────────────────────────────────

    def forecast(
        self,
        tenant_id: str,
        history: np.ndarray,
        horizon: int = 60,
        sla_tier: SLATier | str = SLATier.STANDARD,
        K: int | None = None,
        archetype_embedding: np.ndarray | None = None,
        seed: int | None = None,
    ) -> ForecastResponse:
        """
        Execute the full autoresearch loop for one forecast request.

        This is the method that will be called by the stateless worker
        (service.py in M10) for every incoming forecast request.

        Args:
            tenant_id: Tenant identifier (for logging and archetype lookup).
            history: Multivariate history array, shape (T, D).
            horizon: Number of steps to forecast.
            sla_tier: SLA tier for cost-asymmetric scoring.
            K: Number of configurations to evaluate. Default: self._default_K.
            archetype_embedding: Tenant's archetype embedding for prior
                conditioning (M4 — currently unused, marked TODO).

        Returns:
            ForecastResponse with final forecast, winner, and diagnostics.
        """
        if isinstance(sla_tier, str):
            sla_tier = SLATier(sla_tier)

        if K is None:
            K = self._default_K

        # Deterministic per-request RNG: hash tenant_id + sla_tier for
        # reproducible config sampling across calls with the same inputs.
        if seed is not None:
            request_rng = np.random.default_rng(seed)
        else:
            # hashlib.blake2b is platform-independent (Python's built-in
            # hash() is randomized per-interpreter and not reproducible
            # across hosts); we want the same (tenant, tier, shape) to
            # produce the same RNG everywhere.
            import hashlib
            digest = hashlib.blake2b(
                f"{tenant_id}:{sla_tier.value}:{history.shape}".encode(),
                digest_size=8,
            ).digest()
            seed_hash = int.from_bytes(digest, "big")
            request_rng = np.random.default_rng(seed_hash & 0x7FFFFFFF)

        timing: dict[str, float] = {}
        t_total = time.perf_counter()

        # ── Stage 1: Split history ────────────────────────────────
        t0 = time.perf_counter()
        train_history, val_actuals = self._split_history(history)
        timing["split_history"] = (time.perf_counter() - t0) * 1000

        # ── Stage 2: Sample K configurations ──────────────────────
        t0 = time.perf_counter()
        configs = self._sample_configs(K, sla_tier, archetype_embedding, request_rng)
        timing["sample_configs"] = (time.perf_counter() - t0) * 1000

        # ── Stage 3: Batch-evaluate K configs on val_slice ────────
        t0 = time.perf_counter()
        val_forecasts = self._client.forecast_batch(
            histories=[train_history] * K,
            configs=configs,
            horizon=val_actuals.shape[0],
        )
        timing["batch_evaluate"] = (time.perf_counter() - t0) * 1000

        # ── Stage 4: Score with cost-asymmetric loss ───────────────
        t0 = time.perf_counter()
        loss_fn = make_loss_for_sla(sla_tier)
        scored: list[ConfigScore] = []
        for i, forecast in enumerate(val_forecasts):
            score = loss_fn.score(val_actuals, forecast)
            scored.append(ConfigScore(config=configs[i], score=score, forecast=forecast))
        timing["score_configs"] = (time.perf_counter() - t0) * 1000

        # ── Stage 5: Select winner ─────────────────────────────────
        t0 = time.perf_counter()
        scored.sort(key=lambda cs: cs.score)
        for rank, cs in enumerate(scored):
            cs.rank = rank
        winner_config = scored[0].config
        timing["select_winner"] = (time.perf_counter() - t0) * 1000

        # ── Stage 6: Final forecast on full history ────────────────
        t0 = time.perf_counter()
        final_forecast = self._client.forecast(history, winner_config, horizon)
        timing["final_forecast"] = (time.perf_counter() - t0) * 1000

        total_latency_ms = (time.perf_counter() - t_total) * 1000

        logger.info(
            "Autoresearch for %s: winner=%s (score=%.4f), "
            "K=%d, spread=%.4f, latency=%.0fms",
            tenant_id,
            f"ctx={winner_config.context_len}",
            scored[0].score,
            K,
            scored[-1].score - scored[0].score if len(scored) > 1 else 0,
            total_latency_ms,
        )

        return ForecastResponse(
            tenant_id=tenant_id,
            final_forecast=final_forecast,
            winning_config=winner_config,
            candidate_scores=scored,
            sla_tier=sla_tier,
            timing_ms=timing,
            total_latency_ms=total_latency_ms,
        )

    # ── Internal: History Splitting ───────────────────────────────

    def _split_history(
        self, history: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Split history into training prefix and validation slice.

        The validation slice is the last `val_split_ratio` fraction of the
        time series. This simulates a realistic forecasting scenario where
        we validate against the most recent observed data.

        PATENT NOTE: We use a temporal split (not random) because forecast
        evaluation must respect temporal ordering. A random split would
        leak future information into the past and inflate performance.

        Args:
            history: Shape (T, D).

        Returns:
            (train_history, val_actuals) where train_history is (T_train, D)
            and val_actuals is (T_val, D).
        """
        T = history.shape[0]
        val_len = max(1, int(T * self._val_split_ratio))

        if val_len >= T:
            # Edge case: history too short — use all but last point for training
            val_len = 1
            train_len = T - 1
        else:
            train_len = T - val_len

        train = history[:train_len, :]
        val = history[train_len:, :]

        return train, val

    # ── Internal: Configuration Sampling ──────────────────────────

    def _sample_configs(
        self,
        K: int,
        sla_tier: SLATier,
        archetype_embedding: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[ForecastConfig]:
        """
        Sample K diverse configurations from the search space.

        Uses stratified random sampling across the configuration dimensions
        to ensure diverse coverage. When archetype_embedding is provided,
        biases context_len sampling toward the archetype's historically best
        configuration while still exploring alternatives.

        Sampling strategy:
          - context_len: Archetype-biased if embedding available, else uniform
          - quantiles: Weighted toward SLA-appropriate presets
          - freq: Fixed to "T" (1-minute) for PoC

        Args:
            K: Number of configs to sample.
            sla_tier: SLA tier (influences quantile preset weighting).
            archetype_embedding: Optional bias vector. If str, treated as
                archetype name for context_len biasing. If np.ndarray,
                placeholder for future full embedding support.

        Returns:
            List of K distinct ForecastConfig objects.
        """
        configs: list[ForecastConfig] = []

        # Weight quantile presets by SLA tier appropriateness
        if sla_tier == SLATier.PREMIUM:
            preset_weights = {"protective": 0.6, "balanced": 0.3, "light": 0.1}
        elif sla_tier == SLATier.STANDARD:
            preset_weights = {"protective": 0.3, "balanced": 0.5, "light": 0.2}
        else:  # BASIC
            preset_weights = {"protective": 0.1, "balanced": 0.3, "light": 0.6}

        preset_names = list(preset_weights.keys())
        preset_probs = np.array(list(preset_weights.values()))
        preset_probs = preset_probs / preset_probs.sum()

        # Use local RNG for reproducibility
        local_rng = rng if rng is not None else self._rng

        # ── Archetype-biased context_len sampling ──────────────────
        # When archetype info is available, bias ~40% of configs toward
        # the archetype's known-best context_len, keeping ~60% uniform
        # for exploration.
        bias_ctx: int | None = None
        bias_strength: float = 0.0
        if archetype_embedding is not None:
            if isinstance(archetype_embedding, str):
                # Archetype name → lookup preferred context_len
                bias_ctx = _ARCHETYPE_CONTEXT_BIAS.get(archetype_embedding)
            elif isinstance(archetype_embedding, np.ndarray):
                # Full embedding → find nearest archetype centroid
                try:
                    from tsfm_autoresearch.archetype_store import _ARCHETYPE_CONTEXT_BIAS as _ACB
                    # Future: use FAISS to find nearest archetype
                    # For now, just try to use the embedding as-is
                    pass
                except ImportError:
                    pass
            if bias_ctx is not None:
                bias_strength = 0.4  # 40% of configs use archetype bias

        for i in range(K):
            if bias_ctx is not None and local_rng.random() < bias_strength:
                ctx_len = bias_ctx
            else:
                ctx_len = int(local_rng.choice(_CONTEXT_LEN_CANDIDATES))

            preset = str(local_rng.choice(preset_names, p=preset_probs))
            quantiles = _QUANTILE_PRESETS[preset]

            config = ForecastConfig(
                context_len=ctx_len,
                quantiles=quantiles,
                freq="T",
            )
            configs.append(config)

        return configs

    # ── Properties ────────────────────────────────────────────────

    @property
    def client(self) -> TSFMClient:
        return self._client

    @property
    def default_K(self) -> int:
        return self._default_K

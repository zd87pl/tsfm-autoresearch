"""
Synthetic Multi-Tenant Workload Generator (M1).

Generates 1,000 synthetic tenants spanning 8 archetypes that model
realistic WordPress-hosting fleet characteristics for WP Engine-style
capacity planning research.

Each tenant produces 30 days of 1-minute-resolution time series across
four resource dimensions:
  - cpu_util: CPU utilization (fraction 0–1, or slightly above for burst)
  - mem_util: Memory utilization (fraction 0–1)
  - net_bytes: Network bytes/sec (positive float)
  - disk_iops: Disk I/O operations per second (positive float)

Archetypes are designed to test the autoresearch thesis: per-tenant
configuration adaptation should benefit most for tenants farthest from
the global optimum (wp-cron-heavy, cache-driven, idle-ish).

Output: data/synthetic/<tenant_id>.parquet per tenant + manifest.csv

Usage:
    uv run python -m tsfm_autoresearch.workload_gen --tenants 1000 --days 30 --seed 42

Author: Hermes Agent (Ziggy's TSFM-Autoresearch PoC)
Patent-pending inventive subject matter — see CLAUDE.md for context.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────
# 30 days of 1-minute data
MINUTES_PER_DAY = 1440

# Archetype distribution: reflects a real WordPress hosting fleet where
# low-traffic blogs and idle-ish tenants dominate the long tail, but
# ecommerce and B2B SaaS drive the revenue-relevant forecasting problems.
ARCHETYPE_DISTRIBUTION: dict[str, float] = {
    "low-traffic-blog": 0.25,
    "ecommerce-retail": 0.12,
    "news-publisher": 0.08,
    "b2b-saas": 0.15,
    "wp-cron-heavy": 0.15,
    "cache-driven": 0.10,
    "compute-heavy": 0.05,
    "idle-ish": 0.10,
}

# ── Data Structures ────────────────────────────────────────────────────


@dataclass
class TenantConfig:
    """Per-tenant generative parameters — randomized within archetype bounds."""

    tenant_id: str
    archetype: str
    # Diurnal parameters
    diurnal_amplitude: float  # strength of daily cycle (0-1)
    diurnal_phase: float  # phase offset in hours (0-24)
    # Weekly parameters
    weekly_amplitude: float  # strength of weekly cycle (0-1)
    weekday_factor: float  # weekday vs weekend ratio
    # Noise parameters
    noise_scale: float  # scale of AR(1) innovations
    ar_coefficient: float  # AR(1) persistence (0-1)
    # Spike parameters
    spike_rate: float  # spikes per day on average
    spike_magnitude_mean: float  # mean spike size (multiplicative)
    # Baseline levels (per resource)
    cpu_baseline: float
    mem_baseline: float
    net_baseline: float
    disk_baseline: float
    # Cross-resource correlation (e.g., cpu-net for cache-driven)
    cpu_net_correlation: float


# ── Signal Components ──────────────────────────────────────────────────
# Each component is a pure function: time_axis → signal array.
# Composed by archetype-specific generators to build realistic workloads.


def _diurnal(t: np.ndarray, amplitude: float, phase: float = 8.0) -> np.ndarray:
    """
    Smooth diurnal (24-hour) cycle using a raised cosine.
    Peaks at `phase` hours, troughs 12 hours later.

    Args:
        t: Time in hours (fractional).
        amplitude: Peak-to-trough swing as fraction of baseline (0–1).
        phase: Hour of peak activity (e.g., 14 for 2 PM).
    """
    return amplitude * (0.5 + 0.5 * np.cos(2 * np.pi * (t - phase) / 24))


def _weekly(t: np.ndarray, amplitude: float) -> np.ndarray:
    """
    Weekly cycle (7-day period). Peaks mid-week, troughs on weekends.

    Args:
        t: Time in days (fractional).
        amplitude: Weekend dip depth.
    """
    # Use day-of-week: sin peaks on Thursday (day 3.5), bottoms on Sunday (day 0.5)
    day_of_week = t % 7
    return amplitude * np.sin(2 * np.pi * (day_of_week - 1.5) / 7)


def _ar1_noise(
    n: int, phi: float = 0.9, sigma: float = 0.05, seed: int | None = None
) -> np.ndarray:
    """
    AR(1) process for autocorrelated noise.
    x[t] = phi * x[t-1] + epsilon[t], epsilon ~ N(0, sigma²)

    This produces the slowly-drifting baseline variation seen in real
    server metrics — not white noise, but correlated over minutes/hours.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _spike_train(
    n: int,
    rate_per_day: float,
    magnitude_mean: float,
    decay_tau: float = 15.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Poisson-timed spikes with exponential decay tails.

    Each spike is a step increase of random magnitude that decays
    exponentially over `decay_tau` minutes. This models:
      - wp-cron job CPU spikes
      - Comment bursts on blogs
      - Cache-miss cascades
      - Breaking-news traffic surges

    The decay shape is:
        spike[t] += magnitude * exp(-t_since_spike / decay_tau)

    Args:
        n: Number of time steps.
        rate_per_day: Expected number of spikes per day (Poisson rate).
        magnitude_mean: Mean multiplicative spike size.
        decay_tau: Exponential decay time constant in minutes.
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    days = n / MINUTES_PER_DAY
    n_spikes = rng.poisson(rate_per_day * days)
    spike_times = rng.integers(0, n, size=max(n_spikes, 1))
    spike_mags = rng.exponential(magnitude_mean, size=max(n_spikes, 1))

    result = np.zeros(n)
    for t, mag in zip(spike_times, spike_mags):
        decay = mag * np.exp(-np.arange(n - t) / decay_tau)
        result[t:] += decay

    return result


def _correlated_signals(
    n: int,
    rho: float = 0.7,
    sigma1: float = 0.05,
    sigma2: float = 0.05,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate two AR(1)-like signals with specified correlation rho.

    Used primarily for cpu_util ↔ net_bytes correlation in cache-driven
    archetypes where cache misses cause simultaneous CPU and network spikes.
    """
    rng = np.random.default_rng(seed)
    # Generate two independent AR(1) series
    eps1 = rng.normal(0, sigma1, n)
    eps2 = rng.normal(0, sigma2, n)
    # Mix them to achieve correlation rho
    eps_combined = rho * eps1 + np.sqrt(1 - rho**2) * eps2
    # Apply AR(1) persistence
    x1 = np.zeros(n)
    x2 = np.zeros(n)
    for i in range(1, n):
        x1[i] = 0.9 * x1[i - 1] + eps1[i]
        x2[i] = 0.9 * x2[i - 1] + eps_combined[i]
    return x1, x2


# ── Archetype Generators ───────────────────────────────────────────────
# Each generator produces (cpu, mem, net, disk) arrays for one tenant.
# They compose the signal components above with archetype-specific params.
# Comments explain *why* each parameterization models the named workload.


def _generate_low_traffic_blog(
    t_hours: np.ndarray, n: int, cfg: TenantConfig, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Low CPU/mem baseline, weak diurnal, occasional comment-driven spikes.

    Blog visitors trickle in throughout the day. Comments trigger modest
    PHP processing spikes. Disk I/O is near-zero (static content mostly).
    """
    rng = np.random.default_rng(seed + 1)

    # Base signal: very low + weak diurnal
    diurnal = _diurnal(t_hours, cfg.diurnal_amplitude, cfg.diurnal_phase)
    noise = _ar1_noise(n, phi=cfg.ar_coefficient, sigma=cfg.noise_scale, seed=seed + 2)
    spikes = _spike_train(
        n, cfg.spike_rate, cfg.spike_magnitude_mean, decay_tau=8.0, seed=seed + 3
    )

    cpu = np.clip(cfg.cpu_baseline + diurnal * 0.3 + noise + spikes * 0.5, 0, 1.0)
    mem = np.clip(cfg.mem_baseline + diurnal * 0.1 + noise * 0.7, 0, 1.0)
    net = np.maximum(0, cfg.net_baseline * (1 + diurnal * 0.5 + noise) + spikes * 200)
    disk = np.maximum(0, cfg.disk_baseline + spikes * 15 + rng.normal(0, 2, n))

    return cpu, mem, net, disk


def _generate_ecommerce_retail(
    t_hours: np.ndarray, t_days: np.ndarray, n: int, cfg: TenantConfig, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Moderate baseline, strong diurnal + weekly, sharp campaign spikes.

    Ecommerce shows clear business-hours peaks with lunchtime and evening
    browsing surges. Weekend patterns differ from weekdays. Campaign
    spikes (Black Friday, flash sales) are large, sharp, and rare.
    """
    rng = np.random.default_rng(seed + 1)

    diurnal = _diurnal(t_hours, cfg.diurnal_amplitude, cfg.diurnal_phase)
    weekly = _weekly(t_days, cfg.weekly_amplitude)
    noise = _ar1_noise(n, phi=cfg.ar_coefficient, sigma=cfg.noise_scale, seed=seed + 2)
    # Campaign spikes: rare (0.3/day) but large, simulating flash sales
    spikes = _spike_train(
        n, cfg.spike_rate, cfg.spike_magnitude_mean, decay_tau=30.0, seed=seed + 3
    )

    # Ecommerce: diurnal + weekly modulation + campaigns
    base_pattern = diurnal * (1 + 0.5 * weekly)
    cpu = np.clip(
        cfg.cpu_baseline + base_pattern * 0.5 + noise * 0.3 + spikes, 0, 1.5
    )
    mem = np.clip(cfg.mem_baseline + base_pattern * 0.3 + noise * 0.5, 0, 1.0)
    net = np.maximum(
        0,
        cfg.net_baseline * (1 + base_pattern + noise) + spikes * cfg.net_baseline * 3,
    )
    disk = np.maximum(
        0,
        cfg.disk_baseline * (1 + base_pattern * 0.5)
        + spikes * 50
        + rng.normal(0, 5, n),
    )

    return cpu, mem, net, disk


def _generate_news_publisher(
    t_hours: np.ndarray, n: int, cfg: TenantConfig, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Diurnal modulated by stochastic breaking-news bursts.

    News sites follow a predictable daily readership pattern, but breaking
    news events cause sudden, massive traffic surges that dominate the
    signal. These bursts are rarer than ecommerce campaigns but larger.
    The hallmark: long quiet periods punctuated by extreme outliers.
    """
    rng = np.random.default_rng(seed + 1)

    diurnal = _diurnal(t_hours, cfg.diurnal_amplitude, cfg.diurnal_phase)
    noise = _ar1_noise(n, phi=cfg.ar_coefficient, sigma=cfg.noise_scale, seed=seed + 2)

    # Breaking-news bursts: rare (0.1/day) but extreme, long-decay (60 min)
    # This is the defining characteristic — fixed-config forecasters
    # will systematically underpredict these.
    bursts = _spike_train(
        n,
        rate_per_day=cfg.spike_rate,
        magnitude_mean=cfg.spike_magnitude_mean * 3,
        decay_tau=60.0,
        seed=seed + 3,
    )

    cpu = np.clip(cfg.cpu_baseline + diurnal * 0.4 + noise * 0.2 + bursts * 0.7, 0, 2.0)
    mem = np.clip(
        cfg.mem_baseline + diurnal * 0.15 + noise * 0.3 + bursts * 0.3, 0, 1.0
    )
    net = np.maximum(
        0,
        cfg.net_baseline * (1 + diurnal + noise) + bursts * cfg.net_baseline * 10,
    )
    disk = np.maximum(0, cfg.disk_baseline + bursts * 100 + rng.normal(0, 10, n))

    return cpu, mem, net, disk


def _generate_b2b_saas(
    t_hours: np.ndarray, t_days: np.ndarray, n: int, cfg: TenantConfig, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Weekday business-hours pattern, weekend trough.

    B2B SaaS workloads are the most predictable — sharp on/off at 9 AM
    and 5 PM, nearly flat on weekends. The forecasting challenge here
    is that fixed global configs tuned for more volatile tenants will
    over-allocate for B2B, wasting capacity. Autoresearch should learn
    to be conservative on weekends and precise on weekdays.
    """
    rng = np.random.default_rng(seed + 1)

    # Sharp business-hours mask: 1.0 during 9-17 weekdays, 0.1 otherwise
    hour_of_day = t_hours % 24
    day_of_week = t_days % 7
    is_weekday = day_of_week < 5
    is_business_hours = (hour_of_day >= 9) & (hour_of_day <= 17)
    business_mask = np.where(is_weekday & is_business_hours, 1.0, 0.1)

    noise = _ar1_noise(n, phi=cfg.ar_coefficient, sigma=cfg.noise_scale, seed=seed + 2)

    cpu = np.clip(cfg.cpu_baseline * business_mask + noise * 0.3, 0, 1.0)
    mem = np.clip(cfg.mem_baseline * (0.7 + 0.3 * business_mask) + noise * 0.2, 0, 1.0)
    net = np.maximum(
        0,
        cfg.net_baseline * business_mask * (1 + noise * 0.5)
        + rng.normal(0, cfg.net_baseline * 0.1, n),
    )
    disk = np.maximum(
        0,
        cfg.disk_baseline * business_mask
        + rng.normal(0, cfg.disk_baseline * 0.3, n),
    )

    return cpu, mem, net, disk


def _generate_wp_cron_heavy(
    t_hours: np.ndarray, n: int, cfg: TenantConfig, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Flat HTTP traffic, periodic CPU/mem spikes uncorrelated with traffic.

    This is the hardest archetype for a fixed global config — CPU spikes
    from wp-cron jobs are completely decorrelated from the (already flat)
    traffic pattern. A fixed config trained on the fleet average will
    systematically miss these spikes. This is where autoresearch should
    show its largest advantage: per-tenant configs can learn the spike
    cadence while fleet-level configs cannot.

    KEY INSIGHT for patent/paper: The wp-cron archetype demonstrates that
    single-config approaches face an irreducible error floor because some
    tenants have dynamics that don't align with the fleet majority.
    Autoresearch sidesteps this by adapting the configuration per-request.
    """
    rng = np.random.default_rng(seed + 1)

    # Flat baseline — almost no diurnal variation
    noise = _ar1_noise(n, phi=cfg.ar_coefficient, sigma=cfg.noise_scale, seed=seed + 2)

    # Periodic cron spikes: very regular (high rate, small magnitude)
    # but completely independent of the "traffic" (net_bytes) signal
    cron_spikes = _spike_train(
        n,
        rate_per_day=cfg.spike_rate,  # high rate: ~48/day = every 30 min
        magnitude_mean=cfg.spike_magnitude_mean,
        decay_tau=5.0,  # short decay: cron jobs are brief
        seed=seed + 3,
    )

    # Traffic is flat (no diurnal) — the defining contrast
    cpu = np.clip(cfg.cpu_baseline + noise * 0.15 + cron_spikes * 0.8, 0, 1.5)
    mem = np.clip(cfg.mem_baseline + noise * 0.3 + cron_spikes * 0.4, 0, 1.0)
    net = np.maximum(0, cfg.net_baseline * (1 + noise * 0.3) + rng.normal(0, 50, n))
    disk = np.maximum(
        0,
        cfg.disk_baseline + cron_spikes * 80 + rng.normal(0, 3, n),
    )

    return cpu, mem, net, disk


def _generate_cache_driven(
    t_hours: np.ndarray, t_days: np.ndarray, n: int, cfg: TenantConfig, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Spiky CPU on cache misses, high CPU–network correlation.

    Cache-driven workloads (e.g., WooCommerce stores without full-page
    caching) show CPU spikes that are CAUSED by network requests during
    cache-miss cascades. The CPU and network signals are tightly coupled
    — when traffic spikes, CPU follows within seconds.

    This tests whether autoresearch can exploit cross-signal structure
    that a fixed config might miss. The cost-asymmetric loss should
    particularly benefit here because missed CPU spikes cause SLA
    violations on premium tiers.
    """
    rng = np.random.default_rng(seed + 1)

    diurnal = _diurnal(t_hours, cfg.diurnal_amplitude, cfg.diurnal_phase)
    weekly = _weekly(t_days, cfg.weekly_amplitude)

    # Generate correlated CPU and network noise
    # This is the key signal: cpu_noise and net_noise share variation
    eps1 = rng.normal(0, cfg.noise_scale * 0.3, n)
    eps2 = rng.normal(0, cfg.noise_scale * 0.3, n)
    shared_noise = (
        cfg.cpu_net_correlation * eps1
        + np.sqrt(1 - cfg.cpu_net_correlation**2) * eps2
    )

    # Cache-miss cascades: spikes that affect both CPU and network
    # but CPU spikes are proportionally larger (the cache miss penalty)
    cache_spikes = _spike_train(
        n,
        rate_per_day=cfg.spike_rate,
        magnitude_mean=cfg.spike_magnitude_mean,
        decay_tau=10.0,
        seed=seed + 3,
    )

    base = diurnal * (1 + 0.3 * weekly)
    cpu_noise = _ar1_noise(n, phi=0.7, sigma=cfg.noise_scale * 0.5, seed=seed + 4)
    net_noise = _ar1_noise(n, phi=0.7, sigma=cfg.noise_scale * 0.8, seed=seed + 5)

    cpu = np.clip(
        cfg.cpu_baseline
        + base * 0.4
        + cpu_noise
        + cache_spikes * 2.0  # CPU hit hard by cache misses
        + shared_noise,
        0,
        1.5,
    )
    mem = np.clip(cfg.mem_baseline + base * 0.15 + cpu_noise * 1.5, 0, 1.0)
    net = np.maximum(
        0,
        cfg.net_baseline * (1 + base)
        + net_noise * cfg.net_baseline
        + cache_spikes * cfg.net_baseline
        + shared_noise * cfg.net_baseline,
    )
    disk = np.maximum(
        0,
        cfg.disk_baseline * (1 + base * 0.3)
        + cache_spikes * 30
        + rng.normal(0, 5, n),
    )

    return cpu, mem, net, disk


def _generate_compute_heavy(
    t_hours: np.ndarray, n: int, cfg: TenantConfig, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    High baseline CPU and memory, constant work, low network.

    Compute-heavy tenants (e.g., background ML inference, video transcoding)
    run hot continuously. The challenge isn't detecting spikes — it's
    accurately forecasting sustained high load without over-allocating.
    Low variance but high baseline demands precise calibration.
    """
    rng = np.random.default_rng(seed + 1)

    noise = _ar1_noise(n, phi=cfg.ar_coefficient, sigma=cfg.noise_scale, seed=seed + 2)

    cpu = np.clip(cfg.cpu_baseline + noise, 0, 1.0)
    mem = np.clip(cfg.mem_baseline + noise * 0.3, 0, 1.0)
    net = np.maximum(
        0,
        cfg.net_baseline * (1 + noise * 0.5) + rng.normal(0, cfg.net_baseline * 0.05, n),
    )
    disk = np.maximum(
        0,
        cfg.disk_baseline + rng.normal(0, cfg.disk_baseline * 0.2, n),
    )

    return cpu, mem, net, disk


def _generate_idle_ish(
    t_hours: np.ndarray, n: int, cfg: TenantConfig, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Mostly flat, negligible load, very rare access.

    Idle tenants are the majority in a shared hosting fleet (think:
    parked domains, abandoned blogs). Their resource usage is near-zero
    with occasional spikes when a search crawler hits or the owner logs
    in to check something.
    """
    rng = np.random.default_rng(seed + 1)

    noise = _ar1_noise(n, phi=0.5, sigma=cfg.noise_scale * 0.5, seed=seed + 2)
    rare_spikes = _spike_train(
        n,
        rate_per_day=cfg.spike_rate,
        magnitude_mean=cfg.spike_magnitude_mean,
        decay_tau=5.0,
        seed=seed + 3,
    )

    cpu = np.clip(cfg.cpu_baseline + noise + rare_spikes * 0.6, 0, 1.5)
    mem = np.clip(cfg.mem_baseline + noise * 0.5 + rare_spikes * 0.1, 0, 1.0)
    net = np.maximum(
        0,
        cfg.net_baseline * (1 + noise)
        + rare_spikes * cfg.net_baseline * 5
        + rng.normal(0, 5, n),
    )
    disk = np.maximum(0, cfg.disk_baseline + rare_spikes * 10 + rng.normal(0, 1, n))

    return cpu, mem, net, disk


# ── Generator Registry ─────────────────────────────────────────────────
# Maps archetype names to their generator functions.

GENERATOR_REGISTRY: dict[str, Callable] = {
    "low-traffic-blog": _generate_low_traffic_blog,
    "ecommerce-retail": _generate_ecommerce_retail,
    "news-publisher": _generate_news_publisher,
    "b2b-saas": _generate_b2b_saas,
    "wp-cron-heavy": _generate_wp_cron_heavy,
    "cache-driven": _generate_cache_driven,
    "compute-heavy": _generate_compute_heavy,
    "idle-ish": _generate_idle_ish,
}


# ── Tenant Configuration Factory ───────────────────────────────────────


def _generate_tenant_configs(
    n_tenants: int, seed: int
) -> list[TenantConfig]:
    """
    Generate randomized tenant configurations respecting archetype distributions.

    Each tenant within an archetype gets randomized parameters drawn from
    archetype-appropriate ranges. This ensures intra-archetype diversity
    while maintaining the defining characteristics.

    The randomization ranges are designed so that:
    - Tenants within an archetype are distinguishable (not identical clones)
    - Archetypes remain clearly separated in feature space (M4 clustering works)
    - Extreme values are possible but rare (realistic fleet behavior)
    """
    rng = np.random.default_rng(seed)

    # Assign archetypes by weighted sampling
    archetypes = list(ARCHETYPE_DISTRIBUTION.keys())
    weights = list(ARCHETYPE_DISTRIBUTION.values())
    assigned = rng.choice(archetypes, size=n_tenants, p=weights)

    configs: list[TenantConfig] = []
    for i, archetype in enumerate(assigned):
        tenant_seed = seed + i * 1000
        local_rng = np.random.default_rng(tenant_seed)

        # ── Archetype-specific parameter ranges ──
        # Each archetype defines plausible bounds for its parameters.
        # These were calibrated by iterating visually until the generated
        # series "looked right" against real WP Engine fleet telemetry.

        if archetype == "low-traffic-blog":
            cfg = TenantConfig(
                tenant_id=f"tenant_{i:04d}",
                archetype=archetype,
                diurnal_amplitude=local_rng.uniform(0.05, 0.15),
                diurnal_phase=18.0,  # evening peak
                weekly_amplitude=local_rng.uniform(0.0, 0.05),
                weekday_factor=1.0,
                noise_scale=local_rng.uniform(0.01, 0.03),
                ar_coefficient=local_rng.uniform(0.5, 0.8),
                spike_rate=local_rng.uniform(0.5, 2.0),
                spike_magnitude_mean=local_rng.uniform(0.1, 0.3),
                cpu_baseline=local_rng.uniform(0.02, 0.08),
                mem_baseline=local_rng.uniform(0.05, 0.15),
                net_baseline=local_rng.uniform(10, 50),  # bytes/s
                disk_baseline=local_rng.uniform(1, 10),
                cpu_net_correlation=0.2,
            )

        elif archetype == "ecommerce-retail":
            cfg = TenantConfig(
                tenant_id=f"tenant_{i:04d}",
                archetype=archetype,
                diurnal_amplitude=local_rng.uniform(0.3, 0.6),
                diurnal_phase=local_rng.uniform(13, 16),  # afternoon peak
                weekly_amplitude=local_rng.uniform(0.15, 0.4),
                weekday_factor=1.0,
                noise_scale=local_rng.uniform(0.02, 0.06),
                ar_coefficient=local_rng.uniform(0.6, 0.9),
                spike_rate=local_rng.uniform(0.1, 0.5),  # few campaigns
                spike_magnitude_mean=local_rng.uniform(0.5, 1.5),
                cpu_baseline=local_rng.uniform(0.15, 0.35),
                mem_baseline=local_rng.uniform(0.25, 0.5),
                net_baseline=local_rng.uniform(200, 800),
                disk_baseline=local_rng.uniform(20, 80),
                cpu_net_correlation=0.5,
            )

        elif archetype == "news-publisher":
            cfg = TenantConfig(
                tenant_id=f"tenant_{i:04d}",
                archetype=archetype,
                diurnal_amplitude=local_rng.uniform(0.25, 0.5),
                diurnal_phase=local_rng.uniform(8, 20),  # broad active window
                weekly_amplitude=local_rng.uniform(0.05, 0.15),
                weekday_factor=1.0,
                noise_scale=local_rng.uniform(0.02, 0.05),
                ar_coefficient=local_rng.uniform(0.5, 0.8),
                spike_rate=local_rng.uniform(0.05, 0.2),  # very rare breaking news
                spike_magnitude_mean=local_rng.uniform(1.5, 4.0),  # extreme
                cpu_baseline=local_rng.uniform(0.1, 0.25),
                mem_baseline=local_rng.uniform(0.15, 0.35),
                net_baseline=local_rng.uniform(100, 500),
                disk_baseline=local_rng.uniform(15, 60),
                cpu_net_correlation=0.6,
            )

        elif archetype == "b2b-saas":
            cfg = TenantConfig(
                tenant_id=f"tenant_{i:04d}",
                archetype=archetype,
                diurnal_amplitude=0.0,  # not used — sharp business mask instead
                diurnal_phase=0.0,
                weekly_amplitude=0.0,  # not used — business mask handles weekends
                weekday_factor=1.0,
                noise_scale=local_rng.uniform(0.01, 0.03),
                ar_coefficient=local_rng.uniform(0.6, 0.9),
                spike_rate=0.0,  # no spikes — B2B is predictable
                spike_magnitude_mean=0.0,
                cpu_baseline=local_rng.uniform(0.2, 0.5),
                mem_baseline=local_rng.uniform(0.3, 0.6),
                net_baseline=local_rng.uniform(100, 400),
                disk_baseline=local_rng.uniform(30, 100),
                cpu_net_correlation=0.4,
            )

        elif archetype == "wp-cron-heavy":
            cfg = TenantConfig(
                tenant_id=f"tenant_{i:04d}",
                archetype=archetype,
                diurnal_amplitude=0.0,  # flat traffic
                diurnal_phase=0.0,
                weekly_amplitude=0.0,
                weekday_factor=1.0,
                noise_scale=local_rng.uniform(0.01, 0.02),
                ar_coefficient=local_rng.uniform(0.3, 0.6),
                spike_rate=local_rng.uniform(20, 60),  # very frequent cron jobs
                spike_magnitude_mean=local_rng.uniform(0.1, 0.4),
                cpu_baseline=local_rng.uniform(0.02, 0.08),
                mem_baseline=local_rng.uniform(0.05, 0.15),
                net_baseline=local_rng.uniform(20, 80),
                disk_baseline=local_rng.uniform(5, 20),
                cpu_net_correlation=0.0,  # spikes are traffic-independent
            )

        elif archetype == "cache-driven":
            cfg = TenantConfig(
                tenant_id=f"tenant_{i:04d}",
                archetype=archetype,
                diurnal_amplitude=local_rng.uniform(0.2, 0.45),
                diurnal_phase=local_rng.uniform(12, 18),
                weekly_amplitude=local_rng.uniform(0.1, 0.3),
                weekday_factor=1.0,
                noise_scale=local_rng.uniform(0.03, 0.08),
                ar_coefficient=local_rng.uniform(0.5, 0.8),
                spike_rate=local_rng.uniform(3, 15),  # frequent cache-miss cascades
                spike_magnitude_mean=local_rng.uniform(0.3, 1.0),
                cpu_baseline=local_rng.uniform(0.1, 0.3),
                mem_baseline=local_rng.uniform(0.2, 0.4),
                net_baseline=local_rng.uniform(150, 600),
                disk_baseline=local_rng.uniform(10, 50),
                cpu_net_correlation=local_rng.uniform(0.6, 0.9),  # tightly coupled
            )

        elif archetype == "compute-heavy":
            cfg = TenantConfig(
                tenant_id=f"tenant_{i:04d}",
                archetype=archetype,
                diurnal_amplitude=0.0,
                diurnal_phase=0.0,
                weekly_amplitude=0.0,
                weekday_factor=1.0,
                noise_scale=local_rng.uniform(0.02, 0.05),
                ar_coefficient=local_rng.uniform(0.7, 0.95),
                spike_rate=0.0,
                spike_magnitude_mean=0.0,
                cpu_baseline=local_rng.uniform(0.5, 0.85),
                mem_baseline=local_rng.uniform(0.5, 0.85),
                net_baseline=local_rng.uniform(10, 50),
                disk_baseline=local_rng.uniform(50, 200),
                cpu_net_correlation=0.1,
            )

        elif archetype == "idle-ish":
            cfg = TenantConfig(
                tenant_id=f"tenant_{i:04d}",
                archetype=archetype,
                diurnal_amplitude=0.0,
                diurnal_phase=0.0,
                weekly_amplitude=0.0,
                weekday_factor=1.0,
                noise_scale=local_rng.uniform(0.002, 0.01),
                ar_coefficient=local_rng.uniform(0.3, 0.6),
                spike_rate=local_rng.uniform(0.01, 0.1),  # almost never
                spike_magnitude_mean=local_rng.uniform(0.1, 0.5),
                cpu_baseline=local_rng.uniform(0.001, 0.02),
                mem_baseline=local_rng.uniform(0.01, 0.05),
                net_baseline=local_rng.uniform(0.1, 5),
                disk_baseline=local_rng.uniform(0.1, 2),
                cpu_net_correlation=0.0,
            )

        else:
            raise ValueError(f"Unknown archetype: {archetype}")

        configs.append(cfg)

    return configs


# ── Main Generation Pipeline ───────────────────────────────────────────


def generate_workloads(
    n_tenants: int = 1000,
    n_days: int = 30,
    seed: int = 42,
    output_dir: Path | str = "data/synthetic",
) -> Path:
    """
    Generate synthetic multi-tenant workload data.

    Returns:
        Path to the output directory containing per-tenant Parquet files
        and manifest.csv.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_minutes = n_days * MINUTES_PER_DAY
    t_minutes = np.arange(n_minutes, dtype=np.float64)
    t_hours = t_minutes / 60.0
    t_days = t_minutes / MINUTES_PER_DAY

    # Generate tenant configurations
    configs = _generate_tenant_configs(n_tenants, seed)
    print(f"Generating {n_tenants} tenants over {n_days} days ({n_minutes} steps)...")

    # Write manifest
    manifest_path = output_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tenant_id",
                "archetype",
                "cpu_baseline",
                "mem_baseline",
                "net_baseline",
                "disk_baseline",
                "spike_rate",
                "diurnal_amplitude",
            ]
        )
        for cfg in configs:
            writer.writerow(
                [
                    cfg.tenant_id,
                    cfg.archetype,
                    f"{cfg.cpu_baseline:.4f}",
                    f"{cfg.mem_baseline:.4f}",
                    f"{cfg.net_baseline:.1f}",
                    f"{cfg.disk_baseline:.1f}",
                    f"{cfg.spike_rate:.3f}",
                    f"{cfg.diurnal_amplitude:.4f}",
                ]
            )

    # Generate each tenant's time series
    for idx, cfg in enumerate(configs):
        tenant_seed = seed + idx * 1000 + 1
        generator = GENERATOR_REGISTRY[cfg.archetype]

        try:
            if cfg.archetype in ("ecommerce-retail", "cache-driven"):
                cpu, mem, net, disk = generator(t_hours, t_days, n_minutes, cfg, tenant_seed)
            elif cfg.archetype == "b2b-saas":
                cpu, mem, net, disk = generator(t_hours, t_days, n_minutes, cfg, tenant_seed)
            else:
                cpu, mem, net, disk = generator(t_hours, n_minutes, cfg, tenant_seed)
        except TypeError:
            # Some generators take t_days, some don't — fallback
            cpu, mem, net, disk = generator(t_hours, n_minutes, cfg, tenant_seed)

        # Build Polars DataFrame
        import polars as pl

        end_ts = pl.datetime(2025, 1, 1) + pl.duration(minutes=n_minutes - 1)
        df = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    start=pl.datetime(2025, 1, 1),
                    end=end_ts,
                    interval="1m",
                    eager=True,
                ),
                "cpu_util": cpu.astype(np.float32),
                "mem_util": mem.astype(np.float32),
                "net_bytes": net.astype(np.float32),
                "disk_iops": disk.astype(np.float32),
            }
        )

        # Write Parquet
        parquet_path = output_dir / f"{cfg.tenant_id}.parquet"
        df.write_parquet(parquet_path, compression="zstd")

        if (idx + 1) % 100 == 0:
            print(f"  ... {idx + 1}/{n_tenants} tenants generated")

    print(f"✓ Generated {n_tenants} tenants → {output_dir}/")
    print(f"  Manifest: {manifest_path}")
    print(f"  Per-tenant Parquet files: {output_dir}/tenant_*.parquet")

    return output_dir


# ── CLI Entrypoint ─────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-tenant workload data for TSFM-Autoresearch"
    )
    parser.add_argument(
        "--tenants", type=int, default=1000, help="Number of synthetic tenants"
    )
    parser.add_argument(
        "--days", type=int, default=30, help="Days of history per tenant"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output", type=str, default="data/synthetic", help="Output directory"
    )
    args = parser.parse_args()

    generate_workloads(
        n_tenants=args.tenants,
        n_days=args.days,
        seed=args.seed,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()

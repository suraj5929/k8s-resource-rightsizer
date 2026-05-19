"""Stage 3 — predict expected usage ceiling per workload."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Project forward 7 days when trend is significant
PROJECTION_DAYS = 7
TREND_THRESHOLD = 0.001  # cores/hour — below this, ignore trend


def fit_and_predict(features_df: pd.DataFrame) -> pd.DataFrame:
    """Return features_df with four prediction columns appended.

    For stable workloads (|trend| <= TREND_THRESHOLD) uses simple percentile
    model (p95 → request, p99 → limit).

    For trending workloads uses QuantileRegressor to fit a proper quantile
    regression over time, giving a more accurate ceiling than a flat percentile
    projection. Falls back to the linear projection if sklearn is unavailable.

    Added columns:
        cpu_request_pred, cpu_limit_pred,
        mem_request_pred, mem_limit_pred
    """
    df = features_df.copy()

    df["cpu_request_pred"] = df.apply(
        lambda r: _predict(r, metric="cpu", quantile=0.95), axis=1)
    df["cpu_limit_pred"] = df.apply(
        lambda r: _predict(r, metric="cpu", quantile=0.99), axis=1)

    df["mem_request_pred"] = df.apply(
        lambda r: _predict(r, metric="mem", quantile=0.95), axis=1)
    df["mem_limit_pred"] = df.apply(
        lambda r: _predict(r, metric="mem", quantile=0.99), axis=1)

    return df


# ── per-row prediction ────────────────────────────────────────────────────────

def _predict(row: pd.Series, metric: str, quantile: float) -> float:
    """Choose percentile or QuantileRegressor based on trend significance."""
    p_col     = f"{metric}_p{int(quantile * 100)}"
    trend_col = f"{metric}_trend"
    base      = row[p_col]
    trend     = row[trend_col]

    if abs(trend) <= TREND_THRESHOLD:
        # Stable workload — flat percentile is accurate enough
        return base

    # Trending workload — try QuantileRegressor for a better fit
    timeseries = row.get(f"_{metric}_series")
    if timeseries is not None:
        try:
            return _quantile_regressor_predict(timeseries, quantile, trend)
        except Exception:
            pass  # fall through to linear projection

    # Linear projection fallback (no raw series stored, or sklearn unavailable)
    projected = base + trend * PROJECTION_DAYS * 24
    return max(base, projected)


def _quantile_regressor_predict(series: np.ndarray, quantile: float,
                                 trend: float) -> float:
    """Fit QuantileRegressor on the raw timeseries and predict the ceiling.

    Trains on the observed window, then predicts at now + PROJECTION_DAYS to
    get the expected quantile at that future point.

    Args:
        series:   1-D array of usage values (ordered by time)
        quantile: target quantile (0.95 or 0.99)
        trend:    pre-computed slope (used only for sign check)

    Returns:
        Predicted usage ceiling (always >= observed quantile)
    """
    from sklearn.linear_model import QuantileRegressor  # lazy import

    n = len(series)
    X = np.arange(n).reshape(-1, 1)
    y = series

    qr = QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs")
    qr.fit(X, y)

    # Predict at now + 7 days in the same time units (index steps)
    steps_per_day = max(1, n // 7)
    future_step = n + PROJECTION_DAYS * steps_per_day
    predicted = float(qr.predict([[future_step]])[0])

    # Never return below the observed quantile
    observed = float(np.percentile(series, quantile * 100))
    return max(observed, predicted)


# ── series injection (called by features.py) ─────────────────────────────────

def attach_series(features_df: pd.DataFrame,
                  raw_df: pd.DataFrame) -> pd.DataFrame:
    """Attach raw timeseries arrays to the features DataFrame so that
    fit_and_predict can use QuantileRegressor for trending workloads.

    This is an optional enrichment step. features_df works fine without it;
    trending workloads will fall back to linear projection.

    Args:
        features_df: output of engineer_features()
        raw_df:      original per-interval DataFrame from MetricScraper.fetch()

    Returns:
        features_df with ``_cpu_series`` and ``_mem_series`` columns added.
    """
    df = features_df.copy()
    cpu_series = {}
    mem_series = {}

    for workload, grp in raw_df.groupby("workload"):
        grp_sorted = grp.sort_values("timestamp")
        cpu_series[workload] = grp_sorted["cpu_cores"].to_numpy()
        mem_series[workload] = grp_sorted["memory_mib"].to_numpy()

    df["_cpu_series"] = df["workload"].map(cpu_series)
    df["_mem_series"] = df["workload"].map(mem_series)
    return df

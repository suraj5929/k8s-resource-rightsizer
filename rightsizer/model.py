"""Stage 3 — predict expected usage ceiling per workload."""
from __future__ import annotations

import pandas as pd

# Project forward 7 days when trend is significant
PROJECTION_DAYS = 7
TREND_THRESHOLD = 0.001  # cores/hour — below this, ignore trend


def fit_and_predict(features_df: pd.DataFrame) -> pd.DataFrame:
    """Return features_df with four prediction columns appended.

    Added columns:
        cpu_request_pred, cpu_limit_pred,
        mem_request_pred, mem_limit_pred
    """
    df = features_df.copy()

    df["cpu_request_pred"] = df.apply(_cpu_request, axis=1)
    df["cpu_limit_pred"]   = df["cpu_p99"]

    df["mem_request_pred"] = df.apply(_mem_request, axis=1)
    df["mem_limit_pred"]   = df["mem_p99"]

    return df


# ── per-row helpers ───────────────────────────────────────────────────────────

def _cpu_request(row: pd.Series) -> float:
    base = row["cpu_p95"]
    if row["cpu_trend"] > TREND_THRESHOLD:
        projected = base + row["cpu_trend"] * PROJECTION_DAYS * 24
        return max(base, projected)
    return base


def _mem_request(row: pd.Series) -> float:
    base = row["mem_p95"]
    if row["mem_trend"] > TREND_THRESHOLD:
        projected = base + row["mem_trend"] * PROJECTION_DAYS * 24
        return max(base, projected)
    return base

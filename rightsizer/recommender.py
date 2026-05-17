"""Stage 4 — apply safety headroom, enforce minimums, flag anomalies."""
from __future__ import annotations

import math

import pandas as pd

# Headroom multipliers
CPU_REQUEST_HEADROOM  = 1.10
CPU_LIMIT_HEADROOM    = 1.30
MEM_REQUEST_HEADROOM  = 1.15
MEM_LIMIT_HEADROOM    = 1.40

# Hard minimums
MIN_CPU_REQUEST_CORES = 0.010   # 10m
MIN_MEM_REQUEST_MIB   = 32.0    # 32Mi

OOM_MEM_LIMIT_BUMP    = 1.25    # extra 25% when OOM kills exist


def compute_recommendations(model_df: pd.DataFrame) -> pd.DataFrame:
    """Return model_df with recommendation columns appended.

    Added columns:
        rec_cpu_request, rec_cpu_limit,
        rec_mem_request, rec_mem_limit,
        oom_flag, savings_note
    """
    df = model_df.copy()

    df["_cpu_req_cores"] = (df["cpu_request_pred"] * CPU_REQUEST_HEADROOM
                            ).clip(lower=MIN_CPU_REQUEST_CORES)
    df["_cpu_lim_cores"] = df["cpu_limit_pred"] * CPU_LIMIT_HEADROOM
    # limit must be >= request
    df["_cpu_lim_cores"] = df[["_cpu_req_cores", "_cpu_lim_cores"]].max(axis=1)

    df["_mem_req_mib"] = (df["mem_request_pred"] * MEM_REQUEST_HEADROOM
                          ).clip(lower=MIN_MEM_REQUEST_MIB)
    df["_mem_lim_mib"] = df["mem_limit_pred"] * MEM_LIMIT_HEADROOM

    # OOM override
    oom_mask = df["oom_count"] > 0
    df.loc[oom_mask, "_mem_lim_mib"] *= OOM_MEM_LIMIT_BUMP

    # limit must be >= request
    df["_mem_lim_mib"] = df[["_mem_req_mib", "_mem_lim_mib"]].max(axis=1)

    # Human-readable Kubernetes units
    df["rec_cpu_request"] = df["_cpu_req_cores"].apply(cores_to_millicores)
    df["rec_cpu_limit"]   = df["_cpu_lim_cores"].apply(cores_to_millicores)
    df["rec_mem_request"] = df["_mem_req_mib"].apply(mib_to_k8s)
    df["rec_mem_limit"]   = df["_mem_lim_mib"].apply(mib_to_k8s)

    df["oom_flag"]     = oom_mask
    df["savings_note"] = df.apply(_savings_note, axis=1)

    return df.drop(columns=["_cpu_req_cores", "_cpu_lim_cores",
                             "_mem_req_mib", "_mem_lim_mib"])


# ── unit converters ───────────────────────────────────────────────────────────

def cores_to_millicores(cores: float) -> str:
    return f"{int(round(cores * 1000))}m"


def mib_to_k8s(mib: float) -> str:
    if mib >= 1024:
        # Round up to nearest 0.1 Gi to never under-provision
        gi = math.ceil(mib / 1024 * 10) / 10
        return f"{gi}Gi"
    return f"{int(math.ceil(mib))}Mi"


# ── savings note ──────────────────────────────────────────────────────────────

def _savings_note(row: pd.Series) -> str:
    notes = []
    if row["oom_count"] > 0:
        notes.append(f"OOM kills detected ({int(row['oom_count'])}); memory limit bumped +25%")
    if row["cpu_trend"] > 0.001:
        notes.append(
            f"CPU trending up ({row['cpu_trend']:.4f} cores/hr); "
            "request projected 7 days forward"
        )
    if row["mem_trend"] > 0.001:
        notes.append(
            f"Memory trending up ({row['mem_trend']:.4f} MiB/hr); "
            "request projected 7 days forward"
        )
    return "; ".join(notes) if notes else "stable"

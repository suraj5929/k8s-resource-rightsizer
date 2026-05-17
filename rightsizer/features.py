"""Stage 2 — engineer per-workload feature vectors from raw timeseries."""
from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw timeseries into one feature row per workload.

    Returns a DataFrame with columns:
        workload, namespace, container,
        cpu_p50, cpu_p95, cpu_p99, cpu_max, cpu_stddev, cpu_trend,
        mem_p50, mem_p95, mem_p99, mem_max, mem_stddev, mem_trend,
        oom_count
    """
    records = []
    for (workload, namespace, container), group in df.groupby(
        ["workload", "namespace", "container"]
    ):
        group = group.sort_values("timestamp")
        cpu = group["cpu_cores"].values
        mem = group["memory_mib"].values
        ts = group["timestamp"]

        hours = (ts - ts.min()).dt.total_seconds().values / 3600

        cpu_trend = _slope(hours, cpu)
        mem_trend = _slope(hours, mem)

        records.append({
            "workload":  workload,
            "namespace": namespace,
            "container": container,
            # CPU features
            "cpu_p50":    float(np.percentile(cpu, 50)),
            "cpu_p95":    float(np.percentile(cpu, 95)),
            "cpu_p99":    float(np.percentile(cpu, 99)),
            "cpu_max":    float(cpu.max()),
            "cpu_stddev": float(cpu.std()),
            "cpu_trend":  float(cpu_trend),
            # Memory features
            "mem_p50":    float(np.percentile(mem, 50)),
            "mem_p95":    float(np.percentile(mem, 95)),
            "mem_p99":    float(np.percentile(mem, 99)),
            "mem_max":    float(mem.max()),
            "mem_stddev": float(mem.std()),
            "mem_trend":  float(mem_trend),
            # OOM events
            "oom_count":  int(group["oom_kill"].sum()),
        })

    return pd.DataFrame(records).reset_index(drop=True)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    """Return linear slope (y per unit x) via least-squares fit."""
    if len(x) < 2 or x.max() == x.min():
        return 0.0
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)

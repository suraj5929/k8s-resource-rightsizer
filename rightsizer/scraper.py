"""Stage 1 — abstract data source: Prometheus or local CSV."""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests


REQUIRED_COLS = {"timestamp", "workload", "namespace", "container",
                 "cpu_cores", "memory_mib", "oom_kill"}


class MetricScraper:
    """Fetch workload metrics from Prometheus or a local CSV file."""

    def __init__(self, source: str, data_dir: str = "data/",
                 prometheus_url: str = "http://localhost:9090",
                 lookback_days: int = 7, namespace_filter: str | None = None):
        if source not in ("csv", "prometheus"):
            raise ValueError(f"source must be 'csv' or 'prometheus', got {source!r}")
        self.source = source
        self.data_dir = data_dir
        self.prometheus_url = prometheus_url.rstrip("/")
        self.lookback_days = lookback_days
        self.namespace_filter = namespace_filter

    # ── public ────────────────────────────────────────────────────────────────

    def fetch(self) -> pd.DataFrame:
        if self.source == "csv":
            df = self._load_csv()
        else:
            df = self._query_prometheus()

        self._validate(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        if self.namespace_filter:
            df = df[df["namespace"] == self.namespace_filter]
            if df.empty:
                raise ValueError(f"No data found for namespace {self.namespace_filter!r}")

        return df.reset_index(drop=True)

    # ── CSV mode ──────────────────────────────────────────────────────────────

    def _load_csv(self) -> pd.DataFrame:
        path = os.path.join(self.data_dir, "metrics.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"metrics.csv not found at {path}. "
                "Run: python scripts/generate_data.py"
            )
        df = pd.read_csv(path)
        return df

    # ── Prometheus mode ───────────────────────────────────────────────────────

    def _query_prometheus(self) -> pd.DataFrame:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.lookback_days)
        step = "5m"

        ns_regex = self.namespace_filter or "prod|batch|ml|infra"

        cpu_query = (
            f'rate(container_cpu_usage_seconds_total'
            f'{{namespace=~"{ns_regex}",container!=""}}[5m])'
        )
        mem_query = (
            f'container_memory_working_set_bytes'
            f'{{namespace=~"{ns_regex}",container!=""}}'
        )
        oom_query = (
            f'kube_pod_container_status_last_terminated_reason'
            f'{{reason="OOMKilled",namespace=~"{ns_regex}"}}'
        )

        cpu_df = self._range_query(cpu_query, start, end, step, value_col="cpu_cores")
        mem_df = self._range_query(mem_query, start, end, step, value_col="memory_mib",
                                   scale=1 / (1024 ** 2))

        # OOM kills: sum per workload over the window
        oom_df = self._instant_query(oom_query)

        df = cpu_df.merge(mem_df, on=["timestamp", "workload", "namespace", "container"],
                          how="outer")
        df = df.merge(oom_df, on=["workload", "namespace", "container"], how="left")
        df["oom_kill"] = df["oom_kill"].fillna(0).astype(int)
        df[["cpu_cores", "memory_mib"]] = df[["cpu_cores", "memory_mib"]].fillna(0)
        return df

    def _range_query(self, query: str, start: datetime, end: datetime,
                     step: str, value_col: str, scale: float = 1.0) -> pd.DataFrame:
        params = {
            "query": query,
            "start": start.timestamp(),
            "end": end.timestamp(),
            "step": step,
        }
        resp = requests.get(f"{self.prometheus_url}/api/v1/query_range",
                            params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()["data"]["result"]

        rows = []
        for series in data:
            labels = series["metric"]
            workload = labels.get("deployment") or labels.get("pod", "unknown")
            namespace = labels.get("namespace", "unknown")
            container = labels.get("container", "unknown")
            for ts, val in series["values"]:
                rows.append({
                    "timestamp": datetime.fromtimestamp(float(ts), tz=timezone.utc).isoformat(),
                    "workload":  workload,
                    "namespace": namespace,
                    "container": container,
                    value_col:   float(val) * scale,
                })
        return pd.DataFrame(rows)

    def _instant_query(self, query: str) -> pd.DataFrame:
        resp = requests.get(f"{self.prometheus_url}/api/v1/query",
                            params={"query": query}, timeout=30)
        resp.raise_for_status()
        data = resp.json()["data"]["result"]

        rows = []
        for series in data:
            labels = series["metric"]
            workload = labels.get("deployment") or labels.get("pod", "unknown")
            rows.append({
                "workload":  workload,
                "namespace": labels.get("namespace", "unknown"),
                "container": labels.get("container", "unknown"),
                "oom_kill":  int(float(series["value"][1])),
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["workload", "namespace", "container", "oom_kill"])

    # ── validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

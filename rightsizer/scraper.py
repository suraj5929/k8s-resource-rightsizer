"""Stage 1 — abstract data source: Prometheus or local CSV."""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone, timedelta

import pandas as pd
import requests


REQUIRED_COLS = {"timestamp", "workload", "namespace", "container",
                 "cpu_cores", "memory_mib", "oom_kill"}

# Deployment pods:  name-<9-10 hex RS hash>-<5 alphanum pod hash>
_DEPLOY_POD_RE = re.compile(r"^(.+)-[0-9a-f]{9,10}-[a-z0-9]{5}$")
# StatefulSet / DaemonSet pods:  name-<index or short hash>
_SS_POD_RE = re.compile(r"^(.+)-[0-9a-z]+$")


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
        return pd.read_csv(path)

    # ── Prometheus mode ───────────────────────────────────────────────────────

    def _query_prometheus(self) -> pd.DataFrame:
        self._health_check()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.lookback_days)
        step = "5m"

        # Use exact match for single namespace, regex for multi-namespace default
        if self.namespace_filter:
            ns_sel = f'namespace="{self.namespace_filter}"'
        else:
            ns_sel = 'namespace=~"prod|batch|ml|infra"'

        cpu_query = (
            f'rate(container_cpu_usage_seconds_total'
            f'{{{ns_sel},container!="",container!="POD"}}[5m])'
        )
        mem_query = (
            f'container_memory_working_set_bytes'
            f'{{{ns_sel},container!="",container!="POD"}}'
        )
        # count_over_time returns how many samples exist (i.e. how many scrape
        # intervals the pod's last termination was OOMKilled) — a reliable proxy
        # for OOM kill frequency over the lookback window.
        oom_query = (
            f'count_over_time('
            f'kube_pod_container_status_last_terminated_reason'
            f'{{reason="OOMKilled",{ns_sel}}}[{self.lookback_days}d])'
        )

        cpu_df = self._range_query(cpu_query, start, end, step, value_col="cpu_cores")
        mem_df = self._range_query(mem_query, start, end, step, value_col="memory_mib",
                                   scale=1 / (1024 ** 2))

        if cpu_df.empty and mem_df.empty:
            raise ValueError(
                "No CPU or memory metrics returned from Prometheus. "
                f"Checked namespaces: {ns_sel}. "
                "Verify the namespace selector and that cAdvisor metrics are present."
            )

        oom_df = self._instant_query_oom(oom_query)

        df = cpu_df.merge(mem_df, on=["timestamp", "workload", "namespace", "container"],
                          how="outer")
        df = df.merge(oom_df, on=["workload", "namespace", "container"], how="left")
        # oom_kill is a per-row binary flag (0/1); the Prometheus instant query
        # returns a window count, so cap at 1 to match the CSV schema.
        df["oom_kill"] = (df["oom_kill"].fillna(0) > 0).astype(int)
        df[["cpu_cores", "memory_mib"]] = df[["cpu_cores", "memory_mib"]].fillna(0)
        return df

    def _health_check(self) -> None:
        """Verify Prometheus is reachable before running queries."""
        try:
            resp = requests.get(f"{self.prometheus_url}/-/healthy", timeout=5)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Prometheus at {self.prometheus_url}. "
                "Ensure it is running or use --source csv."
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Prometheus at {self.prometheus_url} timed out after 5s. "
                "Check network connectivity."
            )
        except requests.exceptions.HTTPError as exc:
            raise RuntimeError(f"Prometheus health check failed: {exc}") from exc

    def _range_query(self, query: str, start: datetime, end: datetime,
                     step: str, value_col: str, scale: float = 1.0) -> pd.DataFrame:
        params = {
            "query": query,
            "start": start.timestamp(),
            "end":   end.timestamp(),
            "step":  step,
        }
        try:
            resp = requests.get(f"{self.prometheus_url}/api/v1/query_range",
                                params=params, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Prometheus range query failed: {exc}") from exc

        result = resp.json().get("data", {}).get("result", [])

        rows = []
        for series in result:
            labels = series["metric"]
            workload  = self._extract_workload(labels)
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

        if not rows:
            return pd.DataFrame(
                columns=["timestamp", "workload", "namespace", "container", value_col])
        return pd.DataFrame(rows)

    def _instant_query_oom(self, query: str) -> pd.DataFrame:
        """Run an instant query and return per-workload OOM counts."""
        try:
            resp = requests.get(f"{self.prometheus_url}/api/v1/query",
                                params={"query": query}, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Prometheus instant query failed: {exc}") from exc

        result = resp.json().get("data", {}).get("result", [])

        rows = []
        for series in result:
            labels = series["metric"]
            rows.append({
                "workload":  self._extract_workload(labels),
                "namespace": labels.get("namespace", "unknown"),
                "container": labels.get("container", "unknown"),
                "oom_kill":  int(float(series["value"][1])),
            })
        return (pd.DataFrame(rows) if rows
                else pd.DataFrame(columns=["workload", "namespace", "container", "oom_kill"]))

    @staticmethod
    def _extract_workload(labels: dict) -> str:
        """Derive deployment/workload name from Prometheus metric labels.

        Priority:
          1. ``deployment`` label (set by kube-prometheus-stack recording rules)
          2. ``workload`` label (custom exporters / Istio)
          3. Strip generated suffix from ``pod`` label
        """
        if labels.get("deployment"):
            return labels["deployment"]
        if labels.get("workload"):
            return labels["workload"]
        pod = labels.get("pod", "")
        if pod:
            return MetricScraper._strip_pod_suffix(pod)
        return "unknown"

    @staticmethod
    def _strip_pod_suffix(pod: str) -> str:
        """Strip k8s-generated suffix from a pod name to recover workload name.

        - Deployment:   ``payment-service-7d9f4b5c9-xkvpn``  →  ``payment-service``
        - StatefulSet:  ``postgres-0``                        →  ``postgres``
        - DaemonSet:    ``fluentd-a1b2c``                     →  ``fluentd``
        """
        m = _DEPLOY_POD_RE.match(pod)
        if m:
            return m.group(1)
        m = _SS_POD_RE.match(pod)
        if m:
            return m.group(1)
        return pod

    # ── validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate(df: pd.DataFrame) -> None:
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

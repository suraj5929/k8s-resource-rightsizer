"""Dedicated tests for Prometheus live mode (MetricScraper + scraper internals)."""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
import requests

from rightsizer.scraper import MetricScraper


# ── mock helpers ──────────────────────────────────────────────────────────────

TS = 1_700_000_000.0  # fixed Unix timestamp for deterministic tests


def _make_range_result(pod: str, namespace: str, container: str,
                       values: list[tuple[float, str]]) -> dict:
    return {
        "metric": {"namespace": namespace, "pod": pod, "container": container},
        "values": [[ts, v] for ts, v in values],
    }


def _make_instant_result(pod: str, namespace: str, container: str,
                         count: int) -> dict:
    return {
        "metric": {
            "namespace": namespace,
            "pod": pod,
            "container": container,
            "reason": "OOMKilled",
        },
        "value": [TS, str(count)],
    }


def _mock_responses(cpu_results: list, mem_results: list,
                    oom_results: list | None = None):
    """Return a requests.get side_effect covering health, cpu range,
    mem range, and OOM instant queries in order."""
    oom_results = oom_results or []
    call_n = {"n": 0}

    def side_effect(url, **kwargs):
        m = MagicMock()
        m.raise_for_status = MagicMock()

        if url.endswith("/-/healthy"):
            m.json.return_value = {}
            return m

        call_n["n"] += 1
        if "query_range" in url:
            if call_n["n"] == 1:
                m.json.return_value = {"data": {"result": cpu_results}}
            else:
                m.json.return_value = {"data": {"result": mem_results}}
        else:  # /api/v1/query
            m.json.return_value = {"data": {"result": oom_results}}
        return m

    return side_effect


# ── schema & basic correctness ────────────────────────────────────────────────

class TestPrometheusSchema:
    def test_returns_required_columns(self):
        cpu = [_make_range_result(
            "svc-abc12def9-xkvpn", "prod", "svc",
            [(TS, "0.18"), (TS + 300, "0.20")])]
        mem = [_make_range_result(
            "svc-abc12def9-xkvpn", "prod", "svc",
            [(TS, str(256 * 1024 ** 2)), (TS + 300, str(260 * 1024 ** 2))])]

        with patch("requests.get", side_effect=_mock_responses(cpu, mem)):
            df = MetricScraper(source="prometheus",
                               prometheus_url="http://fake:9090",
                               namespace_filter="prod").fetch()

        required = {"timestamp", "workload", "namespace", "container",
                    "cpu_cores", "memory_mib", "oom_kill"}
        assert required.issubset(set(df.columns))

    def test_memory_converted_from_bytes_to_mib(self):
        bytes_val = 512 * 1024 * 1024  # 512 MiB in bytes
        cpu = [_make_range_result("svc-abc12def9-xkvpn", "prod", "svc",
                                  [(TS, "0.1")])]
        mem = [_make_range_result("svc-abc12def9-xkvpn", "prod", "svc",
                                  [(TS, str(bytes_val))])]

        with patch("requests.get", side_effect=_mock_responses(cpu, mem)):
            df = MetricScraper(source="prometheus",
                               prometheus_url="http://fake:9090",
                               namespace_filter="prod").fetch()

        assert abs(df["memory_mib"].iloc[0] - 512.0) < 0.1

    def test_namespace_filter_applied(self):
        cpu = [
            _make_range_result("svc-abc12def9-xkvpn", "prod", "svc",
                               [(TS, "0.1")]),
            _make_range_result("other-abc12def9-xkvpn", "staging", "other",
                               [(TS, "0.2")]),
        ]
        mem = [
            _make_range_result("svc-abc12def9-xkvpn", "prod", "svc",
                               [(TS, str(128 * 1024 ** 2))]),
            _make_range_result("other-abc12def9-xkvpn", "staging", "other",
                               [(TS, str(64 * 1024 ** 2))]),
        ]

        with patch("requests.get", side_effect=_mock_responses(cpu, mem)):
            df = MetricScraper(source="prometheus",
                               prometheus_url="http://fake:9090",
                               namespace_filter="prod").fetch()

        assert set(df["namespace"].unique()) == {"prod"}


# ── workload name extraction ──────────────────────────────────────────────────

class TestWorkloadExtraction:
    @pytest.mark.parametrize("pod,expected", [
        # Deployment (9-char RS hash + 5-char pod hash)
        ("payment-service-7d9f4b5c9-xkvpn", "payment-service"),
        # Deployment with hyphenated name
        ("my-cool-app-abc12def9-zxcvb",     "my-cool-app"),
        # StatefulSet (numeric index)
        ("postgres-0",                       "postgres"),
        ("kafka-2",                          "kafka"),
        # DaemonSet (short hash suffix)
        ("fluentd-a1b2c",                    "fluentd"),
        # No suffix
        ("standalone",                       "standalone"),
    ])
    def test_strip_pod_suffix(self, pod, expected):
        assert MetricScraper._strip_pod_suffix(pod) == expected

    def test_prefers_deployment_label(self):
        labels = {
            "deployment": "my-app",
            "pod": "my-app-abc12def9-xkvpn",
            "namespace": "prod",
        }
        assert MetricScraper._extract_workload(labels) == "my-app"

    def test_prefers_workload_label_over_pod(self):
        labels = {
            "workload": "my-app",
            "pod": "my-app-abc12def9-xkvpn",
            "namespace": "prod",
        }
        assert MetricScraper._extract_workload(labels) == "my-app"

    def test_falls_back_to_pod_stripping(self):
        labels = {"pod": "payment-service-7d9f4b5c9-xkvpn", "namespace": "prod"}
        assert MetricScraper._extract_workload(labels) == "payment-service"

    def test_unknown_when_no_labels(self):
        assert MetricScraper._extract_workload({}) == "unknown"


# ── OOM handling ──────────────────────────────────────────────────────────────

class TestOOMHandling:
    def test_oom_flag_set_when_count_nonzero(self):
        cpu = [_make_range_result("svc-abc12def9-xkvpn", "prod", "svc",
                                  [(TS, "0.1"), (TS + 300, "0.1")])]
        mem = [_make_range_result("svc-abc12def9-xkvpn", "prod", "svc",
                                  [(TS, str(128 * 1024 ** 2)),
                                   (TS + 300, str(128 * 1024 ** 2))])]
        oom = [_make_instant_result("svc-abc12def9-xkvpn", "prod", "svc", 5)]

        with patch("requests.get", side_effect=_mock_responses(cpu, mem, oom)):
            df = MetricScraper(source="prometheus",
                               prometheus_url="http://fake:9090",
                               namespace_filter="prod").fetch()

        assert df["oom_kill"].max() == 1  # capped to binary flag
        assert df["oom_kill"].sum() > 0

    def test_oom_zero_when_no_kills(self):
        cpu = [_make_range_result("svc-abc12def9-xkvpn", "prod", "svc",
                                  [(TS, "0.1")])]
        mem = [_make_range_result("svc-abc12def9-xkvpn", "prod", "svc",
                                  [(TS, str(128 * 1024 ** 2))])]

        with patch("requests.get", side_effect=_mock_responses(cpu, mem, [])):
            df = MetricScraper(source="prometheus",
                               prometheus_url="http://fake:9090",
                               namespace_filter="prod").fetch()

        assert df["oom_kill"].sum() == 0

    def test_oom_zero_count_is_not_flagged(self):
        cpu = [_make_range_result("svc-abc12def9-xkvpn", "prod", "svc",
                                  [(TS, "0.1")])]
        mem = [_make_range_result("svc-abc12def9-xkvpn", "prod", "svc",
                                  [(TS, str(128 * 1024 ** 2))])]
        oom = [_make_instant_result("svc-abc12def9-xkvpn", "prod", "svc", 0)]

        with patch("requests.get", side_effect=_mock_responses(cpu, mem, oom)):
            df = MetricScraper(source="prometheus",
                               prometheus_url="http://fake:9090",
                               namespace_filter="prod").fetch()

        assert df["oom_kill"].sum() == 0


# ── error handling ────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_connection_refused_raises_connection_error(self):
        with patch("requests.get",
                   side_effect=requests.exceptions.ConnectionError("refused")):
            scraper = MetricScraper(source="prometheus",
                                    prometheus_url="http://unreachable:9090")
            with pytest.raises(ConnectionError, match="Cannot connect to Prometheus"):
                scraper.fetch()

    def test_timeout_raises_timeout_error(self):
        with patch("requests.get",
                   side_effect=requests.exceptions.Timeout("timed out")):
            scraper = MetricScraper(source="prometheus",
                                    prometheus_url="http://slow:9090")
            with pytest.raises(TimeoutError, match="timed out"):
                scraper.fetch()

    def test_empty_cpu_and_mem_raises_value_error(self):
        with patch("requests.get", side_effect=_mock_responses([], [])):
            scraper = MetricScraper(source="prometheus",
                                    prometheus_url="http://fake:9090")
            with pytest.raises(ValueError, match="No CPU or memory metrics"):
                scraper.fetch()

    def test_invalid_source_raises_value_error(self):
        with pytest.raises(ValueError, match="source must be"):
            MetricScraper(source="kafka")

    def test_namespace_filter_no_match_raises(self):
        cpu = [_make_range_result("svc-abc12def9-xkvpn", "prod", "svc",
                                  [(TS, "0.1")])]
        mem = [_make_range_result("svc-abc12def9-xkvpn", "prod", "svc",
                                  [(TS, str(128 * 1024 ** 2))])]

        with patch("requests.get", side_effect=_mock_responses(cpu, mem)):
            scraper = MetricScraper(source="prometheus",
                                    prometheus_url="http://fake:9090",
                                    namespace_filter="staging")
            with pytest.raises(ValueError, match="No data found for namespace"):
                scraper.fetch()

    def test_http_error_on_health_check_raises_runtime_error(self):
        def side_effect(url, **kwargs):
            m = MagicMock()
            m.raise_for_status.side_effect = requests.exceptions.HTTPError("503")
            return m

        with patch("requests.get", side_effect=side_effect):
            scraper = MetricScraper(source="prometheus",
                                    prometheus_url="http://fake:9090")
            with pytest.raises(RuntimeError, match="health check failed"):
                scraper.fetch()

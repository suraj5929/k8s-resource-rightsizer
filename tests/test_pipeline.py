"""End-to-end tests covering the plan's testing checklist."""
import io
import os
import textwrap
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import requests
import yaml

from rightsizer.scraper import MetricScraper
from rightsizer.features import engineer_features
from rightsizer.model import fit_and_predict
from rightsizer.recommender import (
    compute_recommendations,
    MIN_CPU_REQUEST_CORES,
    MIN_MEM_REQUEST_MIB,
)
from rightsizer.patcher import render_patches


# ── fixtures ─────────────────────────────────────────────────────────────────

def _make_raw_df(n_points: int = 100) -> pd.DataFrame:
    """Minimal synthetic DataFrame matching the required schema."""
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2026-01-01T00:00:00", tz="UTC")
    timestamps = pd.date_range(base, periods=n_points, freq="5min")

    rows = []
    workloads = [
        # (name, ns, container, cpu_mean, mem_mean, oom_rate)
        ("payment-service",  "prod",  "payment-service",  0.18, 220, 0.0),
        ("notification-svc", "prod",  "notification-svc", 0.004, 40, 0.0),
        ("ml-inference",     "ml",    "ml-inference",     0.50, 900, 0.5),
    ]
    for name, ns, container, cpu_mean, mem_mean, oom_rate in workloads:
        cpu = rng.normal(cpu_mean, cpu_mean * 0.1, n_points).clip(0.001)
        mem = rng.normal(mem_mean, mem_mean * 0.05, n_points).clip(10)
        oom = (rng.random(n_points) < oom_rate).astype(int)
        for i in range(n_points):
            rows.append({
                "timestamp":  timestamps[i],
                "workload":   name,
                "namespace":  ns,
                "container":  container,
                "cpu_cores":  float(cpu[i]),
                "memory_mib": float(mem[i]),
                "oom_kill":   int(oom[i]),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def raw_df():
    return _make_raw_df()


@pytest.fixture(scope="module")
def features_df(raw_df):
    return engineer_features(raw_df)


@pytest.fixture(scope="module")
def model_df(features_df):
    return fit_and_predict(features_df)


@pytest.fixture(scope="module")
def rec_df(model_df):
    return compute_recommendations(model_df)


# ── Stage 1: scraper ──────────────────────────────────────────────────────────

def test_scraper_loads_csv(tmp_path):
    raw = _make_raw_df(50)
    # write a minimal CSV
    raw["timestamp"] = raw["timestamp"].astype(str)
    raw.to_csv(tmp_path / "metrics.csv", index=False)

    scraper = MetricScraper(source="csv", data_dir=str(tmp_path))
    df = scraper.fetch()

    expected_cols = {"timestamp", "workload", "namespace", "container",
                     "cpu_cores", "memory_mib", "oom_kill"}
    assert expected_cols.issubset(set(df.columns))
    assert len(df) == 150  # 3 workloads × 50 rows


def test_scraper_missing_csv_raises(tmp_path):
    scraper = MetricScraper(source="csv", data_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        scraper.fetch()


def test_scraper_invalid_source():
    with pytest.raises(ValueError):
        MetricScraper(source="kafka")


# ── Stage 1: Prometheus mode ──────────────────────────────────────────────────

def _mock_get(healthy_url, cpu_result=None, mem_result=None, oom_result=None):
    """Return a side_effect function that routes mocked responses by URL."""
    ts = 1700000000.0

    def _range_response(result):
        m = MagicMock()
        m.raise_for_status = MagicMock()
        m.json.return_value = {"data": {"result": result}}
        return m

    def _instant_response(result):
        m = MagicMock()
        m.raise_for_status = MagicMock()
        m.json.return_value = {"data": {"result": result}}
        return m

    default_cpu = [
        {
            "metric": {
                "namespace": "prod",
                "pod": "payment-service-abc12def9-xkvpn",
                "container": "payment-service",
            },
            "values": [[ts, "0.18"], [ts + 300, "0.20"]],
        }
    ]
    default_mem = [
        {
            "metric": {
                "namespace": "prod",
                "pod": "payment-service-abc12def9-xkvpn",
                "container": "payment-service",
            },
            "values": [[ts, str(256 * 1024 * 1024)], [ts + 300, str(260 * 1024 * 1024)]],
        }
    ]
    default_oom = []

    cpu_data = cpu_result if cpu_result is not None else default_cpu
    mem_data = mem_result if mem_result is not None else default_mem
    oom_data = oom_result if oom_result is not None else default_oom

    call_count = {"n": 0}

    def side_effect(url, **kwargs):
        if url.endswith("/-/healthy"):
            m = MagicMock()
            m.raise_for_status = MagicMock()
            return m
        if "query_range" in url:
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _range_response(cpu_data)
            return _range_response(mem_data)
        if "/api/v1/query" in url:
            return _instant_response(oom_data)
        raise ValueError(f"Unexpected URL: {url}")

    return side_effect


def test_prometheus_returns_correct_schema():
    with patch("requests.get", side_effect=_mock_get(True)):
        scraper = MetricScraper(source="prometheus",
                                prometheus_url="http://fake:9090",
                                namespace_filter="prod")
        df = scraper.fetch()

    assert set(df.columns) >= {"timestamp", "workload", "namespace",
                                "container", "cpu_cores", "memory_mib", "oom_kill"}
    assert len(df) > 0
    assert (df["namespace"] == "prod").all()


def test_prometheus_strips_pod_suffix():
    with patch("requests.get", side_effect=_mock_get(True)):
        scraper = MetricScraper(source="prometheus",
                                prometheus_url="http://fake:9090",
                                namespace_filter="prod")
        df = scraper.fetch()

    # pod was "payment-service-abc12def9-xkvpn" — suffix must be stripped
    assert (df["workload"] == "payment-service").all()


def test_prometheus_oom_count_populated():
    oom_result = [
        {
            "metric": {
                "namespace": "prod",
                "pod": "payment-service-abc12def9-xkvpn",
                "container": "payment-service",
                "reason": "OOMKilled",
            },
            "value": [1700000000, "3"],
        }
    ]
    with patch("requests.get", side_effect=_mock_get(True, oom_result=oom_result)):
        scraper = MetricScraper(source="prometheus",
                                prometheus_url="http://fake:9090",
                                namespace_filter="prod")
        df = scraper.fetch()

    # OOM count from Prometheus is > 0 → all rows for this workload flagged as 1
    assert df["oom_kill"].sum() > 0
    assert df["oom_kill"].max() == 1  # capped to binary flag


def test_prometheus_empty_results_raise():
    with patch("requests.get", side_effect=_mock_get(True, cpu_result=[], mem_result=[])):
        scraper = MetricScraper(source="prometheus", prometheus_url="http://fake:9090")
        with pytest.raises(ValueError, match="No CPU or memory metrics"):
            scraper.fetch()


def test_prometheus_connection_error_raises():
    with patch("requests.get",
               side_effect=requests.exceptions.ConnectionError("refused")):
        scraper = MetricScraper(source="prometheus",
                                prometheus_url="http://unreachable:9090")
        with pytest.raises(ConnectionError, match="Cannot connect to Prometheus"):
            scraper.fetch()


def test_strip_pod_suffix_deployment():
    assert MetricScraper._strip_pod_suffix("payment-service-7d9f4b5c9-xkvpn") == "payment-service"


def test_strip_pod_suffix_statefulset():
    assert MetricScraper._strip_pod_suffix("postgres-0") == "postgres"


def test_strip_pod_suffix_no_suffix():
    assert MetricScraper._strip_pod_suffix("standalone") == "standalone"


def test_extract_workload_prefers_deployment_label():
    labels = {"deployment": "my-app", "pod": "my-app-abc12def9-xkvpn", "namespace": "prod"}
    assert MetricScraper._extract_workload(labels) == "my-app"


def test_extract_workload_falls_back_to_pod():
    labels = {"pod": "my-app-abc12def9-xkvpn", "namespace": "prod"}
    assert MetricScraper._extract_workload(labels) == "my-app"


# ── Stage 2: features ─────────────────────────────────────────────────────────

def test_features_one_row_per_workload(raw_df, features_df):
    assert len(features_df) == raw_df["workload"].nunique()


def test_features_all_13_columns(features_df):
    expected = {
        "cpu_p50", "cpu_p95", "cpu_p99", "cpu_max", "cpu_stddev", "cpu_trend",
        "mem_p50", "mem_p95", "mem_p99", "mem_max", "mem_stddev", "mem_trend",
        "oom_count",
    }
    assert expected.issubset(set(features_df.columns))


def test_features_oom_count_positive(features_df):
    ml_row = features_df[features_df["workload"] == "ml-inference"].iloc[0]
    assert ml_row["oom_count"] > 0


# ── Stage 3: model ────────────────────────────────────────────────────────────

def test_model_adds_prediction_columns(model_df):
    for col in ("cpu_request_pred", "cpu_limit_pred", "mem_request_pred", "mem_limit_pred"):
        assert col in model_df.columns


def test_model_predictions_non_negative(model_df):
    assert (model_df["cpu_request_pred"] >= 0).all()
    assert (model_df["mem_request_pred"] >= 0).all()


def test_model_trend_projection_non_negative(features_df):
    """Projected request must be >= base p95."""
    model_df = fit_and_predict(features_df)
    assert (model_df["cpu_request_pred"] >= model_df["cpu_p95"]).all()
    assert (model_df["mem_request_pred"] >= model_df["mem_p95"]).all()


# ── Stage 4: recommender ──────────────────────────────────────────────────────

def test_recommender_oom_bumps_mem_limit(rec_df, model_df):
    ml_rec = rec_df[rec_df["workload"] == "ml-inference"].iloc[0]
    ml_mod = model_df[model_df["workload"] == "ml-inference"].iloc[0]

    # parse recommended limit back to MiB for comparison
    mem_lim_str = ml_rec["rec_mem_limit"]
    if mem_lim_str.endswith("Gi"):
        mem_lim_mib = float(mem_lim_str[:-2]) * 1024
    else:
        mem_lim_mib = float(mem_lim_str[:-2])

    # without OOM: p99 * 1.40; with OOM: p99 * 1.40 * 1.25
    expected_min = ml_mod["mem_p99"] * 1.40 * 1.25
    assert mem_lim_mib >= expected_min * 0.99  # allow rounding


def test_recommender_never_below_minimums(rec_df):
    for _, row in rec_df.iterrows():
        cpu_req_str = row["rec_cpu_request"]
        cpu_cores = int(cpu_req_str.rstrip("m")) / 1000
        assert cpu_cores >= MIN_CPU_REQUEST_CORES

        mem_req_str = row["rec_mem_request"]
        if mem_req_str.endswith("Gi"):
            mem_mib = float(mem_req_str[:-2]) * 1024
        else:
            mem_mib = float(mem_req_str[:-2])
        assert mem_mib >= MIN_MEM_REQUEST_MIB


def test_recommender_oom_flag_set(rec_df):
    ml_row = rec_df[rec_df["workload"] == "ml-inference"].iloc[0]
    assert ml_row["oom_flag"] is True or ml_row["oom_flag"] == True


# ── Stage 5: patcher ─────────────────────────────────────────────────────────

def test_patcher_valid_yaml(rec_df, tmp_path):
    render_patches(rec_df, output_dir=str(tmp_path), dry_run=False)

    for wl in rec_df["workload"].unique():
        patch_path = tmp_path / f"{wl}.yaml"
        assert patch_path.exists(), f"Missing patch file for {wl}"
        with open(patch_path) as f:
            doc = yaml.safe_load(f)
        assert doc["apiVersion"] == "apps/v1"
        assert doc["kind"] == "Deployment"
        containers = doc["spec"]["template"]["spec"]["containers"]
        assert len(containers) == 1
        resources = containers[0]["resources"]
        assert "requests" in resources
        assert "limits" in resources


def test_patcher_summary_yaml(rec_df, tmp_path):
    render_patches(rec_df, output_dir=str(tmp_path), dry_run=False)
    summary_path = tmp_path / "summary.yaml"
    assert summary_path.exists()
    with open(summary_path) as f:
        doc = yaml.safe_load(f)
    assert "rightsizer_report" in doc
    assert len(doc["rightsizer_report"]["workloads"]) == len(rec_df)


def test_patcher_dry_run_no_files(rec_df, tmp_path, capsys):
    render_patches(rec_df, output_dir=str(tmp_path), dry_run=True)
    captured = capsys.readouterr()
    assert "patches/" in captured.out
    # no files written
    assert not any(tmp_path.iterdir())

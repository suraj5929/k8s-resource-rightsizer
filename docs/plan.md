# Resource Right-Sizer Agent — Developer Spec

> **Goal**: Scrape local Prometheus metrics, run a lightweight regression model locally, and output recommended `requests`/`limits` per workload as a YAML patch — no external service needed.

---

## Project Structure

```
rightsizer/
├── data/
│   ├── metrics.csv              # synthetic dataset (or real Prometheus export)
│   └── current_resources.csv   # existing requests/limits per workload
├── rightsizer/
│   ├── __init__.py
│   ├── scraper.py               # Stage 1: fetch metrics from Prometheus (or CSV)
│   ├── features.py              # Stage 2: engineer features from raw timeseries
│   ├── model.py                 # Stage 3: fit regression, predict usage
│   ├── recommender.py           # Stage 4: apply headroom, compute requests/limits
│   └── patcher.py               # Stage 5: render Kubernetes YAML patch
├── main.py                      # CLI entrypoint
├── requirements.txt
└── README.md
```

---

## Dataset Schema

### `data/metrics.csv`
| Column | Type | Description |
|---|---|---|
| `timestamp` | ISO8601 string | 5-min interval datapoints over 7 days |
| `workload` | string | Deployment name |
| `namespace` | string | Kubernetes namespace |
| `container` | string | Container name |
| `cpu_cores` | float | CPU usage in cores (0.18 = 180m) |
| `memory_mib` | float | Memory usage in MiB |
| `oom_kill` | int (0/1) | OOM kill event flag |

### `data/current_resources.csv`
| Column | Type | Description |
|---|---|---|
| `workload` | string | Deployment name |
| `cpu_req` | string | Current CPU request e.g. `500m` |
| `cpu_lim` | string | Current CPU limit e.g. `1000m` |
| `mem_req` | string | Current memory request e.g. `512Mi` |
| `mem_lim` | string | Current memory limit e.g. `1Gi` |

---

## Stage 1 — `scraper.py`

**Responsibility**: Abstract data source. In production, query Prometheus. For local dev, load from CSV.

```python
# Interface to implement
class MetricScraper:
    def __init__(self, source: str, lookback_days: int = 7): ...
    def fetch(self) -> pd.DataFrame: ...
```

### Production mode (Prometheus)
- Use `requests.get` to hit `/api/v1/query_range`
- PromQL queries to run:
  ```
  rate(container_cpu_usage_seconds_total{namespace=~"prod|batch|ml|infra"}[5m])
  container_memory_working_set_bytes{namespace=~"prod|batch|ml|infra"}
  kube_pod_container_status_last_terminated_reason{reason="OOMKilled"}
  ```
- Parse response: `data.result[].values[]` → list of `[timestamp, value]`
- Normalize into the same DataFrame schema as the CSV

### Dev mode (CSV)
- Load `data/metrics.csv` directly with `pd.read_csv`
- Parse `timestamp` column as `datetime`

### Output DataFrame columns
`timestamp`, `workload`, `namespace`, `container`, `cpu_cores`, `memory_mib`, `oom_kill`

---

## Stage 2 — `features.py`

**Responsibility**: Aggregate raw timeseries into per-workload feature vectors.

```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame: ...
```

### Features to compute per workload
| Feature | Description |
|---|---|
| `cpu_p50` | Median CPU usage |
| `cpu_p95` | 95th percentile CPU — drives `requests` |
| `cpu_p99` | 99th percentile CPU — drives `limits` |
| `cpu_max` | Absolute peak (spike detection) |
| `cpu_stddev` | Variability — high stddev = spiky workload |
| `cpu_trend` | Linear slope over time (cores/hour) — catches growing services |
| `mem_p50` | Median memory |
| `mem_p95` | 95th percentile memory |
| `mem_p99` | 99th percentile memory |
| `mem_max` | Absolute peak |
| `mem_stddev` | Memory variability |
| `mem_trend` | Memory growth slope |
| `oom_count` | Total OOM kills in window — flag for limit tightness |

### Trend calculation
Use `numpy.polyfit(x, y, 1)` where `x` is elapsed hours and `y` is the usage series.
```python
hours = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 3600
slope, _ = np.polyfit(hours, df['cpu_cores'], 1)
```

---

## Stage 3 — `model.py`

**Responsibility**: Fit a lightweight model per workload and predict expected usage ceiling.

```python
def fit_and_predict(features_df: pd.DataFrame) -> pd.DataFrame: ...
```

### Approach
- **No ML framework needed** — use `numpy.percentile` directly on the timeseries for the base case
- **Optional upgrade**: fit `sklearn.linear_model.QuantileRegressor` for a proper quantile regression if trend is significant

### Logic
```python
# Simple percentile model (default)
cpu_request_pred = cpu_p95
cpu_limit_pred   = cpu_p99

# If workload is trending up (cpu_trend > 0.001 cores/hour),
# project forward by 7 more days and use that as the ceiling instead:
projected_cpu = cpu_p95 + (cpu_trend * 7 * 24)
cpu_request_pred = max(cpu_p95, projected_cpu)
```

### Output columns added
`cpu_request_pred`, `cpu_limit_pred`, `mem_request_pred`, `mem_limit_pred`

---

## Stage 4 — `recommender.py`

**Responsibility**: Apply safety headroom, enforce minimums, flag anomalies.

```python
def compute_recommendations(model_df: pd.DataFrame) -> pd.DataFrame: ...
```

### Headroom rules
```python
CPU_REQUEST_HEADROOM  = 1.10   # p95 × 1.10
CPU_LIMIT_HEADROOM    = 1.30   # p99 × 1.30
MEM_REQUEST_HEADROOM  = 1.15   # p95 × 1.15 (memory doesn't compress)
MEM_LIMIT_HEADROOM    = 1.40   # p99 × 1.40

# Hard minimums
MIN_CPU_REQUEST_CORES = 0.010  # 10m
MIN_MEM_REQUEST_MIB   = 32     # 32Mi
```

### OOM kill override
If `oom_count > 0`, bump memory limit by an additional 25% on top of headroom:
```python
if oom_count > 0:
    mem_limit_final *= 1.25
```

### Output: human-readable Kubernetes units
```python
def cores_to_millicores(cores: float) -> str:
    return f"{int(round(cores * 1000))}m"

def mib_to_k8s(mib: float) -> str:
    if mib >= 1024:
        return f"{round(mib/1024, 1)}Gi"
    return f"{int(round(mib))}Mi"
```

### Output columns
`rec_cpu_request`, `rec_cpu_limit`, `rec_mem_request`, `rec_mem_limit`, `oom_flag`, `savings_note`

---

## Stage 5 — `patcher.py`

**Responsibility**: Render Kubernetes strategic merge patch YAML per workload.

```python
def render_patches(rec_df: pd.DataFrame, output_dir: str = "patches/") -> None: ...
```

### Output format per workload
```yaml
# patches/payment-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: payment-service
  namespace: prod
spec:
  template:
    spec:
      containers:
      - name: payment-service
        resources:
          requests:
            cpu: "180m"
            memory: "256Mi"
          limits:
            cpu: "420m"
            memory: "512Mi"
```

### Also emit a summary report
`patches/summary.yaml` — all workloads in one file with before/after comparison:
```yaml
rightsizer_report:
  generated_at: "2026-05-09T00:00:00"
  workloads:
    - name: payment-service
      namespace: prod
      current:
        cpu_request: "500m"
        cpu_limit: "1000m"
        mem_request: "512Mi"
        mem_limit: "1Gi"
      recommended:
        cpu_request: "180m"
        cpu_limit: "420m"
        mem_request: "256Mi"
        mem_limit: "512Mi"
      flags:
        oom_kills: 0
        trending_up: false
```

---

## `main.py` — CLI

```python
# Usage examples:
# python main.py --source csv --data-dir data/ --output-dir patches/
# python main.py --source prometheus --prometheus-url http://localhost:9090 --lookback-days 7
# python main.py --source csv --dry-run   # print to stdout only
```

### Arguments
| Flag | Default | Description |
|---|---|---|
| `--source` | `csv` | `csv` or `prometheus` |
| `--data-dir` | `data/` | Path to CSV files (csv mode) |
| `--prometheus-url` | `http://localhost:9090` | Prometheus base URL |
| `--lookback-days` | `7` | Days of history to pull |
| `--output-dir` | `patches/` | Where to write YAML patches |
| `--dry-run` | `False` | Print patches to stdout, don't write files |
| `--namespace` | `None` | Filter to single namespace |

---

## `requirements.txt`

```
pandas>=2.0
numpy>=1.25
scikit-learn>=1.4
pyyaml>=6.0
requests>=2.31
click>=8.1
```

---

## Key Business Logic Summary

| Workload pattern | What agent should do |
|---|---|
| Idle / massively over-provisioned | Cut requests/limits significantly |
| Spiky but rare peaks | Use p95 for request, p99 for limit — don't chase the spike |
| Growing trend | Project 7 days forward and use that as the ceiling |
| OOM kills present | Add extra 25% buffer on memory limit |
| Stable and predictable | Tight fit: p95 request, p99 limit, standard headroom |

---

## Apply Patches

```bash
# Single workload
kubectl patch deployment payment-service -n prod \
  --patch-file patches/payment-service.yaml

# All workloads via loop
for f in patches/*.yaml; do
  name=$(basename $f .yaml)
  kubectl patch deployment $name --patch-file $f || true
done

# Or use server-side apply
kubectl apply --server-side -f patches/
```

---

## Testing Checklist

- [ ] `scraper.py` loads CSV and returns correct DataFrame shape
- [ ] `features.py` produces one row per workload with all 13 features
- [ ] `model.py` trend projection is non-negative
- [ ] `recommender.py` OOM flag adds 25% extra on memory limit
- [ ] `recommender.py` never recommends below minimum values
- [ ] `patcher.py` output is valid YAML (use `yaml.safe_load` to verify)
- [ ] `main.py --dry-run` prints summary without writing files
- [ ] `ml-inference` gets higher mem limit due to OOM kills
- [ ] `notification-svc` CPU recommendation is ~40m (not 500m)
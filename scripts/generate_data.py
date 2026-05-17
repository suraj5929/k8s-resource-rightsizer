"""Generate synthetic metrics and current_resources CSV files for local dev."""
import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

INTERVAL_MINUTES = 5
DAYS = 7
N = int(DAYS * 24 * 60 / INTERVAL_MINUTES)  # 2016 rows per workload

start = pd.Timestamp("2026-05-09T00:00:00")
timestamps = pd.date_range(start, periods=N, freq="5min")

# ── workload definitions ─────────────────────────────────────────────────────
# Each entry: (name, namespace, container, cpu_base, cpu_noise, cpu_trend,
#              mem_base_mib, mem_noise, oom_prob)
WORKLOADS = [
    # stable, massively over-provisioned
    ("payment-service",  "prod",  "payment-service",  0.18, 0.03, 0.0,    220,  20,  0.0),
    # trending up in CPU
    ("batch-processor",  "batch", "batch-processor",  0.10, 0.05, 0.002,  180,  30,  0.0),
    # idle/over-provisioned
    ("notification-svc", "prod",  "notification-svc", 0.004, 0.002, 0.0,  40,   8,  0.0),
    # spiky
    ("api-gateway",      "prod",  "api-gateway",      0.30, 0.20, 0.0,    300,  80,  0.0),
    # OOM kills, needs more memory
    ("ml-inference",     "ml",    "ml-inference",     0.50, 0.10, 0.001,  900, 100,  0.02),
]

rows = []
for name, ns, container, cpu_base, cpu_noise, cpu_trend, mem_base, mem_noise, oom_prob in WORKLOADS:
    hours = np.arange(N) * INTERVAL_MINUTES / 60
    cpu = cpu_base + cpu_trend * hours + rng.normal(0, cpu_noise, N)
    cpu = np.clip(cpu, 0.001, None)

    mem = mem_base + rng.normal(0, mem_noise, N)
    mem = np.clip(mem, 10, None)

    oom = rng.random(N) < oom_prob
    oom = oom.astype(int)

    for i in range(N):
        rows.append({
            "timestamp": timestamps[i].isoformat(),
            "workload":  name,
            "namespace": ns,
            "container": container,
            "cpu_cores": round(float(cpu[i]), 4),
            "memory_mib": round(float(mem[i]), 2),
            "oom_kill":  int(oom[i]),
        })

metrics_df = pd.DataFrame(rows)

# ── current resource allocations ────────────────────────────────────────────
current = pd.DataFrame([
    {"workload": "payment-service",  "namespace": "prod",  "cpu_req": "500m",  "cpu_lim": "1000m", "mem_req": "512Mi", "mem_lim": "1Gi"},
    {"workload": "batch-processor",  "namespace": "batch", "cpu_req": "200m",  "cpu_lim": "400m",  "mem_req": "256Mi", "mem_lim": "512Mi"},
    {"workload": "notification-svc", "namespace": "prod",  "cpu_req": "500m",  "cpu_lim": "1000m", "mem_req": "256Mi", "mem_lim": "512Mi"},
    {"workload": "api-gateway",      "namespace": "prod",  "cpu_req": "500m",  "cpu_lim": "1000m", "mem_req": "512Mi", "mem_lim": "1Gi"},
    {"workload": "ml-inference",     "namespace": "ml",    "cpu_req": "1000m", "cpu_lim": "2000m", "mem_req": "1Gi",   "mem_lim": "2Gi"},
])

os.makedirs("data", exist_ok=True)
metrics_df.to_csv("data/metrics.csv", index=False)
current.to_csv("data/current_resources.csv", index=False)

print(f"Wrote data/metrics.csv        ({len(metrics_df):,} rows)")
print(f"Wrote data/current_resources.csv ({len(current)} rows)")

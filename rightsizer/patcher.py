"""Stage 5 — render Kubernetes strategic merge patch YAML per workload."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import yaml


def render_patches(rec_df: pd.DataFrame, output_dir: str = "patches/",
                   current_df: pd.DataFrame | None = None,
                   dry_run: bool = False) -> None:
    """Write one YAML patch file per workload plus a summary report.

    Args:
        rec_df:     recommendations DataFrame from recommender.compute_recommendations
        output_dir: directory to write YAML files (ignored when dry_run=True)
        current_df: optional DataFrame with current resource allocations
        dry_run:    if True, print to stdout instead of writing files
    """
    if not dry_run:
        os.makedirs(output_dir, exist_ok=True)

    current_lookup: dict[str, dict] = {}
    if current_df is not None:
        for _, row in current_df.iterrows():
            current_lookup[row["workload"]] = row.to_dict()

    summary_workloads = []

    for _, row in rec_df.iterrows():
        patch = _build_patch(row)
        patch_yaml = yaml.dump(patch, default_flow_style=False, sort_keys=False)

        if dry_run:
            print(f"# patches/{row['workload']}.yaml")
            print(patch_yaml)
        else:
            path = os.path.join(output_dir, f"{row['workload']}.yaml")
            with open(path, "w") as f:
                f.write(patch_yaml)

        current = current_lookup.get(row["workload"], {})
        summary_workloads.append(_build_summary_entry(row, current))

    summary = {
        "rightsizer_report": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "workloads": summary_workloads,
        }
    }
    summary_yaml = yaml.dump(summary, default_flow_style=False, sort_keys=False)

    if dry_run:
        print("# patches/summary.yaml")
        print(summary_yaml)
    else:
        path = os.path.join(output_dir, "summary.yaml")
        with open(path, "w") as f:
            f.write(summary_yaml)
        print(f"Wrote {len(rec_df)} patch files + summary.yaml to {output_dir}")


# ── builders ──────────────────────────────────────────────────────────────────

def _build_patch(row: pd.Series) -> dict:
    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": row["workload"],
            "namespace": row["namespace"],
        },
        "spec": {
            "template": {
                "spec": {
                    "containers": [{
                        "name": row["container"],
                        "resources": {
                            "requests": {
                                "cpu":    row["rec_cpu_request"],
                                "memory": row["rec_mem_request"],
                            },
                            "limits": {
                                "cpu":    row["rec_cpu_limit"],
                                "memory": row["rec_mem_limit"],
                            },
                        },
                    }],
                }
            }
        },
    }


def _build_summary_entry(row: pd.Series, current: dict) -> dict:
    entry: dict = {
        "name":      row["workload"],
        "namespace": row["namespace"],
        "recommended": {
            "cpu_request": row["rec_cpu_request"],
            "cpu_limit":   row["rec_cpu_limit"],
            "mem_request": row["rec_mem_request"],
            "mem_limit":   row["rec_mem_limit"],
        },
        "flags": {
            "oom_kills":   int(row["oom_count"]),
            "trending_up": bool(row["cpu_trend"] > 0.001 or row["mem_trend"] > 0.001),
        },
    }
    if current:
        entry["current"] = {
            "cpu_request": current.get("cpu_req", "unknown"),
            "cpu_limit":   current.get("cpu_lim", "unknown"),
            "mem_request": current.get("mem_req", "unknown"),
            "mem_limit":   current.get("mem_lim", "unknown"),
        }
    if row.get("savings_note"):
        entry["savings_note"] = row["savings_note"]
    return entry

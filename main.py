#!/usr/bin/env python3
"""k8s-resource-rightsizer — CLI entrypoint.

Usage examples:
    python main.py --source csv --data-dir data/ --output-dir patches/
    python main.py --source csv --dry-run
    python main.py --source prometheus --prometheus-url http://localhost:9090 --lookback-days 7
    python main.py --source csv --namespace prod --dry-run
"""
import os

import click
import pandas as pd

from rightsizer.scraper import MetricScraper
from rightsizer.features import engineer_features
from rightsizer.model import fit_and_predict
from rightsizer.recommender import compute_recommendations
from rightsizer.patcher import render_patches


@click.command()
@click.option("--source", default="csv", show_default=True,
              type=click.Choice(["csv", "prometheus"]),
              help="Data source: local CSV or live Prometheus.")
@click.option("--data-dir", default="data/", show_default=True,
              help="Path to CSV files (csv mode only).")
@click.option("--prometheus-url", default="http://localhost:9090", show_default=True,
              help="Prometheus base URL (prometheus mode only).")
@click.option("--lookback-days", default=7, show_default=True,
              help="Days of history to pull.")
@click.option("--output-dir", default="patches/", show_default=True,
              help="Where to write YAML patch files.")
@click.option("--dry-run", is_flag=True, default=False,
              help="Print patches to stdout; do not write files.")
@click.option("--namespace", default=None,
              help="Filter to a single Kubernetes namespace.")
def main(source, data_dir, prometheus_url, lookback_days,
         output_dir, dry_run, namespace):
    click.echo(f"[1/5] Fetching metrics  (source={source})")
    scraper = MetricScraper(
        source=source,
        data_dir=data_dir,
        prometheus_url=prometheus_url,
        lookback_days=lookback_days,
        namespace_filter=namespace,
    )
    raw_df = scraper.fetch()
    click.echo(f"      {len(raw_df):,} rows across "
               f"{raw_df['workload'].nunique()} workloads")

    click.echo("[2/5] Engineering features")
    features_df = engineer_features(raw_df)

    click.echo("[3/5] Fitting model / predicting ceilings")
    model_df = fit_and_predict(features_df)

    click.echo("[4/5] Computing recommendations")
    rec_df = compute_recommendations(model_df)

    # Load current resources for comparison (csv mode only)
    current_df = None
    if source == "csv":
        current_path = os.path.join(data_dir, "current_resources.csv")
        if os.path.exists(current_path):
            current_df = pd.read_csv(current_path)

    click.echo("[5/5] Rendering YAML patches")
    render_patches(rec_df, output_dir=output_dir,
                   current_df=current_df, dry_run=dry_run)

    if not dry_run:
        _print_table(rec_df)


def _print_table(rec_df: pd.DataFrame) -> None:
    click.echo("\nRecommendation summary:")
    header = f"{'Workload':<22} {'Namespace':<8} {'CPU req':>8} {'CPU lim':>8} " \
             f"{'Mem req':>9} {'Mem lim':>9}  Note"
    click.echo(header)
    click.echo("-" * len(header))
    for _, row in rec_df.iterrows():
        note = row["savings_note"] if row["savings_note"] != "stable" else ""
        click.echo(
            f"{row['workload']:<22} {row['namespace']:<8} "
            f"{row['rec_cpu_request']:>8} {row['rec_cpu_limit']:>8} "
            f"{row['rec_mem_request']:>9} {row['rec_mem_limit']:>9}  {note}"
        )


if __name__ == "__main__":
    main()

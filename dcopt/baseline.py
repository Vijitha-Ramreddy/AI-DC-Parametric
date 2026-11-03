"""Baseline measurements, yield, failures, capability, and downstream escapes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

from dcopt.config import load_config
from dcopt.io import load_datasets


def cpk(values: pd.Series, lo_limit: float | None, hi_limit: float | None) -> float | None:
    """Calculate the lower-sided, upper-sided, or two-sided Cpk."""
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty or values.std(ddof=1) == 0:
        return None
    sides: list[float] = []
    mean = float(values.mean())
    sigma = float(values.std(ddof=1))
    if pd.notna(lo_limit):
        sides.append((mean - float(lo_limit)) / (3 * sigma))
    if pd.notna(hi_limit):
        sides.append((float(hi_limit) - mean) / (3 * sigma))
    return min(sides) if sides else None


def poisson_upper_bound(events: int, confidence: float) -> float:
    """Return the one-sided Poisson upper bound for an event count."""
    return float(chi2.ppf(confidence, 2 * (events + 1)) / 2)


def yield_table(results: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    """Calculate first-pass yield overall and for each requested dimension."""
    rows: list[dict[str, Any]] = [{"dimension": "overall", "level": "ALL",
                                   "devices": len(results), "passed": int(results["sort_pass"].eq(1).sum()),
                                   "yield_pct": float(results["sort_pass"].mean() * 100)}]
    for dimension in dimensions:
        for level, group in results.groupby(dimension, sort=True):
            rows.append({"dimension": dimension, "level": str(level), "devices": len(group),
                         "passed": int(group["sort_pass"].eq(1).sum()),
                         "yield_pct": float(group["sort_pass"].mean() * 100)})
    return pd.DataFrame(rows)


def reconstructed_group_time(results: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    """Estimate cumulative catalog time by stop-on-first-fail group."""
    cumulative = 0.0
    rows: list[dict[str, Any]] = []
    group_time: dict[str, float] = {}
    cumulative_by_test: dict[str, float] = {}
    for row in catalog.itertuples(index=False):
        cumulative += float(row.test_time_ms)
        cumulative_by_test[str(row.test)] = cumulative
        group_time[str(row.group)] = group_time.get(str(row.group), 0.0) + float(row.test_time_ms)

    total_catalog = float(catalog["test_time_ms"].sum())
    return pd.DataFrame([
        {"group": group, "catalog_time_ms": group_time[group],
         "share_of_full_flow_pct": group_time[group] / total_catalog * 100}
        for group in sorted(group_time)
    ])


def run(config_path: str | Path) -> Path:
    """Compute baseline tables and charts and write results/01_baseline."""
    config_path = Path(config_path)
    config = load_config(config_path)
    data = load_datasets(config_path)
    results, catalog = data.results, data.catalog
    output_dir = Path(config["paths"]["results_dir"])
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir /= "01_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    yields = yield_table(results, ["lot", "tester", "site", "temp_C"])
    yields.to_csv(output_dir / "yield.csv", index=False)

    bin_pareto = results["hard_bin"].value_counts().rename_axis("hard_bin").reset_index(name="count")
    bin_pareto["share_pct"] = bin_pareto["count"] / len(results) * 100
    bin_pareto.to_csv(output_dir / "bin_pareto.csv", index=False)

    time_summary = pd.DataFrame([
        {"population": "overall", "mean_ms": results["dc_test_time_ms"].mean(),
         "median_ms": results["dc_test_time_ms"].median(), "devices": len(results)},
        {"population": "passing", "mean_ms": results.loc[results["sort_pass"].eq(1), "dc_test_time_ms"].mean(),
         "median_ms": results.loc[results["sort_pass"].eq(1), "dc_test_time_ms"].median(),
         "devices": int(results["sort_pass"].eq(1).sum())},
        {"population": "failing", "mean_ms": results.loc[results["sort_pass"].eq(0), "dc_test_time_ms"].mean(),
         "median_ms": results.loc[results["sort_pass"].eq(0), "dc_test_time_ms"].median(),
         "devices": int(results["sort_pass"].eq(0).sum())},
    ])
    time_summary.to_csv(output_dir / "time_summary.csv", index=False)
    group_time = reconstructed_group_time(results, catalog)
    group_time.to_csv(output_dir / "time_by_group.csv", index=False)

    fail_rows: list[dict[str, Any]] = []
    for row in catalog.itertuples(index=False):
        test = str(row.test)
        pf = f"{test}_PF"
        fails = int(results[pf].eq("F").sum()) if pf in results else 0
        fail_rows.append({"test": test, "fail_count": fails, "fail_rate_pct": fails / len(results) * 100,
                          "cpk": cpk(results[test], row.lo_limit, row.hi_limit)})
    fail_summary = pd.DataFrame(fail_rows)
    fail_summary.to_csv(output_dir / "test_failures.csv", index=False)

    confidence = float(config["confidence"])
    downstream_rows: list[dict[str, Any]] = []
    sort_pass = results["sort_pass"].eq(1)
    for column in ("final_test_fail", "burnin_fail", "rma"):
        events = int(results.loc[sort_pass, column].fillna(0).eq(1).sum())
        denominator = int(sort_pass.sum())
        downstream_rows.append({
            "label": column, "sort_pass_devices": denominator, "event_count": events,
            "rate_pct": events / denominator * 100 if denominator else 0.0,
            "dppm": events / denominator * 1_000_000 if denominator else 0.0,
            "poisson_upper_bound_events": poisson_upper_bound(events, confidence),
            "poisson_upper_bound_dppm": poisson_upper_bound(events, confidence) / denominator * 1_000_000 if denominator else 0.0,
        })
    downstream = pd.DataFrame(downstream_rows)
    downstream.to_csv(output_dir / "downstream_labels.csv", index=False)

    lot_yield = yields.query("dimension == 'lot'").sort_values("level")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(lot_yield["level"], lot_yield["yield_pct"], color="#4C78A8")
    ax.set(title="First-pass yield by lot", xlabel="Lot", ylabel="Yield (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "yield_by_lot.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bin_pareto["hard_bin"].astype(str), bin_pareto["count"], color="#F28E2B")
    ax.set(title="Hard-bin Pareto", xlabel="Hard bin", ylabel="Devices")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "bin_pareto.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(group_time["group"], group_time["catalog_time_ms"], color="#59A14F")
    ax.set(title="Catalog DC time by test group", xlabel="Test group", ylabel="Time (ms)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "time_by_group.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    fail_plot = fail_summary.sort_values("fail_rate_pct", ascending=False)
    ax.bar(fail_plot["test"], fail_plot["fail_rate_pct"], color="#E15759")
    ax.tick_params(axis="x", rotation=45)
    ax.set(title="Per-test fail rate", ylabel="Fail rate (%)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fail_rate_by_test.png", dpi=150)
    plt.close(fig)

    overall = yields.iloc[0]
    summary = [
        "# 01 Baseline summary",
        "",
        f"- Overall first-pass yield: **{overall['yield_pct']:.3f}%** "
        f"({int(overall['passed']):,}/{int(overall['devices']):,})",
        f"- Mean DC time overall: **{time_summary.iloc[0]['mean_ms']:.2f} ms/device**",
        f"- Mean passing-device time: **{time_summary.iloc[1]['mean_ms']:.2f} ms**",
        f"- Mean failing-device time: **{time_summary.iloc[2]['mean_ms']:.2f} ms**",
        f"- Highest-fail-rate test: **{fail_plot.iloc[0]['test']}** "
        f"({fail_plot.iloc[0]['fail_rate_pct']:.3f}%)",
        "",
        "## Assumptions",
        "- Catalog row order is the test-flow order.",
        "- Baseline stop-on-first-fail time is reconstructed from cumulative catalog test times.",
        "- Poisson upper bounds are one-sided at the configured confidence level.",
        "- Fail limits and measurements remain authoritative; no model changes these decisions.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Baseline complete. Overall yield: {overall['yield_pct']:.3f}%. "
          f"Mean DC time: {time_summary.iloc[0]['mean_ms']:.2f} ms/device.")
    print(f"Results written to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

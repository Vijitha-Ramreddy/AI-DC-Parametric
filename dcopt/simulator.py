"""Replay all DC test-time optimization techniques on holdout lots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from dcopt.config import load_config
from dcopt.io import load_datasets


TECHNIQUES = ("settling", "ordering", "static", "predictive", "adaptive", "pat")


def poisson_upper_dppm(events: int, devices: int, confidence: float) -> float:
    if devices <= 0:
        return float("nan")
    upper_rate = chi2.ppf(confidence, 2 * (events + 1)) / (2 * devices)
    return float(upper_rate * 1e6)


def artifact(results_dir: Path, phase: str, name: str) -> pd.DataFrame:
    path = results_dir / phase / name
    if not path.exists():
        raise FileNotFoundError(f"Required simulator artifact not found: {path}")
    return pd.read_csv(path)


def run(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    data = load_datasets(config_path)
    results_dir = Path(config["paths"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = config_path.parent / results_dir
    holdout_lots = [str(lot) for lot in config["holdout_lots"]]
    holdout = data.results[data.results["lot"].isin(holdout_lots)].copy()

    settling = artifact(results_dir, "02_settling", "recommendations.csv")
    static = artifact(results_dir, "03_static", "decision_table.csv")
    static_escapes = artifact(results_dir, "03_static", "holdout_escapes.csv")
    ordering = artifact(results_dir, "04_ordering", "time_comparison.csv")
    predictive = artifact(results_dir, "05_predictive", "predictions.csv")
    adaptive = artifact(results_dir, "06_adaptive", "device_decisions.csv")
    pat = artifact(results_dir, "07_pat", "flagged_devices.csv")

    baseline_ms = float(data.catalog["test_time_ms"].sum())
    settling_savings = float(settling["time_saved_ms"].sum())

    static_times = data.catalog.set_index("test")["test_time_ms"]
    static_savings = 0.0
    for row in static.itertuples(index=False):
        if row.decision == "REMOVE":
            static_savings += float(static_times.loc[row.test])
        elif row.decision == "SAMPLE":
            static_savings += float(static_times.loc[row.test]) * (1 - float(row.sampling_rate or 0.0))

    predictions_holdout = predictive[predictive["ecid"].isin(holdout["ecid"].astype(str))]
    predictive_skips = predictions_holdout["decision"].eq("SKIP")
    predictive_savings = 0.0
    predicted_times = {test: float(data.catalog.loc[data.catalog["test"].eq(test), "test_time_ms"].iloc[0])
                       for test in predictions_holdout["target"].unique()}
    predictive_savings = float(sum(
        predicted_times[str(row.target)] for row in predictions_holdout[predictive_skips].itertuples(index=False)
    ) / max(len(holdout), 1))

    original_row = ordering[(ordering["flow"] == "original") & (ordering["population"] == "all")]
    ranked_row = ordering[(ordering["flow"] == "ranked") & (ordering["population"] == "all")]
    if not original_row.empty and not ranked_row.empty:
        ordering_savings = max(0.0, float(original_row["mean_dc_time_ms"].iloc[0] -
                                          ranked_row["mean_dc_time_ms"].iloc[0]))
    else:
        ordering_savings = 0.0

    reduced_devices = adaptive["ecid"].isin(holdout["ecid"].astype(str)) & adaptive["decision"].eq("REDUCED")
    adaptive_reduced_fraction = float(reduced_devices.sum() / max(
        adaptive["ecid"].isin(holdout["ecid"].astype(str)).sum(), 1))
    adaptive_savings = (static_savings + predictive_savings) * adaptive_reduced_fraction
    adaptive_fallback_rate = 1.0 - adaptive_reduced_fraction

    holdout_pat = pat[pat["ecid"].isin(holdout["ecid"].astype(str))]
    pat_flag = (holdout_pat["mahalanobis_flag"] | holdout_pat["isolation_forest_flag"]).astype(bool)
    pat_catches = int((
        pat_flag.to_numpy(dtype=bool)
        & holdout_pat["is_latent_defect_truth"].fillna(0).eq(1).to_numpy(dtype=bool)
    ).sum())

    ordered = [str(item).strip().lower() for item in config.get("simulator_order", list(TECHNIQUES))]
    ordered = [item for item in TECHNIQUES if item in ordered] + [item for item in ordered if item not in TECHNIQUES]
    savings = {
        "settling": settling_savings,
        "ordering": ordering_savings,
        "static": static_savings,
        "predictive": predictive_savings,
        "adaptive": adaptive_savings,
        "pat": 0.0,
    }

    static_escape_count = int(static_escapes["holdout_escape_count"].sum()) if "holdout_escape_count" in static_escapes else 0
    predictive_escape_count = int(
        (predictions_holdout["decision"].eq("SKIP") & predictions_holdout["true_fail"].eq(True)).sum())
    model_escape_count = static_escape_count + predictive_escape_count
    dppm_upper = poisson_upper_dppm(model_escape_count, len(holdout), float(config["confidence"]))
    false_fails = int(
        (predictions_holdout["decision"].eq("SKIP") & predictions_holdout["true_fail"].eq(True)).sum())

    scenarios: list[dict[str, object]] = []
    cumulative = 0.0
    for technique in ordered:
        if technique != "adaptive":
            cumulative += savings[technique]
        else:
            # Static and predictive savings only apply to adaptive REDUCED devices.
            cumulative = cumulative - (static_savings + predictive_savings) * adaptive_fallback_rate
        mean_time = baseline_ms - cumulative
        reduction = 100 * cumulative / baseline_ms
        scenario_escape_count = (
            static_escape_count if technique == "static" else
            model_escape_count if technique in ("predictive", "adaptive", "pat") else 0
        )
        scenarios.append({
            "scenario": "cumulative_" + technique,
            "techniques": "+".join(ordered[:ordered.index(technique) + 1]),
            "mean_dc_time_ms": mean_time,
            "reduction_pct": reduction,
            "test_cost_per_device": mean_time / 1000 * float(config.get("tester_cost_per_sec") or 0.0),
            "fallback_rate_pct": adaptive_fallback_rate * 100 if technique in ("adaptive", "pat") else 0.0,
            "model_skipped_escape_count": scenario_escape_count,
            "escape_dppm_upper_bound": poisson_upper_dppm(
                scenario_escape_count, len(holdout), float(config["confidence"])),
            "pat_catches": pat_catches if technique in ("pat",) else 0,
            "false_fail_overkill": 0,
        })

    for technique in TECHNIQUES:
        mean_time = baseline_ms - savings[technique]
        reduction = 100 * savings[technique] / baseline_ms
        scenarios.append({
            "scenario": technique + "_alone",
            "techniques": technique,
            "mean_dc_time_ms": mean_time,
            "reduction_pct": reduction,
            "test_cost_per_device": mean_time / 1000 * float(config.get("tester_cost_per_sec") or 0.0),
            "fallback_rate_pct": adaptive_fallback_rate * 100 if technique == "adaptive" else 0.0,
            "model_skipped_escape_count": (static_escape_count if technique == "static" else
                                           predictive_escape_count if technique == "predictive" else 0),
            "escape_dppm_upper_bound": poisson_upper_dppm(
                static_escape_count if technique == "static" else predictive_escape_count if technique == "predictive" else 0,
                len(holdout), float(config["confidence"])),
            "pat_catches": pat_catches if technique == "pat" else 0,
            "false_fail_overkill": 0,
        })

    cumulative_all = sum(
        savings[technique] for technique in ordered if technique != "adaptive"
    )
    if "adaptive" in ordered:
        cumulative_all = (settling_savings + ordering_savings
                          + (static_savings + predictive_savings) * adaptive_reduced_fraction)
    scenarios.append({
        "scenario": "combined",
        "techniques": "+".join(ordered),
        "mean_dc_time_ms": baseline_ms - cumulative_all,
        "reduction_pct": 100 * cumulative_all / baseline_ms,
        "test_cost_per_device": (baseline_ms - cumulative_all) / 1000 * float(config.get("tester_cost_per_sec") or 0.0),
        "fallback_rate_pct": adaptive_fallback_rate * 100,
        "model_skipped_escape_count": model_escape_count,
        "escape_dppm_upper_bound": dppm_upper,
        "pat_catches": pat_catches,
        "false_fail_overkill": 0,
    })

    scenario_table = pd.DataFrame(scenarios)
    scenario_table["meets_target_reduction"] = scenario_table["reduction_pct"] >= float(config["target_reduction_pct"])
    scenario_table["meets_dppm_budget"] = scenario_table["escape_dppm_upper_bound"] <= float(config["dppm_budget"])
    scenario_table["yield_change_false_fail_overkill"] = scenario_table["false_fail_overkill"]

    output_dir = results_dir / "08_scorecard"
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_table.to_csv(output_dir / "scenario_table.csv", index=False)

    labels = ["baseline"] + ordered
    values = [baseline_ms] + [-savings[technique] for technique in ordered]
    fig, ax = plt.subplots(figsize=(9, 5))
    running = baseline_ms
    ax.bar(0, baseline_ms, color="#4E79A7", label="baseline")
    for i, technique in enumerate(ordered, start=1):
        ax.bar(i, -savings[technique], bottom=running, color="#59A14F")
        running -= savings[technique]
    ax.axhline(running, color="#E15759", linestyle="--")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("DC time (ms/device)")
    ax.set_title("Cumulative configured savings")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "savings_waterfall.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(scenario_table["reduction_pct"], scenario_table["escape_dppm_upper_bound"],
               s=60, color="#4E79A7")
    ax.axvline(float(config["target_reduction_pct"]), color="#59A14F", linestyle="--", label="Reduction target")
    ax.axhline(float(config["dppm_budget"]), color="#E15759", linestyle="--", label="DPPM budget")
    ax.set(title="Savings vs. modeled risk", xlabel="Reduction (%)", ylabel="Escape DPPM upper bound")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "savings_vs_risk.png", dpi=150)
    plt.close(fig)

    summary = "\n".join([
        "# 08 Scorecard summary",
        "",
        f"- Holdout devices replayed: {len(holdout):,}.",
        f"- Baseline catalog time: {baseline_ms:.2f} ms/device.",
        f"- Combined configured reduction: {scenario_table.loc[scenario_table.scenario.eq('combined'), 'reduction_pct'].iloc[0]:.2f}%.",
        f"- Combined escape upper bound: {dppm_upper:.2f} DPPM.",
        f"- Predictive skipped true failures (model escapes): {predictive_escape_count}.",
        f"- PAT catches in the holdout: {pat_catches}.",
        f"- Adaptive fallback-flow share: {adaptive_fallback_rate * 100:.2f}%.",
        "- Technique order for the waterfall comes from simulator_order in config.yaml.",
    ])
    (output_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")
    print(f"Simulator complete. Results written to {output_dir}.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

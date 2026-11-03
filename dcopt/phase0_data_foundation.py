"""Phase 0: validate and summarize the linked synthetic data foundation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from dcopt.config import load_config


def resolve_splits(config: dict[str, Any], lots: list[str]) -> tuple[list[str], list[str]]:
    """Resolve fixed or leave-one-lot-out train/test splits from config."""
    split_config = config["splits"]
    if bool(split_config.get("leave_one_lot_out", False)):
        held_out = [str(lot) for lot in lots[-1:]]
        training = [lot for lot in lots if lot not in held_out]
        return training, held_out

    training = [str(lot) for lot in (config.get("train_lots") or split_config["train_lots"])]
    testing = [str(lot) for lot in (config.get("holdout_lots") or split_config["test_lots"])]
    unknown = sorted((set(training) | set(testing)) - set(lots))
    if unknown:
        raise ValueError(f"Configured lots are absent from data: {unknown}")
    overlap = sorted(set(training) & set(testing))
    if overlap:
        raise ValueError(f"Train/test lots overlap: {overlap}")
    return training, testing


def validate_catalog(results: pd.DataFrame, catalog: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Validate catalog coverage and required result columns."""
    issues: list[str] = []
    required = [str(column) for column in config["validation"]["required_result_columns"]]
    missing_columns = sorted(set(required) - set(results.columns))
    if missing_columns:
        issues.append(f"Missing required result columns: {missing_columns}")

    measurement_tests = set(catalog["test"])
    missing_tests = sorted(measurement_tests - set(results.columns))
    if missing_tests:
        issues.append(f"Catalog tests missing from results: {missing_tests}")
    return issues


def run(config_path: str | Path) -> Path:
    """Run the Phase 0 foundation audit and write all requested artifacts."""
    config = load_config(config_path)
    config_path = Path(config_path)
    data_dir = (config_path.parent / config["paths"]["data_dir"]).resolve()
    output_dir = Path(config["paths"]["results_dir"])
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir = output_dir / config["output"]["step_name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pd.read_csv(data_dir / config["paths"]["results"])
    catalog = pd.read_csv(data_dir / config["paths"]["catalog"])
    settling = pd.read_csv(data_dir / config["paths"]["settling_sweep"])
    train_lots, test_lots = resolve_splits(config, sorted(results["lot"].unique()))

    issues = validate_catalog(results, catalog, config)
    duplicate_ecids = int(results["ecid"].duplicated().sum()) if "ecid" in results else 0
    if bool(config["validation"]["expect_unique_ecid"]) and duplicate_ecids:
        issues.append(f"Duplicate ECIDs: {duplicate_ecids}")

    forbidden = str(config["validation"]["forbidden_model_feature"])
    model_feature_leakage = forbidden in results.columns
    measurement_tests = list(catalog["test"])
    measurement_missing = {
        test: float(results[test].isna().mean())
        for test in measurement_tests
        if test in results.columns
    }

    lot_summary = (
        results.groupby("lot", sort=True)
        .agg(
            devices=("ecid", "size"),
            mean_dc_time_ms=("dc_test_time_ms", "mean"),
            sort_pass_rate=("sort_pass", lambda values: float(values.mean())),
            latent_truth_count=(forbidden, "sum"),
        )
        .reset_index()
        .assign(split=lambda frame: frame["lot"].map(
            lambda lot: "train" if lot in train_lots else "test" if lot in test_lots else "unused"
        ))
    )
    lot_summary.to_csv(output_dir / "lot_summary.csv", index=False)
    settling_with_bias = settling.assign(
        absolute_bias=lambda frame: (frame["measured_value"] - frame["reference_value"]).abs()
    )
    settling_summary = (
        settling_with_bias.groupby(["test", "temp_C", "wait_ms", "n_samples"], sort=True)
        .agg(
            mean_absolute_error=("absolute_bias", "mean"),
            measurement_std=("measured_value", "std"),
            repeats=("repeat", "size"),
        )
        .reset_index()
    )
    settling_summary.to_csv(output_dir / "settling_summary.csv", index=False)
    split_summary = pd.DataFrame(
        [
            {"split": "train", "lots": ", ".join(train_lots), "devices": int(results["lot"].isin(train_lots).sum())},
            {"split": "test", "lots": ", ".join(test_lots), "devices": int(results["lot"].isin(test_lots).sum())},
        ]
    )
    split_summary.to_csv(output_dir / "split_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(lot_summary["lot"], lot_summary["mean_dc_time_ms"], color="#4C78A8")
    ax.set(title="Mean DC test time by lot", xlabel="Lot", ylabel="DC test time (ms)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "lot_dc_time.png", dpi=150)
    plt.close(fig)

    status = "PASS" if not issues else "FAIL"
    issue_lines = [f"- {issue}" for issue in issues] if issues else ["- None"]
    summary = [
        "# Phase 0 data foundation summary",
        "",
        f"- Validation status: **{status}**",
        f"- Results: {len(results):,} devices across {results['lot'].nunique()} lots",
        f"- Catalog: {len(catalog)} tests",
        f"- Train split: {', '.join(train_lots)} ({int(results['lot'].isin(train_lots).sum()):,} devices)",
        f"- Test split: {', '.join(test_lots)} ({int(results['lot'].isin(test_lots).sum()):,} devices)",
        f"- Mean baseline DC time: {results['dc_test_time_ms'].mean():.2f} ms/device",
        f"- Sort-pass rate: {results['sort_pass'].mean():.4%}",
        f"- Duplicate ECIDs: {duplicate_ecids}",
        f"- Forbidden truth column present in source: {'yes' if model_feature_leakage else 'no'}",
        "- LATENT-DEFECT TRUTH USE: scoring-only for PAT; it was not added to any feature table.",
        "",
        "## Assumptions",
        "- Train and held-out devices are assigned strictly by lot from config.yaml.",
        "- Stop-on-first-fail baseline time is retained as observed synthetic time.",
        "",
        "## Validation issues",
        *issue_lines,
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    pd.DataFrame(
        [{"test": test, "missing_rate": missing_rate} for test, missing_rate in measurement_missing.items()]
    ).to_csv(output_dir / "measurement_missing_rates.csv", index=False)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    output_dir = run(args.config)
    print(f"Phase 0 complete. Results written to {output_dir}.")


if __name__ == "__main__":
    main()

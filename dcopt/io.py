"""Load and validate the three synthetic DC dataset CSV files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from dcopt.config import load_config


CATALOG_COLUMNS = {"test", "group", "unit", "lo_limit", "hi_limit", "test_time_ms"}
RESULT_COLUMNS = {
    "ecid", "lot", "wafer", "die_x", "die_y", "tester", "site", "temp_C",
    "first_fail_test", "sort_pass", "hard_bin", "dc_test_time_ms",
    "final_test_fail", "burnin_fail", "rma", "is_latent_defect_truth",
}
SETTLING_COLUMNS = {
    "test", "device", "temp_C", "wait_ms", "n_samples", "repeat",
    "reference_value", "measured_value",
}


@dataclass(frozen=True)
class Datasets:
    results: pd.DataFrame
    catalog: pd.DataFrame
    settling: pd.DataFrame


def load_datasets(config_path: str | Path) -> Datasets:
    """Load configured CSVs and validate required schema from data/README.md."""
    config_path = Path(config_path)
    config = load_config(config_path)
    data_dir = (config_path.parent / config["paths"]["data_dir"]).resolve()
    results = pd.read_csv(data_dir / config["paths"]["results"])
    catalog = pd.read_csv(data_dir / config["paths"]["catalog"])
    settling = pd.read_csv(data_dir / config["paths"]["settling_sweep"])

    catalog_tests = set(catalog["test"].astype(str))
    expected_results = RESULT_COLUMNS | catalog_tests | {f"{test}_PF" for test in catalog_tests}
    validations = {
        "dc_test_results.csv": (results, expected_results),
        "test_catalog.csv": (catalog, CATALOG_COLUMNS),
        "settling_sweep.csv": (settling, SETTLING_COLUMNS),
    }
    for filename, (frame, expected) in validations.items():
        missing = sorted(expected - set(frame.columns))
        if missing:
            raise ValueError(f"{filename} is missing columns required by data/README.md: {missing}")
    return Datasets(results=results, catalog=catalog, settling=settling)


def data_quality_report(data: Datasets) -> dict[str, Any]:
    """Build a concise report of row counts and key data distributions."""
    null_counts = data.results.isna().sum()
    return {
        "row_counts": {
            "dc_test_results": len(data.results),
            "test_catalog": len(data.catalog),
            "settling_sweep": len(data.settling),
        },
        "null_columns": {
            column: int(count) for column, count in null_counts.items() if int(count) > 0
        },
        "lots": sorted(data.results["lot"].unique().astype(str)),
        "testers": sorted(data.results["tester"].unique().astype(str)),
        "temperatures": sorted(data.results["temp_C"].unique().tolist()),
    }


def print_data_quality_report(report: dict[str, Any]) -> None:
    """Print the data-quality report in a plain-text, human-readable form."""
    print("Data quality report")
    print("- Row counts:", report["row_counts"])
    print("- Lots:", ", ".join(report["lots"]))
    print("- Testers:", ", ".join(report["testers"]))
    print("- Temperatures:", ", ".join(str(value) for value in report["temperatures"]))
    print("- Null values:", report["null_columns"] if report["null_columns"] else "none")


def run(config_path: str | Path) -> Path:
    """Load data and write and print the Phase 0 I/O quality artifacts."""
    config = load_config(config_path)
    output_dir = Path(config["paths"]["results_dir"]) / "io"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_datasets(config_path)
    report = data_quality_report(data)
    print_data_quality_report(report)

    pd.DataFrame([
        {"dataset": "dc_test_results", "rows": report["row_counts"]["dc_test_results"]},
        {"dataset": "test_catalog", "rows": report["row_counts"]["test_catalog"]},
        {"dataset": "settling_sweep", "rows": report["row_counts"]["settling_sweep"]},
    ]).to_csv(output_dir / "row_counts.csv", index=False)
    pd.DataFrame(
        [{"column": column, "null_count": count} for column, count in report["null_columns"].items()]
    ).to_csv(output_dir / "null_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(report["row_counts"].keys(), report["row_counts"].values(), color="#4C78A8")
    ax.set(title="Input dataset row counts", ylabel="Rows")
    fig.tight_layout()
    fig.savefig(output_dir / "row_counts.png", dpi=150)
    plt.close(fig)
    (output_dir / "summary.md").write_text(
        "# I/O data-quality summary\n\n"
        f"- Row counts: {report['row_counts']}\n"
        f"- Lots: {', '.join(report['lots'])}\n"
        f"- Testers: {', '.join(report['testers'])}\n"
        f"- Temperatures: {', '.join(str(value) for value in report['temperatures'])}\n"
        f"- Null counts: {report['null_columns'] if report['null_columns'] else 'none'}\n",
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(f"Results written to {run(args.config)}")


if __name__ == "__main__":
    main()

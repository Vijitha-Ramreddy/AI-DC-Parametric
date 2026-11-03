"""Settling time and averaging optimization from the synthetic sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from dcopt.config import load_config
from dcopt.io import load_datasets


def tolerance(test: pd.Series, typical_value: float) -> float:
    """Return two-sided tolerance, or distance from typical to a one-sided limit."""
    lo = test["lo_limit"]
    hi = test["hi_limit"]
    if pd.notna(lo) and pd.notna(hi):
        return float(hi - lo)
    limit = float(hi if pd.notna(hi) else lo)
    return abs(limit - typical_value)


def exponential_error(wait_ms: np.ndarray, e0: float, tau: float) -> np.ndarray:
    return e0 * np.exp(-wait_ms / tau)


def metrics_by_setting(sweep: pd.DataFrame) -> pd.DataFrame:
    sweep = sweep.assign(error=sweep["measured_value"] - sweep["reference_value"])
    bias = sweep.groupby(["test", "temp_C", "wait_ms", "n_samples"], sort=True)["error"].mean()
    repeatability = (
        sweep.groupby(["test", "temp_C", "wait_ms", "n_samples", "device"])["error"]
        .std(ddof=1)
        .groupby(["test", "temp_C", "wait_ms", "n_samples"])
        .mean()
    )
    return bias.rename("bias").to_frame().join(repeatability.rename("repeatability")).reset_index()


def run(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    data = load_datasets(config_path)
    sweep, catalog, results = data.settling, data.catalog, data.results
    output_dir = Path(config["paths"]["results_dir"])
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir /= "02_settling"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = metrics_by_setting(sweep)
    typical = {test: float(results[test].median()) for test in catalog["test"] if test in results}
    catalog_lookup = catalog.set_index("test")
    for row_index, row in metrics.iterrows():
        test_tolerance = tolerance(catalog_lookup.loc[row["test"]], typical.get(row["test"], sweep.loc[sweep["test"] == row["test"], "reference_value"].mean()))
        metrics.loc[row_index, "tolerance"] = test_tolerance
        metrics.loc[row_index, "bias_pct_tolerance"] = abs(row["bias"]) / test_tolerance * 100
        metrics.loc[row_index, "noise_pct_tolerance"] = row["repeatability"] / test_tolerance * 100
    metrics.to_csv(output_dir / "sweep_metrics.csv", index=False)

    curves: list[dict[str, Any]] = []
    curve_points: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
    for (test, temp), group in metrics.groupby(["test", "temp_C"]):
        curve = group.groupby("wait_ms", as_index=False)["bias_pct_tolerance"].mean()
        x = curve["wait_ms"].to_numpy(dtype=float)
        y = np.abs(curve["bias_pct_tolerance"].to_numpy(dtype=float))
        try:
            (e0, tau), _ = curve_fit(exponential_error, x, y, p0=[max(y[0], 1e-6), 1.0],
                                     bounds=(0, np.inf), maxfev=20_000)
        except (RuntimeError, ValueError):
            tau, e0 = np.nan, np.nan
        curves.append({"test": test, "temp_C": temp, "e0": e0, "tau_ms": tau})
        curve_points[(test, float(temp))] = (x, y)
    curves_df = pd.DataFrame(curves)
    tau_ratio = curves_df.pivot(index="test", columns="temp_C", values="tau_ms").rename(
        columns={25: "tau_25c", 90: "tau_90c"}).reset_index()
    tau_ratio["tau_ratio_90_over_25"] = tau_ratio["tau_90c"] / tau_ratio["tau_25c"]
    curves_df.to_csv(output_dir / "settling_curves.csv", index=False)
    tau_ratio.to_csv(output_dir / "tau_ratio.csv", index=False)

    bias_limit = float(config["gage_bias_limit_pct_of_tolerance"])
    noise_limit = float(config["noise_limit_pct_of_tolerance"])
    margin = float(config["settling_margin_pct"]) / 100
    overhead_fraction = float(config["settling_overhead_fraction"])
    recommendations: list[dict[str, Any]] = []
    for test in catalog["test"].astype(str):
        catalog_row = catalog_lookup.loc[test]
        catalog_time = float(catalog_row["test_time_ms"])
        feasible = metrics[
            (metrics["test"] == test)
            & (metrics["bias_pct_tolerance"] <= bias_limit)
            & (metrics["noise_pct_tolerance"] <= noise_limit)
        ]
        # Group by setting and require every sampled temperature to meet both limits.
        settings = []
        for (wait_ms, samples), setting in feasible.groupby(["wait_ms", "n_samples"]):
            temps = set(setting["temp_C"])
            if temps >= {25, 90}:
                settings.append((float(wait_ms), int(samples)))
        settings.sort(key=lambda item: (item[0], item[1]))
        if settings:
            wait_ms, samples = settings[0]
            status = "configured"
            wait_with_margin = wait_ms * (1 + margin)
            fixed_overhead = catalog_time * (1 - overhead_fraction)
            sample_time = (catalog_time * overhead_fraction / 2) / baseline_samples
            proposed_time = fixed_overhead + wait_with_margin + samples * sample_time
        else:
            wait_ms = float(sweep["wait_ms"].max())
            samples = int(sweep["n_samples"].max())
            wait_with_margin = float(wait_ms)
            status = "baseline_fallback_no_feasible_setting"
            proposed_time = catalog_time
        baseline_wait = float(config["baseline_wait_ms"])
        baseline_samples = float(config["baseline_samples"])
        # Keep the remaining fixed portion unchanged; default the split between wait/samples equally.
        recommendations.append({
            "test": test, "wait_ms": wait_ms, "n_samples": samples,
            "wait_with_margin_ms": wait_with_margin, "status": status,
            "catalog_time_ms": catalog_time, "recommended_time_ms": proposed_time,
            "time_saved_ms": max(catalog_time - proposed_time, 0.0),
            "time_saved_pct": max(catalog_time - proposed_time, 0.0) / catalog_time * 100,
            "dominant_temperature_C": 90,
        })
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df.to_csv(output_dir / "recommendations.csv", index=False)

    for test in recommendations_df["test"]:
        test_metrics = metrics[metrics["test"] == test]
        for metric, title in (("bias_pct_tolerance", "Bias (% tolerance)"),
                              ("noise_pct_tolerance", "Repeatability noise (% tolerance)")):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
            for ax, temp in zip(axes, sorted(test_metrics["temp_C"].unique())):
                subset = test_metrics[test_metrics["temp_C"] == temp].pivot(
                    index="n_samples", columns="wait_ms", values=metric)
                image = ax.imshow(subset.to_numpy(), aspect="auto", origin="lower",
                                  extent=[subset.columns.min(), subset.columns.max(),
                                          subset.index.min(), subset.index.max()])
                ax.set_title(f"{title} at {int(temp)}C")
                ax.set_xlabel("Wait (ms)")
                ax.set_ylabel("Samples")
                fig.colorbar(image, ax=ax, fraction=0.046)
            fig.suptitle(test)
            fig.tight_layout()
            fig.savefig(output_dir / f"{test}_{metric}_heatmap.png", dpi=150)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        chosen = recommendations_df.loc[recommendations_df["test"] == test].iloc[0]
        for temp in sorted(curve_points):
            if temp[0] == test:
                x, y = curve_points[temp]
                ax.plot(x, y, marker="o", label=f"{int(temp[1])}C")
        ax.scatter([chosen["wait_with_margin_ms"]], [0], marker="*", s=160, color="red",
                   label="Chosen point", zorder=5)
        ax.axvline(chosen["wait_with_margin_ms"], color="red", linestyle="--", alpha=0.6)
        ax.set(title=f"{test} settling curve", xlabel="Wait (ms)",
               ylabel="Mean settling error (% tolerance)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{test}_settling_curve.png", dpi=150)
        plt.close(fig)

    summary = [
        "# 02 Settling summary",
        "",
        "- Recommendations are evaluated at 25C and 90C; the 90C condition is always governing.",
        "- One-sided-limit tolerance is the distance from the observed median to that limit.",
        "- Default overhead allocation uses 50% of catalog time for settling/averaging as configured.",
        "",
        f"- Median recommended time saving: {recommendations_df['time_saved_pct'].median():.2f}%",
        f"- Hot/cold tau ratio mean: {tau_ratio['tau_ratio_90_over_25'].mean():.3f}",
        "",
        "## Assumption",
        "The catalog time is split into fixed overhead and settling/averaging; "
        "half of the settling allocation scales by sample count and half is represented by wait time.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Settling complete. Results written to {output_dir}.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

"""Split-conformal predictive RUN/SKIP testing."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from dcopt.config import load_config
from dcopt.io import load_datasets


FEATURES = ["CONT_VDIODE", "IIL_A", "IIH_A", "VOH", "VOL", "IDDQ_V1",
            "tester", "site", "temp_C", "die_radius"]
TARGETS = ["IDDQ_V2", "IDDQ_V3", "IDDQ_V4", "IDDQ_V5", "IIL_B", "IDD_STATIC"]


def clopper_pearson_upper(successes: int, trials: int, confidence: float) -> float:
    if trials == 0:
        return 1.0
    return float(beta.ppf(confidence, successes + 1, trials - successes))


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["die_radius"] = np.sqrt(frame["die_x"] ** 2 + frame["die_y"] ** 2)
    return frame


def conformal_quantile(residuals: pd.Series, alpha: float) -> float:
    residuals = np.sort(np.abs(residuals.to_numpy(dtype=float)))
    rank = math.ceil((len(residuals) + 1) * (1 - alpha))
    return float(residuals[min(rank - 1, len(residuals) - 1)])


def interval_inside_limits(lower: pd.Series, upper: pd.Series, lo: float | None,
                           hi: float | None, margin_fraction: float) -> pd.Series:
    tolerance = (float(hi) - float(lo)) if pd.notna(lo) and pd.notna(hi) else abs(float(hi if pd.notna(hi) else lo))
    margin = tolerance * margin_fraction
    inside = pd.Series(True, index=lower.index)
    if pd.notna(lo):
        inside &= lower >= float(lo) + margin
    if pd.notna(hi):
        inside &= upper <= float(hi) - margin
    return inside


def run(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    data = load_datasets(config_path)
    full, catalog = add_features(data.results), data.catalog
    train_lots = [str(lot) for lot in config["train_lots"]]
    holdout_lots = [str(lot) for lot in config["holdout_lots"]]
    train_pool = full[full["lot"].isin(train_lots)].copy()
    holdout = full[full["lot"].isin(holdout_lots)].copy()
    calibration_lots = sorted(train_pool["lot"].unique())[-2:]
    fit_lots = [lot for lot in train_pool["lot"].unique() if lot not in calibration_lots]
    fit = train_pool[train_pool["lot"].isin(fit_lots)].copy()
    calibration = train_pool[train_pool["lot"].isin(calibration_lots)].copy()

    output_dir = Path(config["paths"]["results_dir"])
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir /= "05_predictive"
    output_dir.mkdir(parents=True, exist_ok=True)

    alpha = float(config["conformal_alpha"])
    margin_fraction = float(config["limit_margin_pct"]) / 100
    catalog_lookup = catalog.set_index("test")
    metrics_rows = []
    predictions = []

    categorical = ["tester", "site"]
    numeric = [column for column in FEATURES if column not in categorical]
    for target in TARGETS:
        limit_row = catalog_lookup.loc[target]
        lo = limit_row["lo_limit"]
        hi = limit_row["hi_limit"]
        preprocessor = ColumnTransformer([
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
        ])
        model = Pipeline([
            ("preprocess", preprocessor),
            ("regressor", HistGradientBoostingRegressor(random_state=int(config["seed"]))),
        ])
        model.fit(fit[FEATURES], fit[target])
        calibration_prediction = model.predict(calibration[FEATURES])
        q = conformal_quantile(pd.Series(calibration[target].to_numpy() - calibration_prediction), alpha)
        holdout_prediction = model.predict(holdout[FEATURES])
        lower = pd.Series(holdout_prediction - q, index=holdout.index)
        upper = pd.Series(holdout_prediction + q, index=holdout.index)
        skip = interval_inside_limits(lower, upper, lo, hi, margin_fraction)
        actual_fail = holdout[f"{target}_PF"].eq("F")
        escaped = int((skip & actual_fail).sum())
        distance = holdout_prediction - holdout[target].to_numpy()
        per_time = float(catalog_lookup.loc[target, "test_time_ms"])

        group_frames = [
            ("overall", pd.Series("overall", index=holdout.index)),
            ("temp_C", holdout["temp_C"].astype(str)),
            ("tester", holdout["tester"].astype(str)),
            ("wafer_zone", np.where(holdout["die_radius"] <= float(config["wafer_zone_radius_threshold"]),
                                    "inner", "edge")),
        ]
        for dimension, grouping in group_frames:
            for level, indices in pd.Series(grouping, index=holdout.index).groupby(grouping).groups.items():
                indexer = holdout.index.intersection(indices)
                if len(indexer) == 0:
                    continue
                actual = holdout.loc[indexer, target]
                predicted = holdout_prediction[holdout.index.get_indexer(indexer)]
                covered = ((lower.loc[indexer] <= actual).to_numpy() &
                           (actual.to_numpy() <= upper.loc[indexer])).mean()
                group_skip = skip.loc[indexer]
                group_escape = int((group_skip & actual_fail.loc[indexer]).sum())
                metrics_rows.append({
                    "target": target, "dimension": dimension, "level": str(level),
                    "devices": len(indexer),
                    "mae": float(np.mean(np.abs(predicted - actual.to_numpy()))),
                    "coverage": float(covered),
                    "skip_rate": float(group_skip.mean()),
                    "skipped_escape_count": group_escape,
                    "escape_upper_bound": clopper_pearson_upper(
                        group_escape, int(group_skip.sum()), float(config["confidence"])),
                    "time_saved_ms_per_device": float(per_time * group_skip.mean()),
                })
        predictions.append(pd.DataFrame({
            "target": target, "ecid": holdout["ecid"],
            "actual": holdout[target].to_numpy(),
            "predicted": holdout_prediction,
            "lower": lower.to_numpy(),
            "upper": upper.to_numpy(),
            "decision": np.where(skip, "SKIP", "RUN"),
            "true_fail": actual_fail.to_numpy(),
        }))

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions_df = pd.concat(predictions, ignore_index=True)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)

    for target in TARGETS:
        target_predictions = predictions_df[predictions_df["target"] == target]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(target_predictions["actual"], target_predictions["predicted"],
                   s=8, alpha=0.35, c=np.where(target_predictions["decision"] == "SKIP", "green", "red"))
        bounds = [target_predictions["actual"].min(), target_predictions["actual"].max()]
        ax.plot(bounds, bounds, "k--", linewidth=1)
        ax.set(title=f"{target}: predicted vs actual", xlabel="Actual", ylabel="Predicted")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{target}_predicted_vs_actual.png", dpi=150)
        plt.close(fig)

    coverage = metrics[metrics["dimension"] != "overall"]
    fig, ax = plt.subplots(figsize=(12, 5))
    coverage_pivot = coverage.pivot_table(index="target", columns=["dimension", "level"], values="coverage")
    coverage_pivot.plot(ax=ax, marker="o")
    ax.axhline(1 - alpha, color="red", linestyle="--", label=f"Target {1-alpha:.2f}")
    ax.set(title="Interval coverage by group", ylabel="Coverage")
    ax.legend([], frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "coverage_by_group.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    overall = metrics[metrics["dimension"] == "overall"]
    ax.scatter(overall["skip_rate"], overall["skipped_escape_count"] /
               overall["devices"].clip(lower=1), s=70)
    for _, row in overall.iterrows():
        ax.annotate(row["target"], (row["skip_rate"], row["skipped_escape_count"] / row["devices"]))
    ax.set(title="Skip rate vs holdout escapes", xlabel="Skip rate", ylabel="Skipped escape rate")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "skip_rate_vs_escapes.png", dpi=150)
    plt.close(fig)

    # Audit holdout devices that fail IDDQ_V4 but pass every other catalog test.
    other_tests = [test for test in catalog_lookup.index.astype(str) if test != "IDDQ_V4"]
    v4_only = (holdout["IDDQ_V4_PF"].eq("F") &
               holdout[[f"{test}_PF" for test in other_tests]].eq("P").all(axis=1))
    v4_decisions = predictions_df[
        (predictions_df["target"] == "IDDQ_V4") &
        predictions_df["ecid"].isin(holdout.loc[v4_only, "ecid"])
    ][["ecid", "decision", "true_fail"]]
    v4_decisions.to_csv(output_dir / "IDDQ_V4_only_fail_devices.csv", index=False)

    (output_dir / "summary.md").write_text(
        "# 05 Predictive summary\n\n"
        "- Features contain only cheap measurements and process context; no downstream or latent-truth labels.\n"
        "- Calibration lots are selected separately from training lots within the configured train lots.\n"
        f"- IDDQ_V4-only holdout failures audited: {v4_only.sum()}\n"
        f"- IDDQ_V4-only failures skipped: {int((v4_decisions['decision'] == 'SKIP').sum())}\n",
        encoding="utf-8",
    )
    print(f"Predictive complete. Results written to {output_dir}.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

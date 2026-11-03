"""Part Average Testing using only temperature-specific sort-pass training features."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from dcopt.config import load_config
from dcopt.io import load_datasets


FEATURES = ["IDDQ_V1", "IDDQ_V2", "IDDQ_V3", "IDDQ_V4", "IDDQ_V5",
            "IDD_STATIC", "IIL_A", "IIH_A"]


def add_pat_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    leakage = frame["IIL_A"].copy()
    leakage = leakage.mask(leakage.abs() < 1e-6, 1e-6)
    for test in ["IDDQ_V1", "IDDQ_V2", "IDDQ_V3", "IDDQ_V4", "IDDQ_V5", "IDD_STATIC", "IIH_A"]:
        if test not in frame.columns:
            continue
        frame[f"{test}_over_IIL_A"] = frame[test] / leakage
    return frame


def metric_scores(y_true: np.ndarray, flagged: np.ndarray, score: np.ndarray) -> dict[str, float]:
    prevalence = float(y_true.mean())
    precision = float(precision_score(y_true, flagged, zero_division=0))
    recall = float(recall_score(y_true, flagged, zero_division=0))
    lift = precision / prevalence if prevalence else 0.0
    try:
        roc_auc = float(roc_auc_score(y_true, score))
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, score))
    except ValueError:
        pr_auc = float("nan")
    return {"precision": precision, "recall": recall, "lift": lift,
            "roc_auc": roc_auc, "pr_auc": pr_auc}


def top_score_flags(scores: np.ndarray, max_fraction: float) -> np.ndarray:
    """Select at most the configured fraction using deterministic score order."""
    max_count = int(np.floor(len(scores) * max_fraction))
    if max_count <= 0:
        return np.zeros(len(scores), dtype=bool)
    order = np.argsort(scores, kind="stable")[::-1]
    flags = np.zeros(len(scores), dtype=bool)
    flags[order[:max_count]] = True
    return flags


def run(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    data = load_datasets(config_path)
    full = add_pat_features(data.results)
    train = full[
        full["lot"].isin([str(lot) for lot in config["train_lots"]]) &
        full["sort_pass"].eq(1)
    ].copy()
    holdout = full[
        full["lot"].isin([str(lot) for lot in config["holdout_lots"]]) &
        full["sort_pass"].eq(1)
    ].copy()
    augmented_features = FEATURES + [f"{test}_over_IIL_A" for test in
                                     ["IDDQ_V1", "IDDQ_V2", "IDDQ_V3", "IDDQ_V4", "IDDQ_V5",
                                      "IDD_STATIC", "IIH_A"]]
    output_dir = Path(config["paths"]["results_dir"])
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir /= "07_pat"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_flag_fraction = float(config["pat_max_flag_pct"]) / 100
    scored_frames = []
    metrics_rows = []
    flagged_frames = []
    for temp in sorted(full["temp_C"].unique()):
        train_group = train[train["temp_C"].eq(temp)].copy()
        holdout_group = holdout[holdout["temp_C"].eq(temp)].copy()
        if train_group.empty or holdout_group.empty:
            continue
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_group[augmented_features])
        holdout_scaled = scaler.transform(holdout_group[augmented_features])

        robust = MinCovDet(random_state=int(config["seed"]), support_fraction=None).fit(train_scaled)
        mah_train = robust.mahalanobis(train_scaled)
        mah_hold = robust.mahalanobis(holdout_scaled)

        isolation = IsolationForest(
            n_estimators=300,
            contamination=max_flag_fraction,
            random_state=int(config["seed"]),
        ).fit(train_scaled)
        iso_train = -isolation.decision_function(train_scaled)
        iso_hold = -isolation.decision_function(holdout_scaled)

        # Combine fixed-rate thresholds by temperature. This keeps the global PAT flag
        # budget bounded while selecting the larger score-tail population.
        mah_flags = top_score_flags(mah_hold, max_flag_fraction)
        iso_flags = top_score_flags(iso_hold, max_flag_fraction)

        # Six-sigma flags any feature outside mean +/- 6 sigma.
        means = train_group[augmented_features].mean()
        sigmas = train_group[augmented_features].std(ddof=1).replace(0, np.nan)
        univariate_exclusions = (
            (holdout_group[augmented_features] < means - 6 * sigmas).any(axis=1).to_numpy()
            | (holdout_group[augmented_features] > means + 6 * sigmas).any(axis=1).to_numpy()
        )
        holdout_z = ((holdout_group[augmented_features] - means) / sigmas).abs()
        univariate_score = holdout_z.max(axis=1).to_numpy()
        univariate_cap_flags = top_score_flags(univariate_score, max_flag_fraction)
        univariate_flags = univariate_exclusions & univariate_cap_flags

        score_frame = holdout_group[["ecid", "temp_C", "lot", "wafer", "burnin_fail",
                                     "rma", "is_latent_defect_truth"]].copy()
        score_frame["mahalanobis_score"] = mah_hold
        score_frame["isolation_forest_score"] = iso_hold
        score_frame["mahalanobis_flag"] = mah_flags
        score_frame["isolation_forest_flag"] = iso_flags
        score_frame["univariate_6sigma_flag"] = univariate_flags
        scored_frames.append(score_frame)

        for method, flagged, score in (
            ("Mahalanobis", mah_flags, mah_hold),
            ("IsolationForest", iso_flags, iso_hold),
            ("Univariate6Sigma", univariate_flags, univariate_flags.astype(float)),
        ):
            for label in ("is_latent_defect_truth", "burnin_fail", "rma"):
                truth = score_frame[label].fillna(0).to_numpy(dtype=int)
                scores = metric_scores(truth, flagged.astype(bool), score)
                metrics_rows.append({
                    "temp_C": temp, "method": method, "label": label,
                    "flagged_count": int(flagged.sum()),
                    "flag_pct": float(flagged.mean() * 100),
                    "burnin_fails_caught": int(((truth == 1) & flagged).sum())
                    if label == "burnin_fail" else "",
                    **scores,
                })

    scored = pd.concat(scored_frames, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    scored.to_csv(output_dir / "flagged_devices.csv", index=False)
    flagged = scored[
        scored["mahalanobis_flag"] | scored["isolation_forest_flag"] | scored["univariate_6sigma_flag"]
    ]
    flagged.to_csv(output_dir / "flagged_devices_table.csv", index=False)

    for method, score_col, flag_col in (
        ("Mahalanobis", "mahalanobis_score", "mahalanobis_flag"),
        ("IsolationForest", "isolation_forest_score", "isolation_forest_flag"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(scored[score_col], bins=40, color="#A0A0A0", alpha=0.65, label="Sort-pass")
        latent = scored[scored["is_latent_defect_truth"].eq(1)]
        ax.hist(latent[score_col], bins=20, color="#E15759", alpha=0.9, label="Latent truth")
        ax.set(title=f"{method} PAT score", xlabel="PAT score", ylabel="Devices")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{method.lower()}_score_histogram.png", dpi=150)
        plt.close(fig)

    (output_dir / "summary.md").write_text(
        "# 07 PAT summary\n\n"
        "- Features are limited to sort-pass measurements and leakage-normalized ratios.\n"
        "- Latent defect truth is used only after fitting to calculate holdout metrics.\n"
        f"- Maximum configured flag rate: {config['pat_max_flag_pct']}% per temperature.\n"
        f"- Holdout sort-pass devices flagged: {len(flagged):,}; unique ECIDs: {flagged['ecid'].nunique():,}\n"
        "- Burn-in and RMA are downstream scoring labels, never model features.\n",
        encoding="utf-8",
    )
    print(f"PAT complete. Results written to {output_dir}.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

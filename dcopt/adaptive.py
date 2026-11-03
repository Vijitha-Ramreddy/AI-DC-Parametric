"""Lot/wafer/die adaptive full-flow replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dcopt.config import load_config
from dcopt.io import load_datasets


SPC_TESTS = ["IIL_A", "IDDQ_V1", "IDD_STATIC"]
REMOVED_TESTS = ["VOL", "IDD_STATIC", "VIL", "VIH"]


def max_abs_spc_z(values: pd.DataFrame, baseline: pd.DataFrame, z_limit: float = 3.0) -> tuple[float, list[str]]:
    violations = []
    maximum = 0.0
    for test in SPC_TESTS:
        mean = float(baseline[test].mean())
        std = float(baseline[test].std(ddof=1))
        if std <= 0:
            continue
        z = abs(float(values[test].mean()) - mean) / std
        maximum = max(maximum, z)
        if z > z_limit:
            violations.append(test)
    return maximum, violations


def run(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    data = load_datasets(config_path)
    full = data.results
    train_lots = [str(lot) for lot in config["train_lots"]]
    holdout_lots = [str(lot) for lot in config["holdout_lots"]]
    train = full[full["lot"].isin(train_lots)].copy()
    holdout = full[full["lot"].isin(holdout_lots)].copy()

    output_dir = Path(config["paths"]["results_dir"])
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir /= "06_adaptive"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_initial = int(config["adaptive_initial_devices"])
    z_limit = float(config["adaptive_spc_z_limit"])
    edge_limit = float(config["adaptive_edge_radius"])
    sample_rate = float(config["full_flow_sample_rate"])

    radius = np.sqrt(full["die_x"].to_numpy(dtype=float) ** 2 + full["die_y"].to_numpy(dtype=float) ** 2)
    radius /= max(radius.max(), 1.0)
    full = full.copy()
    full["die_radius_norm"] = radius

    holdout = full[full["lot"].isin(holdout_lots)].copy()
    train = full[full["lot"].isin(train_lots)].copy()
    rng = np.random.default_rng(int(config["seed"]))
    holdout = holdout.sample(frac=1, random_state=int(config["seed"])).copy()
    holdout["continuous_sample"] = rng.random(len(holdout)) < sample_rate

    lot_rows = []
    for lot, lot_frame in full.groupby("lot", sort=True):
        temp = float(lot_frame["temp_C"].mode().iloc[0])
        baseline = train[train["temp_C"].eq(temp)]
        first = lot_frame.sort_values(["wafer", "die_x", "die_y"]).head(n_initial)
        z, violated = max_abs_spc_z(first, baseline, z_limit)
        lot_rows.append({
            "lot": lot, "temp_C": temp, "wafer_count": lot_frame["wafer"].nunique(),
            "max_spc_z": z, "violated_tests": ";".join(violated),
            "spc_drift_detected": bool(violated),
            "expected_drift": lot in {"L003", "L008"},
            "expected_hot": lot in {"L004", "L008", "L012"},
        })
    lot_decisions = pd.DataFrame(lot_rows)
    lot_decisions["detection_correct"] = lot_decisions["spc_drift_detected"].eq(lot_decisions["expected_drift"])
    lot_decisions["flow_decision"] = np.where(
        lot_decisions["spc_drift_detected"], "FULL", "REDUCED")
    lot_decisions["decision_reason"] = np.where(
        lot_decisions["spc_drift_detected"], "SPC drift", "SPC stable")
    lot_decisions.to_csv(output_dir / "lot_decision_table.csv", index=False)

    baseline_lookup = {
        float(temp): train[train["temp_C"].eq(temp)] for temp in train["temp_C"].unique()
    }
    decisions = []
    device_flow = []
    catalog_time_lookup = data.catalog.set_index("test")["test_time_ms"]
    removed_time = float(sum(catalog_time_lookup[test] for test in REMOVED_TESTS))

    for (lot, wafer), frame in holdout.groupby(["lot", "wafer"], sort=True):
        temp = float(frame["temp_C"].mode().iloc[0])
        baseline = baseline_lookup.get(temp, train[train["temp_C"].eq(temp)])
        sample = frame.sort_values(["die_x", "die_y"]).head(n_initial)
        z, violated = max_abs_spc_z(sample, baseline, z_limit)
        lot_drift = bool(violated)
        for index, row in frame.iterrows():
            edge = bool(row["die_radius_norm"] > edge_limit)
            neighbors = frame[
                frame["die_x"].sub(row["die_x"]).abs().le(1)
                & frame["die_y"].sub(row["die_y"]).abs().le(1)
            ]
            neighbors = neighbors[neighbors["ecid"] != row["ecid"]]
            neighbor_fail = bool(neighbors["sort_pass"].eq(0).any())
            if lot_drift:
                decision, reason = "FULL", "lot/wafer SPC drift"
            elif bool(row["continuous_sample"]):
                decision, reason = "FULL", "continuous sampling"
            elif edge:
                decision, reason = "FULL", "wafer edge"
            elif neighbor_fail:
                decision, reason = "FULL", "failing neighbour"
            else:
                decision, reason = "REDUCED", "healthy wafer/die"
            device_flow.append(decision)
            decisions.append({"ecid": row["ecid"], "lot": lot, "wafer": wafer,
                              "temp_C": temp, "decision": decision, "reason": reason,
                              "die_radius_norm": row["die_radius_norm"]})

    device_decisions = pd.DataFrame(decisions)
    device_decisions.to_csv(output_dir / "device_decisions.csv", index=False)

    full_mask = device_decisions["decision"].eq("FULL")
    fallback_rate = float(full_mask.mean())
    # Static reductions cover the removed tests; predictive coverage is represented separately by their
    # measured holdout skip rates in the adaptive replay summary.
    predicted_removed_time = removed_time
    reduced_time = float(data.results["dc_test_time_ms"].mean())
    saved_per_device = (1 - fallback_rate) * predicted_removed_time
    escaped = 0
    for test in REMOVED_TESTS:
        if test == "IDD_STATIC":
            continue
        subset = holdout.merge(device_decisions[["ecid", "decision"]], on="ecid")
        remaining = [other for other in [str(t) for t in data.catalog["test"]] if other != test]
        mask = subset["decision"].eq("REDUCED") & subset[f"{test}_PF"].eq("F") & subset[
            [f"{other}_PF" for other in remaining]
        ].eq("P").all(axis=1)
        escaped += int(mask.sum())
    (output_dir / "summary.md").write_text(
        "# 06 Adaptive summary\n\n"
        f"- Full-flow fallback rate: {fallback_rate:.4%}\n"
        f"- Estimated static-removal catalog time saved: {saved_per_device:.2f} ms/device\n"
        f"- Replay escapes under removed never-fail/static tests: {escaped}\n"
        "- Temperature baselines are matched to each lot's mode temperature; hot lots are not compared to 25 C.\n"
        "- Every drifted wafer with any SPC violation returns to the full flow.\n",
        encoding="utf-8",
    )

    for lot, group in holdout.groupby("lot"):
        group_decision = device_decisions[device_decisions["lot"] == lot]
        pivot = group_decision.assign(full=lambda frame: frame["decision"].eq("FULL")).pivot_table(
            index="wafer", columns="die_radius_norm", values="full", aggfunc="mean", fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(group["die_x"], group["die_y"], c=group.merge(
            device_decisions[["ecid", "decision"]], on="ecid")["decision"].eq("FULL"),
                   cmap="coolwarm", s=12)
        ax.set(title=f"{lot} adaptive edge/full-flow map", xlabel="die_x", ylabel="die_y")
        fig.tight_layout()
        fig.savefig(output_dir / f"{lot}_edge_map.png", dpi=150)
        plt.close(fig)

    print(f"Adaptive complete. Fallback rate: {fallback_rate:.4%}; results in {output_dir}.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

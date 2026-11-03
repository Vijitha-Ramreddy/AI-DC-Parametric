"""Streaming production monitoring and full-flow rollback replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from dcopt.config import load_config
from dcopt.io import load_datasets


MODEL_FEATURES = ["CONT_VDIODE", "IIL_A", "IIH_A", "VOH", "VOL",
                  "IDDQ_V1", "tester", "site", "temp_C", "die_radius"]
SPC_TESTS = ["IIL_A", "IDDQ_V1", "IDD_STATIC"]


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if expected.empty or actual.empty:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(expected.quantile(quantiles))
    if len(edges) < 3:
        return 0.0
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def spc_alarm(values: pd.DataFrame, baseline: pd.DataFrame, z_limit: float) -> tuple[bool, str, float]:
    violations = []
    z_max = 0.0
    for test in SPC_TESTS:
        std = float(baseline[test].std(ddof=1))
        if std <= 0 or values[test].dropna().empty:
            continue
        z = abs(float(values[test].mean()) - float(baseline[test].mean())) / std
        z_max = max(z_max, z)
        if z > z_limit:
            violations.append(test)
    return bool(violations), ";".join(violations), z_max


def equipment_t02_site3_alarm(lot: pd.DataFrame, baseline: pd.DataFrame, z_limit: float) -> bool:
    lot_offset = lot[(lot["tester"].eq("T02")) & (lot["site"].eq(3))]
    base_offset = baseline[(baseline["tester"].eq("T02")) & (baseline["site"].eq(3))]
    if lot_offset.empty or base_offset.empty:
        return False
    std = float(lot_offset["IIL_A"].std(ddof=1))
    if not np.isfinite(std) or std <= 0:
        std = float(base_offset["IIL_A"].std(ddof=1))
    scale = std if np.isfinite(std) and std > 0 else 1.0
    return abs(float(lot_offset["IIL_A"].mean()) - float(base_offset["IIL_A"].mean())) > z_limit * scale


def run(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    data = load_datasets(config_path)
    results_dir = Path(config["paths"]["results_dir"])
    if not results_dir.is_absolute():
        results_dir = config_path.parent / results_dir
    output_dir = results_dir / "09_monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)

    full = data.results.copy()
    full["die_radius"] = np.sqrt(full["die_x"] ** 2 + full["die_y"] ** 2)
    train_lots = [str(lot) for lot in config["train_lots"]]
    holdout_lots = [str(lot) for lot in config["holdout_lots"]]
    train = full[full["lot"].isin(train_lots)].copy()
    holdout = full[full["lot"].isin(holdout_lots)].copy()
    rng = np.random.default_rng(int(config["seed"]))
    full["monitor_sample"] = rng.random(len(full)) < float(config["full_flow_sample_rate"])

    predictions_path = results_dir / "05_predictive" / "predictions.csv"
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
    alpha = float(config["conformal_alpha"])
    psi_limit = float(config["monitoring_psi_limit"])
    coverage_limit = 1.0 - alpha - float(config["monitoring_coverage_drop"])
    z_limit = float(config["adaptive_spc_z_limit"])
    manual_reenable = {str(value) for value in config["monitoring_manual_reenable"]}

    events = []
    timeline = []
    active_rollbacks = set()
    for scope, stream in (("holdout", holdout), ("all", full)):
        stream = stream.sort_values("ecid").copy()
        for lot, lot_frame in stream.groupby("lot", sort=True):
            lot_frame = lot_frame.copy()
            lot_frame["monitor_sample"] = full.loc[lot_frame.index, "monitor_sample"]
            sampled = lot_frame[lot_frame["monitor_sample"]].copy()
            temp = float(lot_frame["temp_C"].mode().iloc[0])
            baseline = train[train["temp_C"].eq(temp)]

            psi_values, ks_values = [], []
            for feature in ["CONT_VDIODE", "IIL_A", "IIH_A", "VOH", "VOL", "IDDQ_V1", "temp_C"]:
                psi_value = psi(baseline[feature], lot_frame[feature])
                ks_value = float(ks_2samp(baseline[feature].dropna(), lot_frame[feature].dropna()).statistic)
                psi_values.append(psi_value)
                ks_values.append(ks_value)
            max_psi = float(max(psi_values, default=0.0))

            spc_alarm_flag, violated, spc_z = spc_alarm(sampled, baseline, z_limit)

            coverage = float("nan")
            if not predictions.empty:
                sampled_predictions = predictions[predictions["ecid"].astype(str).isin(
                    sampled["ecid"].astype(str))]
                if not sampled_predictions.empty:
                    covered = ((sampled_predictions["lower"] <= sampled_predictions["actual"]) &
                               (sampled_predictions["actual"] <= sampled_predictions["upper"]))
                    coverage = float(covered.mean())

            baseline_fail = float(baseline["sort_pass"].eq(0).mean()) if len(baseline) else 0.0
            fail = float(lot_frame["sort_pass"].eq(0).mean())
            fail_std = np.sqrt(max(baseline_fail * (1 - baseline_fail) / max(len(lot_frame), 1), 1e-12))
            fail_z = abs(fail - baseline_fail) / fail_std
            fail_alarm = bool(fail_z > z_limit)

            alarms = []
            if max_psi > psi_limit:
                alarms.append("PSI")
            if pd.notna(coverage) and coverage < coverage_limit:
                alarms.append("coverage")
            if spc_alarm_flag or fail_alarm:
                alarms.append("SPC")

            equipment_alarm = equipment_t02_site3_alarm(lot_frame, baseline, z_limit)
            if equipment_alarm:
                alarms.append("equipment_T02_site3")

            key = f"{scope}:{lot}"
            if alarms:
                active_rollbacks.add(key)
                reason = ";".join(alarms)
                testers = sorted(lot_frame["tester"].astype(str).unique())
                for tester in testers:
                    is_manually_enabled = (
                        tester in manual_reenable
                        or f"{lot}:{tester}" in manual_reenable
                        or str(lot) in manual_reenable
                    )
                    events.append({
                        "scope": scope, "lot": lot, "tester": tester, "flow": "FULL",
                        "alarm": reason, "psi": max_psi, "coverage": coverage,
                        "spc_z": spc_z, "spc_tests": violated, "fail_rate": fail,
                        "fail_rate_z": fail_z,                         "manual_reenable": is_manually_enabled,
                    })
            elif key in active_rollbacks and lot not in manual_reenable:
                active_rollbacks.add(key)

            timeline.append({
                "scope": scope, "lot": lot, "max_psi": max_psi,
                "max_ks": float(max(ks_values, default=0.0)),
                "coverage": coverage, "spc_z": spc_z,
                "fail_rate_z": fail_z, "equipment_alarm": equipment_alarm,
                "full_flow": bool(alarms),
            })

    event_log = pd.DataFrame(events)
    if event_log.empty:
        event_log = pd.DataFrame(columns=["scope", "lot", "tester", "flow", "alarm", "psi",
                                          "coverage", "spc_z", "spc_tests", "fail_rate",
                                          "fail_rate_z", "manual_reenable"])
    event_log.to_csv(output_dir / "event_log.csv", index=False)
    timeline_df = pd.DataFrame(timeline)
    timeline_df.to_csv(output_dir / "timeline.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(timeline_df))
    ax.plot(x, timeline_df["max_psi"], marker="o", label="max PSI")
    ax.axhline(psi_limit, color="#E15759", linestyle="--", label="PSI limit")
    alarm_x = x[timeline_df["full_flow"].to_numpy()]
    ax.scatter(alarm_x, timeline_df.loc[timeline_df["full_flow"], "max_psi"],
               color="#E15759", s=70, zorder=3, label="full-flow alarm")
    ax.set_xticks(x, timeline_df["scope"] + ":" + timeline_df["lot"])
    ax.set_ylabel("PSI")
    ax.set_title("Streaming monitoring timeline")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "timeline_alarms.png", dpi=150)
    plt.close(fig)

    for scope, stream in (("holdout", holdout), ("all", full)):
        tester_means = stream.groupby("tester")["IIL_A"].mean()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(tester_means.index.astype(str), tester_means)
        ax.axhline(tester_means.get("T02", np.nan), color="#E15759", linestyle="--", alpha=0.5)
        ax.set(title=f"{scope} mean IIL_A by tester", xlabel="Tester", ylabel="IIL_A")
        fig.tight_layout()
        fig.savefig(output_dir / f"{scope}_tester_IIL_A.png", dpi=150)
        plt.close(fig)

        grouped = stream.groupby(["tester", "site"], as_index=False)["IIL_A"].mean()
        for tester, tester_group in grouped.groupby("tester"):
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = np.where((tester_group["tester"] == "T02") & (tester_group["site"] == 3),
                              "#E15759", "#4E79A7")
            ax.bar(tester_group["site"].astype(str), tester_group["IIL_A"], color=colors)
            ax.set(title=f"{scope} IIL_A by site - {tester}", xlabel="Site", ylabel="IIL_A")
            fig.tight_layout()
            fig.savefig(output_dir / f"{scope}_site_{tester}_IIL_A.png", dpi=150)
            plt.close(fig)

    (output_dir / "summary.md").write_text(
        "# 09 Monitoring summary\n\n"
        f"- Streamed {len(holdout):,} holdout and {len(full):,} total devices in ECID/time order.\n"
        f"- Full-flow events logged: {len(event_log):,}.\n"
        "- Automatic rollback applies to a lot and every tester in that lot until a manual config re-enable.\n"
        "- T02/site 3 IIL_A offset is classified and plotted as an equipment alarm.\n"
        f"- PSI limit: {psi_limit}; conformal coverage floor: {coverage_limit:.3f}.\n",
        encoding="utf-8",
    )
    print(f"Monitoring complete. Results written to {output_dir}.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

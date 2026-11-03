"""Train-only static reduction and holdout replay."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

from dcopt.baseline import cpk
from dcopt.config import load_config
from dcopt.io import load_datasets
from dcopt.risk import escape_dppm


def pf_column(frame: pd.DataFrame, test: str) -> pd.Series:
    return frame[f"{test}_PF"].astype(str).str.upper()


def run(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    data = load_datasets(config_path)
    full, catalog = data.results, data.catalog
    train_lots = [str(lot) for lot in config["train_lots"]]
    holdout_lots = [str(lot) for lot in config["holdout_lots"]]
    train = full[full["lot"].isin(train_lots)].copy()
    holdout = full[full["lot"].isin(holdout_lots)].copy()
    tests = catalog["test"].astype(str).tolist()

    output_dir = Path(config["paths"]["results_dir"])
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir /= "03_static"
    output_dir.mkdir(parents=True, exist_ok=True)

    table_rows = []
    unique_rows = []
    correlation_all = train[tests].corr(method="pearson")
    correlation_spearman = train[tests].corr(method="spearman")
    correlation_excluding = train.loc[
        ~((train["tester"].eq("T02")) & (train["site"].eq(3))), tests
    ].corr(method="pearson")
    correlation_all.to_csv(output_dir / "correlation_pearson.csv")
    correlation_spearman.to_csv(output_dir / "correlation_spearman.csv")
    correlation_excluding.to_csv(output_dir / "correlation_pearson_excluding_T02_site3.csv")

    for row in catalog.itertuples(index=False):
        test = str(row.test)
        pf = pf_column(train, test)
        fails = int(pf.eq("F").sum())
        other_tests = [other for other in tests if other != test]
        other_pass = train[[f"{other}_PF" for other in other_tests]].eq("P").all(axis=1)
        unique_fails = int((pf.eq("F") & other_pass).sum())
        mi_sort = float(mutual_info_regression(train[[test]].fillna(train[test].median()),
                                               train["sort_pass"].fillna(0).astype(int),
                                               random_state=config["seed"])[0])
        burnin = train["burnin_fail"].fillna(0)
        mi_burnin = float(mutual_info_regression(train[[test]].fillna(train[test].median()),
                                                 burnin.astype(int),
                                                 random_state=config["seed"])[0])
        cpk_value = cpk(train[test], row.lo_limit, row.hi_limit)
        never_fail = fails == 0 and cpk_value is not None and cpk_value >= 1.33
        unique_rows.append({"test": test, "unique_fail_count": unique_fails,
                            "total_fail_count": fails, "never_fail": never_fail,
                            "mi_sort_pass": mi_sort, "mi_burnin_fail": mi_burnin,
                            "cpk": cpk_value})
    unique_df = pd.DataFrame(unique_rows)
    unique_df.to_csv(output_dir / "unique_fail_counts.csv", index=False)

    sampling_rates = [float(rate) for rate in config["sampling_rates"]] + [0.0]
    decision_rows = []
    budget = float(config["dppm_budget"])
    confidence = float(config["confidence"])
    for test_row in unique_df.itertuples(index=False):
        best_sampling = None
        best_savings = 0.0
        for sampling_rate in sorted(set(sampling_rates), reverse=True):
            dppm = escape_dppm(int(test_row.unique_fail_count), len(train), sampling_rate, confidence)
            if dppm <= budget:
                savings = float(catalog.loc[catalog["test"] == test_row.test, "test_time_ms"].iloc[0]) * (1 - sampling_rate)
                if savings > best_savings:
                    best_sampling, best_savings = sampling_rate, savings
        if bool(test_row.never_fail):
            decision, sampling = "REMOVE", 0.0
        elif test_row.unique_fail_count < 3:
            decision, sampling = "KEEP", None
        elif best_sampling is not None:
            decision = "REMOVE" if best_sampling == 0.0 else "SAMPLE"
            sampling = best_sampling
        else:
            decision, sampling = "KEEP", None
        if decision == "KEEP":
            estimated_dppm = escape_dppm(int(test_row.unique_fail_count), len(train), 0.05, confidence)
        else:
            estimated_dppm = escape_dppm(int(test_row.unique_fail_count), len(train),
                                         0.0 if sampling == 0 else sampling, confidence)
        decision_rows.append({"test": test_row.test, "decision": decision, "sampling_rate": sampling,
                              "dppm_upper_bound": estimated_dppm,
                              "unique_fail_count": int(test_row.unique_fail_count),
                              "rare_failure_high_risk": bool(test_row.unique_fail_count < 3)})
    decisions = pd.DataFrame(decision_rows)
    decisions["dppm_budget"] = budget
    for column in ("total_fail_count", "never_fail", "mi_sort_pass", "mi_burnin_fail", "cpk"):
        lookup = unique_df.set_index("test")[column]
        decisions[column] = decisions["test"].map(lookup)
    decisions.to_csv(output_dir / "decision_table.csv", index=False)

    # Replay the train-selected coverage policy on holdout measurements.
    removed = decisions.loc[decisions["decision"].isin(["REMOVE"]), "test"].tolist()
    held_full = holdout.copy()
    escaped = pd.DataFrame()
    escape_counts = []
    for test in removed:
        remaining = [other for other in tests if other != test]
        mask = pf_column(held_full, test).eq("F") & held_full[[f"{other}_PF" for other in remaining]].eq("P").all(axis=1)
        escape_counts.append({"removed_test": test, "holdout_escape_count": int(mask.sum())})
    escaped = pd.DataFrame(escape_counts)
    escaped.to_csv(output_dir / "holdout_escapes.csv", index=False)

    catalog_time = catalog.set_index("test")["test_time_ms"]
    saved = decisions.apply(lambda row: float(catalog_time[row["test"]]) *
                            (1 - float(row["sampling_rate"] or 0.0)) if row["decision"] == "REMOVE"
                            else float(catalog_time[row["test"]]) * (1 - float(row["sampling_rate"]))
                            if row["decision"] == "SAMPLE" else 0.0, axis=1).sum()
    with open(output_dir / "summary.md", "w", encoding="utf-8") as handle:
        handle.write("# 03 Static reduction summary\n\n")
        handle.write(f"- Decisions: {decisions['decision'].value_counts().to_dict()}\n")
        handle.write(f"- Holdout devices replayed: {len(holdout):,}\n")
        handle.write(f"- Holdout real escapes: {int(escaped['holdout_escape_count'].sum()) if len(escaped) else 0}\n")
        handle.write(f"- Estimated full-flow catalog time saved: {saved:.2f} ms/device\n")
        handle.write("- All predictions were trained using train lots only; holdout data was replayed afterward.\n")
        handle.write("- Rare-failure count below 3 defaults to KEEP/high-risk unless explicitly proven safe.\n")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(correlation_all, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Train Pearson measurement correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(correlation_spearman, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Train Spearman measurement correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_spearman_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    unique_df.sort_values("unique_fail_count", ascending=False).plot.bar(
        x="test", y="unique_fail_count", ax=ax, color="#E15759", legend=False)
    ax.set(title="Unique-fail count by test", ylabel="Devices", xlabel="Test")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "unique_fail_bar.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(decisions["dppm_upper_bound"], decisions.apply(lambda row:
        float(catalog_time[row["test"]]) * (1 - float(row["sampling_rate"] or 0.0))
        if row["decision"] == "REMOVE" else 0.0, axis=1), s=60)
    ax.axvline(budget, color="red", linestyle="--", label="DPPM budget")
    ax.set(xlabel="Estimated DPPM upper bound", ylabel="Estimated time saved (ms)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "savings_vs_dppm.png", dpi=150)
    plt.close(fig)

    print(f"Static reduction complete. Results written to {output_dir}.")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

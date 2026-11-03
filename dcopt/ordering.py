"""Fail-probability-per-second DC test-flow ordering."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dcopt.config import load_config
from dcopt.io import load_datasets


def ranked_order(train: pd.DataFrame, catalog: pd.DataFrame,
                 keep_relay_groups_together: bool = False) -> pd.DataFrame:
    """Rank tests by training fail probability per catalog test millisecond."""
    rows = []
    for row in catalog.itertuples(index=False):
        test = str(row.test)
        fail_probability = float(train[f"{test}_PF"].eq("F").mean())
        rows.append({
            "test": test,
            "group": str(row.group),
            "test_time_ms": float(row.test_time_ms),
            "fail_probability": fail_probability,
            "fail_probability_per_ms": fail_probability / float(row.test_time_ms),
        })
    ranking = pd.DataFrame(rows)
    if keep_relay_groups_together:
        ranking = ranking.sort_values("fail_probability_per_ms", ascending=False, kind="stable")
        priority = ranking.groupby("group", sort=False)["fail_probability_per_ms"].max()
        group_order = priority.sort_values(ascending=False).index.tolist()
        pieces = [ranking[ranking["group"] == group].sort_values(
            "fail_probability_per_ms", ascending=False, kind="stable") for group in group_order]
        ranking = pd.concat(pieces, ignore_index=True)
        ranking["rank"] = range(1, len(ranking) + 1)
    else:
        ranking = ranking.sort_values("fail_probability_per_ms", ascending=False, kind="stable")
        ranking["rank"] = range(1, len(ranking) + 1)
    return ranking


def stop_first_fail_time(frame: pd.DataFrame, order: list[str],
                         catalog: pd.DataFrame) -> pd.Series:
    """Replay stop-on-first-fail time for each device."""
    times = catalog.set_index("test")["test_time_ms"].to_dict()
    total = pd.Series(0.0, index=frame.index)
    any_fail = pd.Series(False, index=frame.index)
    cumulative = 0.0
    for test in order:
        cumulative += float(times[test])
        still_waiting = ~any_fail
        this_fail = frame[f"{test}_PF"].eq("F") & still_waiting
        total.loc[this_fail] = cumulative
        any_fail |= this_fail
    total.loc[~any_fail] = cumulative
    return total


def run(config_path: str | Path) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    data = load_datasets(config_path)
    full, catalog = data.results, data.catalog
    train_lots = [str(lot) for lot in config["train_lots"]]
    holdout_lots = [str(lot) for lot in config["holdout_lots"]]
    train = full[full["lot"].isin(train_lots)].copy()
    holdout = full[full["lot"].isin(holdout_lots)].copy()

    output_dir = Path(config["paths"]["results_dir"])
    if not output_dir.is_absolute():
        output_dir = config_path.parent / output_dir
    output_dir /= "04_ordering"
    output_dir.mkdir(parents=True, exist_ok=True)

    keep_groups = bool(config.get("keep_relay_groups_together", False))
    ranking = ranked_order(train, catalog, keep_groups)
    original_order = catalog["test"].astype(str).tolist()
    ranked_order_list = ranking["test"].tolist()
    ranking.insert(0, "new_rank", ranking["rank"])
    ranking.to_csv(output_dir / "new_order.csv", index=False)

    original_time = stop_first_fail_time(holdout, original_order, catalog)
    ranked_time = stop_first_fail_time(holdout, ranked_order_list, catalog)
    passing = holdout["sort_pass"].eq(1)
    comparison = pd.DataFrame([
        {"flow": "original", "population": "all", "mean_dc_time_ms": original_time.mean(),
         "median_dc_time_ms": original_time.median(), "devices": len(holdout)},
        {"flow": "ranked", "population": "all", "mean_dc_time_ms": ranked_time.mean(),
         "median_dc_time_ms": ranked_time.median(), "devices": len(holdout)},
        {"flow": "original", "population": "passing", "mean_dc_time_ms": original_time.loc[passing].mean(),
         "median_dc_time_ms": original_time.loc[passing].median(), "devices": int(passing.sum())},
        {"flow": "ranked", "population": "passing", "mean_dc_time_ms": ranked_time.loc[passing].mean(),
         "median_dc_time_ms": ranked_time.loc[passing].median(), "devices": int(passing.sum())},
        {"flow": "original", "population": "failing", "mean_dc_time_ms": original_time.loc[~passing].mean(),
         "median_dc_time_ms": original_time.loc[~passing].median(), "devices": int((~passing).sum())},
        {"flow": "ranked", "population": "failing", "mean_dc_time_ms": ranked_time.loc[~passing].mean(),
         "median_dc_time_ms": ranked_time.loc[~passing].median(), "devices": int((~passing).sum())},
    ])
    comparison.to_csv(output_dir / "time_comparison.csv", index=False)

    pf_tests = [f"{test}_PF" for test in catalog["test"].astype(str)]
    expected_pass = holdout[pf_tests].eq("P").all(axis=1).astype(int)
    actual_pass = holdout["sort_pass"].eq(1).astype(int)
    bin_compare = pd.DataFrame({
        "hard_bin": holdout["hard_bin"],
        "actual_sort_pass": actual_pass,
        "replayed_sort_pass": expected_pass,
    })
    bin_unchanged = bool(actual_pass.equals(expected_pass))
    bin_distribution = holdout["hard_bin"].value_counts().rename_axis("hard_bin").reset_index(name="count")
    bin_distribution.to_csv(output_dir / "bin_verification.csv", index=False)
    bin_compare.to_csv(output_dir / "device_pass_verification.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = comparison.pivot(index="population", columns="flow", values="mean_dc_time_ms")
    pivot.plot.bar(ax=ax, color={"original": "#4C78A8", "ranked": "#59A14F"})
    ax.set(title="Holdout stop-on-first-fail mean DC time", ylabel="Mean DC time (ms)")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ordering_time_comparison.png", dpi=150)
    plt.close(fig)

    summary = [
        "# 04 Ordering summary",
        "",
        f"- Relay-group constraint: {'ON' if keep_groups else 'OFF'}",
        f"- Original mean holdout time: {original_time.mean():.2f} ms/device",
        f"- Ranked mean holdout time: {ranked_time.mean():.2f} ms/device",
        f"- Mean change: {(ranked_time.mean() - original_time.mean()):.2f} ms/device",
        f"- Device pass/fail reproduced exactly: {'YES' if bin_unchanged else 'NO'}",
        f"- Hard-bin distribution retained in replay source: {len(bin_distribution)} bins",
        "",
        "## Assumption",
        "- Reordering changes stop-on-first-fail sequence but does not override measurement limits.",
        "- If relay grouping is enabled, groups are ordered by their highest test density and internally ranked.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"Ordering complete. Holdout mean: original {original_time.mean():.2f} ms, "
          f"ranked {ranked_time.mean():.2f} ms; pass/fail unchanged={bin_unchanged}.")
    print(f"Results written to {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()

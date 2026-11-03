"""Escape-risk and exact Poisson confidence-bound calculations."""

from __future__ import annotations

import argparse

from scipy.stats import chi2


def garwood_upper_bound(events: int, confidence: float = 0.95) -> float:
    """Return the exact one-sided Poisson/Garwood upper bound."""
    return float(chi2.ppf(confidence, 2 * (events + 1)) / 2)


def escape_dppm(unique_fail_count: int, volume: int, sampling_rate: float,
                confidence: float = 0.95) -> float:
    """Estimate DPPM escape risk with a Poisson upper bound."""
    probability_upper = garwood_upper_bound(unique_fail_count, confidence) / volume
    return probability_upper * (1.0 - sampling_rate) * 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    print(f"Risk helper loaded with config {args.config}.")


if __name__ == "__main__":
    main()

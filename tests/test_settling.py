import pandas as pd

from dcopt.settling import metrics_by_setting, tolerance


def test_two_sided_tolerance() -> None:
    row = pd.Series({"lo_limit": 0.0, "hi_limit": 10.0})
    assert tolerance(row, 5.0) == 10.0


def test_one_sided_tolerance_uses_typical_distance() -> None:
    row = pd.Series({"lo_limit": None, "hi_limit": 5.0})
    assert tolerance(row, 3.0) == 2.0


def test_repeatability_is_computed_per_setting() -> None:
    sweep = pd.DataFrame({
        "test": ["T"] * 6,
        "temp_C": [25] * 6,
        "wait_ms": [1.0] * 6,
        "n_samples": [1] * 6,
        "device": [1, 1, 1, 2, 2, 2],
        "repeat": [1, 2, 3, 1, 2, 3],
        "measured_value": [1.0, 1.0, 1.0, 2.0, 2.1, 1.9],
        "reference_value": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    })
    metrics = metrics_by_setting(sweep)
    assert len(metrics) == 1
    assert metrics.iloc[0]["bias"] == 0.0

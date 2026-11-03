import pandas as pd

from dcopt.monitoring import equipment_t02_site3_alarm, psi


def test_psi_detects_distribution_shift() -> None:
    baseline = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    shifted = pd.Series([20.0, 21.0, 22.0, 23.0, 24.0])
    assert psi(baseline, shifted) > 0.25


def test_t02_site3_offset_is_equipment_alarm() -> None:
    base = pd.DataFrame({
        "tester": ["T02"] * 4,
        "site": [3] * 4,
        "IIL_A": [1.0, 1.1, 0.9, 1.0],
    })
    shifted = pd.DataFrame({
        "tester": ["T02"] * 4,
        "site": [3] * 4,
        "IIL_A": [61.0, 62.0, 60.0, 61.0],
    })
    assert equipment_t02_site3_alarm(shifted, base, 0.7)

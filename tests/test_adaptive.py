import pandas as pd

from dcopt.adaptive import max_abs_spc_z


def test_temperature_baseline_does_not_create_false_hot_drift() -> None:
    baseline = pd.DataFrame({
        "IIL_A": [-430.0] * 10,
        "IDDQ_V1": [2.5] * 10,
        "IDD_STATIC": [8.75] * 10,
    })
    observed = baseline.copy()
    z, violations = max_abs_spc_z(observed, baseline, 3.0)
    assert z == 0.0
    assert violations == []


def test_spc_flags_drift() -> None:
    baseline = pd.DataFrame({
        "IIL_A": list(range(10)),
        "IDDQ_V1": list(range(10)),
        "IDD_STATIC": list(range(10)),
    })
    observed = baseline + 20
    z, violations = max_abs_spc_z(observed, baseline, 0.7)
    assert z > 0.7
    assert violations

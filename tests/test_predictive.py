import pandas as pd

from dcopt.predictive import FEATURES, clopper_pearson_upper, interval_inside_limits


def test_features_never_include_labels() -> None:
    prohibited = {"is_latent_defect_truth", "sort_pass", "burnin_fail", "rma", "first_fail_test"}
    assert prohibited.isdisjoint(FEATURES)


def test_interval_must_fit_both_margins() -> None:
    lower = pd.Series([4.6, 4.0])
    upper = pd.Series([4.4, 5.0])
    inside = interval_inside_limits(lower, upper, 0.0, 5.0, 0.10)
    assert inside.tolist() == [True, False]


def test_clopper_pearson_upper_is_bounded() -> None:
    upper = clopper_pearson_upper(0, 100, 0.95)
    assert 0.0 <= upper <= 1.0

import math

import pandas as pd
from pytest import approx

from dcopt.baseline import cpk, poisson_upper_bound, yield_table


def test_yield_table_overall_and_dimension() -> None:
    results = pd.DataFrame({
        "sort_pass": [1, 0, 1],
        "lot": ["L001", "L001", "L002"],
    })
    table = yield_table(results, ["lot"])
    overall = table.query("dimension == 'overall'").iloc[0]
    assert overall["yield_pct"] == approx(200 / 3)


def test_cpk_two_sided() -> None:
    values = pd.Series([9.0, 10.0, 11.0])
    value = cpk(values, 0.0, 20.0)
    assert value is not None and math.isfinite(value)


def test_poisson_upper_bound_increases_with_events() -> None:
    assert poisson_upper_bound(10, 0.95) > poisson_upper_bound(1, 0.95)

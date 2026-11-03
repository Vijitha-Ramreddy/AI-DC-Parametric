from dcopt.simulator import poisson_upper_dppm


def test_simulator_poisson_upper_bound_matches_risk_helper() -> None:
    upper_bound = poisson_upper_dppm(0, 8820, 0.95)
    assert 0 < upper_bound < 1000

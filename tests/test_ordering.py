import pandas as pd

from dcopt.ordering import ranked_order, stop_first_fail_time


def test_ranked_order_uses_fail_probability_per_ms() -> None:
    train = pd.DataFrame({"A_PF": ["F", "P"], "B_PF": ["P", "P"]})
    catalog = pd.DataFrame({"test": ["A", "B"], "group": ["G1", "G2"], "test_time_ms": [10.0, 20.0]})
    ranked = ranked_order(train, catalog, keep_relay_groups_together=False)
    assert ranked.iloc[0]["test"] == "A"


def test_stop_first_fail_time() -> None:
    frame = pd.DataFrame({"A_PF": ["F"], "B_PF": ["P"]})
    catalog = pd.DataFrame({"test": ["A", "B"], "group": ["G1", "G1"], "test_time_ms": [10.0, 20.0]})
    catalog = catalog.set_index("test")
    catalog = catalog.reset_index()
    assert stop_first_fail_time(frame, ["A", "B"], catalog).iloc[0] == 10.0

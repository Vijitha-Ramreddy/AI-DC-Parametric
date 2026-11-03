from dcopt.phase0_data_foundation import resolve_splits, validate_catalog


def test_resolve_fixed_lots_is_lot_disjoint() -> None:
    config = {
        "splits": {
            "leave_one_lot_out": False,
            "train_lots": ["L001", "L002"],
            "test_lots": ["L003", "L004"],
        }
    }
    train, test = resolve_splits(config, ["L001", "L002", "L003", "L004"])
    assert train == ["L001", "L002"]
    assert test == ["L003", "L004"]
    assert set(train).isdisjoint(test)


def test_lot_out_only_uses_last_configured_lot() -> None:
    config = {"splits": {"leave_one_lot_out": True, "train_lots": [], "test_lots": []}}
    train, test = resolve_splits(config, ["L001", "L002", "L003"])
    assert train == ["L001", "L002"]
    assert test == ["L003"]


def test_catalog_requires_measurements() -> None:
    import pandas as pd

    catalog = pd.DataFrame({"test": ["TEST_A"]})
    results = pd.DataFrame({"ecid": [1], "lot": ["L001"], "wafer": [1], "tester": ["T01"],
                            "site": [1], "temp_C": [25], "sort_pass": [1], "hard_bin": [1],
                            "dc_test_time_ms": [100], "is_latent_defect_truth": [0]})
    config = {"validation": {"required_result_columns": list(results.columns)}}
    assert "Catalog tests missing from results: ['TEST_A']" in validate_catalog(results, catalog, config)

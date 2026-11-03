import pandas as pd

from dcopt.io import Datasets, data_quality_report


def test_data_quality_report_includes_required_dimensions() -> None:
    results = pd.DataFrame({
        "lot": ["L001", "L002"],
        "tester": ["T01", "T02"],
        "temp_C": [25, 90],
        "value": [1.0, None],
    })
    data = Datasets(results=results, catalog=pd.DataFrame({"test": ["TEST"]}), settling=pd.DataFrame({"test": ["TEST"]}))

    report = data_quality_report(data)

    assert report["row_counts"] == {
        "dc_test_results": 2,
        "test_catalog": 1,
        "settling_sweep": 1,
    }
    assert report["lots"] == ["L001", "L002"]
    assert report["null_columns"] == {"value": 1}

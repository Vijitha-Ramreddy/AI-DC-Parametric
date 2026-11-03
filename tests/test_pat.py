import numpy as np
import pandas as pd

from dcopt.pat import add_pat_features, metric_scores


def test_pat_adds_leakage_normalized_features() -> None:
    frame = pd.DataFrame({"IIL_A": [-100.0], "IDDQ_V1": [1.0]})
    augmented = add_pat_features(frame)
    assert "IDDQ_V1_over_IIL_A" in augmented.columns
    assert augmented["IDDQ_V1_over_IIL_A"].iloc[0] == -0.01


def test_pat_metrics_reflect_precision_and_recall() -> None:
    metrics = metric_scores(
        np.array([1, 0, 1, 0]),
        np.array([1, 0, 0, 0]),
        np.array([0.9, 0.8, 0.2, 0.1]),
    )
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.evaluation import (
    evaluate_capped_test_set,
    evaluate_probability_model,
    evaluate_test_set,
    get_latest_engine_states,
)


class FixedPredictionModel:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions, dtype=float)

    def predict(self, features):
        assert list(features.columns) == FEATURE_COLUMNS
        assert len(features) == len(self.predictions)
        return self.predictions


def make_test_rows(engine_cycles):
    rows = []

    for engine_id, cycles in engine_cycles.items():
        for cycle in cycles:
            row = {
                "engine_id": engine_id,
                "cycle": cycle,
            }
            row.update({column: 0.0 for column in FEATURE_COLUMNS})
            rows.append(row)

    return pd.DataFrame(rows)


def test_get_latest_engine_states_returns_one_final_row_per_engine():
    data = make_test_rows({1: [3, 1, 2], 2: [1, 4, 2]})

    result = get_latest_engine_states(data)

    latest = dict(zip(result["engine_id"], result["cycle"]))

    assert latest == {1: 3, 2: 4}
    assert len(result) == 2


def test_evaluate_test_set_clips_operational_prediction_but_scores_raw():
    test_data = make_test_rows({1: [1], 2: [1]})
    test_rul = pd.DataFrame(
        {
            "engine_id": [1, 2],
            "actual_rul": [0.0, 10.0],
        }
    )
    model = FixedPredictionModel([-5.0, 10.0])

    results, mae = evaluate_test_set(model, test_data, test_rul)

    assert results["predicted_rul_raw"].tolist() == [-5.0, 10.0]
    assert results["predicted_rul"].tolist() == [0.0, 10.0]
    assert mae == pytest.approx(2.5)


def test_evaluate_capped_test_set_clips_to_zero_and_rul_cap():
    test_data = make_test_rows({1: [1], 2: [1]})
    test_rul = pd.DataFrame(
        {
            "engine_id": [1, 2],
            "actual_rul": [180.0, 0.0],
            "actual_rul_capped": [125.0, 0.0],
        }
    )
    model = FixedPredictionModel([150.0, -5.0])

    results, mae = evaluate_capped_test_set(model, test_data, test_rul)

    assert results["predicted_rul_capped_raw"].tolist() == [150.0, -5.0]
    assert results["predicted_rul_capped"].tolist() == [125.0, 0.0]
    assert mae == pytest.approx(15.0)


def test_probability_evaluation_reports_expected_metrics_and_counts():
    financial = pd.DataFrame(
        {
            "engine_id": [1, 2, 3, 4],
            "actual_rul_capped": [10, 20, 80, 100],
            "high_risk_probability": [0.9, 0.8, 0.2, 0.1],
        }
    )

    metrics = evaluate_probability_model(financial)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert 0 <= metrics["brier_score"] <= 1
    assert metrics["log_loss"] >= 0
    assert int(metrics["calibration_table"]["engines"].sum()) == 4

import pandas as pd
import pytest

from src.risk import (
    analyse_maintenance_classification,
    assign_actual_risk,
    assign_predicted_risk,
    maintenance_decision,
    optimise_high_risk_threshold,
)


@pytest.mark.parametrize(
    ("rul", "expected"),
    [
        (0, "HIGH"),
        (30, "HIGH"),
        (31, "MEDIUM"),
        (60, "MEDIUM"),
        (61, "LOW"),
        (125, "LOW"),
    ],
)
def test_assign_actual_risk_boundaries(rul, expected):
    assert assign_actual_risk(rul) == expected


@pytest.mark.parametrize(
    ("rul", "expected"),
    [
        (0, "HIGH"),
        (38, "HIGH"),
        (39, "MEDIUM"),
        (60, "MEDIUM"),
        (61, "LOW"),
    ],
)
def test_assign_predicted_risk_uses_operational_threshold(rul, expected):
    assert assign_predicted_risk(rul, high_threshold=38) == expected


@pytest.mark.parametrize(
    ("rul", "expected_risk", "expected_recommendation"),
    [
        (-5, "HIGH", "Schedule preventative maintenance"),
        (38, "HIGH", "Schedule preventative maintenance"),
        (45, "MEDIUM", "Increase monitoring"),
        (80, "LOW", "Continue operating"),
    ],
)
def test_maintenance_decision(rul, expected_risk, expected_recommendation):
    result = maintenance_decision(rul, high_threshold=38)

    assert result["predicted_rul"] >= 0
    assert result["risk"] == expected_risk
    assert result["recommendation"] == expected_recommendation


def test_threshold_optimisation_selects_lowest_fpr_then_lowest_threshold():
    oof = pd.DataFrame(
        {
            "engine_id": [1, 2, 3, 4],
            "cycle": [1, 1, 1, 1],
            "rul": [10, 20, 50, 70],
            "predicted_rul": [15, 25, 45, 70],
        }
    )

    result = optimise_high_risk_threshold(
        oof,
        actual_high_threshold=30,
        target_recall=1.0,
    )

    selected = result.loc[result["selected"]]

    assert len(selected) == 1
    assert int(selected.iloc[0]["threshold"]) == 25
    assert selected.iloc[0]["high_recall"] == pytest.approx(1.0)
    assert selected.iloc[0]["false_positive_rate"] == pytest.approx(0.0)


def test_maintenance_classification_summary_counts():
    results = pd.DataFrame(
        {
            "engine_id": [1, 2, 3, 4],
            "actual_rul_capped": [10, 20, 45, 90],
            "predicted_rul_capped": [15, 50, 40, 80],
        }
    )

    _, matrix, summary = analyse_maintenance_classification(
        results,
        predicted_high_threshold=38,
    )

    assert matrix.loc["HIGH", "HIGH"] == 1
    assert matrix.loc["HIGH", "MEDIUM"] == 1
    assert summary["high_risk_engines"] == 2
    assert summary["high_risk_correct"] == 1
    assert summary["high_risk_missed"] == 1
    assert summary["high_risk_recall"] == pytest.approx(0.5)

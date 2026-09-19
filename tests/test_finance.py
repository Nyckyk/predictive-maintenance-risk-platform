import numpy as np
import pandas as pd
import pytest

from src.finance import (
    calculate_financial_risk,
    run_cost_sensitivity_analysis,
    summarise_financial_risk,
)


class FixedProbabilityModel:
    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, features):
        assert list(features.columns) == ["predicted_rul"]
        assert len(features) == len(self.probabilities)
        return np.column_stack(
            [1.0 - self.probabilities, self.probabilities]
        )


def test_calculate_financial_risk_uses_break_even_probability():
    results = pd.DataFrame(
        {
            "engine_id": [1, 2, 3],
            "predicted_rul_capped": [100.0, 40.0, 10.0],
        }
    )
    model = FixedProbabilityModel([0.10, 0.20, 0.80])

    financial = calculate_financial_risk(
        results,
        model,
        maintenance_cost=8000,
        failure_cost=40000,
    )

    assert financial["break_even_probability"].eq(0.20).all()
    assert financial["risk_adjusted_failure_exposure"].tolist() == pytest.approx(
        [4000, 8000, 32000]
    )
    assert financial["maintenance_economically_justified"].tolist() == [
        False,
        True,
        True,
    ]
    assert financial["economic_action"].tolist() == [
        "Continue / monitor",
        "Preventative maintenance",
        "Preventative maintenance",
    ]


def test_summarise_financial_risk_uses_only_selected_engines():
    financial = pd.DataFrame(
        {
            "maintenance_economically_justified": [False, True, True],
            "break_even_probability": [0.2, 0.2, 0.2],
            "maintenance_cost": [8000, 8000, 8000],
            "risk_adjusted_failure_exposure": [4000, 12000, 32000],
            "risk_adjusted_net_benefit": [-4000, 4000, 24000],
        }
    )

    summary = summarise_financial_risk(financial)

    assert summary["engines"] == 3
    assert summary["maintenance_justified"] == 2
    assert summary["maintenance_outlay"] == 16000
    assert summary["risk_adjusted_failure_exposure"] == 44000
    assert summary["risk_adjusted_net_benefit"] == 28000


def test_cost_sensitivity_analysis_has_all_twenty_scenarios():
    financial = pd.DataFrame(
        {"high_risk_probability": [0.10, 0.50, 0.90]}
    )

    result = run_cost_sensitivity_analysis(financial)

    assert len(result) == 20
    assert set(result["maintenance_cost"]) == {
        5000, 8000, 10000, 12000, 15000
    }
    assert set(result["failure_cost"]) == {
        30000, 40000, 50000, 60000
    }

    base = result[
        (result["maintenance_cost"] == 8000)
        & (result["failure_cost"] == 40000)
    ].iloc[0]

    assert base["break_even_probability"] == pytest.approx(0.20)
    assert int(base["engines_selected"]) == 2
    assert base["maintenance_outlay"] == pytest.approx(16000)
    assert base["risk_adjusted_exposure"] == pytest.approx(56000)
    assert base["risk_adjusted_net_benefit"] == pytest.approx(40000)

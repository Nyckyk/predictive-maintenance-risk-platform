import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

import src.models as models_module
from src.config import FEATURE_COLUMNS
from src.models import create_model, generate_oof_predictions


def make_grouped_training_data():
    rows = []

    for engine_id in range(1, 6):
        for cycle in (1, 2):
            row = {
                "engine_id": engine_id,
                "cycle": cycle,
                "rul": 2 - cycle,
            }
            row.update({column: 0.0 for column in FEATURE_COLUMNS})
            row["setting_1"] = float(engine_id)
            rows.append(row)

    return pd.DataFrame(rows)


def test_create_model_returns_expected_estimator_types():
    assert isinstance(create_model("Linear Regression"), LinearRegression)
    assert isinstance(create_model("Random Forest"), RandomForestRegressor)
    assert isinstance(create_model("Gradient Boosting"), GradientBoostingRegressor)


def test_create_model_raises_for_unknown_name():
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("Not a model")


def test_random_forest_and_gradient_boosting_are_reproducible():
    random_forest = create_model("Random Forest")
    gradient_boosting = create_model("Gradient Boosting")

    assert random_forest.random_state == 42
    assert gradient_boosting.random_state == 42


def test_oof_predictions_keep_validation_engines_out_of_training(monkeypatch):
    data = make_grouped_training_data()

    class AuditModel:
        def fit(self, X, y):
            self.training_engines = set(X["setting_1"].astype(int))
            return self

        def predict(self, X):
            validation_engines = set(X["setting_1"].astype(int))
            assert self.training_engines.isdisjoint(validation_engines)
            return np.zeros(len(X), dtype=float)

    monkeypatch.setattr(
        models_module,
        "create_model",
        lambda name: AuditModel(),
    )

    results = generate_oof_predictions(data, "Audit")

    assert len(results) == len(data)
    assert results["predicted_rul"].notna().all()
    assert results["engine_id"].tolist() == data["engine_id"].tolist()

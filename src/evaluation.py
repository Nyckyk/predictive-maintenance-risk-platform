import pandas as pd

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)

from src.config import ACTUAL_HIGH_THRESHOLD, FEATURE_COLUMNS, RUL_CAP


def get_latest_engine_states(test_data: pd.DataFrame) -> pd.DataFrame:
    """Return the final observed state of each test engine."""

    return (
        test_data
        .sort_values(["engine_id", "cycle"])
        .groupby("engine_id")
        .tail(1)
        .copy()
    )


def evaluate_test_set(
    model,
    test_data: pd.DataFrame,
    test_rul: pd.DataFrame,
):
    """Evaluate uncapped RUL predictions."""

    latest_states = get_latest_engine_states(test_data)
    predictions = model.predict(latest_states[FEATURE_COLUMNS])

    latest_states["predicted_rul_raw"] = predictions
    latest_states["predicted_rul"] = latest_states[
        "predicted_rul_raw"
    ].clip(lower=0)

    results = latest_states.merge(
        test_rul,
        on="engine_id",
        how="left",
    )

    test_mae = mean_absolute_error(
        results["actual_rul"],
        results["predicted_rul_raw"],
    )
    return results, test_mae


def evaluate_capped_test_set(
    model,
    test_data: pd.DataFrame,
    test_rul: pd.DataFrame,
):
    """Evaluate capped-RUL model predictions."""

    latest_states = get_latest_engine_states(test_data)
    predictions = model.predict(latest_states[FEATURE_COLUMNS])

    latest_states["predicted_rul_capped_raw"] = predictions
    latest_states["predicted_rul_capped"] = latest_states[
        "predicted_rul_capped_raw"
    ].clip(lower=0, upper=RUL_CAP)

    results = latest_states.merge(
        test_rul,
        on="engine_id",
        how="left",
    )

    capped_mae = mean_absolute_error(
        results["actual_rul_capped"],
        results["predicted_rul_capped_raw"],
    )
    return results, capped_mae


def evaluate_probability_model(
    financial_results: pd.DataFrame,
) -> dict:
    """Evaluate discrimination and calibration of HIGH-risk probabilities."""

    results = financial_results.copy()
    results["actual_high"] = (
        results["actual_rul_capped"] <= ACTUAL_HIGH_THRESHOLD
    ).astype(int)

    y_true = results["actual_high"]
    y_probability = results["high_risk_probability"]

    brier = brier_score_loss(y_true, y_probability)
    logloss = log_loss(y_true, y_probability)
    roc_auc = roc_auc_score(y_true, y_probability)

    results["probability_bin"] = pd.cut(
        results["high_risk_probability"],
        bins=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        include_lowest=True,
    )

    calibration_table = (
        results
        .groupby("probability_bin", observed=True)
        .agg(
            engines=("engine_id", "count"),
            mean_predicted_probability=("high_risk_probability", "mean"),
            actual_high_rate=("actual_high", "mean"),
        )
        .reset_index()
    )

    return {
        "brier_score": brier,
        "log_loss": logloss,
        "roc_auc": roc_auc,
        "calibration_table": calibration_table,
    }

import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from src.config import (
    ACTUAL_HIGH_THRESHOLD,
    FEATURE_COLUMNS,
    MODEL_NAMES,
    RUL_CAP,
)


def create_model(name: str):
    """Create one of the candidate RUL regression models."""

    if name == "Linear Regression":
        return LinearRegression()

    if name == "Random Forest":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        )

    if name == "Gradient Boosting":
        return GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )

    raise ValueError(f"Unknown model: {name}")


def train_baseline_model(df: pd.DataFrame):
    """Train Linear Regression using an 80/20 engine split."""

    train_data = df[df["engine_id"] <= 80]
    validation_data = df[df["engine_id"] > 80]

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["rul"]
    X_validation = validation_data[FEATURE_COLUMNS]
    y_validation = validation_data["rul"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_validation)
    mae = mean_absolute_error(y_validation, predictions)

    return model, validation_data, predictions, mae


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """Compare models using engines 81-100 for validation."""

    train_data = df[df["engine_id"] <= 80]
    validation_data = df[df["engine_id"] > 80]

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["rul"]
    X_validation = validation_data[FEATURE_COLUMNS]
    y_validation = validation_data["rul"]

    results = []

    for name in MODEL_NAMES:
        model = create_model(name)
        model.fit(X_train, y_train)
        predictions = model.predict(X_validation)
        mae = mean_absolute_error(y_validation, predictions)
        results.append({"model": name, "validation_mae": mae})

    return (
        pd.DataFrame(results)
        .sort_values("validation_mae")
        .reset_index(drop=True)
    )


def cross_validate_models(df: pd.DataFrame) -> pd.DataFrame:
    """Compare models using five engine-level folds."""

    X = df[FEATURE_COLUMNS]
    y = df["rul"]
    groups = df["engine_id"]

    group_kfold = GroupKFold(n_splits=5)
    results = []

    for name in MODEL_NAMES:
        fold_maes = []

        for train_index, validation_index in group_kfold.split(
            X,
            y,
            groups=groups,
        ):
            X_train = X.iloc[train_index]
            y_train = y.iloc[train_index]
            X_validation = X.iloc[validation_index]
            y_validation = y.iloc[validation_index]

            model = create_model(name)
            model.fit(X_train, y_train)
            predictions = model.predict(X_validation)
            fold_maes.append(
                mean_absolute_error(y_validation, predictions)
            )

        results.append(
            {
                "model": name,
                "mean_mae": sum(fold_maes) / len(fold_maes),
                "min_mae": min(fold_maes),
                "max_mae": max(fold_maes),
            }
        )

    return (
        pd.DataFrame(results)
        .sort_values("mean_mae")
        .reset_index(drop=True)
    )


def generate_oof_predictions(
    df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Generate engine-safe out-of-fold predictions.

    No engine is predicted by a model trained using data
    from that same engine.
    """

    X = df[FEATURE_COLUMNS]
    y = df["rul"]
    groups = df["engine_id"]

    predictions = pd.Series(index=df.index, dtype=float)
    group_kfold = GroupKFold(n_splits=5)

    for train_index, validation_index in group_kfold.split(
        X,
        y,
        groups=groups,
    ):
        model = create_model(model_name)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        predictions.iloc[validation_index] = model.predict(
            X.iloc[validation_index]
        )

    results = df[["engine_id", "cycle", "rul"]].copy()
    results["predicted_rul"] = predictions
    return results


def train_failure_risk_model(oof_results: pd.DataFrame):
    """
    Estimate P(actual RUL <= 30 | predicted capped RUL).

    The model is trained only on engine-safe OOF predictions.
    """

    risk_data = oof_results.copy()
    risk_data["predicted_rul"] = risk_data["predicted_rul"].clip(
        lower=0,
        upper=RUL_CAP,
    )
    risk_data["actual_high"] = (
        risk_data["rul"] <= ACTUAL_HIGH_THRESHOLD
    ).astype(int)

    X = risk_data[["predicted_rul"]]
    y = risk_data["actual_high"]

    # Equalise total influence across engines.
    engine_rows = (
        risk_data
        .groupby("engine_id")["engine_id"]
        .transform("count")
    )
    sample_weights = 1.0 / engine_rows
    sample_weights *= len(sample_weights) / sample_weights.sum()

    risk_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )
    risk_model.fit(
        X,
        y,
        sample_weight=sample_weights,
    )
    return risk_model


def train_final_model(
    df: pd.DataFrame,
    model_name: str,
):
    """Train the selected RUL model on all training engines."""

    X_train = df[FEATURE_COLUMNS]
    y_train = df["rul"]

    model = create_model(model_name)
    model.fit(X_train, y_train)
    return model

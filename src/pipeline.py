from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


TRAIN_DATA_PATH = Path("data/raw/train_FD001.txt")
TEST_DATA_PATH = Path("data/raw/test_FD001.txt")
TEST_RUL_PATH = Path("data/raw/RUL_FD001.txt")
OUTPUT_DIR = Path("outputs")


COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


FEATURE_COLUMNS = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_data(path: Path) -> pd.DataFrame:
    """Load a NASA C-MAPSS FD001 engine dataset."""

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )

    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Remaining Useful Life for every training engine cycle."""

    df = df.copy()

    max_cycles = (
        df.groupby("engine_id")["cycle"]
        .transform("max")
    )

    df["rul"] = max_cycles - df["cycle"]

    return df


def load_test_rul(path: Path) -> pd.DataFrame:
    """Load the true RUL values for the FD001 test engines."""

    rul = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["actual_rul"],
    )

    rul["engine_id"] = range(1, len(rul) + 1)

    return rul[["engine_id", "actual_rul"]]


def train_baseline_model(df: pd.DataFrame):
    """Train and validate a baseline Linear Regression RUL model."""

    # Keep entire engines separate between training and validation.
    train_data = df[df["engine_id"] <= 80]
    validation_data = df[df["engine_id"] > 80]

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["rul"]

    X_validation = validation_data[FEATURE_COLUMNS]
    y_validation = validation_data["rul"]

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    mae = mean_absolute_error(
        y_validation,
        predictions,
    )

    return model, validation_data, predictions, mae


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """Compare candidate RUL prediction models on unseen engines."""

    train_data = df[df["engine_id"] <= 80]
    validation_data = df[df["engine_id"] > 80]

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["rul"]

    X_validation = validation_data[FEATURE_COLUMNS]
    y_validation = validation_data["rul"]

    models = {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_validation)

        mae = mean_absolute_error(
            y_validation,
            predictions,
        )

        results.append(
            {
                "model": name,
                "validation_mae": mae,
            }
        )

    results_df = pd.DataFrame(results)

    results_df = (
        results_df
        .sort_values("validation_mae")
        .reset_index(drop=True)
    )

    return results_df


def evaluate_test_set(
    model,
    test_data: pd.DataFrame,
    test_rul: pd.DataFrame,
):
    """Evaluate a trained model using the NASA FD001 test engines."""

    # Use the final available sensor reading for each test engine.
    latest_states = (
        test_data
        .sort_values(["engine_id", "cycle"])
        .groupby("engine_id")
        .tail(1)
        .copy()
    )

    X_test = latest_states[FEATURE_COLUMNS]

    predictions = model.predict(X_test)

    # Preserve raw predictions for honest evaluation.
    latest_states["predicted_rul_raw"] = predictions

    # Displayed RUL cannot physically be below zero.
    latest_states["predicted_rul"] = (
        latest_states["predicted_rul_raw"]
        .clip(lower=0)
    )

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


def maintenance_decision(
    predicted_rul: float,
    maintenance_cost: float = 8000,
    failure_cost: float = 40000,
) -> dict:
    """Convert predicted RUL into an illustrative maintenance decision."""

    predicted_rul = max(0.0, float(predicted_rul))

    if predicted_rul <= 30:
        risk = "HIGH"
        recommendation = "Schedule preventative maintenance"

    elif predicted_rul <= 60:
        risk = "MEDIUM"
        recommendation = "Increase monitoring"

    else:
        risk = "LOW"
        recommendation = "Continue operating"

    cost_difference = failure_cost - maintenance_cost

    return {
        "predicted_rul": predicted_rul,
        "risk": risk,
        "recommendation": recommendation,
        "maintenance_cost": maintenance_cost,
        "failure_cost": failure_cost,
        "cost_difference": cost_difference,
    }


def plot_test_predictions(results: pd.DataFrame) -> None:
    """Plot actual versus predicted RUL for the NASA test engines."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        results["actual_rul"],
        results["predicted_rul"],
        alpha=0.7,
    )

    max_rul = max(
        results["actual_rul"].max(),
        results["predicted_rul"].max(),
    )

    plt.plot(
        [0, max_rul],
        [0, max_rul],
        linestyle="--",
        label="Perfect prediction",
    )

    plt.xlabel("Actual RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title(
        "NASA C-MAPSS FD001: Actual vs Predicted RUL"
    )

    plt.legend()
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "baseline_actual_vs_predicted.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Prediction plot saved to: {output_path}"
    )


if __name__ == "__main__":

    # ---------------------------------------------------------
    # Load training data and calculate RUL
    # ---------------------------------------------------------

    data = load_data(TRAIN_DATA_PATH)
    data = add_rul(data)

    print(data.head())

    print()
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print(f"Engines: {data['engine_id'].nunique()}")

    print()
    print("Engine 1 final cycles:")

    print(
        data.loc[
            data["engine_id"] == 1,
            ["engine_id", "cycle", "rul"],
        ].tail()
    )

    # ---------------------------------------------------------
    # Train baseline Linear Regression model
    # ---------------------------------------------------------

    (
        model,
        validation_data,
        predictions,
        validation_mae,
    ) = train_baseline_model(data)

    print()
    print("Baseline RUL Model")
    print("------------------")
    print("Model: Linear Regression")
    print("Training engines: 80")
    print("Validation engines: 20")
    print(
        "Validation Mean Absolute Error: "
        f"{validation_mae:.2f} cycles"
    )

    # ---------------------------------------------------------
    # Compare candidate models using validation engines
    # ---------------------------------------------------------

    model_comparison = compare_models(data)

    print()
    print("Model Comparison")
    print("----------------")

    print(
        model_comparison.to_string(
            index=False,
            formatters={
                "validation_mae": (
                    lambda x: f"{x:.2f}"
                )
            },
        )
    )

    best_model_name = (
        model_comparison.iloc[0]["model"]
    )

    print()
    print(
        f"Best validation model: "
        f"{best_model_name}"
    )

    # ---------------------------------------------------------
    # Evaluate baseline on official NASA test set
    # ---------------------------------------------------------

    test_data = load_data(TEST_DATA_PATH)
    test_rul = load_test_rul(TEST_RUL_PATH)

    test_results, test_mae = evaluate_test_set(
        model,
        test_data,
        test_rul,
    )

    plot_test_predictions(test_results)

    print()
    print("NASA FD001 Test Evaluation")
    print("--------------------------")
    print(f"Test engines: {len(test_results)}")
    print(
        f"Test Mean Absolute Error: "
        f"{test_mae:.2f} cycles"
    )

    # ---------------------------------------------------------
    # Example maintenance decision
    # ---------------------------------------------------------

    example_engine = test_results.iloc[0]

    decision = maintenance_decision(
        example_engine["predicted_rul"]
    )

    print()
    print("Maintenance Decision")
    print("--------------------")
    print(
        f"Engine: "
        f"{int(example_engine['engine_id'])}"
    )
    print(
        f"Current cycle: "
        f"{int(example_engine['cycle'])}"
    )
    print(
        f"Actual RUL: "
        f"{example_engine['actual_rul']:.0f} cycles"
    )
    print(
        f"Predicted RUL: "
        f"{decision['predicted_rul']:.1f} cycles"
    )
    print(
        f"Risk level: "
        f"{decision['risk']}"
    )
    print(
        f"Recommendation: "
        f"{decision['recommendation']}"
    )

    print()
    print("Illustrative Financial Assumptions")
    print("----------------------------------")
    print(
        f"Planned maintenance cost: "
        f"£{decision['maintenance_cost']:,.0f}"
    )
    print(
        f"Unplanned failure cost: "
        f"£{decision['failure_cost']:,.0f}"
    )
    print(
        f"Cost difference: "
        f"£{decision['cost_difference']:,.0f}"
    )
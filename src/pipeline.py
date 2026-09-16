from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

TRAIN_DATA_PATH = Path("data/raw/train_FD001.txt")
TEST_DATA_PATH = Path("data/raw/test_FD001.txt")
TEST_RUL_PATH = Path("data/raw/RUL_FD001.txt")

OUTPUT_DIR = Path("outputs")

RUL_CAP = 125


COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


FEATURE_COLUMNS = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """Load a NASA C-MAPSS FD001 engine dataset."""

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )

    return df


def load_test_rul(path: Path) -> pd.DataFrame:
    """Load the true RUL values for the FD001 test engines."""

    rul = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["actual_rul"],
    )

    rul["engine_id"] = range(
        1,
        len(rul) + 1,
    )

    return rul[
        ["engine_id", "actual_rul"]
    ]


# ---------------------------------------------------------
# RUL target creation
# ---------------------------------------------------------

def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate uncapped Remaining Useful Life."""

    df = df.copy()

    max_cycles = (
        df.groupby("engine_id")["cycle"]
        .transform("max")
    )

    df["rul"] = (
        max_cycles
        - df["cycle"]
    )

    return df


def add_capped_rul(
    df: pd.DataFrame,
    cap: int = RUL_CAP,
) -> pd.DataFrame:
    """Calculate capped Remaining Useful Life."""

    df = df.copy()

    max_cycles = (
        df.groupby("engine_id")["cycle"]
        .transform("max")
    )

    uncapped_rul = (
        max_cycles
        - df["cycle"]
    )

    df["rul"] = uncapped_rul.clip(
        upper=cap
    )

    return df


def add_capped_test_rul(
    test_rul: pd.DataFrame,
    cap: int = RUL_CAP,
) -> pd.DataFrame:
    """Add a capped target for NASA test-set evaluation."""

    test_rul = test_rul.copy()

    test_rul["actual_rul_capped"] = (
        test_rul["actual_rul"]
        .clip(upper=cap)
    )

    return test_rul


# ---------------------------------------------------------
# Baseline Linear Regression
# ---------------------------------------------------------

def train_baseline_model(
    df: pd.DataFrame,
):
    """Train and validate a baseline Linear Regression model."""

    # Keep entire engines separate to prevent leakage.
    train_data = df[
        df["engine_id"] <= 80
    ]

    validation_data = df[
        df["engine_id"] > 80
    ]

    X_train = train_data[
        FEATURE_COLUMNS
    ]

    y_train = train_data[
        "rul"
    ]

    X_validation = validation_data[
        FEATURE_COLUMNS
    ]

    y_validation = validation_data[
        "rul"
    ]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_validation
    )

    mae = mean_absolute_error(
        y_validation,
        predictions,
    )

    return (
        model,
        validation_data,
        predictions,
        mae,
    )


# ---------------------------------------------------------
# Single validation split comparison
# ---------------------------------------------------------

def compare_models(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare candidate models on unseen validation engines."""

    train_data = df[
        df["engine_id"] <= 80
    ]

    validation_data = df[
        df["engine_id"] > 80
    ]

    X_train = train_data[
        FEATURE_COLUMNS
    ]

    y_train = train_data[
        "rul"
    ]

    X_validation = validation_data[
        FEATURE_COLUMNS
    ]

    y_validation = validation_data[
        "rul"
    ]

    models = {
        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
    }

    results = []

    for name, model in models.items():

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

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

    results_df = pd.DataFrame(
        results
    )

    return (
        results_df
        .sort_values("validation_mae")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# Engine-level cross-validation
# ---------------------------------------------------------

def cross_validate_models(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare models using 5-fold engine-level cross-validation."""

    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        "rul"
    ]

    groups = df[
        "engine_id"
    ]

    models = {
        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
    }

    group_kfold = GroupKFold(
        n_splits=5
    )

    results = []

    for name, model in models.items():

        fold_maes = []

        for (
            train_index,
            validation_index,
        ) in group_kfold.split(
            X,
            y,
            groups=groups,
        ):

            X_train = X.iloc[
                train_index
            ]

            y_train = y.iloc[
                train_index
            ]

            X_validation = X.iloc[
                validation_index
            ]

            y_validation = y.iloc[
                validation_index
            ]

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_validation
            )

            mae = mean_absolute_error(
                y_validation,
                predictions,
            )

            fold_maes.append(
                mae
            )

        results.append(
            {
                "model": name,
                "mean_mae":
                    sum(fold_maes)
                    / len(fold_maes),

                "min_mae":
                    min(fold_maes),

                "max_mae":
                    max(fold_maes),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    return (
        results_df
        .sort_values("mean_mae")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# Final Gradient Boosting model
# ---------------------------------------------------------

def train_final_model(
    df: pd.DataFrame,
):
    """Train Gradient Boosting on all 100 training engines."""

    X_train = df[
        FEATURE_COLUMNS
    ]

    y_train = df[
        "rul"
    ]

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ---------------------------------------------------------
# Uncapped NASA test evaluation
# ---------------------------------------------------------

def evaluate_test_set(
    model,
    test_data: pd.DataFrame,
    test_rul: pd.DataFrame,
):
    """Evaluate a trained model on the NASA FD001 test engines."""

    latest_states = (
        test_data
        .sort_values(
            ["engine_id", "cycle"]
        )
        .groupby(
            "engine_id"
        )
        .tail(1)
        .copy()
    )

    X_test = latest_states[
        FEATURE_COLUMNS
    ]

    predictions = model.predict(
        X_test
    )

    # Preserve raw predictions for honest metric calculation.
    latest_states[
        "predicted_rul_raw"
    ] = predictions

    # Negative displayed RUL is not physically meaningful.
    latest_states[
        "predicted_rul"
    ] = (
        latest_states[
            "predicted_rul_raw"
        ]
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

    return (
        results,
        test_mae,
    )


# ---------------------------------------------------------
# Capped NASA test evaluation
# ---------------------------------------------------------

def evaluate_capped_test_set(
    model,
    test_data: pd.DataFrame,
    test_rul: pd.DataFrame,
):
    """Evaluate a model trained using capped RUL targets."""

    latest_states = (
        test_data
        .sort_values(
            ["engine_id", "cycle"]
        )
        .groupby(
            "engine_id"
        )
        .tail(1)
        .copy()
    )

    X_test = latest_states[
        FEATURE_COLUMNS
    ]

    predictions = model.predict(
        X_test
    )

    latest_states[
        "predicted_rul_capped"
    ] = (
        pd.Series(
            predictions,
            index=latest_states.index,
        )
        .clip(
            lower=0,
            upper=RUL_CAP,
        )
    )

    results = latest_states.merge(
        test_rul,
        on="engine_id",
        how="left",
    )

    capped_mae = mean_absolute_error(
        results[
            "actual_rul_capped"
        ],
        results[
            "predicted_rul_capped"
        ],
    )

    return (
        results,
        capped_mae,
    )


# ---------------------------------------------------------
# Maintenance and financial decision
# ---------------------------------------------------------

def maintenance_decision(
    predicted_rul: float,
    maintenance_cost: float = 8000,
    failure_cost: float = 40000,
) -> dict:
    """Convert predicted RUL into an illustrative maintenance decision."""

    predicted_rul = max(
        0.0,
        float(predicted_rul),
    )

    if predicted_rul <= 30:
        risk = "HIGH"

        recommendation = (
            "Schedule preventative maintenance"
        )

    elif predicted_rul <= 60:
        risk = "MEDIUM"

        recommendation = (
            "Increase monitoring"
        )

    else:
        risk = "LOW"

        recommendation = (
            "Continue operating"
        )

    cost_difference = (
        failure_cost
        - maintenance_cost
    )

    return {
        "predicted_rul":
            predicted_rul,

        "risk":
            risk,

        "recommendation":
            recommendation,

        "maintenance_cost":
            maintenance_cost,

        "failure_cost":
            failure_cost,

        "cost_difference":
            cost_difference,
    }


# ---------------------------------------------------------
# Visualisation
# ---------------------------------------------------------

def plot_test_predictions(
    results: pd.DataFrame,
) -> None:
    """Plot actual versus predicted uncapped RUL."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        results[
            "actual_rul"
        ],
        results[
            "predicted_rul"
        ],
        alpha=0.7,
    )

    max_rul = max(
        results[
            "actual_rul"
        ].max(),

        results[
            "predicted_rul"
        ].max(),
    )

    plt.plot(
        [0, max_rul],
        [0, max_rul],
        linestyle="--",
        label="Perfect prediction",
    )

    plt.xlabel(
        "Actual RUL (cycles)"
    )

    plt.ylabel(
        "Predicted RUL (cycles)"
    )

    plt.title(
        "Gradient Boosting: "
        "Actual vs Predicted RUL"
    )

    plt.legend()
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "final_actual_vs_predicted.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Prediction plot saved to: "
        f"{output_path}"
    )


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

if __name__ == "__main__":

    # ---------------------------------------------------------
    # Load uncapped training dataset
    # ---------------------------------------------------------

    data = load_data(
        TRAIN_DATA_PATH
    )

    data = add_rul(
        data
    )

    print(
        data.head()
    )

    print()

    print(
        f"Rows: {len(data):,}"
    )

    print(
        f"Columns: {len(data.columns)}"
    )

    print(
        f"Engines: "
        f"{data['engine_id'].nunique()}"
    )

    print()
    print(
        "Engine 1 final cycles:"
    )

    print(
        data.loc[
            data["engine_id"] == 1,
            [
                "engine_id",
                "cycle",
                "rul",
            ],
        ].tail()
    )

    # ---------------------------------------------------------
    # Baseline Linear Regression
    # ---------------------------------------------------------

    (
        baseline_model,
        validation_data,
        baseline_predictions,
        validation_mae,
    ) = train_baseline_model(
        data
    )

    print()
    print(
        "Baseline RUL Model"
    )

    print(
        "------------------"
    )

    print(
        "Model: Linear Regression"
    )

    print(
        "Training engines: 80"
    )

    print(
        "Validation engines: 20"
    )

    print(
        "Validation Mean Absolute Error: "
        f"{validation_mae:.2f} cycles"
    )

    # ---------------------------------------------------------
    # Single validation split comparison
    # ---------------------------------------------------------

    model_comparison = compare_models(
        data
    )

    print()
    print(
        "Model Comparison"
    )

    print(
        "----------------"
    )

    print(
        model_comparison.to_string(
            index=False,
            formatters={
                "validation_mae":
                    lambda x:
                    f"{x:.2f}"
            },
        )
    )

    best_validation_model = (
        model_comparison.iloc[0][
            "model"
        ]
    )

    print()

    print(
        "Best validation model: "
        f"{best_validation_model}"
    )

    # ---------------------------------------------------------
    # Uncapped 5-fold grouped cross-validation
    # ---------------------------------------------------------

    cv_results = cross_validate_models(
        data
    )

    print()
    print(
        "5-Fold Engine-Level "
        "Cross-Validation"
    )

    print(
        "------------------------------------"
    )

    print(
        cv_results.to_string(
            index=False,
            formatters={
                "mean_mae":
                    lambda x:
                    f"{x:.2f}",

                "min_mae":
                    lambda x:
                    f"{x:.2f}",

                "max_mae":
                    lambda x:
                    f"{x:.2f}",
            },
        )
    )

    best_cv_model = (
        cv_results.iloc[0][
            "model"
        ]
    )

    print()

    print(
        "Best cross-validation model: "
        f"{best_cv_model}"
    )

    # ---------------------------------------------------------
    # Train final uncapped Gradient Boosting model
    # ---------------------------------------------------------

    final_model = train_final_model(
        data
    )

    print()
    print(
        "Final Model"
    )

    print(
        "-----------"
    )

    print(
        "Model: Gradient Boosting"
    )

    print(
        "Training engines: 100"
    )

    # ---------------------------------------------------------
    # Official NASA uncapped test evaluation
    # ---------------------------------------------------------

    test_data = load_data(
        TEST_DATA_PATH
    )

    test_rul = load_test_rul(
        TEST_RUL_PATH
    )

    test_results, test_mae = (
        evaluate_test_set(
            final_model,
            test_data,
            test_rul,
        )
    )

    plot_test_predictions(
        test_results
    )

    print()
    print(
        "NASA FD001 Final Test Evaluation"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Test engines: "
        f"{len(test_results)}"
    )

    print(
        "Final Test Mean Absolute Error: "
        f"{test_mae:.2f} cycles"
    )

    # ---------------------------------------------------------
    # Capped-RUL experiment
    # ---------------------------------------------------------

    capped_data = load_data(
        TRAIN_DATA_PATH
    )

    capped_data = add_capped_rul(
        capped_data
    )

    capped_cv_results = (
        cross_validate_models(
            capped_data
        )
    )

    print()
    print(
        "Capped-RUL Cross-Validation"
    )

    print(
        "---------------------------"
    )

    print(
        f"RUL cap: "
        f"{RUL_CAP} cycles"
    )

    print(
        capped_cv_results.to_string(
            index=False,
            formatters={
                "mean_mae":
                    lambda x:
                    f"{x:.2f}",

                "min_mae":
                    lambda x:
                    f"{x:.2f}",

                "max_mae":
                    lambda x:
                    f"{x:.2f}",
            },
        )
    )

    capped_best_model = (
        capped_cv_results.iloc[0][
            "model"
        ]
    )

    print()

    print(
        "Best capped-RUL "
        "cross-validation model: "
        f"{capped_best_model}"
    )

    capped_final_model = (
        train_final_model(
            capped_data
        )
    )

    capped_test_rul = (
        add_capped_test_rul(
            test_rul
        )
    )

    (
        capped_results,
        capped_test_mae,
    ) = evaluate_capped_test_set(
        capped_final_model,
        test_data,
        capped_test_rul,
    )

    print()
    print(
        "Capped-RUL Test Evaluation"
    )

    print(
        "--------------------------"
    )

    print(
        f"RUL cap: "
        f"{RUL_CAP} cycles"
    )

    print(
        "Capped Test MAE: "
        f"{capped_test_mae:.2f} cycles"
    )

    # ---------------------------------------------------------
    # Example uncapped maintenance decision
    # ---------------------------------------------------------

    example_engine = (
        test_results.iloc[0]
    )

    decision = (
        maintenance_decision(
            example_engine[
                "predicted_rul"
            ]
        )
    )

    print()
    print(
        "Maintenance Decision"
    )

    print(
        "--------------------"
    )

    print(
        "Engine: "
        f"{int(example_engine['engine_id'])}"
    )

    print(
        "Current cycle: "
        f"{int(example_engine['cycle'])}"
    )

    print(
        "Actual RUL: "
        f"{example_engine['actual_rul']:.0f} "
        "cycles"
    )

    print(
        "Predicted RUL: "
        f"{decision['predicted_rul']:.1f} "
        "cycles"
    )

    print(
        "Risk level: "
        f"{decision['risk']}"
    )

    print(
        "Recommendation: "
        f"{decision['recommendation']}"
    )

    print()
    print(
        "Illustrative Financial Assumptions"
    )

    print(
        "----------------------------------"
    )

    print(
        "Planned maintenance cost: "
        f"£{decision['maintenance_cost']:,.0f}"
    )

    print(
        "Unplanned failure cost: "
        f"£{decision['failure_cost']:,.0f}"
    )

    print(
        "Cost difference: "
        f"£{decision['cost_difference']:,.0f}"
    ) 
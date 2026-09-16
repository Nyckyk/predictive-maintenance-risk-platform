from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold


# =========================================================
# CONFIGURATION
# =========================================================

TRAIN_DATA_PATH = Path("data/raw/train_FD001.txt")
TEST_DATA_PATH = Path("data/raw/test_FD001.txt")
TEST_RUL_PATH = Path("data/raw/RUL_FD001.txt")
OUTPUT_DIR = Path("outputs")

RUL_CAP = 125
ACTUAL_HIGH_THRESHOLD = 30
MEDIUM_THRESHOLD = 60

TARGET_HIGH_RECALL = 0.90
THRESHOLD_SEARCH_MIN = 20
THRESHOLD_SEARCH_MAX = 60

# Illustrative assumptions only; these are not NASA dataset values.
MAINTENANCE_COST = 8000
FAILURE_COST = 40000

COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

FEATURE_COLUMNS = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

MODEL_NAMES = [
    "Linear Regression",
    "Random Forest",
    "Gradient Boosting",
]

def run_cost_sensitivity_analysis(
    financial_results: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate maintenance decisions across multiple cost scenarios."""

    maintenance_costs = [
        5000,
        8000,
        10000,
        12000,
        15000,
    ]

    failure_costs = [
        30000,
        40000,
        50000,
        60000,
    ]

    scenarios = []

    for maintenance_cost in maintenance_costs:

        for failure_cost in failure_costs:

            break_even_probability = (
                maintenance_cost
                / failure_cost
            )

            probabilities = (
                financial_results[
                    "high_risk_probability"
                ]
            )

            selected = (
                probabilities
                >= break_even_probability
            )

            selected_probabilities = (
                probabilities[
                    selected
                ]
            )

            engines_selected = int(
                selected.sum()
            )

            maintenance_outlay = (
                engines_selected
                * maintenance_cost
            )

            risk_adjusted_exposure = (
                selected_probabilities.sum()
                * failure_cost
            )

            net_benefit = (
                risk_adjusted_exposure
                - maintenance_outlay
            )

            scenarios.append(
                {
                    "maintenance_cost":
                        maintenance_cost,

                    "failure_cost":
                        failure_cost,

                    "break_even_probability":
                        break_even_probability,

                    "engines_selected":
                        engines_selected,

                    "maintenance_outlay":
                        maintenance_outlay,

                    "risk_adjusted_exposure":
                        risk_adjusted_exposure,

                    "risk_adjusted_net_benefit":
                        net_benefit,
                }
            )

    return pd.DataFrame(
        scenarios
    )  

# =========================================================
# MODEL CREATION
# =========================================================

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


# =========================================================
# DATA LOADING
# =========================================================

def load_data(path: Path) -> pd.DataFrame:
    """Load a NASA C-MAPSS FD001 dataset."""

    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )


def load_test_rul(path: Path) -> pd.DataFrame:
    """Load NASA's true RUL values for FD001 test engines."""

    rul = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["actual_rul"],
    )
    rul["engine_id"] = range(1, len(rul) + 1)
    return rul[["engine_id", "actual_rul"]]


# =========================================================
# RUL TARGET CREATION
# =========================================================

def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate uncapped Remaining Useful Life."""

    df = df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    df["rul"] = max_cycles - df["cycle"]
    return df


def add_capped_rul(
    df: pd.DataFrame,
    cap: int = RUL_CAP,
) -> pd.DataFrame:
    """Calculate capped Remaining Useful Life."""

    df = df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].transform("max")
    uncapped_rul = max_cycles - df["cycle"]
    df["rul"] = uncapped_rul.clip(upper=cap)
    return df


def add_capped_test_rul(
    test_rul: pd.DataFrame,
    cap: int = RUL_CAP,
) -> pd.DataFrame:
    """Add capped RUL values to NASA test targets."""

    test_rul = test_rul.copy()
    test_rul["actual_rul_capped"] = test_rul["actual_rul"].clip(upper=cap)
    return test_rul


# =========================================================
# RISK FUNCTIONS
# =========================================================

def assign_actual_risk(rul: float) -> str:
    """Convert actual RUL into the reference risk category."""

    if rul <= ACTUAL_HIGH_THRESHOLD:
        return "HIGH"
    if rul <= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def assign_predicted_risk(
    rul: float,
    high_threshold: int,
) -> str:
    """Convert predicted RUL into an operational risk category."""

    if rul <= high_threshold:
        return "HIGH"
    if rul <= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


# =========================================================
# BASELINE AND MODEL COMPARISON
# =========================================================

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


# =========================================================
# GROUPED CROSS-VALIDATION
# =========================================================

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


# =========================================================
# OUT-OF-FOLD PREDICTIONS
# =========================================================

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


# =========================================================
# HIGH-RISK THRESHOLD OPTIMISATION
# =========================================================

def optimise_high_risk_threshold(
    oof_results: pd.DataFrame,
    actual_high_threshold: int = ACTUAL_HIGH_THRESHOLD,
    target_recall: float = TARGET_HIGH_RECALL,
) -> pd.DataFrame:
    """Select an operational alert threshold using training OOF predictions."""

    actual_high = oof_results["rul"] <= actual_high_threshold
    results = []

    for threshold in range(
        THRESHOLD_SEARCH_MIN,
        THRESHOLD_SEARCH_MAX + 1,
    ):
        predicted_high = oof_results["predicted_rul"] <= threshold

        true_positive = int((actual_high & predicted_high).sum())
        false_negative = int((actual_high & ~predicted_high).sum())
        false_positive = int((~actual_high & predicted_high).sum())
        true_negative = int((~actual_high & ~predicted_high).sum())

        recall_denominator = true_positive + false_negative
        precision_denominator = true_positive + false_positive
        negative_denominator = false_positive + true_negative

        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )
        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
        false_positive_rate = (
            false_positive / negative_denominator
            if negative_denominator > 0
            else 0.0
        )

        results.append(
            {
                "threshold": threshold,
                "high_recall": recall,
                "high_precision": precision,
                "false_positive_rate": false_positive_rate,
                "true_positive": true_positive,
                "false_negative": false_negative,
                "false_positive": false_positive,
                "true_negative": true_negative,
            }
        )

    results_df = pd.DataFrame(results)
    results_df["selected"] = False

    eligible = results_df[results_df["high_recall"] >= target_recall]

    if not eligible.empty:
        selected_row = (
            eligible
            .sort_values(["false_positive_rate", "threshold"])
            .iloc[0]
        )
        selected_threshold = int(selected_row["threshold"])
        results_df.loc[
            results_df["threshold"] == selected_threshold,
            "selected",
        ] = True

    return results_df


# =========================================================
# FAILURE-RISK PROBABILITY MODEL
# =========================================================

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


# =========================================================
# FINAL RUL MODEL
# =========================================================

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


# =========================================================
# TEST SET EVALUATION
# =========================================================

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


# =========================================================
# RISK ANALYSIS
# =========================================================

def analyse_risk_regions(results: pd.DataFrame) -> pd.DataFrame:
    """Measure RUL error within actual maintenance-risk regions."""

    analysis = results.copy()
    analysis["actual_risk"] = analysis["actual_rul_capped"].apply(
        assign_actual_risk
    )
    analysis["absolute_error"] = (
        analysis["predicted_rul_capped_raw"]
        - analysis["actual_rul_capped"]
    ).abs()
    analysis["signed_error"] = (
        analysis["predicted_rul_capped_raw"]
        - analysis["actual_rul_capped"]
    )
    analysis["overprediction"] = (
        analysis["predicted_rul_capped_raw"]
        > analysis["actual_rul_capped"]
    )

    risk_summary = (
        analysis
        .groupby("actual_risk", observed=True)
        .agg(
            engines=("engine_id", "count"),
            mae=("absolute_error", "mean"),
            mean_error=("signed_error", "mean"),
            overprediction_rate=("overprediction", "mean"),
        )
        .reset_index()
    )

    risk_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    risk_summary["risk_order"] = risk_summary["actual_risk"].map(risk_order)
    risk_summary = (
        risk_summary
        .sort_values("risk_order")
        .drop(columns="risk_order")
        .reset_index(drop=True)
    )
    risk_summary["overprediction_rate"] *= 100
    return risk_summary


def analyse_maintenance_classification(
    results: pd.DataFrame,
    predicted_high_threshold: int,
):
    """Evaluate maintenance-risk classification."""

    analysis = results.copy()
    analysis["actual_risk"] = analysis["actual_rul_capped"].apply(
        assign_actual_risk
    )
    analysis["predicted_risk"] = analysis["predicted_rul_capped"].apply(
        lambda rul: assign_predicted_risk(
            rul,
            predicted_high_threshold,
        )
    )

    risk_order = ["HIGH", "MEDIUM", "LOW"]
    confusion_matrix = pd.crosstab(
        analysis["actual_risk"],
        analysis["predicted_risk"],
        rownames=["Actual"],
        colnames=["Predicted"],
    ).reindex(
        index=risk_order,
        columns=risk_order,
        fill_value=0,
    )

    correct = analysis["actual_risk"] == analysis["predicted_risk"]
    accuracy = correct.mean()

    actual_high = analysis["actual_risk"] == "HIGH"
    predicted_high = analysis["predicted_risk"] == "HIGH"

    true_positive = actual_high & predicted_high
    false_negative = actual_high & ~predicted_high
    false_positive = ~actual_high & predicted_high
    high_to_low = actual_high & (analysis["predicted_risk"] == "LOW")

    high_recall = (
        true_positive.sum() / actual_high.sum()
        if actual_high.sum() > 0
        else 0.0
    )
    high_precision = (
        true_positive.sum() / predicted_high.sum()
        if predicted_high.sum() > 0
        else 0.0
    )

    summary = {
        "accuracy": accuracy,
        "high_risk_engines": int(actual_high.sum()),
        "high_risk_correct": int(true_positive.sum()),
        "high_risk_missed": int(false_negative.sum()),
        "high_risk_recall": high_recall,
        "high_risk_precision": high_precision,
        "false_high_alerts": int(false_positive.sum()),
        "high_to_low_misses": int(high_to_low.sum()),
    }

    return analysis, confusion_matrix, summary


# =========================================================
# OPERATIONAL MAINTENANCE DECISION
# =========================================================

def maintenance_decision(
    predicted_rul: float,
    high_threshold: int,
) -> dict:
    """Convert predicted RUL into an operational decision."""

    predicted_rul = max(0.0, float(predicted_rul))
    risk = assign_predicted_risk(
        predicted_rul,
        high_threshold,
    )

    if risk == "HIGH":
        recommendation = "Schedule preventative maintenance"
    elif risk == "MEDIUM":
        recommendation = "Increase monitoring"
    else:
        recommendation = "Continue operating"

    return {
        "predicted_rul": predicted_rul,
        "risk": risk,
        "recommendation": recommendation,
    }


# =========================================================
# FINANCIAL RISK MODEL
# =========================================================

def calculate_financial_risk(
    results: pd.DataFrame,
    risk_model,
    maintenance_cost: float = MAINTENANCE_COST,
    failure_cost: float = FAILURE_COST,
) -> pd.DataFrame:
    """
    Convert HIGH-risk probability into an illustrative
    risk-adjusted financial decision.

    P(actual RUL <= 30) is used as a proxy for short-horizon
    failure exposure. It is not a directly observed failure
    probability.
    """

    financial_results = results.copy()

    risk_features = financial_results[["predicted_rul_capped"]].rename(
        columns={"predicted_rul_capped": "predicted_rul"}
    )

    financial_results["high_risk_probability"] = (
        risk_model.predict_proba(risk_features)[:, 1]
    )
    financial_results["maintenance_cost"] = maintenance_cost
    financial_results["failure_cost"] = failure_cost
    financial_results["break_even_probability"] = (
        maintenance_cost / failure_cost
    )

    financial_results["risk_adjusted_failure_exposure"] = (
        financial_results["high_risk_probability"] * failure_cost
    )
    financial_results["risk_adjusted_net_benefit"] = (
        financial_results["risk_adjusted_failure_exposure"]
        - maintenance_cost
    )
    financial_results["maintenance_economically_justified"] = (
        financial_results["risk_adjusted_failure_exposure"]
        >= maintenance_cost
    )
    financial_results["economic_action"] = financial_results[
        "maintenance_economically_justified"
    ].map(
        {
            True: "Preventative maintenance",
            False: "Continue / monitor",
        }
    )

    return financial_results


def summarise_financial_risk(
    financial_results: pd.DataFrame,
) -> dict:
    """Summarise illustrative fleet-level financial implications."""

    justified = financial_results["maintenance_economically_justified"]
    selected = financial_results[justified]

    return {
        "engines": len(financial_results),
        "maintenance_justified": int(justified.sum()),
        "break_even_probability": financial_results[
            "break_even_probability"
        ].iloc[0],
        "maintenance_outlay": selected["maintenance_cost"].sum(),
        "risk_adjusted_failure_exposure": selected[
            "risk_adjusted_failure_exposure"
        ].sum(),
        "risk_adjusted_net_benefit": selected[
            "risk_adjusted_net_benefit"
        ].sum(),
    }


# =========================================================
# PROBABILITY MODEL EVALUATION
# =========================================================

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


# =========================================================
# VISUALISATION
# =========================================================

def plot_rul_predictions(
    results: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
    title: str,
    filename: str,
) -> None:
    """Plot actual versus predicted RUL."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(
        results[actual_column],
        results[predicted_column],
        alpha=0.7,
    )

    max_rul = max(
        results[actual_column].max(),
        results[predicted_column].max(),
    )
    plt.plot(
        [0, max_rul],
        [0, max_rul],
        linestyle="--",
        label="Perfect prediction",
    )
    plt.xlabel("Actual RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Prediction plot saved to: {output_path}")


def plot_probability_calibration(
    calibration_table: pd.DataFrame,
) -> None:
    """Plot observed HIGH-risk rate against predicted probability."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    plt.plot(
        calibration_table["mean_predicted_probability"],
        calibration_table["actual_high_rate"],
        marker="o",
        label="Observed calibration",
    )
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )
    plt.xlabel("Mean predicted HIGH-risk probability")
    plt.ylabel("Observed HIGH-risk rate")
    plt.title("HIGH-Risk Probability Calibration")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / "probability_calibration.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Calibration plot saved to: {output_path}")


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # UNCAPPED EXPERIMENT
    # =====================================================

    data = add_rul(load_data(TRAIN_DATA_PATH))

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

    (
        baseline_model,
        validation_data,
        baseline_predictions,
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

    model_comparison = compare_models(data)
    print()
    print("Model Comparison")
    print("----------------")
    print(
        model_comparison.to_string(
            index=False,
            formatters={
                "validation_mae": lambda x: f"{x:.2f}",
            },
        )
    )

    best_validation_model = model_comparison.iloc[0]["model"]
    print()
    print(f"Best validation model: {best_validation_model}")

    cv_results = cross_validate_models(data)
    print()
    print("5-Fold Engine-Level Cross-Validation")
    print("------------------------------------")
    print(
        cv_results.to_string(
            index=False,
            formatters={
                "mean_mae": lambda x: f"{x:.2f}",
                "min_mae": lambda x: f"{x:.2f}",
                "max_mae": lambda x: f"{x:.2f}",
            },
        )
    )

    best_cv_model = cv_results.iloc[0]["model"]
    print()
    print(f"Best cross-validation model: {best_cv_model}")

    final_model = train_final_model(data, best_cv_model)
    print()
    print("Final Uncapped Model")
    print("--------------------")
    print(f"Model: {best_cv_model}")
    print("Training engines: 100")

    test_data = load_data(TEST_DATA_PATH)
    test_rul = load_test_rul(TEST_RUL_PATH)

    test_results, test_mae = evaluate_test_set(
        final_model,
        test_data,
        test_rul,
    )

    plot_rul_predictions(
        test_results,
        actual_column="actual_rul",
        predicted_column="predicted_rul",
        title=f"{best_cv_model}: Uncapped Actual vs Predicted RUL",
        filename="uncapped_actual_vs_predicted.png",
    )

    print()
    print("NASA FD001 Uncapped Test Evaluation")
    print("-----------------------------------")
    print(f"Test engines: {len(test_results)}")
    print(f"Uncapped Test MAE: {test_mae:.2f} cycles")

    # =====================================================
    # CAPPED-RUL EXPERIMENT
    # =====================================================

    capped_data = add_capped_rul(load_data(TRAIN_DATA_PATH))
    capped_cv_results = cross_validate_models(capped_data)

    print()
    print("Capped-RUL Cross-Validation")
    print("---------------------------")
    print(f"RUL cap: {RUL_CAP} cycles")
    print(
        capped_cv_results.to_string(
            index=False,
            formatters={
                "mean_mae": lambda x: f"{x:.2f}",
                "min_mae": lambda x: f"{x:.2f}",
                "max_mae": lambda x: f"{x:.2f}",
            },
        )
    )

    capped_best_model = capped_cv_results.iloc[0]["model"]
    print()
    print(
        "Best capped-RUL cross-validation model: "
        f"{capped_best_model}"
    )

    # =====================================================
    # OUT-OF-FOLD TRAINING PREDICTIONS
    # =====================================================

    oof_results = generate_oof_predictions(
        capped_data,
        capped_best_model,
    )

    # =====================================================
    # HIGH-RISK THRESHOLD OPTIMISATION
    # =====================================================

    threshold_results = optimise_high_risk_threshold(
        oof_results,
        actual_high_threshold=ACTUAL_HIGH_THRESHOLD,
        target_recall=TARGET_HIGH_RECALL,
    )
    threshold_results.to_csv(
        OUTPUT_DIR / "threshold_optimization.csv",
        index=False,
    )

    selected_thresholds = threshold_results[
        threshold_results["selected"]
    ]

    print()
    print("HIGH-Risk Threshold Optimisation")
    print("--------------------------------")
    print(
        "Actual HIGH definition: "
        f"RUL <= {ACTUAL_HIGH_THRESHOLD} cycles"
    )
    print(
        "Target OOF HIGH-risk recall: "
        f"{TARGET_HIGH_RECALL:.0%}"
    )

    if not selected_thresholds.empty:
        selected = selected_thresholds.iloc[0]
        optimised_high_threshold = int(selected["threshold"])

        print(
            "Selected predicted-RUL threshold: "
            f"{optimised_high_threshold} cycles"
        )
        print(
            "OOF HIGH-risk recall: "
            f"{selected['high_recall']:.1%}"
        )
        print(
            "OOF HIGH-risk precision: "
            f"{selected['high_precision']:.1%}"
        )
        print(
            "OOF false-positive rate: "
            f"{selected['false_positive_rate']:.1%}"
        )
        print(
            "OOF false negatives (cycle observations): "
            f"{int(selected['false_negative'])}"
        )
        print(
            "OOF false positives (cycle observations): "
            f"{int(selected['false_positive'])}"
        )
    else:
        optimised_high_threshold = ACTUAL_HIGH_THRESHOLD
        print("No threshold achieved the target recall.")
        print(
            "Using default HIGH threshold: "
            f"{optimised_high_threshold} cycles"
        )

    print(
        "Threshold results saved to: "
        "outputs/threshold_optimization.csv"
    )

    # =====================================================
    # FAILURE-RISK PROBABILITY MODEL
    # =====================================================

    risk_probability_model = train_failure_risk_model(oof_results)

    print()
    print("Failure-Risk Probability Model")
    print("------------------------------")
    print("Model: Logistic Regression")
    print("Input: out-of-fold predicted capped RUL")
    print(
        "Target: probability actual RUL "
        f"<= {ACTUAL_HIGH_THRESHOLD} cycles"
    )

    # =====================================================
    # FINAL CAPPED MODEL
    # =====================================================

    capped_final_model = train_final_model(
        capped_data,
        capped_best_model,
    )
    capped_test_rul = add_capped_test_rul(test_rul)

    capped_results, capped_test_mae = evaluate_capped_test_set(
        capped_final_model,
        test_data,
        capped_test_rul,
    )

    plot_rul_predictions(
        capped_results,
        actual_column="actual_rul_capped",
        predicted_column="predicted_rul_capped",
        title=f"{capped_best_model}: Capped Actual vs Predicted RUL",
        filename="final_actual_vs_predicted.png",
    )

    print()
    print("Capped-RUL Test Evaluation")
    print("--------------------------")
    print(f"RUL cap: {RUL_CAP} cycles")
    print(f"Model: {capped_best_model}")
    print(f"Capped Test MAE: {capped_test_mae:.2f} cycles")

    # =====================================================
    # RISK-REGION ERROR ANALYSIS
    # =====================================================

    risk_analysis = analyse_risk_regions(capped_results)

    print()
    print("Risk-Region Error Analysis")
    print("--------------------------")
    print(
        risk_analysis.to_string(
            index=False,
            formatters={
                "mae": lambda x: f"{x:.2f}",
                "mean_error": lambda x: f"{x:+.2f}",
                "overprediction_rate": lambda x: f"{x:.1f}%",
            },
        )
    )

    # =====================================================
    # DEFAULT THRESHOLD CLASSIFICATION
    # =====================================================

    (
        default_classification_results,
        default_confusion_matrix,
        default_summary,
    ) = analyse_maintenance_classification(
        capped_results,
        predicted_high_threshold=ACTUAL_HIGH_THRESHOLD,
    )

    print()
    print("Maintenance Classification (Default Threshold)")
    print("----------------------------------------------")
    print(
        "Predicted HIGH threshold: "
        f"{ACTUAL_HIGH_THRESHOLD} cycles"
    )
    print()
    print(default_confusion_matrix.to_string())
    print()
    print(
        "Overall classification accuracy: "
        f"{default_summary['accuracy']:.1%}"
    )
    print(
        "HIGH-risk recall: "
        f"{default_summary['high_risk_recall']:.1%}"
    )
    print(
        "HIGH-risk precision: "
        f"{default_summary['high_risk_precision']:.1%}"
    )
    print(f"HIGH-risk missed: {default_summary['high_risk_missed']}")
    print(f"False HIGH alerts: {default_summary['false_high_alerts']}")
    print(
        "HIGH-risk engines classified as LOW: "
        f"{default_summary['high_to_low_misses']}"
    )

    # =====================================================
    # OPTIMISED THRESHOLD CLASSIFICATION
    # =====================================================

    (
        optimised_classification_results,
        optimised_confusion_matrix,
        optimised_summary,
    ) = analyse_maintenance_classification(
        capped_results,
        predicted_high_threshold=optimised_high_threshold,
    )

    print()
    print("Maintenance Classification (Optimised Threshold)")
    print("------------------------------------------------")
    print(
        "Predicted HIGH threshold: "
        f"{optimised_high_threshold} cycles"
    )
    print()
    print(optimised_confusion_matrix.to_string())
    print()
    print(
        "Overall classification accuracy: "
        f"{optimised_summary['accuracy']:.1%}"
    )
    print(
        "HIGH-risk engines: "
        f"{optimised_summary['high_risk_engines']}"
    )
    print(
        "HIGH-risk correctly identified: "
        f"{optimised_summary['high_risk_correct']}"
    )
    print(
        "HIGH-risk missed: "
        f"{optimised_summary['high_risk_missed']}"
    )
    print(
        "HIGH-risk recall: "
        f"{optimised_summary['high_risk_recall']:.1%}"
    )
    print(
        "HIGH-risk precision: "
        f"{optimised_summary['high_risk_precision']:.1%}"
    )
    print(
        "False HIGH alerts: "
        f"{optimised_summary['false_high_alerts']}"
    )
    print(
        "HIGH-risk engines classified as LOW: "
        f"{optimised_summary['high_to_low_misses']}"
    )

    # =====================================================
    # OPERATIONAL MAINTENANCE EXAMPLE
    # =====================================================

    example_engine = capped_results.iloc[0]
    decision = maintenance_decision(
        example_engine["predicted_rul_capped"],
        high_threshold=optimised_high_threshold,
    )

    print()
    print("Maintenance Decision (Capped Model)")
    print("-----------------------------------")
    print(f"Engine: {int(example_engine['engine_id'])}")
    print(f"Current cycle: {int(example_engine['cycle'])}")
    print(
        "Actual capped RUL: "
        f"{example_engine['actual_rul_capped']:.0f} cycles"
    )
    print(
        "Predicted capped RUL: "
        f"{decision['predicted_rul']:.1f} cycles"
    )
    print(
        "Operational HIGH threshold: "
        f"{optimised_high_threshold} cycles"
    )
    print(f"Risk level: {decision['risk']}")
    print(f"Recommendation: {decision['recommendation']}")

    # =====================================================
    # PROBABILITY-BASED FINANCIAL RISK
    # =====================================================

    financial_results = calculate_financial_risk(
        capped_results,
        risk_probability_model,
        maintenance_cost=MAINTENANCE_COST,
        failure_cost=FAILURE_COST,
    )
    financial_results.to_csv(
        OUTPUT_DIR / "financial_risk_results.csv",
        index=False,
    )

    # =====================================================
    # FINANCIAL COST SENSITIVITY ANALYSIS
    # =====================================================

    sensitivity_results = run_cost_sensitivity_analysis(
        financial_results
    )

    sensitivity_results.to_csv(
        OUTPUT_DIR / "cost_sensitivity_analysis.csv",
        index=False,
    )

    financial_summary = summarise_financial_risk(financial_results)
    probability_evaluation = evaluate_probability_model(financial_results)

    calibration_table = probability_evaluation["calibration_table"]
    calibration_table.to_csv(
        OUTPUT_DIR / "probability_calibration.csv",
        index=False,
    )
    plot_probability_calibration(calibration_table)

    # Use highest estimated-risk engine as financial example.
    financial_example = financial_results.loc[
        financial_results["high_risk_probability"].idxmax()
    ]

    print()
    print("Probability Model Evaluation")
    print("----------------------------")
    print(
        "Brier score: "
        f"{probability_evaluation['brier_score']:.4f}"
    )
    print(
        "Log loss: "
        f"{probability_evaluation['log_loss']:.4f}"
    )
    print(
        "ROC AUC: "
        f"{probability_evaluation['roc_auc']:.3f}"
    )

    print()
    print("Probability Calibration")
    print("-----------------------")
    print(
        calibration_table.to_string(
            index=False,
            formatters={
                "mean_predicted_probability": lambda x: f"{x:.1%}",
                "actual_high_rate": lambda x: f"{x:.1%}",
            },
        )
    )

    print()
    print("Probability-Based Financial Risk")
    print("--------------------------------")
    print("Illustrative assumptions:")
    print(
        "Preventative maintenance cost: "
        f"£{MAINTENANCE_COST:,.0f}"
    )
    print(
        "Unplanned failure cost: "
        f"£{FAILURE_COST:,.0f}"
    )
    print(
        "Break-even HIGH-risk probability: "
        f"{financial_summary['break_even_probability']:.1%}"
    )

    print()
    print("Highest estimated-risk test engine:")
    print(f"Engine: {int(financial_example['engine_id'])}")
    print(f"Current cycle: {int(financial_example['cycle'])}")
    print(
        "Actual capped RUL: "
        f"{financial_example['actual_rul_capped']:.0f} cycles"
    )
    print(
        "Predicted capped RUL: "
        f"{financial_example['predicted_rul_capped']:.1f} cycles"
    )
    print(
        "Estimated probability of being within "
        f"{ACTUAL_HIGH_THRESHOLD} cycles of failure: "
        f"{financial_example['high_risk_probability']:.1%}"
    )
    print(
        "Risk-adjusted failure exposure: "
        f"£{financial_example['risk_adjusted_failure_exposure']:,.0f}"
    )
    print(
        "Preventative maintenance cost: "
        f"£{financial_example['maintenance_cost']:,.0f}"
    )
    print(
        "Illustrative risk-adjusted net benefit: "
        f"£{financial_example['risk_adjusted_net_benefit']:,.0f}"
    )
    print(
        "Maintenance economically justified: "
        f"{financial_example['maintenance_economically_justified']}"
    )
    print(f"Economic action: {financial_example['economic_action']}")

    print()
    print("Illustrative Fleet Financial Summary")
    print("------------------------------------")
    print(f"Test engines: {financial_summary['engines']}")
    print(
        "Engines where maintenance is economically justified: "
        f"{financial_summary['maintenance_justified']}"
    )
    print(
        "Maintenance outlay for those engines: "
        f"£{financial_summary['maintenance_outlay']:,.0f}"
    )
    print(
        "Risk-adjusted failure exposure for those engines: "
        f"£{financial_summary['risk_adjusted_failure_exposure']:,.0f}"
    )
    print(
        "Illustrative risk-adjusted net benefit: "
        f"£{financial_summary['risk_adjusted_net_benefit']:,.0f}"
    )

    print()
    print(
        "Financial results saved to: "
        "outputs/financial_risk_results.csv"
    )
    print(
        "Calibration results saved to: "
        "outputs/probability_calibration.csv"
    )

    print()
    print("Financial Cost Sensitivity Analysis")
    print("-----------------------------------")
    print(
        sensitivity_results.to_string(
            index=False,
            formatters={
                "maintenance_cost":
                    lambda x: f"£{x:,.0f}",

                "failure_cost":
                    lambda x: f"£{x:,.0f}",

                "break_even_probability":
                    lambda x: f"{x:.1%}",

                "maintenance_outlay":
                    lambda x: f"£{x:,.0f}",

                "risk_adjusted_exposure":
                    lambda x: f"£{x:,.0f}",

                "risk_adjusted_net_benefit":
                    lambda x: f"£{x:,.0f}",
            },
        )
    )

    print()
    print(
        "Sensitivity analysis saved to: "
        "outputs/cost_sensitivity_analysis.csv"
    ) 
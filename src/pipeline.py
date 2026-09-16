from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)
from sklearn.metrics import mean_absolute_error
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

# Illustrative financial assumptions.
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

    raise ValueError(
        f"Unknown model: {name}"
    )


# =========================================================
# DATA LOADING
# =========================================================

def load_data(
    path: Path,
) -> pd.DataFrame:
    """Load a NASA C-MAPSS FD001 dataset."""

    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )


def load_test_rul(
    path: Path,
) -> pd.DataFrame:
    """Load NASA's true RUL values for FD001 test engines."""

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


# =========================================================
# RUL TARGET CREATION
# =========================================================

def add_rul(
    df: pd.DataFrame,
) -> pd.DataFrame:
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
    """Add capped RUL values to NASA test targets."""

    test_rul = test_rul.copy()

    test_rul["actual_rul_capped"] = (
        test_rul["actual_rul"]
        .clip(upper=cap)
    )

    return test_rul


# =========================================================
# RISK FUNCTIONS
# =========================================================

def assign_actual_risk(
    rul: float,
) -> str:
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
# BASELINE MODEL
# =========================================================

def train_baseline_model(
    df: pd.DataFrame,
):
    """Train Linear Regression using an 80/20 engine split."""

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


# =========================================================
# SINGLE-SPLIT MODEL COMPARISON
# =========================================================

def compare_models(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare models using engines 81-100 for validation."""

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

    results = []

    for name in MODEL_NAMES:

        model = create_model(
            name
        )

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

    return (
        pd.DataFrame(results)
        .sort_values(
            "validation_mae"
        )
        .reset_index(
            drop=True
        )
    )


# =========================================================
# GROUPED CROSS-VALIDATION
# =========================================================

def cross_validate_models(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare models using five engine-level folds."""

    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        "rul"
    ]

    groups = df[
        "engine_id"
    ]

    group_kfold = GroupKFold(
        n_splits=5
    )

    results = []

    for name in MODEL_NAMES:

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

            model = create_model(
                name
            )

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

    return (
        pd.DataFrame(results)
        .sort_values(
            "mean_mae"
        )
        .reset_index(
            drop=True
        )
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

    No engine is predicted by a model trained using
    data from that same engine.
    """

    X = df[
        FEATURE_COLUMNS
    ]

    y = df[
        "rul"
    ]

    groups = df[
        "engine_id"
    ]

    predictions = pd.Series(
        index=df.index,
        dtype=float,
    )

    group_kfold = GroupKFold(
        n_splits=5
    )

    for (
        train_index,
        validation_index,
    ) in group_kfold.split(
        X,
        y,
        groups=groups,
    ):

        model = create_model(
            model_name
        )

        model.fit(
            X.iloc[train_index],
            y.iloc[train_index],
        )

        fold_predictions = model.predict(
            X.iloc[validation_index]
        )

        predictions.iloc[
            validation_index
        ] = fold_predictions

    results = df[
        [
            "engine_id",
            "cycle",
            "rul",
        ]
    ].copy()

    results[
        "predicted_rul"
    ] = predictions

    return results


# =========================================================
# HIGH-RISK THRESHOLD OPTIMISATION
# =========================================================

def optimise_high_risk_threshold(
    oof_results: pd.DataFrame,
    actual_high_threshold: int = ACTUAL_HIGH_THRESHOLD,
    target_recall: float = TARGET_HIGH_RECALL,
) -> pd.DataFrame:
    """
    Select an operational alert threshold using training
    out-of-fold predictions only.
    """

    actual_high = (
        oof_results["rul"]
        <= actual_high_threshold
    )

    results = []

    for threshold in range(
        THRESHOLD_SEARCH_MIN,
        THRESHOLD_SEARCH_MAX + 1,
    ):

        predicted_high = (
            oof_results["predicted_rul"]
            <= threshold
        )

        true_positive = (
            actual_high
            & predicted_high
        ).sum()

        false_negative = (
            actual_high
            & ~predicted_high
        ).sum()

        false_positive = (
            ~actual_high
            & predicted_high
        ).sum()

        true_negative = (
            ~actual_high
            & ~predicted_high
        ).sum()

        recall_denominator = (
            true_positive
            + false_negative
        )

        precision_denominator = (
            true_positive
            + false_positive
        )

        negative_denominator = (
            false_positive
            + true_negative
        )

        recall = (
            true_positive
            / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        precision = (
            true_positive
            / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        false_positive_rate = (
            false_positive
            / negative_denominator
            if negative_denominator > 0
            else 0.0
        )

        results.append(
            {
                "threshold":
                    threshold,

                "high_recall":
                    recall,

                "high_precision":
                    precision,

                "false_positive_rate":
                    false_positive_rate,

                "true_positive":
                    int(true_positive),

                "false_negative":
                    int(false_negative),

                "false_positive":
                    int(false_positive),

                "true_negative":
                    int(true_negative),
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df[
        "selected"
    ] = False

    eligible = results_df[
        results_df["high_recall"]
        >= target_recall
    ]

    if not eligible.empty:

        selected_row = (
            eligible
            .sort_values(
                [
                    "false_positive_rate",
                    "threshold",
                ]
            )
            .iloc[0]
        )

        selected_threshold = int(
            selected_row["threshold"]
        )

        results_df.loc[
            results_df["threshold"]
            == selected_threshold,
            "selected",
        ] = True

    return results_df


# =========================================================
# FAILURE-RISK PROBABILITY MODEL
# =========================================================

def train_failure_risk_model(
    oof_results: pd.DataFrame,
):
    """
    Estimate the probability that an engine is within
    30 cycles of failure from its predicted capped RUL.

    The model is trained using engine-safe OOF predictions.
    """

    risk_data = oof_results.copy()

    # Keep the probability model's input consistent with
    # the capped RUL value used during deployment.
    risk_data[
        "predicted_rul"
    ] = (
        risk_data[
            "predicted_rul"
        ]
        .clip(
            lower=0,
            upper=RUL_CAP,
        )
    )

    risk_data[
        "actual_high"
    ] = (
        risk_data["rul"]
        <= ACTUAL_HIGH_THRESHOLD
    ).astype(int)

    X = risk_data[
        ["predicted_rul"]
    ]

    y = risk_data[
        "actual_high"
    ]

    # Equalise the total influence of each engine.
    engine_rows = (
        risk_data
        .groupby("engine_id")["engine_id"]
        .transform("count")
    )

    sample_weights = (
        1.0
        / engine_rows
    )

    # Rescale weights so their mean is approximately 1.
    # This avoids unintentionally changing the relative
    # strength of LogisticRegression regularisation.
    sample_weights = (
        sample_weights
        * (
            len(sample_weights)
            / sample_weights.sum()
        )
    )

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

    X_train = df[
        FEATURE_COLUMNS
    ]

    y_train = df[
        "rul"
    ]

    model = create_model(
        model_name
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# =========================================================
# TEST STATE EXTRACTION
# =========================================================

def get_latest_engine_states(
    test_data: pd.DataFrame,
) -> pd.DataFrame:
    """Return the final observed state of each test engine."""

    return (
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


# =========================================================
# UNCAPPED TEST EVALUATION
# =========================================================

def evaluate_test_set(
    model,
    test_data: pd.DataFrame,
    test_rul: pd.DataFrame,
):
    """Evaluate uncapped RUL predictions."""

    latest_states = (
        get_latest_engine_states(
            test_data
        )
    )

    X_test = latest_states[
        FEATURE_COLUMNS
    ]

    predictions = model.predict(
        X_test
    )

    latest_states[
        "predicted_rul_raw"
    ] = predictions

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


# =========================================================
# CAPPED TEST EVALUATION
# =========================================================

def evaluate_capped_test_set(
    model,
    test_data: pd.DataFrame,
    test_rul: pd.DataFrame,
):
    """Evaluate capped-RUL model predictions."""

    latest_states = (
        get_latest_engine_states(
            test_data
        )
    )

    X_test = latest_states[
        FEATURE_COLUMNS
    ]

    predictions = model.predict(
        X_test
    )

    latest_states[
        "predicted_rul_capped_raw"
    ] = predictions

    latest_states[
        "predicted_rul_capped"
    ] = (
        latest_states[
            "predicted_rul_capped_raw"
        ]
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
            "predicted_rul_capped_raw"
        ],
    )

    return (
        results,
        capped_mae,
    )


# =========================================================
# RISK-REGION ERROR ANALYSIS
# =========================================================

def analyse_risk_regions(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Measure RUL error within actual maintenance-risk regions."""

    analysis = results.copy()

    analysis[
        "actual_risk"
    ] = (
        analysis[
            "actual_rul_capped"
        ]
        .apply(assign_actual_risk)
    )

    analysis[
        "absolute_error"
    ] = (
        analysis[
            "predicted_rul_capped_raw"
        ]
        - analysis[
            "actual_rul_capped"
        ]
    ).abs()

    analysis[
        "signed_error"
    ] = (
        analysis[
            "predicted_rul_capped_raw"
        ]
        - analysis[
            "actual_rul_capped"
        ]
    )

    analysis[
        "overprediction"
    ] = (
        analysis[
            "predicted_rul_capped_raw"
        ]
        > analysis[
            "actual_rul_capped"
        ]
    )

    risk_summary = (
        analysis
        .groupby(
            "actual_risk",
            observed=True,
        )
        .agg(
            engines=(
                "engine_id",
                "count",
            ),
            mae=(
                "absolute_error",
                "mean",
            ),
            mean_error=(
                "signed_error",
                "mean",
            ),
            overprediction_rate=(
                "overprediction",
                "mean",
            ),
        )
        .reset_index()
    )

    risk_order = {
        "HIGH": 0,
        "MEDIUM": 1,
        "LOW": 2,
    }

    risk_summary[
        "risk_order"
    ] = (
        risk_summary[
            "actual_risk"
        ]
        .map(risk_order)
    )

    risk_summary = (
        risk_summary
        .sort_values(
            "risk_order"
        )
        .drop(
            columns="risk_order"
        )
        .reset_index(
            drop=True
        )
    )

    risk_summary[
        "overprediction_rate"
    ] *= 100

    return risk_summary


# =========================================================
# MAINTENANCE CLASSIFICATION
# =========================================================

def analyse_maintenance_classification(
    results: pd.DataFrame,
    predicted_high_threshold: int,
):
    """Evaluate maintenance-risk classification."""

    analysis = results.copy()

    analysis[
        "actual_risk"
    ] = (
        analysis[
            "actual_rul_capped"
        ]
        .apply(assign_actual_risk)
    )

    analysis[
        "predicted_risk"
    ] = (
        analysis[
            "predicted_rul_capped"
        ]
        .apply(
            lambda rul:
            assign_predicted_risk(
                rul,
                predicted_high_threshold,
            )
        )
    )

    risk_order = [
        "HIGH",
        "MEDIUM",
        "LOW",
    ]

    confusion_matrix = pd.crosstab(
        analysis["actual_risk"],
        analysis["predicted_risk"],
        rownames=["Actual"],
        colnames=["Predicted"],
    )

    confusion_matrix = (
        confusion_matrix
        .reindex(
            index=risk_order,
            columns=risk_order,
            fill_value=0,
        )
    )

    correct = (
        analysis["actual_risk"]
        == analysis["predicted_risk"]
    )

    accuracy = correct.mean()

    actual_high = (
        analysis["actual_risk"]
        == "HIGH"
    )

    predicted_high = (
        analysis["predicted_risk"]
        == "HIGH"
    )

    true_positive = (
        actual_high
        & predicted_high
    )

    false_negative = (
        actual_high
        & ~predicted_high
    )

    false_positive = (
        ~actual_high
        & predicted_high
    )

    high_to_low = (
        actual_high
        & (
            analysis["predicted_risk"]
            == "LOW"
        )
    )

    high_recall = (
        true_positive.sum()
        / actual_high.sum()
        if actual_high.sum() > 0
        else 0.0
    )

    high_precision = (
        true_positive.sum()
        / predicted_high.sum()
        if predicted_high.sum() > 0
        else 0.0
    )

    summary = {
        "accuracy":
            accuracy,

        "high_risk_engines":
            int(actual_high.sum()),

        "high_risk_correct":
            int(true_positive.sum()),

        "high_risk_missed":
            int(false_negative.sum()),

        "high_risk_recall":
            high_recall,

        "high_risk_precision":
            high_precision,

        "false_high_alerts":
            int(false_positive.sum()),

        "high_to_low_misses":
            int(high_to_low.sum()),
    }

    return (
        analysis,
        confusion_matrix,
        summary,
    )


# =========================================================
# OPERATIONAL MAINTENANCE DECISION
# =========================================================

def maintenance_decision(
    predicted_rul: float,
    high_threshold: int,
) -> dict:
    """Convert predicted RUL into an operational decision."""

    predicted_rul = max(
        0.0,
        float(predicted_rul),
    )

    risk = assign_predicted_risk(
        predicted_rul,
        high_threshold,
    )

    if risk == "HIGH":

        recommendation = (
            "Schedule preventative maintenance"
        )

    elif risk == "MEDIUM":

        recommendation = (
            "Increase monitoring"
        )

    else:

        recommendation = (
            "Continue operating"
        )

    return {
        "predicted_rul":
            predicted_rul,

        "risk":
            risk,

        "recommendation":
            recommendation,
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
    Convert estimated HIGH-risk probability into an
    illustrative expected financial decision.

    This assumes the probability of being within 30 cycles
    of failure is a proxy for failure exposure over the
    decision horizon.
    """

    financial_results = (
        results.copy()
    )

    risk_features = (
        financial_results[
            ["predicted_rul_capped"]
        ]
        .rename(
            columns={
                "predicted_rul_capped":
                    "predicted_rul"
            }
        )
    )

    high_risk_probability = (
        risk_model.predict_proba(
            risk_features
        )[:, 1]
    )

    financial_results[
        "high_risk_probability"
    ] = high_risk_probability

    financial_results[
        "expected_failure_loss"
    ] = (
        financial_results[
            "high_risk_probability"
        ]
        * failure_cost
    )

    financial_results[
        "maintenance_cost"
    ] = maintenance_cost

    financial_results[
        "failure_cost"
    ] = failure_cost

    financial_results[
        "break_even_probability"
    ] = (
        maintenance_cost
        / failure_cost
    )

    financial_results[
        "financial_benefit"
    ] = (
        financial_results[
            "expected_failure_loss"
        ]
        - maintenance_cost
    )

    financial_results[
        "maintenance_economically_justified"
    ] = (
        financial_results[
            "expected_failure_loss"
        ]
        >= maintenance_cost
    )

    financial_results[
        "economic_action"
    ] = financial_results[
        "maintenance_economically_justified"
    ].map(
        {
            True:
                "Preventative maintenance",

            False:
                "Continue / monitor",
        }
    )

    return financial_results


def summarise_financial_risk(
    financial_results: pd.DataFrame,
) -> dict:
    """
    Summarise illustrative financial implications
    across the test fleet.
    """

    justified = (
        financial_results[
            "maintenance_economically_justified"
        ]
    )

    selected = (
        financial_results[
            justified
        ]
    )

    maintenance_outlay = (
        selected[
            "maintenance_cost"
        ].sum()
    )

    expected_failure_exposure = (
        selected[
            "expected_failure_loss"
        ].sum()
    )

    expected_net_benefit = (
        selected[
            "financial_benefit"
        ].sum()
    )

    break_even_probability = (
        financial_results[
            "break_even_probability"
        ].iloc[0]
    )

    return {
        "engines":
            len(financial_results),

        "maintenance_justified":
            int(justified.sum()),

        "break_even_probability":
            break_even_probability,

        "maintenance_outlay":
            maintenance_outlay,

        "expected_failure_exposure":
            expected_failure_exposure,

        "expected_net_benefit":
            expected_net_benefit,
    }


# =========================================================
# VISUALISATION
# =========================================================

def plot_test_predictions(
    results: pd.DataFrame,
    model_name: str,
) -> None:
    """Plot uncapped actual versus predicted RUL."""

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
        f"{model_name}: "
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


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =====================================================
    # UNCAPPED EXPERIMENT
    # =====================================================

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

    # -----------------------------------------------------
    # Baseline Linear Regression
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Single-split model comparison
    # -----------------------------------------------------

    model_comparison = (
        compare_models(
            data
        )
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
                    f"{x:.2f}",
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

    # -----------------------------------------------------
    # Uncapped grouped cross-validation
    # -----------------------------------------------------

    cv_results = (
        cross_validate_models(
            data
        )
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

    # -----------------------------------------------------
    # Final uncapped model
    # -----------------------------------------------------

    final_model = (
        train_final_model(
            data,
            best_cv_model,
        )
    )

    print()
    print(
        "Final Uncapped Model"
    )
    print(
        "--------------------"
    )

    print(
        f"Model: {best_cv_model}"
    )

    print(
        "Training engines: 100"
    )

    # -----------------------------------------------------
    # NASA test data
    # -----------------------------------------------------

    test_data = load_data(
        TEST_DATA_PATH
    )

    test_rul = load_test_rul(
        TEST_RUL_PATH
    )

    (
        test_results,
        test_mae,
    ) = evaluate_test_set(
        final_model,
        test_data,
        test_rul,
    )

    plot_test_predictions(
        test_results,
        best_cv_model,
    )

    print()
    print(
        "NASA FD001 Uncapped Test Evaluation"
    )
    print(
        "-----------------------------------"
    )

    print(
        f"Test engines: "
        f"{len(test_results)}"
    )

    print(
        "Uncapped Test MAE: "
        f"{test_mae:.2f} cycles"
    )

    # =====================================================
    # CAPPED-RUL EXPERIMENT
    # =====================================================

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

    # =====================================================
    # OUT-OF-FOLD TRAINING PREDICTIONS
    # =====================================================

    oof_results = (
        generate_oof_predictions(
            capped_data,
            capped_best_model,
        )
    )

    # =====================================================
    # HIGH-RISK THRESHOLD OPTIMISATION
    # =====================================================

    threshold_results = (
        optimise_high_risk_threshold(
            oof_results,
            actual_high_threshold=
                ACTUAL_HIGH_THRESHOLD,
            target_recall=
                TARGET_HIGH_RECALL,
        )
    )

    threshold_results.to_csv(
        OUTPUT_DIR
        / "threshold_optimization.csv",
        index=False,
    )

    selected_thresholds = (
        threshold_results[
            threshold_results[
                "selected"
            ]
        ]
    )

    print()
    print(
        "HIGH-Risk Threshold Optimisation"
    )
    print(
        "--------------------------------"
    )

    print(
        "Actual HIGH definition: "
        f"RUL <= "
        f"{ACTUAL_HIGH_THRESHOLD} cycles"
    )

    print(
        "Target OOF HIGH-risk recall: "
        f"{TARGET_HIGH_RECALL:.0%}"
    )

    if not selected_thresholds.empty:

        selected = (
            selected_thresholds.iloc[0]
        )

        optimised_high_threshold = int(
            selected[
                "threshold"
            ]
        )

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
            "OOF false negatives "
            "(cycle observations): "
            f"{int(selected['false_negative'])}"
        )

        print(
            "OOF false positives "
            "(cycle observations): "
            f"{int(selected['false_positive'])}"
        )

    else:

        optimised_high_threshold = (
            ACTUAL_HIGH_THRESHOLD
        )

        print(
            "No threshold achieved the "
            "target recall."
        )

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

    risk_probability_model = (
        train_failure_risk_model(
            oof_results
        )
    )

    print()
    print(
        "Failure-Risk Probability Model"
    )
    print(
        "------------------------------"
    )

    print(
        "Model: Logistic Regression"
    )

    print(
        "Input: out-of-fold predicted capped RUL"
    )

    print(
        "Target: probability actual RUL "
        f"<= {ACTUAL_HIGH_THRESHOLD} cycles"
    )

    # =====================================================
    # FINAL CAPPED MODEL
    # =====================================================

    capped_final_model = (
        train_final_model(
            capped_data,
            capped_best_model,
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
        f"Model: "
        f"{capped_best_model}"
    )

    print(
        "Capped Test MAE: "
        f"{capped_test_mae:.2f} cycles"
    )

    # =====================================================
    # RISK-REGION ERROR ANALYSIS
    # =====================================================

    risk_analysis = (
        analyse_risk_regions(
            capped_results
        )
    )

    print()
    print(
        "Risk-Region Error Analysis"
    )
    print(
        "--------------------------"
    )

    print(
        risk_analysis.to_string(
            index=False,
            formatters={
                "mae":
                    lambda x:
                    f"{x:.2f}",

                "mean_error":
                    lambda x:
                    f"{x:+.2f}",

                "overprediction_rate":
                    lambda x:
                    f"{x:.1f}%",
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
        predicted_high_threshold=
            ACTUAL_HIGH_THRESHOLD,
    )

    print()
    print(
        "Maintenance Classification "
        "(Default Threshold)"
    )
    print(
        "----------------------------------------------"
    )

    print(
        f"Predicted HIGH threshold: "
        f"{ACTUAL_HIGH_THRESHOLD} cycles"
    )

    print()
    print(
        default_confusion_matrix.to_string()
    )

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

    print(
        "HIGH-risk missed: "
        f"{default_summary['high_risk_missed']}"
    )

    print(
        "False HIGH alerts: "
        f"{default_summary['false_high_alerts']}"
    )

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
        predicted_high_threshold=
            optimised_high_threshold,
    )

    print()
    print(
        "Maintenance Classification "
        "(Optimised Threshold)"
    )
    print(
        "------------------------------------------------"
    )

    print(
        "Predicted HIGH threshold: "
        f"{optimised_high_threshold} cycles"
    )

    print()
    print(
        optimised_confusion_matrix.to_string()
    )

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

    example_engine = (
        capped_results.iloc[0]
    )

    decision = (
        maintenance_decision(
            example_engine[
                "predicted_rul_capped"
            ],
            high_threshold=
                optimised_high_threshold,
        )
    )

    print()
    print(
        "Maintenance Decision "
        "(Capped Model)"
    )
    print(
        "----------------------------"
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
        "Actual capped RUL: "
        f"{example_engine['actual_rul_capped']:.0f} "
        "cycles"
    )

    print(
        "Predicted capped RUL: "
        f"{decision['predicted_rul']:.1f} "
        "cycles"
    )

    print(
        "Operational HIGH threshold: "
        f"{optimised_high_threshold} cycles"
    )

    print(
        "Risk level: "
        f"{decision['risk']}"
    )

    print(
        "Recommendation: "
        f"{decision['recommendation']}"
    )

    # =====================================================
    # PROBABILITY-BASED FINANCIAL RISK
    # =====================================================

    financial_results = (
        calculate_financial_risk(
            capped_results,
            risk_probability_model,
            maintenance_cost=
                MAINTENANCE_COST,
            failure_cost=
                FAILURE_COST,
        )
    )

    financial_results.to_csv(
        OUTPUT_DIR
        / "financial_risk_results.csv",
        index=False,
    )

    financial_summary = (
        summarise_financial_risk(
            financial_results
        )
    )

    # Use the engine with the highest estimated
    # HIGH-risk probability as the financial example.
    financial_example = (
        financial_results.loc[
            financial_results[
                "high_risk_probability"
            ].idxmax()
        ]
    )

    print()
    print(
        "Probability-Based Financial Risk"
    )
    print(
        "--------------------------------"
    )

    print(
        "Illustrative assumptions:"
    )

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
    print(
        "Highest estimated-risk test engine:"
    )

    print(
        "Engine: "
        f"{int(financial_example['engine_id'])}"
    )

    print(
        "Current cycle: "
        f"{int(financial_example['cycle'])}"
    )

    print(
        "Actual capped RUL: "
        f"{financial_example['actual_rul_capped']:.0f} "
        "cycles"
    )

    print(
        "Predicted capped RUL: "
        f"{financial_example['predicted_rul_capped']:.1f} "
        "cycles"
    )

    print(
        "Estimated probability of being "
        f"within {ACTUAL_HIGH_THRESHOLD} cycles "
        "of failure: "
        f"{financial_example['high_risk_probability']:.1%}"
    )

    print(
        "Expected failure loss: "
        f"£{financial_example['expected_failure_loss']:,.0f}"
    )

    print(
        "Preventative maintenance cost: "
        f"£{financial_example['maintenance_cost']:,.0f}"
    )

    print(
        "Expected financial benefit "
        "of maintenance: "
        f"£{financial_example['financial_benefit']:,.0f}"
    )

    print(
        "Maintenance economically justified: "
        f"{financial_example['maintenance_economically_justified']}"
    )

    print(
        "Economic action: "
        f"{financial_example['economic_action']}"
    )

    print()
    print(
        "Illustrative Fleet Financial Summary"
    )
    print(
        "------------------------------------"
    )

    print(
        "Test engines: "
        f"{financial_summary['engines']}"
    )

    print(
        "Engines where maintenance is "
        "economically justified: "
        f"{financial_summary['maintenance_justified']}"
    )

    print(
        "Maintenance outlay for those engines: "
        f"£{financial_summary['maintenance_outlay']:,.0f}"
    )

    print(
        "Expected failure exposure "
        "for those engines: "
        f"£{financial_summary['expected_failure_exposure']:,.0f}"
    )

    print(
        "Illustrative expected net benefit: "
        f"£{financial_summary['expected_net_benefit']:,.0f}"
    )

    print()
    print(
        "Financial results saved to: "
        "outputs/financial_risk_results.csv"
    ) 
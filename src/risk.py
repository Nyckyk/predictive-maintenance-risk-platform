import pandas as pd

from src.config import (
    ACTUAL_HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    TARGET_HIGH_RECALL,
    THRESHOLD_SEARCH_MAX,
    THRESHOLD_SEARCH_MIN,
)


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

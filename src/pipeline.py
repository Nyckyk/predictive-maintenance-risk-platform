from pathlib import Path
import sys


# Support both:
#   python src/pipeline.py
# and:
#   python -m src.pipeline
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.config import (
    ACTUAL_HIGH_THRESHOLD,
    FAILURE_COST,
    MAINTENANCE_COST,
    OUTPUT_DIR,
    RUL_CAP,
    TARGET_HIGH_RECALL,
    TEST_DATA_PATH,
    TEST_RUL_PATH,
    TRAIN_DATA_PATH,
)
from src.data import (
    add_capped_rul,
    add_capped_test_rul,
    add_rul,
    load_data,
    load_test_rul,
)
from src.evaluation import (
    evaluate_capped_test_set,
    evaluate_probability_model,
    evaluate_test_set,
)
from src.finance import (
    calculate_financial_risk,
    run_cost_sensitivity_analysis,
    summarise_financial_risk,
)
from src.models import (
    compare_models,
    cross_validate_models,
    generate_oof_predictions,
    train_baseline_model,
    train_failure_risk_model,
    train_final_model,
)
from src.plotting import (
    plot_probability_calibration,
    plot_rul_predictions,
)
from src.risk import (
    analyse_maintenance_classification,
    analyse_risk_regions,
    maintenance_decision,
    optimise_high_risk_threshold,
)


def main():
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


if __name__ == "__main__":
    main()

import pandas as pd

from src.config import FAILURE_COST, MAINTENANCE_COST


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

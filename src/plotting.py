import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import OUTPUT_DIR


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

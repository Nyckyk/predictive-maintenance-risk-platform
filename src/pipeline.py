from pathlib import Path

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


DATA_PATH = Path("data/raw/train_FD001.txt")

COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def load_data(path: Path) -> pd.DataFrame:
    """Load the NASA C-MAPSS FD001 training dataset."""

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )

    return df


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Remaining Useful Life for every engine cycle."""

    df = df.copy()

    max_cycles = df.groupby("engine_id")["cycle"].transform("max")

    df["rul"] = max_cycles - df["cycle"]

    return df


def train_baseline_model(df: pd.DataFrame):
    """Train and evaluate a baseline RUL prediction model."""

    feature_columns = (
        [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )

    # Keep entire engines separate between training and validation.
    train_data = df[df["engine_id"] <= 80]
    validation_data = df[df["engine_id"] > 80]

    X_train = train_data[feature_columns]
    y_train = train_data["rul"]

    X_validation = validation_data[feature_columns]
    y_validation = validation_data["rul"]

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)

    mae = mean_absolute_error(
        y_validation,
        predictions,
    )

    return model, validation_data, predictions, mae


if __name__ == "__main__":
    data = load_data(DATA_PATH)
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

    model, validation_data, predictions, mae = train_baseline_model(data)

    print()
    print("Baseline RUL model")
    print("------------------")
    print("Model: Linear Regression")
    print("Training engines: 80")
    print("Validation engines: 20")
    print(f"Mean Absolute Error: {mae:.2f} cycles")
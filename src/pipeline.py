from pathlib import Path

import pandas as pd


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
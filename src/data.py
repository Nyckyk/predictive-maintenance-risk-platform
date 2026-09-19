from pathlib import Path

import pandas as pd

from src.config import COLUMN_NAMES, RUL_CAP


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

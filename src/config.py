from pathlib import Path


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

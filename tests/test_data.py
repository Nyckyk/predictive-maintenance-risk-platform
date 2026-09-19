import pandas as pd

from src.data import add_capped_rul, add_capped_test_rul, add_rul, load_test_rul


def test_add_rul_calculates_remaining_cycles_per_engine():
    data = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 1, 2],
        }
    )

    result = add_rul(data)

    assert result["rul"].tolist() == [2, 1, 0, 1, 0]


def test_add_rul_does_not_modify_input_dataframe():
    data = pd.DataFrame({"engine_id": [1, 1], "cycle": [1, 2]})

    add_rul(data)

    assert "rul" not in data.columns


def test_add_capped_rul_applies_cap():
    data = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5],
        }
    )

    result = add_capped_rul(data, cap=2)

    assert result["rul"].tolist() == [2, 2, 2, 1, 0]
    assert result["rul"].max() <= 2


def test_add_capped_test_rul_applies_cap():
    test_rul = pd.DataFrame(
        {
            "engine_id": [1, 2, 3],
            "actual_rul": [10, 125, 180],
        }
    )

    result = add_capped_test_rul(test_rul, cap=125)

    assert result["actual_rul_capped"].tolist() == [10, 125, 125]


def test_load_test_rul_assigns_engine_ids(tmp_path):
    path = tmp_path / "RUL.txt"
    path.write_text("12\n34\n56\n", encoding="utf-8")

    result = load_test_rul(path)

    assert result["engine_id"].tolist() == [1, 2, 3]
    assert result["actual_rul"].tolist() == [12, 34, 56]

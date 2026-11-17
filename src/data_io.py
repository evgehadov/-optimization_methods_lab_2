from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "times.csv"


def load_times(path: Path = DEFAULT_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, encoding="utf-8")
    validate_times(df)
    return df


def validate_times(df: pd.DataFrame) -> None:
    if df.shape[0] != df.shape[1]:
        raise ValueError(f"Матрица должна быть квадратной, получено {df.shape}")
    if df.isna().any().any():
        raise ValueError("В матрице есть пропуски")
    if (df.to_numpy() <= 0).any():
        raise ValueError("Время должно быть положительным")


def to_array(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=float)


if __name__ == "__main__":
    data = load_times()
    print(f"Размер: {data.shape}")
    print(f"Бригады: {list(data.index)}")
    print(f"Объекты: {list(data.columns)}")
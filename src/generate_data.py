from pathlib import Path

import numpy as np
import pandas as pd

SEED = 10007
N_CREWS = 6
N_SITES = 6
T_MIN, T_MAX = 4, 20

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "times.csv"


def generate_times(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = rng.integers(T_MIN, T_MAX + 1, size=(N_CREWS, N_SITES))
    rows = [f"Бригада_{i + 1}" for i in range(N_CREWS)]
    cols = [f"Объект_{j + 1}" for j in range(N_SITES)]
    return pd.DataFrame(times, index=rows, columns=cols)


def main() -> None:
    df = generate_times()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, encoding="utf-8")
    print(df)
    print(f"Сохранено: {OUT_PATH}")


if __name__ == "__main__":
    main()
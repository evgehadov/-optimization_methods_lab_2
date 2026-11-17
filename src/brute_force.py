from __future__ import annotations

from itertools import permutations
from pathlib import Path

import numpy as np

from data_io import load_times, to_array

ROOT = Path(__file__).resolve().parent.parent


def brute_force(times: np.ndarray):
    n = times.shape[0]
    best, best_perms = float("inf"), []
    for p in permutations(range(n)):
        cost = float(sum(times[i, p[i]] for i in range(n)))
        if cost < best - 1e-9:
            best, best_perms = cost, [p]
        elif abs(cost - best) <= 1e-9:
            best_perms.append(p)
    return best, best_perms


if __name__ == "__main__":
    df = load_times()
    best, perms = brute_force(to_array(df))
    lines = [f"Перебрано назначений: 720",
             f"Минимальное суммарное время, ч: {best}",
             f"Число оптимальных назначений: {len(perms)}"]
    for p in perms:
        lines.append("; ".join(f"{df.index[i]}->{df.columns[j]}" for i, j in enumerate(p)))
    text = "\n".join(lines)
    print(text)
    (ROOT / "results" / "brute_force.txt").write_text(text + "\n", encoding="utf-8")
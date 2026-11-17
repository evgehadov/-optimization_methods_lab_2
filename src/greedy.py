from __future__ import annotations

import time

import numpy as np


def greedy_assignment(times: np.ndarray, forbidden=None) -> dict:
    n = times.shape[0]
    forbidden = set(forbidden or [])
    start = time.perf_counter()

    pairs = sorted(
        ((times[i, j], i, j) for i in range(n) for j in range(n) if (i, j) not in forbidden),
        key=lambda p: (p[0], p[1], p[2]),
    )
    x = np.zeros((n, n))
    used_crews, used_sites = set(), set()
    for _, i, j in pairs:
        if i not in used_crews and j not in used_sites:
            x[i, j] = 1
            used_crews.add(i)
            used_sites.add(j)

    elapsed = time.perf_counter() - start
    complete = len(used_crews) == n
    return {
        "x": x,
        "objective": float((times * x).sum()) if complete else None,
        "complete": complete,
        "seconds": elapsed,
    }


def compare_with_optimum(greedy_obj: float, optimal_obj: float) -> dict:
    gap = greedy_obj - optimal_obj
    return {"gap_hours": gap, "gap_percent": 100 * gap / optimal_obj}


if __name__ == "__main__":
    from data_io import load_times, to_array
    from model import build_model
    from solve import solve

    df = load_times()
    times = to_array(df)
    opt = solve(build_model(times))
    g = greedy_assignment(times)

    print("Жадное назначение:")
    for i in range(times.shape[0]):
        j = int(np.argmax(g["x"][i]))
        print(f"  {df.index[i]} -> {df.columns[j]} ({times[i, j]:.0f} ч)")
    print("Суммарное время жадного, ч:", g["objective"])
    print("Оптимум солвера, ч:", opt.objective)
    cmp_ = compare_with_optimum(g["objective"], opt.objective)
    print(f"Разрыв: {cmp_['gap_hours']:.0f} ч ({cmp_['gap_percent']:.1f}%)")
    print(f"Время работы жадного, с: {g['seconds']:.6f}")
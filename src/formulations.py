from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from data_io import load_times, to_array
from model import build_model
from scenarios import GROWTH, find_critical_pairs, pairs_of
from solve import solve

ROOT = Path(__file__).resolve().parent.parent
OUT_SCENARIOS = ROOT / "results" / "formulations.csv"
OUT_SCALING = ROOT / "results" / "formulations_scaling.csv"


def build_node_arc(times: np.ndarray, forbidden=None):
    """Формулировка «узел-дуга»: задача о назначениях как поток по сети.

    Вершины: исток S, n бригад, n объектов, сток T.
    Дуги: S -> бригада (стоимость 0), бригада -> объект (стоимость t_ij),
    объект -> T (стоимость 0). Пропускная способность каждой дуги 1.
    В каждой вершине поток входит и выходит поровну, из S выходит n единиц.

    Порядок переменных: n дуг из истока, n*n дуг бригада-объект, n дуг в сток.
    """
    n = times.shape[0]
    n_vars = n + n * n + n
    mid = n  # с какого номера идут дуги бригада-объект
    end = n + n * n  # с какого номера идут дуги в сток

    c = np.zeros(n_vars)
    c[mid:end] = times.flatten()

    A = np.zeros((2 * n + 2, n_vars))
    b = np.zeros(2 * n + 2)

    A[0, 0:n] = 1                       # из истока выходит n единиц
    b[0] = n
    for i in range(n):                  # бригада i: вошло = вышло
        A[1 + i, i] = 1
        A[1 + i, mid + i * n: mid + (i + 1) * n] = -1
    for j in range(n):                  # объект j: вошло = вышло
        A[n + 1 + j, mid + j: end: n] = 1
        A[n + 1 + j, end + j] = -1
    A[2 * n + 1, end:end + n] = 1       # в сток приходит n единиц
    b[2 * n + 1] = n

    bounds = [(0, 1)] * n_vars
    for i, j in (forbidden or []):
        bounds[mid + i * n + j] = (0, 0)
    return c, A, b, bounds


def solve_node_arc(times: np.ndarray, forbidden=None):
    c, A, b, bounds = build_node_arc(times, forbidden)
    n = times.shape[0]
    res = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs-ds")
    x = res.x[n:n + n * n].reshape(n, n) if res.status == 0 else None
    return res, x, (c, A, b)


def mean_time_ms(func, repeats: int) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        func()
    return 1000 * (time.perf_counter() - start) / repeats


def compare_on_scenarios() -> pd.DataFrame:
    df = load_times()
    times = to_array(df)
    n = times.shape[0]

    base_pairs = pairs_of(solve(build_model(times)).x)
    critical = find_critical_pairs(times, base_pairs)
    worst = critical[0][1]
    top2 = [critical[0][1], critical[1][1]]
    closed_column = [(i, base_pairs[0][1]) for i in range(n)]

    cases = [
        ("базовый", times, []),
        ("дефицитный_1", times, [worst]),
        ("дефицитный_2", times * GROWTH, []),
        ("стрессовый", times * GROWTH, top2),
        ("несовместный", times, closed_column),
    ]

    rows = []
    for name, t, forb in cases:
        direct_model = build_model(t, forbidden=forb)
        direct = solve(direct_model)
        res, x_na, (_, A_na, _) = solve_node_arc(t, forb)
        ok = direct.status == 0 and res.status == 0
        rows.append({
            "сценарий": name,
            "прямая_статус": direct.status_text.split(":")[0],
            "узел_дуга_статус": "optimal" if res.status == 0 else "infeasible",
            "прямая_цель_ч": round(direct.objective, 6) if direct.status == 0 else None,
            "узел_дуга_цель_ч": round(float(res.fun), 6) if res.status == 0 else None,
            "цели_совпали": abs(direct.objective - res.fun) <= 1e-6 if ok else None,
            "назначения_совпали": bool(np.allclose(np.round(direct.x), np.round(x_na))) if ok else None,
            "прямая_мс": round(mean_time_ms(lambda: solve(direct_model), 200), 3),
            "узел_дуга_мс": round(mean_time_ms(lambda: solve_node_arc(t, forb), 200), 3),
        })
    return pd.DataFrame(rows)


def size_of_models(n: int) -> dict:
    t = np.ones((n, n))
    direct = build_model(t)
    _, A_na, _, _ = build_node_arc(t)
    return {
        "прямая_переменных": direct.A_eq.shape[1],
        "прямая_ограничений": direct.A_eq.shape[0],
        "прямая_ненулей": int(np.count_nonzero(direct.A_eq)),
        "узел_дуга_переменных": A_na.shape[1],
        "узел_дуга_ограничений": A_na.shape[0],
        "узел_дуга_ненулей": int(np.count_nonzero(A_na)),
    }


def compare_scaling(sizes=(6, 20, 50, 100), seed: int = 10007) -> pd.DataFrame:
    """Как растут размер модели и время решения при увеличении n."""
    rows = []
    for n in sizes:
        rng = np.random.default_rng(seed)
        t = rng.integers(4, 21, size=(n, n)).astype(float)
        repeats = 20 if n <= 50 else 3

        direct_model = build_model(t)
        direct = solve(direct_model)
        res, _, _ = solve_node_arc(t)
        row = {"n": n}
        row.update(size_of_models(n))
        row["цели_совпали"] = abs(direct.objective - res.fun) <= 1e-6
        row["прямая_мс"] = round(mean_time_ms(lambda: solve(direct_model), repeats), 2)
        row["узел_дуга_мс"] = round(mean_time_ms(lambda: solve_node_arc(t), repeats), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 220)

    print("Размер модели при n = 6:")
    for k, v in size_of_models(6).items():
        print(f"  {k}: {v}")

    sc = compare_on_scenarios()
    OUT_SCENARIOS.parent.mkdir(parents=True, exist_ok=True)
    sc.to_csv(OUT_SCENARIOS, index=False, encoding="utf-8")
    print("\nСравнение на сценариях:")
    print(sc.to_string(index=False))

    scaling = compare_scaling()
    scaling.to_csv(OUT_SCALING, index=False, encoding="utf-8")
    print("\nРост размера и времени с увеличением n:")
    print(scaling.to_string(index=False))
    print(f"\nСохранено: {OUT_SCENARIOS}\nСохранено: {OUT_SCALING}")


if __name__ == "__main__":
    main()
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from data_io import load_times, to_array
from model import AssignmentModel, build_model
from scenarios import GROWTH, find_critical_pairs, pairs_of
from solve import solve

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "lp_vs_milp.csv"

REPEATS = 200  # сколько раз повторяем решение, чтобы замерить время честнее


def solve_milp(model: AssignmentModel):
    """Та же модель, но x_ij целые (0 или 1): неделимые назначения."""
    lower = np.array([b[0] for b in model.bounds], dtype=float)
    upper = np.array([b[1] for b in model.bounds], dtype=float)
    res = milp(
        c=model.c,
        constraints=LinearConstraint(model.A_eq, model.b_eq, model.b_eq),
        integrality=np.ones(model.c.size),
        bounds=Bounds(lower, upper),
    )
    return res


def mean_time_ms(func, repeats: int = REPEATS) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        func()
    return 1000 * (time.perf_counter() - start) / repeats


def main() -> None:
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
        model = build_model(t, forbidden=forb)
        lp = solve(model)
        mi = solve_milp(model)
        mi_ok = mi.status == 0

        lp_integral = bool(lp.status == 0 and np.allclose(lp.x, np.round(lp.x)))
        same_obj = (lp.status == 0 and mi_ok and abs(lp.objective - mi.fun) <= 1e-6)
        same_plan = (lp.status == 0 and mi_ok and np.allclose(
            np.round(lp.x), np.round(mi.x.reshape(n, n))))

        rows.append({
            "сценарий": name,
            "LP_статус": lp.status_text.split(":")[0],
            "MILP_статус": "optimal" if mi_ok else "infeasible",
            "LP_время_ч": round(lp.objective, 6) if lp.status == 0 else None,
            "MILP_время_ч": round(float(mi.fun), 6) if mi_ok else None,
            "цели_совпали": same_obj if lp.status == 0 else None,
            "назначения_совпали": same_plan if lp.status == 0 else None,
            "LP_решение_целое": lp_integral if lp.status == 0 else None,
            "MILP_gap": getattr(mi, "mip_gap", None) if mi_ok else None,
            "MILP_узлов": getattr(mi, "mip_node_count", None) if mi_ok else None,
            "LP_мс": round(mean_time_ms(lambda: solve(model)), 3),
            "MILP_мс": round(mean_time_ms(lambda: solve_milp(model)), 3),
        })

    out = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))
    print(f"\nСохранено: {OUT_PATH}")


if __name__ == "__main__":
    main()
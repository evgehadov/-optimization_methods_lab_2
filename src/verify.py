from __future__ import annotations

import numpy as np

TOL = 1e-7


def verify_primal(x: np.ndarray, times: np.ndarray, reported_objective: float,
                  forbidden=None) -> dict:
    n = times.shape[0]
    forbidden = forbidden or []
    checks = {
        "матрица x квадратная": x.shape == (n, n),
        "x >= 0": bool((x >= -TOL).all()),
        "x <= 1": bool((x <= 1 + TOL).all()),
        "каждая бригада на один объект": bool(np.allclose(x.sum(axis=1), 1, atol=TOL)),
        "на каждом объекте одна бригада": bool(np.allclose(x.sum(axis=0), 1, atol=TOL)),
        "запрещённые пары не использованы": all(abs(x[i, j]) <= TOL for i, j in forbidden),
        "решение целое (0 или 1)": bool(np.allclose(x, np.round(x), atol=TOL)),
    }
    recomputed = float((times * x).sum())
    checks["цель совпадает с солвером"] = abs(recomputed - reported_objective) <= 1e-6
    return {"checks": checks, "recomputed_objective": recomputed,
            "all_ok": all(checks.values())}


def verify_dual(duals: np.ndarray, times: np.ndarray, primal_objective: float,
                forbidden=None) -> dict:
    n = times.shape[0]
    u, v = duals[:n], duals[n:]
    forbidden = set(forbidden or [])
    reduced = times - u[:, None] - v[None, :]
    mask = np.ones((n, n), dtype=bool)
    for i, j in forbidden:
        mask[i, j] = False
    dual_obj = float(u.sum() + v.sum())
    checks = {
        "двойственная допустимость (c - u - v >= 0)": bool((reduced[mask] >= -1e-6).all()),
        "двойственная цель = прямая цель": abs(dual_obj - primal_objective) <= 1e-6,
    }
    return {"checks": checks, "dual_objective": dual_obj, "all_ok": all(checks.values())}


def print_report(title: str, report: dict) -> None:
    print(title)
    for name, ok in report["checks"].items():
        print(f"  [{'OK' if ok else 'ОШИБКА'}] {name}")


if __name__ == "__main__":
    from data_io import load_times, to_array
    from model import build_model
    from solve import solve

    times = to_array(load_times())
    result = solve(build_model(times))

    p = verify_primal(result.x, times, result.objective)
    print_report("Проверка допустимости:", p)
    print("  Пересчитанная цель, ч:", p["recomputed_objective"])

    d = verify_dual(result.duals, times, result.objective)
    print_report("Проверка оптимальности:", d)
    print("  Двойственная цель, ч:", d["dual_objective"])
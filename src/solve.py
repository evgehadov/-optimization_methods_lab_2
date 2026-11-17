from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


import numpy as np
import pandas as pd
from scipy.optimize import linprog

from model import AssignmentModel

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "results" / "solver_log.txt"

STATUS_TEXT = {
    0: "optimal: оптимальное решение найдено",
    1: "лимит итераций исчерпан",
    2: "infeasible: допустимых решений нет (ограничения противоречат друг другу)",
    3: "unbounded: целевая функция не ограничена",
    4: "числовые трудности солвера",
}


@dataclass
class SolveResult:
    status: int
    status_text: str
    objective: float | None
    x: np.ndarray | None
    duals: np.ndarray | None
    iterations: int | None


def solve(model: AssignmentModel, method: str = "highs") -> SolveResult:
    res = linprog(
        c=model.c,
        A_eq=model.A_eq,
        b_eq=model.b_eq,
        bounds=model.bounds,
        method=method,
    )
    ok = res.status == 0
    return SolveResult(
        status=res.status,
        status_text=STATUS_TEXT.get(res.status, res.message),
        objective=float(res.fun) if ok else None,
        x=res.x.reshape(model.n, model.n) if ok else None,
        duals=np.asarray(res.eqlin.marginals) if ok else None,
        iterations=getattr(res, "nit", None),
    )


def assignment_table(result: SolveResult, times: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, crew in enumerate(times.index):
        j = int(np.argmax(result.x[i]))
        rows.append({
            "бригада": crew,
            "объект": times.columns[j],
            "время_ч": float(times.iloc[i, j]),
            "значение_x": round(float(result.x[i, j]), 6),
        })
    return pd.DataFrame(rows)


def write_log(result: SolveResult, name: str = "base") -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"[{name}]",
        f"статус: {result.status} - {result.status_text}",
        f"итераций: {result.iterations}",
        f"целевая функция: {result.objective}",
        "",
    ]
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    from data_io import load_times, to_array
    from model import build_model

    times = load_times()
    model = build_model(to_array(times))
    result = solve(model)

    print("Статус:", result.status_text)
    print("Итераций:", result.iterations)
    print("Суммарное время, ч:", result.objective)
    print(assignment_table(result, times).to_string(index=False))
    print("Решение только из 0 и 1:", bool(np.allclose(result.x, np.round(result.x))))
    print("Теневые цены (строки, затем столбцы):", np.round(result.duals, 3))

    LOG_PATH.unlink(missing_ok=True)
    write_log(result)
    tbl = assignment_table(result, times)
    tbl.to_csv(ROOT / "results" / "assignment.csv", index=False, encoding="utf-8")
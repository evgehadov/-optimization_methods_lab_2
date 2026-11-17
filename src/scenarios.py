from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data_io import load_times, to_array
from greedy import greedy_assignment
from model import build_model
from solve import LOG_PATH, solve, write_log
from verify import verify_dual, verify_primal

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "results" / "scenarios.csv"
DETAILS_PATH = ROOT / "results" / "scenario_assignments.csv"

GROWTH = 1.2


def pairs_of(x: np.ndarray) -> list[tuple[int, int]]:
    return [(i, int(np.argmax(x[i]))) for i in range(x.shape[0])]


def pairs_text(pairs, rows, cols) -> str:
    return "; ".join(f"{rows[i]}->{cols[j]}" for i, j in pairs)


def find_critical_pairs(times: np.ndarray, base_pairs) -> list[tuple[int, tuple[int, int]]]:
    base_obj = solve(build_model(times)).objective
    damage = []
    for pair in base_pairs:
        r = solve(build_model(times, forbidden=[pair]))
        loss = (r.objective - base_obj) if r.objective is not None else float("inf")
        damage.append((loss, pair))
    return sorted(damage, key=lambda d: -d[0])


def run_scenario(name, description, times, forbidden, rows, cols, base_obj, base_pairs):
    model = build_model(times, forbidden=forbidden)
    result = solve(model)
    write_log(result, name)
    greedy = greedy_assignment(times, forbidden=forbidden)

    row = {
        "сценарий": name,
        "описание": description,
        "статус": result.status_text.split(":")[0],
        "время_ч": round(result.objective, 6) if result.objective is not None else None,
        "изменение_к_базе_ч": None,
        "жадное_ч": round(greedy["objective"], 6) if greedy["objective"] is not None else None,
        "жадное_статус": "построено" if greedy["complete"] else "не удалось назначить всех",
        "разрыв_жадного_ч": None,
        "проверка_допустимости": None,
        "проверка_оптимальности": None,
        "изменённые_назначения": "",
    }
    details = []
    if result.status == 0:
        row["изменение_к_базе_ч"] = round(result.objective - base_obj, 6)
        p = verify_primal(result.x, times, result.objective, forbidden)
        d = verify_dual(result.duals, times, result.objective, forbidden)
        row["проверка_допустимости"] = p["all_ok"]
        row["проверка_оптимальности"] = d["all_ok"]
        pairs = pairs_of(result.x)
        changed = [pr for pr in pairs if pr not in base_pairs]
        row["изменённые_назначения"] = pairs_text(changed, rows, cols) or "нет"
        if greedy["objective"] is not None:
            row["разрыв_жадного_ч"] = round(greedy["objective"] - result.objective, 6)
        for i, j in pairs:
            details.append({"сценарий": name, "бригада": rows[i], "объект": cols[j],
                            "время_ч": times[i, j]})
    else:
        row["изменённые_назначения"] = "решения нет"
    return row, details


def main() -> None:
    df = load_times()
    rows, cols = list(df.index), list(df.columns)
    times = to_array(df)
    n = times.shape[0]

    LOG_PATH.unlink(missing_ok=True)

    base = solve(build_model(times))
    base_obj = base.objective
    base_pairs = pairs_of(base.x)

    critical = find_critical_pairs(times, base_pairs)
    worst_pair = critical[0][1]
    top2 = [critical[0][1], critical[1][1]]

    print("Ущерб от закрытия каждой назначенной пары:")
    for loss, (i, j) in critical:
        print(f"  {rows[i]}->{cols[j]}: +{loss:.0f} ч")

    blocked_site = base_pairs[0][1]
    closed_column = [(i, blocked_site) for i in range(n)]

    scenarios = [
        ("базовый", "исходные данные", times, []),
        ("дефицитный_1", f"закрыта самая критичная пара {rows[worst_pair[0]]}->{cols[worst_pair[1]]}",
         times, [worst_pair]),
        ("дефицитный_2", "время работ выросло на 20% на всех парах", times * GROWTH, []),
        ("стрессовый", "закрыты две самые критичные пары и время выросло на 20%",
         times * GROWTH, top2),
        ("несовместный", f"{cols[blocked_site]} недоступен для всех бригад",
         times, closed_column),
    ]

    summary, all_details = [], []
    for name, desc, t, forb in scenarios:
        row, details = run_scenario(name, desc, t, forb, rows, cols, base_obj, base_pairs)
        summary.append(row)
        all_details += details

    out = pd.DataFrame(summary)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8")
    pd.DataFrame(all_details).to_csv(DETAILS_PATH, index=False, encoding="utf-8")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print()
    print(out[["сценарий", "статус", "время_ч", "изменение_к_базе_ч",
               "жадное_ч", "жадное_статус", "разрыв_жадного_ч"]].to_string(index=False))
    print()
    print("Проверки (допустимость / оптимальность):")
    for _, r in out.iterrows():
        print(f"  {r['сценарий']}: {r['проверка_допустимости']} / {r['проверка_оптимальности']}")
    print(f"\nСохранено: {OUT_PATH}")


if __name__ == "__main__":
    main()
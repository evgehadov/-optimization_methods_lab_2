from __future__ import annotations

from pathlib import Path

import animation
import formulations
import generate_data
import integer_version
import plots
import scenarios
from data_io import load_times, to_array
from greedy import compare_with_optimum, greedy_assignment
from model import build_model
from solve import assignment_table, solve
from verify import verify_dual, verify_primal

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def report_lines(title: str, report: dict) -> list[str]:
    lines = [title]
    for name, ok in report["checks"].items():
        lines.append(f"  [{'OK' if ok else 'ОШИБКА'}] {name}")
    return lines


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("1. Генерация данных (seed 10007)")
    generate_data.main()

    print("\n2. Базовое решение")
    times_df = load_times()
    times = to_array(times_df)
    result = solve(build_model(times))
    print("Статус:", result.status_text)
    if result.status != 0:
        raise SystemExit("Базовая задача не решена, дальше считать нет смысла")

    table = assignment_table(result, times_df)
    table.to_csv(RESULTS / "assignment.csv", index=False, encoding="utf-8")
    print(table.to_string(index=False))
    print("Суммарное время, ч:", result.objective)

    print("\n3. Независимая проверка")
    primal = verify_primal(result.x, times, result.objective)
    dual = verify_dual(result.duals, times, result.objective)
    lines = report_lines("Проверка допустимости:", primal)
    lines.append(f"  Пересчитанная цель, ч: {primal['recomputed_objective']}")
    lines += report_lines("Проверка оптимальности:", dual)
    lines.append(f"  Двойственная цель, ч: {dual['dual_objective']}")
    text = "\n".join(lines)
    print(text)
    (RESULTS / "verification.txt").write_text(text + "\n", encoding="utf-8")
    if not (primal["all_ok"] and dual["all_ok"]):
        raise SystemExit("Проверка не пройдена")

    print("\n4. Жадная эвристика")
    greedy = greedy_assignment(times)
    cmp_ = compare_with_optimum(greedy["objective"], result.objective)
    print(f"Жадное: {greedy['objective']} ч, оптимум: {result.objective} ч, "
          f"разрыв: {cmp_['gap_hours']:.0f} ч")

    print("\n5. Сценарии")
    scenarios.main()

    print("\n6. Схемы")
    plots.main()

    print("\n7. Дополнительные задания")
    print("\n7.1. Целочисленная версия (LP и MILP)")
    integer_version.main()
    print("\n7.2. Две формулировки (узел-дуга и прямая)")
    formulations.main()
    print("\n7.3. Анимация роста нагрузки")
    animation.main()

    print("\nГотово. Все файлы лежат в папке results/")


if __name__ == "__main__":
    main()
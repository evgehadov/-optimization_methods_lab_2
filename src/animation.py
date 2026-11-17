from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # рисуем в файл, окно не нужно
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from data_io import load_times, to_array
from model import build_model
from scenarios import pairs_of, pairs_text
from solve import solve

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

BLUE = "#1f5fbf"
RED = "#c0392b"

# во сколько раз вырастает время на парах базового плана: от x1.0 до x2.0
FACTORS = np.round(np.arange(1.0, 2.01, 0.1), 2)


def grow_base_pairs(times: np.ndarray, base_pairs, k: float) -> np.ndarray:
    """Время на парах базового плана растёт в k раз (перегрузка этих пар)."""
    grown = times.copy()
    for i, j in base_pairs:
        grown[i, j] *= k
    return grown


def compute_steps(times: np.ndarray, base_pairs) -> list[dict]:
    steps, prev = [], None
    for k in FACTORS:
        t = grow_base_pairs(times, base_pairs, k)
        result = solve(build_model(t))
        pairs = pairs_of(result.x)
        steps.append({
            "k": float(k),
            "times": t,
            "pairs": pairs,
            "prev_pairs": prev,
            "total": result.objective,
            "rebuilt": prev is not None and pairs != prev,
        })
        prev = pairs
    return steps


def make_positions(n: int):
    pos, labels = {}, {}
    for i in range(n):
        pos[("c", i)], pos[("s", i)] = (0, -i), (1, -i)
    return pos, labels


def draw_frame(ax, step: dict, rows, cols) -> None:
    ax.clear()
    n = len(rows)
    pos, labels = make_positions(n)
    g = nx.DiGraph()
    for i in range(n):
        g.add_node(("c", i))
        g.add_node(("s", i))
        labels[("c", i)] = rows[i]
        labels[("s", i)] = cols[i]

    edges, colors, edge_labels = [], [], {}
    prev = step["prev_pairs"]
    for i, j in step["pairs"]:
        e = (("c", i), ("s", j))
        g.add_edge(*e)
        edges.append(e)
        is_new = prev is not None and (i, j) not in prev
        colors.append(RED if is_new else BLUE)
        edge_labels[e] = f"{step['times'][i, j]:.1f} ч"

    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=2000, node_color="#eef3fb",
                           edgecolors="#555555")
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=7)
    nx.draw_networkx_edges(g, pos, edgelist=edges, edge_color=colors, width=2,
                           arrows=True, arrowsize=12, node_size=2000, ax=ax)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, label_pos=0.14,
                                 rotate=False, font_size=8, ax=ax,
                                 bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
    note = "\nплан перестроен (красным новые назначения)" if step["rebuilt"] else ""
    ax.set_title(f"Время на парах базового плана умножено на {step['k']:.1f}\n"
                 f"Суммарное время: {step['total']:.1f} ч{note}", fontsize=11)
    ax.set_xlim(-0.35, 1.35)
    ax.axis("off")


def save_curve(steps: list[dict], path: Path) -> None:
    ks = [s["k"] for s in steps]
    totals = [s["total"] for s in steps]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, totals, marker="o", color=BLUE)
    for s in steps:
        if s["rebuilt"]:
            ax.axvline(s["k"], color=RED, linestyle="--", linewidth=1)
            ax.annotate("перестройка плана", (s["k"], s["total"]),
                        textcoords="offset points", xytext=(6, -16), color=RED, fontsize=9)
    ax.set_xlabel("Во сколько раз выросло время на парах базового плана")
    ax.set_ylabel("Суммарное время, ч")
    ax.set_title("Как растёт суммарное время при перегрузке базовых назначений")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    df = load_times()
    rows, cols = list(df.index), list(df.columns)
    times = to_array(df)

    base_pairs = pairs_of(solve(build_model(times)).x)
    steps = compute_steps(times, base_pairs)

    table = pd.DataFrame([{
        "рост_времени_на_базовых_парах": s["k"],
        "суммарное_время_ч": round(s["total"], 6),
        "план_перестроен": s["rebuilt"],
        "назначения": pairs_text(s["pairs"], rows, cols),
    } for s in steps])
    RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS / "growth_steps.csv", index=False, encoding="utf-8")
    print(table[["рост_времени_на_базовых_парах", "суммарное_время_ч",
                 "план_перестроен"]].to_string(index=False))

    save_curve(steps, RESULTS / "growth_curve.png")

    fig, ax = plt.subplots(figsize=(8, 6.5))
    anim = FuncAnimation(fig, lambda f: draw_frame(ax, steps[f], rows, cols),
                         frames=len(steps), interval=1200)
    anim.save(RESULTS / "growth_animation.gif", writer=PillowWriter(fps=1), dpi=90)
    plt.close(fig)
    print("Сохранено: growth_steps.csv, growth_curve.png, growth_animation.gif")


if __name__ == "__main__":
    main()
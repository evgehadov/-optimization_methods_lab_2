from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from data_io import load_times, to_array
from model import build_model
from scenarios import GROWTH, find_critical_pairs, pairs_of
from solve import solve

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

BLUE = "#1f5fbf"
RED = "#c0392b"


def draw_assignment(times, rows, cols, pairs, title, path, base_pairs=None):
    n = len(rows)
    g = nx.DiGraph()
    pos, labels = {}, {}
    for i in range(n):
        g.add_node(("c", i))
        pos[("c", i)] = (0, -i)
        labels[("c", i)] = rows[i]
        g.add_node(("s", i))
        pos[("s", i)] = (1, -i)
        labels[("s", i)] = cols[i]

    edges, colors, edge_labels = [], [], {}
    for i, j in pairs:
        e = (("c", i), ("s", j))
        g.add_edge(*e)
        edges.append(e)
        is_new = base_pairs is not None and (i, j) not in base_pairs
        colors.append(RED if is_new else BLUE)
        edge_labels[e] = f"{times[i, j]:.1f} ч"

    fig, ax = plt.subplots(figsize=(10, 6.5))
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=3400, node_color="#eef3fb",
                           edgecolors="#555555")
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=9)
    nx.draw_networkx_edges(g, pos, edgelist=edges, edge_color=colors, width=2,
                           arrows=True, arrowsize=14, node_size=3400, ax=ax)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, label_pos=0.14, rotate=False,
                                 font_size=9, ax=ax,
                                 bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    total = sum(times[i, j] for i, j in pairs)
    ax.set_title(f"{title}\nСуммарное время: {total:.1f} ч", fontsize=12)
    ax.set_xlim(-0.35, 1.35)
    ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("Сохранено:", path)


def main() -> None:
    df = load_times()
    rows, cols = list(df.index), list(df.columns)
    times = to_array(df)

    base = solve(build_model(times))
    base_pairs = pairs_of(base.x)
    draw_assignment(times, rows, cols, base_pairs, "Базовый сценарий",
                    RESULTS / "assignment_base.png")

    critical = find_critical_pairs(times, base_pairs)
    worst = critical[0][1]
    r1 = solve(build_model(times, forbidden=[worst]))
    draw_assignment(times, rows, cols, pairs_of(r1.x),
                    f"Дефицитный сценарий: закрыта пара {rows[worst[0]]} - {cols[worst[1]]}\n"
                    "(красным показаны новые назначения)",
                    RESULTS / "assignment_deficit.png", base_pairs)

    top2 = [critical[0][1], critical[1][1]]
    t_stress = times * GROWTH
    r2 = solve(build_model(t_stress, forbidden=top2))
    draw_assignment(t_stress, rows, cols, pairs_of(r2.x),
                    "Стрессовый сценарий: две пары закрыты, время +20%\n"
                    "(красным показаны новые назначения)",
                    RESULTS / "assignment_stress.png", base_pairs)


if __name__ == "__main__":
    main()
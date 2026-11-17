from dataclasses import dataclass

import numpy as np


@dataclass
class AssignmentModel:
    c: np.ndarray
    A_eq: np.ndarray
    b_eq: np.ndarray
    bounds: list
    n: int


def build_model(times: np.ndarray, forbidden=None) -> AssignmentModel:
    n = times.shape[0]
    c = times.flatten()

    A_eq = np.zeros((2 * n, n * n))
    for i in range(n):
        A_eq[i, i * n:(i + 1) * n] = 1
    for j in range(n):
        A_eq[n + j, j::n] = 1
    b_eq = np.ones(2 * n)

    bounds = [(0, 1)] * (n * n)
    for i, j in (forbidden or []):
        bounds[i * n + j] = (0, 0)

    return AssignmentModel(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, n=n)


if __name__ == "__main__":
    from data_io import load_times, to_array

    model = build_model(to_array(load_times()))
    print("Число переменных:", model.c.size)
    print("Форма матрицы ограничений:", model.A_eq.shape)
    print("Единиц в каждом столбце:", set(model.A_eq.sum(axis=0)))
    print("Правые части:", model.b_eq)
import numpy as np


def estimate_transition_matrix(states, n_states=3):
    counts = np.zeros((n_states, n_states), dtype=int)

    for i in range(len(states) - 1):
        counts[states[i], states[i + 1]] += 1

    P = np.zeros((n_states, n_states), dtype=float)

    for i in range(n_states):
        row_sum = counts[i].sum()
        if row_sum > 0:
            P[i] = counts[i] / row_sum

    return counts, P
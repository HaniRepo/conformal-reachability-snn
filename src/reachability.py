import numpy as np


def finite_horizon_reachability(P, target_state=2, horizon=10):
    """
    Compute finite-horizon reachability probabilities:
    p_H[i] = probability of reaching target_state within <= horizon steps
             starting from state i.
    """
    n_states = P.shape[0]

    # Base case: within 0 steps, only target reaches target with prob 1
    p = np.zeros(n_states)
    p[target_state] = 1.0

    for _ in range(horizon):
        new_p = np.zeros(n_states)
        for i in range(n_states):
            if i == target_state:
                new_p[i] = 1.0
            else:
                # reach target now or later
                new_p[i] = P[i, target_state] + np.sum(
                    [P[i, j] * p[j] for j in range(n_states) if j != target_state]
                )
        p = new_p

    return p
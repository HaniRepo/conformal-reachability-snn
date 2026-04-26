import numpy as np
from src.dtmc import estimate_transition_matrix
from src.reachability import finite_horizon_reachability


def sample_random_subsequences(states, subseq_length=300, n_samples=30, random_seed=42):
    rng = np.random.default_rng(random_seed)
    subsequences = []

    max_start = len(states) - subseq_length
    if max_start <= 0:
        raise ValueError("Subsequence length is too large for the given state sequence.")

    for _ in range(n_samples):
        start = rng.integers(0, max_start + 1)
        subseq = states[start:start + subseq_length]
        subsequences.append(subseq)

    return subsequences


def compute_subsequence_reachabilities(subsequences, target_state=2, horizon=10, n_states=3):
    reachabilities = []

    for subseq in subsequences:
        if len(subseq) < 2:
            continue

        _, P_sub = estimate_transition_matrix(subseq, n_states=n_states)
        p_sub = finite_horizon_reachability(P_sub, target_state=target_state, horizon=horizon)
        reachabilities.append(p_sub)

    return np.array(reachabilities)
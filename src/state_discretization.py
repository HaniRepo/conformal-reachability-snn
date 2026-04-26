import numpy as np

'''
def discretize_states(s_t):
    """
    Convert anomaly score into 3 states using percentiles
    """
    q1 = np.percentile(s_t, 33)
    q2 = np.percentile(s_t, 66)

    states = np.zeros(len(s_t), dtype=int)

    states[s_t > q1] = 1
    states[s_t > q2] = 2

    return states, q1, q2

'''

def discretize_states(s_t):
    q1 = 1.0
    q2 = 2.0

    states = np.zeros(len(s_t), dtype=int)
    states[s_t >= q1] = 1
    states[s_t >= q2] = 2

    return states, q1, q2
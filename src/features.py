import numpy as np


def compute_rms(windows):
    """
    Root Mean Square (energy of signal)
    """
    return np.sqrt(np.mean(windows**2, axis=1))


def compute_variance(windows):
    return np.var(windows, axis=1)


def compute_anomaly_score(windows):
    """
    Simple anomaly score (you can change later)
    """
    rms = compute_rms(windows)
    return rms
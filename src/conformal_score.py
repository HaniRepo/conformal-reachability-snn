import numpy as np


def split_conformal_score_interval(y_cal, score_cal, alpha=0.1):
    """
    Multi-class severity score calibration.
    Residual = |true_label - predicted_score|
    Returns scalar radius qhat.
    """
    residuals = np.abs(y_cal - score_cal)
    n = len(residuals)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    qhat = np.quantile(residuals, q_level, method="higher")
    return qhat


def apply_score_interval(scores, qhat, min_score=0.0, max_score=3.0):
    lower = np.clip(scores - qhat, min_score, max_score)
    upper = np.clip(scores + qhat, min_score, max_score)
    return lower, upper
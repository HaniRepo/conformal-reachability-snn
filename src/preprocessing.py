# src/preprocessing.py

import numpy as np

def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)


def combine_signals(normal_signal, fault_signal):
    N = min(len(normal_signal), len(fault_signal))
    normal_signal = normal_signal[:N]
    fault_signal = fault_signal[:N]

    return np.concatenate([normal_signal, fault_signal])


def create_windows(signal, window_size=200, stride=100):
    windows = []
    for i in range(0, len(signal) - window_size, stride):
        windows.append(signal[i:i+window_size])
    return np.array(windows)


def create_labels(num_windows):
    labels = np.zeros(num_windows)
    labels[num_windows // 2:] = 1
    return labels
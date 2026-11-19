import numpy as np
from sklearn.model_selection import train_test_split

def standardize(X, mean=None, std=None):
    X = np.asarray(X, dtype=float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0) + 1e-12
    return (X - mean) / std, mean, std

def train_val_split(X, TH, val_frac=0.2, seed=0):
    return train_test_split(X, TH, test_size=val_frac, random_state=seed)

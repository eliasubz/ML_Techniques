import numpy as np
import pandas as pd


def euclidean_distance(X, instance):
    """Compute Euclidean distance of one instance to all in X (assume no missing, treat all columns as numeric)."""
    X_np = X.values.astype(float)
    instance_np = instance.values.astype(float)

    dists = np.sqrt(np.sum((X_np - instance_np) ** 2, axis=1))
    return pd.DataFrame({"Index": np.arange(X.shape[0]), "Distance": dists})


def cosine_distance(X, instance):
    """Compute cosine distance (1 - sim) of one instance to all in X."""

    X_np = X.values.astype(float)
    x = instance.values.astype(float)
    x_norm = np.linalg.norm(x)
    X_norms = np.linalg.norm(X_np, axis=1)

    x_norm = x_norm if x_norm != 0 else 1.0
    X_norms[X_norms == 0] = 1.0

    sims = np.dot(X_np, x) / (X_norms * x_norm)
    dists = 1.0 - sims
    return pd.DataFrame({"Index": np.arange(X.shape[0]), "Distance": dists})

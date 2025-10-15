import numpy as np
import pandas as pd


def euclidean_distance(X: np.ndarray, x: np.ndarray):
    """Compute Euclidean distance of one instance to all in X (assume no missing, treat all columns as numeric)."""
    diff = X - x
    return np.sqrt(np.einsum('ij,ij->i', diff, diff))


def cosine_distance(X, x):
    """Compute cosine distance (1 - sim) of one instance to all in X."""

    # X_np = X.values.astype(float)
    # x = instance.values.astype(float)
    # x_norm = np.linalg.norm(x)
    # X_norms = np.linalg.norm(X_np, axis=1)

    # x_norm = x_norm if x_norm != 0 else 1.0
    # X_norms[X_norms == 0] = 1.0

    # sims = np.dot(X_np, x) / (X_norms * x_norm)
    # dists = 1.0 - sims
    # return pd.DataFrame({"Index": np.arange(X.shape[0]), "Distance": dists})

    X = np.asarray(X, dtype=float)
    x = np.asarray(x, dtype=float)

    X_norms = np.linalg.norm(X, axis=1) + 1e-12
    x_norm = np.linalg.norm(x) + 1e-12

    sims = (X @ x) / (X_norms * x_norm)
    sims = np.clip(sims, -1.0, 1.0)
    return 1.0 - sims


def heom_distance(X, x, types):
    """
    HEOM distance from one query `instance` to all rows in X, vectorized.
    """
    # # Convert once
    # X_np = X.values
    # x_np = instance.values

    # types_arr = np.asarray(types)

    # if types_arr.shape[0] != X_np.shape[1]:
    #     raise ValueError(f"'types' length ({types_arr.shape[0]}) != n_features ({X_np.shape[1]})") # warning (useful for when OHE + HEOM which is not compatible)

    # num_mask = (types_arr == "numeric")
    # cat_mask = ~num_mask

    # # If numerical, take squared distance
    # d_num2 = 0.0
    # if num_mask.any():
    #     X_num = X_np[:, num_mask].astype(float, copy=False)
    #     x_num = x_np[num_mask].astype(float, copy=False)
    #     d_num2 = np.sum((X_num - x_num) ** 2, axis=1)

    # # If categorical: 0 if equal, 1 if different
    # d_cat = 0.0
    # if cat_mask.any():
    #     X_cat = X_np[:, cat_mask]
    #     x_cat = x_np[cat_mask]

    #     mismatches = (X_cat != x_cat)
    #     d_cat = mismatches.sum(axis=1).astype(float)

    # dists = np.sqrt(d_num2 + d_cat)
    # return pd.DataFrame({"Index": np.arange(X.shape[0]), "Distance": dists})

    X = np.asarray(X)
    x = np.asarray(x)
    types_arr = np.asarray(types)
    if types_arr.shape[0] != X.shape[1]:
        raise ValueError(
            f"'types' length ({types_arr.shape[0]}) != n_features ({X.shape[1]})")

    is_num = (types_arr == "numeric")
    is_cat = ~is_num

    num_contrib = 0.0
    if np.any(is_num):
        Xn = X[:, is_num].astype(float, copy=False)
        xn = x[is_num].astype(float, copy=False)
        num_contrib = np.sum((Xn - xn) ** 2, axis=1)

    cat_contrib = 0.0
    if np.any(is_cat):
        Xc = X[:, is_cat]
        xc = x[is_cat]
        mismatches = (Xc != xc)

        cat_contrib = mismatches.sum(axis=1).astype(float)

    return np.sqrt(num_contrib + cat_contrib)

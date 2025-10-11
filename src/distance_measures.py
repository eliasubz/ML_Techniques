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

def heom_distance(X, instance, types):
    """
    HEOM distance from one query `instance` to all rows in X, vectorized.
    """
    # Convert once
    X_np = X.values
    x_np = instance.values

    types_arr = np.asarray(types)

    if types_arr.shape[0] != X_np.shape[1]:
        raise ValueError(f"'types' length ({types_arr.shape[0]}) != n_features ({X_np.shape[1]})") # warning (useful for when OHE + HEOM which is not compatible)

    num_mask = (types_arr == "numeric")
    cat_mask = ~num_mask 

    # If numerical, take squared distance
    d_num2 = 0.0
    if num_mask.any():
        X_num = X_np[:, num_mask].astype(float, copy=False)
        x_num = x_np[num_mask].astype(float, copy=False)
        d_num2 = np.sum((X_num - x_num) ** 2, axis=1)

    # If categorical: 0 if equal, 1 if different
    d_cat = 0.0
    if cat_mask.any():
        X_cat = X_np[:, cat_mask]
        x_cat = x_np[cat_mask]

        mismatches = (X_cat != x_cat) 
        d_cat = mismatches.sum(axis=1).astype(float)

    dists = np.sqrt(d_num2 + d_cat)
    return pd.DataFrame({"Index": np.arange(X.shape[0]), "Distance": dists})

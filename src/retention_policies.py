import pandas as pd
from collections import Counter

def retention_polcies(retention_policy, train_matrix, instance, k_nearest, pred, y):
    if retention_policy == "never_retain":
        pass
    elif retention_policy == "always_retain":
        train_matrix = pd.concat([train_matrix, instance.to_frame().T])
    elif retention_policy == "different_class_retention":
        if instance.iloc[-1] != pred:
            train_matrix = pd.concat([train_matrix, instance.to_frame().T])
    elif retention_policy == "DD_retention":
        neighbor_labels = y.loc[k_nearest["Index"]].tolist()

        # Compute degree of disagreement (DD)
        vote_counts = Counter(neighbor_labels)
        total_votes = sum(vote_counts.values())
        proportions = [count / total_votes for count in vote_counts.values()]
        DD = 1 - max(proportions)

        # Retain if disagreement above threshold
        dd_threshold = 0.4  # you can tune this value
        if DD >= dd_threshold:
            train_matrix = pd.concat([train_matrix, instance.to_frame().T])
    else:
        raise ValueError(
            f"Unknown retention policy: {retention_policy}")
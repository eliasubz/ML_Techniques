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

        # Count occurrences of each class
        vote_counts = Counter(neighbor_labels)

        # Total number of distinct classes among neighbors
        num_classes = len(vote_counts)

        # Identify majority class and its count
        majority_class, majority_cases = vote_counts.most_common(1)[0]

        # Remaining cases = total cases - majority class cases
        remaining_cases = sum(vote_counts.values()) - majority_cases

        # Avoid division by zero
        if num_classes > 1 and majority_cases > 0:
            d = remaining_cases / ((num_classes - 1) * majority_cases)
        else:
            d = 0  # or some defined fallback

        # Retain instance if d >= threshold
        d_threshold = 0.4  # you can tune this
        if d >= d_threshold:
            train_matrix = pd.concat([train_matrix, instance.to_frame().T])
    else:
        raise ValueError(
            f"Unknown retention policy: {retention_policy}")
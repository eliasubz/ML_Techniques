import numpy as np
from collections import Counter
from processing_types import RetentionPolicy


def retention_policies(retention_policy: RetentionPolicy, instance_class: np.float64, pred: int, k_nearest_labels: list) -> bool:
    match retention_policy:
        case RetentionPolicy.NEVER_RETAIN:
            return False
        case RetentionPolicy.ALWAYS_RETAIN:
            return True
        case RetentionPolicy.DIFFERENT_CLASS_RETENTION:
            if instance_class != pred:
                return True
        case RetentionPolicy.DD_RETENTION:
            # Count occurrences of each class
            vote_counts = Counter(k_nearest_labels)

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
                return True
            else:
                return False
        case _:
            raise ValueError(
                f"Unknown retention policy: {retention_policy}")

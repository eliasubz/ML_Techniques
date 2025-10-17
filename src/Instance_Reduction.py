# Here we have the IR algorithms
import pandas as pd
import random
from Parser import Parser
from processing_types import (
    NormalizationStrategy,
    EncodingStrategy,
    MissingValuesNumericStrategy,
    MissingValuesCategoricalStrategy,
)
from distance_measures import cosine_distance, euclidean_distance, heom_distance
import time
import numpy as np





def condensed_nearest_neighbor(D_matrix: pd.DataFrame, distance_metric="euclidean", types=None):

    X = D_matrix.iloc[:, :-1].to_numpy()
    print(X.shape)
    y = D_matrix.iloc[:, -1].to_numpy()

    n = len(X)
    all_indices = list(range(n))

    # Step b) Choose one random point as seed
    idx = np.random.choice(all_indices)
    E_idx = [idx]
    D_idx = [i for i in all_indices if i != idx]

    pass_num = 1
    while True:
        print(f"Pass {pass_num}")
        D_next = []
        count = 0

        # Shuffle for random selection
        np.random.shuffle(D_idx)

        for i in D_idx:
            print(X.size)
            x = X[i]

            # 1-NN classification using current E
            if distance_metric == "euclidean":
                distances = euclidean_distance(X[E_idx], x)
            elif distance_metric == "cosine":
                distances = cosine_distance(X[E_idx], x)
            elif distance_metric == "heom":
                distances = heom_distance(
                    x, X[E_idx], types)
            else:
                raise ValueError(f"Unknown metric: {distance_metric}")


            print(distances)
            nn_index = E_idx[np.argmin(distances)]
            y_pred = y[nn_index]

            # Compare prediction to true label
            if y_pred == y[i]:
                D_next.append(i)  # correctly classified
            else:
                E_idx.append(i)  # misclassified → add to E
                count += 1

        if count == 0:
            print("No misclassifications → done.")
            break

        # Prepare for next pass
        D_idx = D_next
        pass_num += 1

    print("Instances before: ", n)
    print("Instances after: ", len(E_idx))

    # Return the reduced dataset
    return X[E_idx], y[E_idx], E_idx


def mcnn(D_matrix: pd.DataFrame, distance_metric="euclidean", types=None, max_passes=10):
    """
    Modified Condensed Nearest Neighbor (MCNN)
    Based on Devi & Murty (2002)
    https://campusvirtual.ub.edu/pluginfile.php/9896051/mod_resource/content/2/MCNNDevi02.pdf

    Args:
        D_matrix (pd.DataFrame): DataFrame with features + class label (last column)
        distance_metric (str): 'euclidean' | 'cosine'
        max_passes (int): maximum number of passes allowed
        types (optional): used for mixed-type data (ignored here)
    Returns:
        (X_reduced, y_reduced, E_idx): reduced dataset and selected indices
    """
    X = D_matrix.iloc[:, :-1].to_numpy()
    y = D_matrix.iloc[:, -1].to_numpy()

    n = len(X)
    classes = np.unique(y)

    # Initialize E with one random prototype from each class
    E_idx = []
    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        idx = np.random.choice(cls_indices)
        E_idx.append(idx)

    D_idx = [i for i in range(n) if i not in E_idx]

    print(f"Initial prototypes: {len(E_idx)} ({E_idx})")

    for pass_num in range(1, max_passes + 1):
        print(f"\n=== Pass {pass_num} ===")
        np.random.shuffle(D_idx)
        misclassified = []

        for i in D_idx:
            x = X[i]

            # Compute distances to current prototypes
            if distance_metric == "euclidean":
                distances = euclidean_distance(X[E_idx], x)
            elif distance_metric == "cosine":
                distances = cosine_distance(X[E_idx], x)
            else:
                raise ValueError(f"Unknown metric: {distance_metric}")

            nn_index = E_idx[np.argmin(distances)]
            y_pred = y[nn_index]

            # If misclassified, add to prototype set
            if y_pred != y[i]:
                E_idx.append(i)
                misclassified.append(i)

        print(f"Added {len(misclassified)} new prototypes.")

        # If no misclassifications, stop early
        if len(misclassified) == 0:
            print("All correctly classified → stopping.")
            break

        # Update D_idx (remove newly added prototypes)
        D_idx = [i for i in D_idx if i not in misclassified]

    print("\n--- Final Summary ---")
    print(f"Instances before: {n}")
    print(f"Instances after: {len(E_idx)} ({round(len(E_idx)/n*100, 2)}% retained)")

    return X[E_idx], y[E_idx], E_idx

# Edited Nearest Neighbor 

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from collections import Counter



def edited_nearest_neighbor(D_matrix: pd.DataFrame, k=3, distance_metric="euclidean", types=None):
    """
    Edited Nearest Neighbor (ENN)
    Removes instances that disagree with the majority of their k nearest neighbors.
    
    Args:
        D_matrix (pd.DataFrame): dataset with features + class label (last column)
        k (int): number of nearest neighbors
        distance_metric (str): 'euclidean' | 'cosine'
        types (optional): kept for compatibility
    Returns:
        (X_reduced, y_reduced, kept_indices): reduced dataset and indices kept
    """
    X = D_matrix.iloc[:, :-1].to_numpy()
    y = D_matrix.iloc[:, -1].to_numpy()
    n = len(X)

    kept_indices = []

    for i in range(n):
        x = X[i]
        label = y[i]

        # Compute distances to all others (exclude itself)
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        if distance_metric == "euclidean":
            distances = euclidean_distance(X[mask], x)
        elif distance_metric == "cosine":
            distances = cosine_distance(X[mask], x)
        else:
            raise ValueError(f"Unknown metric: {distance_metric}")

        # Find indices of k nearest neighbors
        nn_indices = np.argsort(distances)[:k]
        neighbor_labels = y[mask][nn_indices]

        # Majority vote
        majority_label = Counter(neighbor_labels).most_common(1)[0][0]

        # Keep if label agrees with majority
        if label == majority_label:
            kept_indices.append(i)

    print(f"Instances before: {n}")
    print(f"Instances after: {len(kept_indices)} ({round(len(kept_indices)/n*100, 2)}% retained)")

    return X[kept_indices], y[kept_indices], kept_indices

# def renn(D_matrix: pd.DataFrame, k=3, distance_metric="euclidean", max_iter=10):
#     """
#     Repeated Edited Nearest Neighbor (RENN)
#     Repeatedly applies ENN until no more instances are removed.
#     """
#     X = D_matrix.iloc[:, :-1].to_numpy()
#     y = D_matrix.iloc[:, -1].to_numpy()

#     iteration = 1
#     while iteration <= max_iter:
#         X_new, y_new, kept_indices = edited_nearest_neighbor(X, y, k=k, distance_metric=distance_metric)
#         removed = len(X) - len(X_new)
#         print(f"RENN Iteration {iteration}: removed {removed} instances")

#         if removed == 0:
#             print("No more instances removed → stopping.")
#             break

#         X, y = X_new, y_new
#         iteration += 1

#     print(f"Instances before: {len(D_matrix)}")
#     print(f"Instances after RENN: {len(X)} ({round(len(X)/len(D_matrix)*100,2)}% retained)")

#     return X, y, kept_indices



if __name__ == "__main__":
    import time

    now = time.time()
    base_path = "datasetsCBR/datasetsCBR"
    dataset_name = "adult"

    parser = Parser(
        base_path="datasetsCBR/datasetsCBR",
        dataset_name="autos",
        normalization_strategy=NormalizationStrategy.STANDARDIZE,
        encoding_strategy=EncodingStrategy.ONE_HOT_ENCODE,
        missing_values_numeric_strategy=MissingValuesNumericStrategy.MEAN,
        missing_values_categorical_strategy=MissingValuesCategoricalStrategy.MODE,
    )
    then = time.time()
    print("time: ", then - now)

    train_matrix, test_matrix = parser.get_split(0)

    condensed_nearest_neighbor(train_matrix, distance_metric="euclidean")
    mcnn(train_matrix, distance_metric="euclidean")
    edited_nearest_neighbor(train_matrix, distance_metric="cosine")
    renn(train_matrix, distance_metric="cosine")
    renn(train_matrix, distance_metric="euclidean")

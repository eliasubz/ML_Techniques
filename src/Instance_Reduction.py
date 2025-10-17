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


def CNN(D: pd.DataFrame):
    #
    passing_count = 1
    sample_idx = random.randrange(D.size[0])
    sample = D.iloc[sample_idx]
    # Drop sample from D
    D = D.drop(index=sample_idx)
    # Add
    E = pd.DataFrame(sample, columns=D.columns)
    return E


import numpy as np
from sklearn.metrics import pairwise_distances


def condensed_nearest_neighbor(D_matrix: pd.DataFrame):

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
            x = X[i].reshape(1, -1)

            # 1-NN classification using current E
            dist = pairwise_distances(x, X[E_idx])
            print(dist)
            nn_index = E_idx[np.argmin(dist)]
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
    condensed_nearest_neighbor(train_matrix)

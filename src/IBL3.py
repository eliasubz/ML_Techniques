from sklearn.datasets import fetch_openml
from Parser import Parser
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from distance_measures import cosine_distance, euclidean_distance, heom_distance
from preallocated_matrix import PreallocatedMatrix
from processing_types import (
    RetentionPolicy,
    EncodingStrategy,
    MissingValuesCategoricalStrategy,
    MissingValuesNumericStrategy,
    NormalizationStrategy,
)
from retention_policies import retention_policies


class IBL:
    def __init__(self):
        """
        k-Instance Based Learner (k-NN) with:
        - metrics: 'euclidean', 'cosine', 'heom'
        - votes: 'modified_plurality', 'borda'
        - types: (list of 'numeric'/'categorical') when using HEOM.
        """

    def ib2_instance_reduction(self, np_train_matrix: np.ndarray):

        CD_idx = [0]
        X = np_train_matrix[:, :-1]
        y = np_train_matrix[:, -1]

        for i in range(1, len(X)):

            distances = euclidean_distance(X[CD_idx], X[i])

            # Find index of the nearest neighbor within CD
            nearest_idx_in_CD = np.argmin(distances)
            nearest_idx = CD_idx[nearest_idx_in_CD]

            if y[nearest_idx] == y[i]:
                classification = True
            else:
                print(np_train_matrix[i])
                classification = False
                CD_idx.append(i)

        return np_train_matrix[CD_idx, :]

    def fit(self, train_matrix: pd.DataFrame):
        np_train_matrix = train_matrix.reset_index(drop=True).to_numpy()

        np_train_matrix = self.ib2_instance_reduction(np_train_matrix)

        self.X = np_train_matrix[:, :-1]
        self.y = np_train_matrix[:, -1]

    def run(
        self,
        test_matrix: pd.DataFrame,
        k=5,
        metric="euclidean",
        vote="modified_plurality",
        retention_policy="DD_retention",
        types=None,
    ):
        import time

        self.k = int(k)
        self.metric = metric
        self.vote = vote
        self.types = types

        test_arr = test_matrix.to_numpy()
        self.X_test = test_arr[:, :-1].astype(np.float64)
        self.y_test = test_arr[:, -1]

        preallocatedMatrix = PreallocatedMatrix(
            self.X.shape[0] + self.X_test.shape[0], self.X.shape[1]
        )
        preallocatedMatrix.append_matrix(self.X)
        self.X = preallocatedMatrix

        predictions = []
        n_test = self.X_test.shape[0]

        total_start = time.time()

        for i in range(n_test):

            step_start = time.time()

            x_instance = self.X_test[i, :]
            y_instance = self.y_test[i]

            dist_start = time.time()
            if self.metric == "euclidean":
                distances = euclidean_distance(self.X.get_filled(), x_instance)
            elif self.metric == "cosine":
                distances = cosine_distance(self.X.get_filled(), x_instance)
            elif self.metric == "heom":
                distances = heom_distance(self.X.get_filled(), x_instance, types)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            dist_end = time.time()

            sort_start = time.time()
            if self.k >= distances.shape[0]:
                idx_k = np.arange(distances.shape[0])
            else:
                idx_k = np.argpartition(distances, self.k - 1)[: self.k]
            order_k = np.argsort(distances[idx_k], kind="stable")
            idx_k = idx_k[order_k]
            sort_end = time.time()

            vote_start = time.time()
            ordered_neighbor_labels = self.y[idx_k].tolist()

            # Voting (unchanged)
            if self.vote == "modified_plurality":
                pred = self._vote_modified_plurality(ordered_neighbor_labels)
            elif self.vote == "borda":
                pred = self._vote_borda(ordered_neighbor_labels)
            else:
                pred = Counter(ordered_neighbor_labels).most_common(1)[0][0]
            vote_end = time.time()

            predictions.append(pred)

            # Retention
            retention_start = time.time()

            should_retain = retention_policies(
                retention_policy,
                instance_class=y_instance,
                pred=pred,
                k_nearest_labels=ordered_neighbor_labels,
            )

            if should_retain:
                self.X.append_column(x_instance)
                # TODO: optimize y storage as well
                self.y = np.append(self.y, y_instance)

            retention_end = time.time()

            step_end = time.time()

        total_end = time.time()
        print(f"Total time for all instances: {total_end-total_start:.2f}s")
        # print(f"Total time for all instances: {total_end-total_start:.2f}s")
        print("Final training set size:", self.X.get_filled().shape)
        return predictions

    @staticmethod
    def _vote_modified_plurality(labels_in_rank):
        """
        Count votes among current neighbors; if tie, drop the farthest and re-vote.
        Deterministic because labels_in_rank is ordered (closest -> farthest).
        """
        idxs = list(range(len(labels_in_rank)))
        while True:
            vals, counts = np.unique(
                [labels_in_rank[i] for i in idxs], return_counts=True
            )
            m = counts.max()
            winners = [v for v, c in zip(vals, counts) if c == m]
            if len(winners) == 1:
                return winners[0]
            # drop farthest
            idxs.pop(-1)
            if len(idxs) == 1:
                return labels_in_rank[idxs[0]]

    @staticmethod
    def _vote_borda(labels_in_rank):
        """
        Borda count: closest gets k-1 points ... farthest 0.
        Tie-break: class of the closest neighbor among tied totals.
        """
        k = len(labels_in_rank)
        scores = {}
        for r, cls in enumerate(labels_in_rank):  # r=0 is closest
            scores[cls] = scores.get(cls, 0) + (k - 1 - r)
        best = max(scores.values())
        tied = [c for c, s in scores.items() if s == best]
        if len(tied) == 1:
            return tied[0]
        # tie-break: pick the tied class that appears first (closest)
        for cls in labels_in_rank:
            if cls in tied:
                return cls


if __name__ == "__main__":

    # Columns: [feature1, feature2, label]
    data = np.array(
        [
            [1.0, 2.0, 0],
            [1.2, 1.8, 0],
            [4.0, 4.5, 1],
            [3.8, 4.2, 1],
            [0.9, 2.1, 0],
            [3.9, 3.9, 1],
        ]
    )

    # --- Run IB2 instance reduction ---
    ibl = IBL()
    reduced_data = ibl.ib2_instance_reduction(data)

    # --- Print results ---
    print("Original data shape:", data.shape)
    print("Reduced data shape:", reduced_data.shape)
    print("\nReduced dataset:\n", reduced_data)

    parser = Parser(
        base_path="datasetsCBR/datasetsCBR",
        dataset_name="adult",
        normalization_strategy=NormalizationStrategy.MINMAX_SCALING,
        encoding_strategy=EncodingStrategy.LABEL_ENCODE,
        missing_values_numeric_strategy=MissingValuesNumericStrategy.MEDIAN,
        missing_values_categorical_strategy=MissingValuesCategoricalStrategy.MODE,
        # faster_parser=True,
    )

    train_matrix, test_matrix = parser.get_split(0)
    types = parser.get_types()
    # Testing IBL
    ibl = IBL()
    ibl.fit(train_matrix)
    # preds = ibl.run(

    #     test_matrix,
    #     k=5,
    #     metric="cosine",
    #     vote="modified_plurality",
    #     retention_policy=RetentionPolicy.NEVER_RETAIN,
    #     types=types,
    # )

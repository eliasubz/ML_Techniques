from sklearn.datasets import fetch_openml
from Parser import Parser
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from distance_measures import cosine_distance, euclidean_distance, heom_distance
from preallocated_matrix import PreallocatedMatrix
from processing_types import RetentionPolicy, EncodingStrategy, MissingValuesCategoricalStrategy, MissingValuesNumericStrategy, NormalizationStrategy
from retention_policies import retention_policies


class IBL:
    def __init__(self):
        """
        k-Instance Based Learner (k-NN) with:
        - metrics: 'euclidean', 'cosine', 'heom'
        - votes: 'modified_plurality', 'borda'
        - types: (list of 'numeric'/'categorical') when using HEOM.
        """

    def fit(self, train_matrix: pd.DataFrame):
        np_train_matrix = train_matrix.reset_index(drop=True).to_numpy()
        self.X = np_train_matrix[:, :-1]
        self.y = np_train_matrix[:, -1]

    def run(self, test_matrix: pd.DataFrame, k=5, metric="euclidean", vote="modified_plurality", retention_policy="DD_retention", types=None):
        import time
        self.k = int(k)
        self.metric = metric
        self.vote = vote
        self.types = types

        test_arr = test_matrix.to_numpy()
        self.X_test = test_arr[:, :-1].astype(np.float64)
        self.y_test = test_arr[:, -1]

        preallocatedMatrix = PreallocatedMatrix(
            self.X.shape[0] + self.X_test.shape[0], self.X.shape[1])
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
                distances = heom_distance(
                    self.X.get_filled(), x_instance, types)
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
            neighbor_labels = self.y[idx_k].tolist()

            # Voting (unchanged)
            if self.vote == "modified_plurality":
                pred = self._vote_modified_plurality(neighbor_labels)
            elif self.vote == "borda":
                pred = self._vote_borda(neighbor_labels)
            else:
                pred = Counter(neighbor_labels).most_common(1)[0][0]
            vote_end = time.time()

            predictions.append(pred)

            # Retention
            retention_start = time.time()

            should_retain = retention_policies(
                retention_policy, instance_class=y_instance, pred=pred, k_nearest_labels=neighbor_labels)

            if should_retain:
                self.X.append_column(x_instance)
                # TODO: optimize y storage as well
                self.y = np.append(self.y, y_instance)

            retention_end = time.time()

            step_end = time.time()
            # print(f"Instance {i}/{len(test_matrix)}: dist={dist_end-dist_start:.5f}s, sort={sort_end-sort_start:.5f}s, vote={vote_end-vote_start:.5f}s, retention={retention_end-retention_start:.5f}s, total={step_end-step_start:.5f}s")

        total_end = time.time()
        print(f"Total time for all instances: {total_end-total_start:.2f}s")

        # for i, instance in test_matrix.iterrows():
        #     x_instance, y_instance = instance.iloc[:-1], instance.iloc[-1]
        #     step_start = time.time()

        #     # Distance calculation
        #     dist_start = time.time()
        #     if self.metric == "euclidean":
        #         distances = euclidean_distance(X, x_instance)
        #     elif self.metric == "cosine":
        #         distances = cosine_distance(X, x_instance)
        #     elif self.metric == "heom":
        #         distances = heom_distance(X, x_instance, types)
        #     else:
        #         raise ValueError(f"Unknown metric: {self.metric}")
        #     dist_end = time.time()

        #     # Sort by distance and get k nearest
        #     sort_start = time.time()
        #     k_nearest = distances.nsmallest(self.k, "Distance")
        #     sort_end = time.time()

        #     # Voting
        #     vote_start = time.time()
        #     neighbor_labels = y.loc[k_nearest["Index"]].tolist()
        #     # print(X.loc[k_nearest["Index"]], y.loc[k_nearest["Index"]])

        #     if self.vote == "modified_plurality":
        #         pred = self._vote_modified_plurality(neighbor_labels)
        #     elif self.vote == "borda":
        #         pred = self._vote_borda(neighbor_labels)
        #     else:
        #         # basic majority
        #         pred = pd.Series(neighbor_labels).mode().iloc[0]
        #     vote_end = time.time()

        #     predictions.append(pred)

        #     # Retention policy
        #     retention_start = time.time()

        #     retention_polcies(retention_policy, self.train_matrix, instance, k_nearest, pred, y)

        #     retention_end = time.time()

        #     step_end = time.time()
        #     print(f"Instance {i}/{len(test_matrix)}: dist={dist_end-dist_start:.4f}s, sort={sort_end-sort_start:.4f}s, vote={vote_end-vote_start:.4f}s, retention={retention_end-retention_start:.4f}s, total={step_end-step_start:.4f}s")

        # total_end = time.time()
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
            vals, counts = np.unique([labels_in_rank[i]
                                     for i in idxs], return_counts=True)
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
    preds = ibl.run(test_matrix, k=5, metric="cosine", vote="modified_plurality",
                    retention_policy=RetentionPolicy.NEVER_RETAIN, types=types)

    # print(preds)
    # print(test_matrix.iloc[:, -1])

    # # Helpful basic metrics
    # acc = accuracy_score(test_matrix.iloc[:, -1], preds)
    # prec = precision_score(
    #     test_matrix.iloc[:, -1], preds, average='weighted', zero_division=0)
    # rec = recall_score(
    #     test_matrix.iloc[:, -1], preds, average='weighted', zero_division=0)
    # f1 = f1_score(test_matrix.iloc[:, -1], preds,
    #               average='weighted', zero_division=0)

    # # Display results
    # print("Performance Metrics:")
    # print(f"Accuracy:  {acc:.4f}")
    # print(f"Precision: {prec:.4f}")
    # print(f"Recall:    {rec:.4f}")
    # print(f"F1-score:  {f1:.4f}")

    # # Confusion matrix + detailed report
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(test_matrix.iloc[:, -1], preds))

    # print("\nClassification Report:")
    # print(classification_report(
    #     test_matrix.iloc[:, -1], preds, zero_division=0))

    # print("Predictions:", preds)
    # print("Ground truth:", list(test_matrix.iloc[:, -1]))

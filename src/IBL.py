from sklearn.datasets import fetch_openml
from Parser import Parser
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

from preprocessing_types import EncodingStrategy, MissingValuesCategoricalStrategy, MissingValuesNumericStrategy, NormalizationStrategy


class IBL:
    def __init__(self):
        """
        k-Instance Based Learner (k-NN) with:
        - metrics: 'euclidean', 'cosine', 'heom'
        - votes: 'modified_plurality', 'borda'
        - types: (list of 'numeric'/'categorical') when using HEOM.
        """

    def fit(self, train_matrix):
        self.train_matrix = train_matrix.reset_index(drop=True)

    def run(self, test_matrix, k=5, metric="euclidean", vote="modified_plurality", retention_policy="DD_retention", types=None):
        import time
        self.k = int(k)
        self.metric = metric
        self.vote = vote
        self.types = types
        X = self.train_matrix.iloc[:, :-1]
        y = self.train_matrix.iloc[:, -1]
        predictions = []

        total_start = time.time()
        for i, instance in test_matrix.iterrows():
            step_start = time.time()

            # Distance calculation
            dist_start = time.time()
            if self.metric == "euclidean":
                distances = self._euclidean_distance(X, instance)
            elif self.metric == "cosine":
                distances = self._cosine_distance(X, instance)
            elif self.metric == "heom":
                distances = self._heom_distance(X, instance)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")
            dist_end = time.time()

            # Sort by distance and get k nearest
            sort_start = time.time()
            k_nearest = distances.nsmallest(self.k, "Distance")
            sort_end = time.time()

            # Voting
            vote_start = time.time()
            neighbor_labels = y.loc[k_nearest["Index"]].tolist()
            # print(X.loc[k_nearest["Index"]], y.loc[k_nearest["Index"]])

            if self.vote == "modified_plurality":
                pred = self._vote_modified_plurality(neighbor_labels)
            elif self.vote == "borda":
                pred = self._vote_borda(neighbor_labels)
            else:
                # basic majority
                pred = pd.Series(neighbor_labels).mode().iloc[0]
            vote_end = time.time()

            predictions.append(pred)

            # Retention policy
            retention_start = time.time()
            if retention_policy == "never_retain":
                pass
            elif retention_policy == "always_retain":
                self.train_matrix.append(instance)
            elif retention_policy == "different_class_retention":
                if instance.iloc[:, -1] != pred:
                    self.train_matrix.append(instance)
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
                    self.train_matrix = pd.concat([self.train_matrix, instance.to_frame().T])
            else:
                raise ValueError(
                    f"Unknown retention policy: {retention_policy}")
            retention_end = time.time()

            step_end = time.time()
            print(f"Instance {i}: dist={dist_end-dist_start:.4f}s, sort={sort_end-sort_start:.4f}s, vote={vote_end-vote_start:.4f}s, retention={retention_end-retention_start:.4f}s, total={step_end-step_start:.4f}s")

        total_end = time.time()
        print(f"Total time for all instances: {total_end-total_start:.2f}s")
        return predictions

    def _euclidean_distance(self, X, instance):
        """Compute Euclidean distance of one instance to all in X"""
        distances = []
        for index, row in X.iterrows():
            d = 0.0
            for idx, value in enumerate(row.values):
                test_val = instance.iloc[idx]

                # Skip missing values
                if pd.isna(value) or pd.isna(test_val):
                    continue

                # Numeric distance
                if np.issubdtype(type(value), np.number):
                    d += (value - test_val) ** 2
                else:
                    # Categorical mismatch
                    d += 1

            d = np.sqrt(d)
            distances.append((index, d))

        return pd.DataFrame(distances, columns=["Index", "Distance"])

    def _cosine_distance(self, X, instance):
        """Compute cosine distance (1 - sim) of one instance to all in X."""

        distances = []

        x = instance.values.astype(float)
        x_norm = np.linalg.norm(x) or 1.0

        for index, row in X.iterrows():
            a = row.values.astype(float)
            a_norm = np.linalg.norm(a) or 1.0
            sim = float(np.dot(a, x)) / (a_norm * x_norm)

            d = 1.0 - sim
            distances.append((index, d))

        return pd.DataFrame(distances, columns=["Index", "Distance"])

    def _heom_distance(self, X, instance):
        """IMPORTANT: numeric uses squared diff (in [0,1]); categorical uses overlap (0 if equal else 1)."""
        if self.types is None:
            raise ValueError(
                "HEOM requires 'types' aligned to columns (pass at init).")

        distances = []
        x_vals = instance.values
        for index, row in X.iterrows():
            d2 = 0.0
            r_vals = row.values
            for j, t in enumerate(self.types):
                a, b = r_vals[j], x_vals[j]

                # Skip missing values if any slip through
                if pd.isna(a) or pd.isna(b):
                    continue

                if t == "numeric":
                    diff = float(a) - float(b)
                    d2 += diff * diff
                else:  # categorical
                    d2 += 0.0 if a == b else 1.0
            distances.append((index, np.sqrt(d2)))

        return pd.DataFrame(distances, columns=["Index", "Distance"])

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

    base_path = "datasetsCBR/datasetsCBR"
    dataset_name = "adult"

    parser = Parser(
        base_path="datasetsCBR/datasetsCBR",
        dataset_name="adult",
        normalization_strategy=NormalizationStrategy.STANDARDIZE,
        encoding_strategy=EncodingStrategy.ONE_HOT_ENCODE,
        missing_values_numeric_strategy=MissingValuesNumericStrategy.MEAN,
        missing_values_categorical_strategy=MissingValuesCategoricalStrategy.MODE
    )

    train_matrix, test_matrix = parser.get_split(0)

    # Testing IBL

    retention_policy = ""
    ibl = IBL()
    ibl.fit(train_matrix)
    preds = ibl.run(test_matrix)
"""
    titanic = fetch_openml(name="titanic", version=1, as_frame=True)

    X = titanic.get('data')[:1000]
    y = titanic.get('target')[:1000]


    X_test = titanic.get('data')[1100:1110]  # smaller test for demo
    y_test = titanic.get('target')[1100:1110]
    
    # Helpful basic metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)

    # Display results
    print("Performance Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Confusion matrix + detailed report
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    

    print("Predictions:", preds)
    print("Ground truth:", list(y_test))

"""

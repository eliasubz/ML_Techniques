from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np


class IBL:
    def __init__(self,  k=5, metric="euclidean", vote="modified_plurality", types=None):
        """
        k-Instance Based Learner (k-NN) with:
        - metrics: 'euclidean', 'cosine', 'heom'
        - votes: 'modified_plurality', 'borda'
        - types: (list of 'numeric'/'categorical') when using HEOM.
        """
        self.k = int(k)
        self.metric = metric
        self.vote = vote
        self.types = types

    def fit(self, X, y):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)

    def run(self, test_X):
        predictions = []

        for i, instance in test_X.iterrows():
            
            if self.metric == "euclidean":
                distances = self._euclidean_distance(self.X, instance)
            elif self.metric == "cosine":
                distances = self._cosine_distance(self.X, instance)
            elif self.metric == "heom":
                distances = self._heom_distance(self.X, instance)
            else:
                raise ValueError(f"Unknown metric: {self.metric}")

            # Sort by distance and get k nearest
            k_nearest = distances.nsmallest(self.k, "Distance")

            # Majority voting (basic)
            neighbor_labels = self.y.loc[k_nearest["Index"]].tolist()
            print(self.X.loc[k_nearest["Index"]], self.y.loc[k_nearest["Index"]])

            if self.vote == "modified_plurality":
                pred = self._vote_modified_plurality(neighbor_labels)
            elif self.vote == "borda":
                pred = self._vote_borda(neighbor_labels)
            else:
                # basic majority
                pred = pd.Series(neighbor_labels).mode().iloc[0]

            predictions.append(pred)

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
            raise ValueError("HEOM requires 'types' aligned to columns (pass at init).")

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
            vals, counts = np.unique([labels_in_rank[i] for i in idxs], return_counts=True)
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
    # Load Titanic dataset from OpenML
    titanic = fetch_openml(name="titanic", version=1, as_frame=True)

    X = titanic.get('data')[:1000]
    y = titanic.get('target')[:1000]
    X_test = titanic.get('data')[1100:1110]  # smaller test for demo
    y_test = titanic.get('target')[1100:1110]

    # Testing IBL
    ibl = IBL()
    ibl.fit(X, y)
    preds = ibl.run(X_test)

    print("Predictions:", preds)
    print("Ground truth:", list(y_test))

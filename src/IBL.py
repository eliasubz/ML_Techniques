from sklearn.datasets import fetch_openml
import pandas as pd
import numpy as np


class IBL:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)

    def run(self, test_X, k=3, weighing_function='none'):
        predictions = []

        for i, instance in test_X.iterrows():
            distances = self._euclidean_distance(self.X, instance)

            # Sort by distance and get k nearest
            k_nearest = distances.nsmallest(k, "Distance")

            # Majority voting (basic)
            neighbor_labels = self.y.loc[k_nearest["Index"]]
            print(self.X.loc[k_nearest["Index"]], self.y.loc[k_nearest["Index"]])

            if weighing_function == "none":
                pred = neighbor_labels.mode().iloc[0]
            else:
                # Placeholder for weighted voting
                pred = neighbor_labels.mode().iloc[0]

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
    preds = ibl.run(X_test, k=3)

    print("Predictions:", preds)
    print("Ground truth:", list(y_test))

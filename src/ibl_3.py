""""3-Instance Based Learner (k-NN)"""
from collections import Counter
import pandas as pd
import numpy as np
from Parser import Parser
from distance_measures import cosine_distance, euclidean_distance, heom_distance
from preallocated_matrix import PreallocatedMatrix
from processing_types import (
    # RetentionPolicy,
    EncodingStrategy,
    MissingValuesCategoricalStrategy,
    MissingValuesNumericStrategy,
    NormalizationStrategy,
)
from retention_policies import retention_policies


class IBL:
    """3-Instance Based Learner (k-NN) with IB2 instance reduction."""
    def __init__(self):
        """
        k-Instance Based Learner (k-NN) with:
        - metrics: 'euclidean', 'cosine', 'heom'
        - votes: 'modified_plurality', 'borda'
        - types: (list of 'numeric'/'categorical') when using HEOM.
        """

    def ib2_instance_reduction(self, np_train_matrix: np.ndarray):
        """Perform IB2 instance reduction on the training data:
        Only if an instance is missclassified by its nearest neighbor it will be 
        added to the concept description (CD)."""

        cd_idx = [0]
        X = np_train_matrix[:, :-1]
        y = np_train_matrix[:, -1]

        for i in range(1, len(X)):

            distances = euclidean_distance(X[cd_idx], X[i])

            # Find index of the nearest neighbor within CD
            print(distances.shape)
            sorted_nearest_idx_in_cd = np.argsort(distances)
            print(sorted_nearest_idx_in_cd.shape)  
            print("nearest_idx_in_CD:", sorted_nearest_idx_in_cd)
            
            nearest_idx = cd_idx[sorted_nearest_idx_in_cd[0]]

            if y[nearest_idx] != y[i]:
                print(np_train_matrix[i])
                cd_idx.append(i)

        return np_train_matrix[cd_idx, :]
    
    def ib3_instance_reduction(self, np_train_matrix: np.ndarray):

        """ Perform IB3 instance reduction on the training data:
            Important: z-score for confidence intervals is set to 0.674 (75% confidence)
            and we use the normal approximation for the binomial distribution.
        """
        # Concept description
        cd_idx = np.array([0], dtype=int)
        # Classification Record
        record = [{"correct": 0, "false": 0} for _ in range(len(np_train_matrix))]

        X = np_train_matrix[:, :-1]
        y = np_train_matrix[:, -1]
        
        for i in range(1, len(X)):
            # Get the distances sorted by cd_idx
            distances = euclidean_distance(X[cd_idx], X[i])
            # Get the sorted indices of the distances (cd_idx order)
            sorted_nearest_idx_in_cd = np.argsort(distances)
            # nearest_idx = cd_idx[sorted_nearest_idx_in_cd[0]]
            correct_classifications = np.where(y[sorted_nearest_idx_in_cd] == y[i], 1, 0)
            first_correct_pred_idx = np.argmax(correct_classifications)

            # Check if the first correct prediction is indeed correct
            # and get the index of the max prediction
            if y[sorted_nearest_idx_in_cd[first_correct_pred_idx]] == y[i]:
                y_max_idx = first_correct_pred_idx
            else:
                i_rand = np.random.randint(0, len(cd_idx))
                y_max_idx = sorted_nearest_idx_in_cd[i_rand]

            # 
            if y[y_max_idx] == y[i]:
                record[i]["correct"] += 1
            else:
                print(np_train_matrix[i], "\nwas classified incorrectly.")
                record[i]["false"] += 1
                cd_idx = np.append(cd_idx, i) 

            # Getting statistics for the significance tests
            counts_for_class = dict(zip(np.unique(y), [(y[:i] == target).sum() for target in np.unique(y[cd_idx])]))
            print("counts_for_class:", counts_for_class)
            # idxs of instances that ought to be deleted
            dropped_idxs = []

            for y_better_then_first_correct in sorted_nearest_idx_in_cd[:first_correct_pred_idx]:
                if y[y_better_then_first_correct] == y[i]:
                    print("this should not happen")
                    record[y_better_then_first_correct]["correct"] +=1
                else:
                    # Increase false count
                    record[y_better_then_first_correct]["false"] += 1


                    
                    # Compute bounds for classification accuracy and relative class frequency with:
                    # https://www.questionpro.com/blog/confidence-interval-formula/

                    # Class accuracy bounds
                    correct_count = record[y_better_then_first_correct]["correct"]
                    false_count = record[y_better_then_first_correct]["false"]
                    classification_accuracy = correct_count / (correct_count + false_count)
                    accuracy_binary_std = np.sqrt((correct_count+false_count) * classification_accuracy * (1 - classification_accuracy))
                    acc_bounds = [
                        classification_accuracy + 0.674 * (accuracy_binary_std / np.sqrt(correct_count+false_count)),
                        classification_accuracy - 0.674 * (accuracy_binary_std / np.sqrt(correct_count+false_count))
                    ]

                    # Relative class frequency bounds
                    class_count = counts_for_class[y[y_better_then_first_correct]]
                    total_count = i # total instances seen so far
                    class_frequency = class_count / total_count
                    class_std = np.sqrt(class_frequency * (1 - class_frequency) / total_count)
                    class_bounds = [
                        class_frequency - 0.674 * class_std,  # lower bound
                        class_frequency + 0.674 * class_std   # upper bound
                    ]

                    # Logic for removing noisy instances from CD
                    acc_low, acc_high = sorted(acc_bounds)
                    class_low, class_high = sorted(class_bounds)
                    # If the class accuracies lowerbound is greater then the class frequencies upperbound
                    # then the instance is accepted.
                    # If the class accuracies upperbound is greater then the class frequencies lowerbound is greater then
                    # the class is dropped and seen as noise.
                    # If the bounds overlap we keep the instance in CD as we are not sure.
                    if acc_high > class_low:
                        dropped_idxs.append(y_better_then_first_correct)
            
            # Remove noisy instances from CD
            cd_idx = np.delete(cd_idx, dropped_idxs, axis=0)
            
        return np_train_matrix[cd_idx, :]
    


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

    data = np.array(
        [
            [1.0, 2.0, 0],
            [4.0, 4.5, 1],
            [1.2, 1.8, 0],
            [3.8, 4.2, 1],
            [0.9, 2.1, 0],
            [3.9, 3.9, 1],
        ]
    )

    # --- Run IB3 instance reduction ----------------------------------
    ibl = IBL()
    reduced_data = ibl.ib3_instance_reduction(data)

    # --- Print results -----------------------------------------------
    print("Original data shape:", data.shape)
    print("Reduced data shape:", reduced_data.shape)
    print("\nReduced dataset:\n", reduced_data)

    # Visualization
    # Suppose these are your reduced indices (from IB3)
    reduced_indices = []
    for row in reduced_data:
        # Find the index of the row in the original data
        idx = np.where((reduced_data == row).all(axis=1))[0][0]
        reduced_indices.append(idx)
    

    # Split original features and labels
    X = data[:, :2]
    y = data[:, 2]

    # Split reduced dataset
    X_reduced = X[reduced_indices]
    y_reduced = y[reduced_indices]

    import matplotlib.pyplot as plt

    # Plot original dataset
    plt.figure(figsize=(6, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', label='Class 0 (original)')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='o', label='Class 1 (original)')

    # Plot reduced dataset
    plt.scatter(X_reduced[y_reduced==0, 0], X_reduced[y_reduced==0, 1],
                facecolors='none', edgecolors='blue', s=150, linewidths=2, label='Class 0 (CD)')
    plt.scatter(X_reduced[y_reduced==1, 0], X_reduced[y_reduced==1, 1],
                facecolors='none', edgecolors='red', s=150, linewidths=2, label='Class 1 (CD)')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('IB3 Instance Reduction')
    plt.legend()
    plt.grid(True)
    plt.show()


    # parser = Parser(
    #     base_path="datasetsCBR/datasetsCBR",
    #     dataset_name="adult",
    #     normalization_strategy=NormalizationStrategy.MINMAX_SCALING,
    #     encoding_strategy=EncodingStrategy.LABEL_ENCODE,
    #     missing_values_numeric_strategy=MissingValuesNumericStrategy.MEDIAN,
    #     missing_values_categorical_strategy=MissingValuesCategoricalStrategy.MODE,
    #     # faster_parser=True,
    # )

    # train_matrix, test_matrix = parser.get_split(0)
    # types = parser.get_types()
    # # Testing IBL
    # ibl = IBL()
    # ibl.fit(train_matrix)
    # # preds = ibl.run(

    # #     test_matrix,
    # #     k=5,
    # #     metric="cosine",
    # #     vote="modified_plurality",
    # #     retention_policy=RetentionPolicy.NEVER_RETAIN,
    # #     types=types,
    # # )

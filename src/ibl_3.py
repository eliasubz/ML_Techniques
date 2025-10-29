""""3-Instance Based Learner (k-NN)"""
from collections import Counter
import time
import pandas as pd
import numpy as np
from Parser import Parser
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


class IBL3:
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


import time
import numpy as np
from collections import Counter
from scipy.spatial.distance import cdist  # optional, faster than custom euclidean_distance


class IBL3:
    """3-Instance Based Learner (k-NN) with IB2 and IB3 instance reduction."""

    def __init__(self):
        """Initialize IBL3 model."""

    def _ib3_instance_reduction(self, np_train_matrix: np.ndarray):
        """
        Perform IB3 instance reduction on the training data.
        Timers added for diagnostics (prints every 100 iterations).
        """
        print("Starting IB3 instance reduction...")

        # --- Initialization ---
        cd_idx = [0]  # Concept description
        record_correct = np.zeros(len(np_train_matrix), dtype=int)
        record_false = np.zeros(len(np_train_matrix), dtype=int)
        acceptable_idx = np.zeros(len(np_train_matrix), dtype=int)
        class_counts = Counter()
        Z = 0.674  # 75% confidence interval z-score

        X = np_train_matrix[:, :-1].astype(float)
        y = np_train_matrix[:, -1]
        n = len(X)

        total_start = time.time()

        # --- Main Loop ---
        for i in range(1, n):
            iter_start = time.time()

            # Compute distances
            t0 = time.time()
            # If you have your own euclidean_distance function, replace cdist below
            distances = euclidean_distance(X[cd_idx], X[i].reshape(1, -1)).ravel()
            sorted_nearest_idx_in_cd = np.argsort(distances)
            t1 = time.time()

            # Classification decision
            correct_classifications = (y[cd_idx][sorted_nearest_idx_in_cd] == y[i]).astype(int)
            first_correct_pred_idx = np.argmax(correct_classifications)
            # correct_mask = (y[cd_idx][sorted_nearest_idx_in_cd] == y[i])
            # first_correct_pred_idx = np.argmax(correct_mask)

            t2 = time.time()

            # Find first acceptable neighbor
            acceptable_mask_for_CD = acceptable_idx[cd_idx]
            acceptable_positions_in_cd = np.where(acceptable_mask_for_CD)[0]
            if len(acceptable_positions_in_cd) > 0:
                y_max_idx = None
                for idx_in_sorted in sorted_nearest_idx_in_cd:
                    if idx_in_sorted in acceptable_positions_in_cd:
                        y_max_idx = cd_idx[idx_in_sorted]
                        break
                if y_max_idx is None:
                    y_max_idx = cd_idx[sorted_nearest_idx_in_cd[0]]
            else:
                if len(cd_idx) > 0:
                    y_max_idx = cd_idx[np.random.randint(0, len(cd_idx))]
                else:
                    y_max_idx = 0
            t3 = time.time()

            # Record classification result
            if y[y_max_idx] == y[i]:
                record_correct[i] += 1
            else:
                record_false[i] += 1
                cd_idx.append(i)
            t4 = time.time()

            # Update class counts
            class_counts[y[i]] += 1
            t5 = time.time()

            # --- Confidence interval computations (optimized) ---
            dropped_idxs = []

            # Only consider relevant subset once
            upto_idx = np.searchsorted(sorted_nearest_idx_in_cd, first_correct_pred_idx, side='right')
            subset_cd_idx = np.array(cd_idx)[sorted_nearest_idx_in_cd[:upto_idx + 1]]
            subset_labels = y[subset_cd_idx]
            same_class_mask = subset_labels == y[i]

            # Fast vectorized updates for correct/false counts
            record_correct[subset_cd_idx[same_class_mask]] += 1
            wrong_idx = subset_cd_idx[~same_class_mask]
            record_false[wrong_idx] += 1

            if len(wrong_idx) > 0:
                correct_counts = record_correct[wrong_idx]
                false_counts = record_false[wrong_idx]
                total_obs = correct_counts + false_counts

                # Avoid divide-by-zero
                valid_mask = total_obs > 0
                wrong_idx = wrong_idx[valid_mask]
                total_obs = total_obs[valid_mask]
                correct_counts = correct_counts[valid_mask]
                false_counts = false_counts[valid_mask]

                classification_accuracy = correct_counts / total_obs
                sqrt_total = np.sqrt(total_obs)
                acc_std = np.sqrt(total_obs * classification_accuracy * (1 - classification_accuracy))
                acc_low = classification_accuracy - Z * (acc_std / sqrt_total)
                acc_high = classification_accuracy + Z * (acc_std / sqrt_total)

                # Vectorized class stats
                class_labels = y[wrong_idx]
                class_count_array = np.array([class_counts.get(lbl, 0) for lbl in class_labels])
                total_count = float(i)

                class_freq = np.divide(class_count_array, total_count, out=np.zeros_like(class_count_array, dtype=float), where=total_count > 0)
                class_std = np.sqrt(class_freq * (1 - class_freq) / total_count) if total_count > 0 else np.zeros_like(class_freq)
                class_low = class_freq - Z * class_std
                class_high = class_freq + Z * class_std

                # Vectorized IB3 decision
                mask_acceptable = acc_low > class_high
                mask_drop = (acc_high < class_low) & (total_obs >= 4)

                acceptable_idx[wrong_idx[mask_acceptable]] = 1
                dropped_idxs.extend(np.where(mask_drop)[0])
            t6 = time.time()

            # Remove noisy instances
            for drop_idx in dropped_idxs:
                if drop_idx in cd_idx:
                    cd_idx.remove(drop_idx)
            t7 = time.time()

            # Print timing every 100th iteration
            if i % 100 == 0 or i == n - 1:
                print(f"\nIteration {i}/{n-1}")
                print(f"  Distance calc:   {t1 - t0:.5f} s")
                print(f"  Classification:  {t2 - t1:.5f} s")
                print(f"  Acceptable check:{t3 - t2:.5f} s")
                print(f"  Record update:   {t4 - t3:.5f} s")
                print(f"  Class counting:  {t5 - t4:.5f} s")
                print(f"  CI computations: {t6 - t5:.5f} s")
                print(f"  Noise removal:   {t7 - t6:.5f} s")
                print(f"  Total iteration: {t7 - iter_start:.5f} s\n")

        total_end = time.time()
        print(f"\nIB3 instance reduction complete. Total time: {total_end - total_start:.2f}s")

        return np_train_matrix[np.array(cd_idx), :]

    

    def ib3_instance_reduction(self, np_train_matrix: np.ndarray):
        """
        Perform IB3 instance reduction on the training data.
        Important:
            - z-score for confidence intervals is set to 0.674 (75% confidence)
            - Normal approximation for the binomial distribution is used
        """
        print("Starting IB3 instance reduction...")
        start_time = time.time()

        # --- Initialization ---
        cd_idx = np.array([0], dtype=int)  # Concept description (indices)
        record = [{"correct": 0, "false": 0} for _ in range(len(np_train_matrix))]  # Classification record
        acceptable_idx = np.zeros((len(np_train_matrix)), dtype=int)  # Acceptable instances

        X = np_train_matrix[:, :-1].astype(float)
        y = np_train_matrix[:, -1]

        for i in range(1, len(X)):
            print(f"\nProcessing instance {i}/{len(X)-1}...")

            # --- Compute distances ---
            distances = euclidean_distance(X[cd_idx], X[i])
            sorted_nearest_idx_in_cd = np.argsort(distances)

            correct_classifications = (y[cd_idx][sorted_nearest_idx_in_cd] == y[i]).astype(int)
            first_correct_pred_idx = np.argmax(correct_classifications)

            # --- Find first acceptable neighbor ---
            acceptable_mask_for_CD = acceptable_idx[cd_idx]
            acceptable_positions_in_cd = np.where(acceptable_mask_for_CD)[0]

            if len(acceptable_positions_in_cd) > 0:
                y_max_idx = None
                for idx_in_sorted in sorted_nearest_idx_in_cd:
                    if idx_in_sorted in acceptable_positions_in_cd:
                        y_max_idx = cd_idx[idx_in_sorted]
                        break
                if y_max_idx is None:
                    y_max_idx = cd_idx[sorted_nearest_idx_in_cd[0]]
            else:
                if len(cd_idx) > 0:
                    i_rand = np.random.randint(0, len(cd_idx))
                    y_max_idx = cd_idx[i_rand]
                else:
                    y_max_idx = 0  # fallback for safety

            # --- Classification and record update ---
            if y[y_max_idx] == y[i]:
                record[i]["correct"] += 1
            else:
                record[i]["false"] += 1
                cd_idx = np.append(cd_idx, i)

            # --- Class statistics for significance testing ---
            unique_labels = np.unique(y[:i+1])
            counts_for_class = {label: np.sum(y[:i+1] == label) for label in unique_labels}

            dropped_idxs = []

            # --- Iterate over neighbors up to first correct prediction ---
            if first_correct_pred_idx < len(sorted_nearest_idx_in_cd):
                upto_idx = np.where(sorted_nearest_idx_in_cd == first_correct_pred_idx)[0]
                if len(upto_idx) > 0:
                    upto_idx = upto_idx[0]
                else:
                    upto_idx = first_correct_pred_idx
            else:
                upto_idx = first_correct_pred_idx

            for y_better_then_first_correct in sorted_nearest_idx_in_cd[:upto_idx + 1]:
                idx_in_cd = int(cd_idx[y_better_then_first_correct])
                label = y[idx_in_cd]

                if y[idx_in_cd] == y[i]:
                    record[idx_in_cd]["correct"] += 1
                else:
                    record[idx_in_cd]["false"] += 1

                    correct_count = record[idx_in_cd]["correct"]
                    false_count = record[idx_in_cd]["false"]
                    total_obs = correct_count + false_count
                    if total_obs == 0:
                        continue

                    classification_accuracy = correct_count / total_obs
                    accuracy_binary_std = np.sqrt(total_obs * classification_accuracy * (1 - classification_accuracy))
                    acc_bounds = [
                        classification_accuracy + 0.674 * (accuracy_binary_std / np.sqrt(total_obs)),
                        classification_accuracy - 0.674 * (accuracy_binary_std / np.sqrt(total_obs))
                    ]

                    class_count = counts_for_class.get(label, 0)
                    total_count = i
                    class_frequency = class_count / total_count if total_count > 0 else 0
                    class_std = np.sqrt(class_frequency * (1 - class_frequency) / total_count) if total_count > 0 else 0
                    class_bounds = [
                        class_frequency - 0.674 * class_std,
                        class_frequency + 0.674 * class_std
                    ]

                    acc_low, acc_high = sorted(acc_bounds)
                    class_low, class_high = sorted(class_bounds)

                    # --- IB3 decision logic ---
                    if acc_low > class_high:
                        acceptable_idx[idx_in_cd] = 1
                    elif acc_high < class_low and total_obs >= 4:
                        dropped_idxs.append(y_better_then_first_correct)

            # --- Remove noisy instances ---
            if len(dropped_idxs) > 0:
                cd_idx = np.delete(cd_idx, dropped_idxs, axis=0)

        print("\nIB3 instance reduction complete.")
        step_end = time.time()
        print(f"Total time for IB3 instance reduction: {step_end - start_time:.2f}s")
        return np_train_matrix[cd_idx, :]



    def fit(self, train_matrix: pd.DataFrame):

        np_train_matrix_old = train_matrix.reset_index(drop=True).to_numpy()
        print("Fitting IBL model with IB3 instance reduction...")
        np_train_matrix = self._ib3_instance_reduction(np_train_matrix_old)
        print(np_train_matrix.shape)
        print(np_train_matrix_old.shape)

        self.X = np_train_matrix[:, :-1]
        self.y = np_train_matrix[:, -1]

    def get_concept_description_size(self):
        storage_mb = self.X.nbytes / (1024 ** 2)
        print(f"Storage used by X: {storage_mb:.2f} MB")
        return self.X.nbytes
    
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
    ibl = IBL3()
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



    parser = Parser(
        base_path="datasetsCBR/datasetsCBR",
        dataset_name="pen-based",
        normalization_strategy=NormalizationStrategy.MINMAX_SCALING,
        encoding_strategy=EncodingStrategy.ONE_HOT_ENCODE,
        missing_values_numeric_strategy=MissingValuesNumericStrategy.MEDIAN,
        missing_values_categorical_strategy=MissingValuesCategoricalStrategy.MODE,
        # faster_parser=True,
    )

    train_matrix, test_matrix = parser.get_split(1)
    print("Got parser data.")
    types = parser.get_types()
    # Testing IBL
    ibl = IBL3()
    ibl.fit(train_matrix)
    print("Fitting is done!")
    preds = ibl.run(
        test_matrix,
        k=5,
        metric="cosine",
        vote="modified_plurality",
        retention_policy=RetentionPolicy.NEVER_RETAIN,
        types=types,
    )
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # Helpful basic metrics
    acc = accuracy_score(test_matrix.iloc[:, -1], preds)
    prec = precision_score(
        test_matrix.iloc[:, -1], preds, average='weighted', zero_division=0)
    rec = recall_score(
        test_matrix.iloc[:, -1], preds, average='weighted', zero_division=0)
    f1 = f1_score(test_matrix.iloc[:, -1], preds,
                  average='weighted', zero_division=0)

    # Display results
    print("Performance Metrics:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # # Confusion matrix + detailed report
    # print("\nConfusion Matrix:")
    # print(confusion_matrix(test_matrix.iloc[:, -1], preds))

    # print("\nClassification Report:")
    # print(classification_report(
    #     test_matrix.iloc[:, -1], preds, zero_division=0))

    # print("Predictions:", preds)
    # print("Ground truth:", list(test_matrix.iloc[:, -1]))



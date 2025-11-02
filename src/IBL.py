from Parser import Parser
import pandas as pd
import numpy as np
from collections import Counter
import time

from distance_measures import (
    cosine_distance,
    euclidean_distance,
    heom_distance,
    weighted_cosine_distance,
    weighted_euclidean_distance,
    weighted_heom_distance,
)
from feature_weighing import compute_feature_weights
from preallocated_matrix import PreallocatedMatrix
from processing_types import (
    FeatureWeightingMethod,
    RetentionPolicy,
    EncodingStrategy,
    MissingValuesCategoricalStrategy,
    MissingValuesNumericStrategy,
    NormalizationStrategy,
)
from retention_policies import retention_policies
from Instance_Reduction import condensed_nearest_neighbor, mcnn, edited_nearest_neighbor, renn


class IBL:
    def __init__(
        self,
    ):
        """
        k-Instance Based Learner (k-NN) with:
        - metrics: 'euclidean', 'cosine', 'heom'
        - votes: 'modified_plurality', 'borda'
        - types: (list of 'numeric'/'categorical') when using HEOM.
        """
        self.feature_weights = None
        self.cp_before_ir = 0
        self.cp_after_ir = 0
        self.cp_after_training = 0

    def ib3_instance_reduction(
        self,
        np_train_matrix: np.ndarray,
        timings: bool = False,
        confidence_z: float = 0.674,
        min_observations: int = 4,
    ) -> np.ndarray:
        """
        Perform IB3 instance reduction on the training data.
        """

        # --- Initialization ---
        cd_idx = [0]  # Concept description
        record_correct = np.zeros(len(np_train_matrix), dtype=int)
        record_false = np.zeros(len(np_train_matrix), dtype=int)
        acceptable_idx = np.zeros(len(np_train_matrix), dtype=int)
        class_counts = Counter()
        Z = confidence_z
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
            distances = euclidean_distance(
                X[cd_idx], X[i].reshape(1, -1)).ravel()
            sorted_nearest_idx_in_cd = np.argsort(distances)
            t1 = time.time()

            # Classification decision
            correct_classifications = (
                y[cd_idx][sorted_nearest_idx_in_cd] == y[i]
            ).astype(int)
            first_correct_pred_idx = np.argmax(correct_classifications)

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
            upto_idx = np.searchsorted(
                sorted_nearest_idx_in_cd, first_correct_pred_idx, side="right"
            )
            subset_cd_idx = np.array(
                cd_idx)[sorted_nearest_idx_in_cd[: upto_idx + 1]]
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
                acc_std = np.sqrt(
                    total_obs * classification_accuracy *
                    (1 - classification_accuracy)
                )
                acc_low = classification_accuracy - Z * (acc_std / sqrt_total)
                acc_high = classification_accuracy + Z * (acc_std / sqrt_total)

                # Vectorized class stats
                class_labels = y[wrong_idx]
                class_count_array = np.array(
                    [class_counts.get(lbl, 0) for lbl in class_labels]
                )
                total_count = float(i)

                class_freq = np.divide(
                    class_count_array,
                    total_count,
                    out=np.zeros_like(class_count_array, dtype=float),
                    where=total_count > 0,
                )
                class_std = (
                    np.sqrt(class_freq * (1 - class_freq) / total_count)
                    if total_count > 0
                    else np.zeros_like(class_freq)
                )
                class_low = class_freq - Z * class_std
                class_high = class_freq + Z * class_std

                # Vectorized IB3 decision
                mask_acceptable = acc_low > class_high
                mask_drop = (acc_high < class_low) & (
                    total_obs >= min_observations)

                acceptable_idx[wrong_idx[mask_acceptable]] = 1
                dropped_idxs.extend(np.where(mask_drop)[0])
            t6 = time.time()

            # Remove noisy instances
            for drop_idx in dropped_idxs:
                if drop_idx in cd_idx:
                    cd_idx.remove(drop_idx)
            t7 = time.time()

        total_end = time.time()

        return np_train_matrix[cd_idx, :]

    def get_concept_description_size(self, matrix=None):
        if matrix is not None:
            storage_mb = matrix.nbytes / (1024**2)
            return storage_mb

        storage_mb = self.X.nbytes / (1024**2)
        return storage_mb

    def fit(
        self,
        train_matrix: pd.DataFrame,
        min_observations=None,
        confidence_z=None,
        distance_measure="euclidean",
        instance_red: str = None,
    ):
        np_train_matrix_b = train_matrix.reset_index(drop=True).to_numpy()

        if instance_red == "IBL3":
            np_train_matrix = self.ib3_instance_reduction(
                np_train_matrix_b, min_observations=min_observations, confidence_z=confidence_z)
        elif instance_red == "IBL3_verbose":
            np_train_matrix = self.ib3_instance_reduction(
                np_train_matrix_b, timings=True, min_observations=min_observations, confidence_z=confidence_z
            )
        elif instance_red == "CNN":
            np_train_matrix = condensed_nearest_neighbor(
                train_matrix, distance_metric=distance_measure)
        elif instance_red == "MCNN":
            np_train_matrix = mcnn(
                train_matrix, distance_metric=distance_measure)
        elif instance_red == "enn":
            np_train_matrix = edited_nearest_neighbor(
                train_matrix, distance_metric=distance_measure)
        elif instance_red == "RENN":
            np_train_matrix = renn(
                train_matrix, distance_metric=distance_measure)
        else:
            np_train_matrix = np_train_matrix_b

        self.X = np_train_matrix[:, :-1]
        self.y = np_train_matrix[:, -1]

        if instance_red is not None:
            self.cp_before_ir = self.get_concept_description_size(
                np_train_matrix_b)
            self.cp_after_ir = self.get_concept_description_size()

    def run(
        self,
        test_matrix: pd.DataFrame,
        k=5,
        metric="euclidean",
        vote="modified_plurality",
        retention_policy="DD_retention",
        types=None,
    ):
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
            if self.feature_weights is not None:
                # Use weighted distance functions
                if self.metric == "euclidean":
                    distances = weighted_euclidean_distance(
                        self.X.get_filled(), x_instance, self.feature_weights
                    )
                elif self.metric == "cosine":
                    distances = weighted_cosine_distance(
                        self.X.get_filled(), x_instance, self.feature_weights
                    )
                elif self.metric == "heom":
                    distances = weighted_heom_distance(
                        self.X.get_filled(), x_instance, types, self.feature_weights
                    )
                else:
                    raise ValueError(f"Unknown metric: {self.metric}")
            else:
                # Use original distance functions
                if self.metric == "euclidean":
                    distances = euclidean_distance(
                        self.X.get_filled(), x_instance)
                elif self.metric == "cosine":
                    distances = cosine_distance(
                        self.X.get_filled(), x_instance)
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
                retention_policy,
                instance_class=y_instance,
                pred=pred,
                k_nearest_labels=neighbor_labels,
            )

            if should_retain:
                self.X.append_column(x_instance)
                # TODO: optimize y storage as well
                self.y = np.append(self.y, y_instance)

            retention_end = time.time()

            step_end = time.time()
            # print(f"Instance {i}/{len(test_matrix)}: dist={dist_end-dist_start:.5f}s, sort={sort_end-sort_start:.5f}s, vote={vote_end-vote_start:.5f}s, retention={retention_end-retention_start:.5f}s, total={step_end-step_start:.5f}s")

        total_end = time.time()

        self.cp_after_training = self.get_concept_description_size(
            self.X.get_filled())

        return predictions

    def fw_KIBLAlgorithm(
        self,
        test_matrix: pd.DataFrame,
        types,
        k=5,
        metric="euclidean",
        vote="modified_plurality",
        retention_policy="DD_retention",
        feature_weighting_method: FeatureWeightingMethod = FeatureWeightingMethod.INFORMATION_GAIN,
        post_encoding_types: list = None,
    ):
        """
        Feature-Weighted k-Instance Based Learning Algorithm.

        This method computes feature weights from the training set and uses them to modify
        the distance metrics during classification.

        Args:
            test_matrix: Test data
            k: Number of nearest neighbors
            metric: Distance metric ('euclidean', 'cosine', 'heom')
            vote: Voting method ('modified_plurality', 'borda')
            retention_policy: Retention policy for instances
            types: Feature types for HEOM distance
            feature_weighting_method: Method for computing feature weights (FeatureWeightingMethod)

        Returns:
            list: Predictions for test instances
        """
        self.types = types
        self.feature_weights = compute_feature_weights(
            self.X,
            self.y,
            method=feature_weighting_method,
            post_encoding_types=np.asarray(post_encoding_types),
        )

        # Run the standard IBL algorithm with feature weights
        predictions = self.run(
            test_matrix=test_matrix,
            k=k,
            metric=metric,
            vote=vote,
            retention_policy=retention_policy,
            types=types,
        )

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
    parser = Parser(
        base_path="datasetsCBR/datasetsCBR",
        dataset_name="pen-based",
        normalization_strategy=NormalizationStrategy.MINMAX_SCALING,
        encoding_strategy=EncodingStrategy.ONE_HOT_ENCODE,
        missing_values_numeric_strategy=MissingValuesNumericStrategy.MEDIAN,
        missing_values_categorical_strategy=MissingValuesCategoricalStrategy.MODE,
        faster_parser=True,
    )

    train_matrix, test_matrix = parser.get_split(0)
    types = parser.get_types()

    # Testing standard IBL
    print("=== Testing Standard IBL ===")
    ibl = IBL()
    instance_red = [None, "IBL3", "IBL3_verbose", "CNN", "MCNN", "enn", "RENN"]
    ibl.fit(train_matrix, instance_red="RENN")
    preds = ibl.run(
        test_matrix,
        k=5,
        metric="heom",
        vote="modified_plurality",
        retention_policy=RetentionPolicy.NEVER_RETAIN,
        types=types,
    )

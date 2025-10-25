import numpy as np
from sklearn_relief import Relief
from sklearn.feature_selection import mutual_info_classif
from processing_types import EncodingStrategy, FeatureWeightingMethod


def compute_feature_weights(X: np.ndarray, y: np.ndarray, method: FeatureWeightingMethod, post_encoding_types: np.ndarray | None = None, preprocessing_config: dict = {}) -> np.ndarray:
    """
    Compute feature weights using the specified method.

    Args:
        X: Training features
        y: Training labels
        method: FeatureWeightingMethod
        preprocessing_config: Preprocessing configuration dictionary

    Returns:
        np.ndarray: Feature weights (values between 0 and 1)
    """
    print(f"Computing feature weights using {method}...")

    if post_encoding_types is None:
        raise ValueError(
            "'post_encoding_types' must be provided when not using One-Hot Encoding")
    if post_encoding_types.shape[0] != X.shape[1]:
        raise ValueError(
            f"'types' length ({post_encoding_types.shape[0]}) != n_features ({X.shape[1]})")

    categorical_features_indices = np.where(
        post_encoding_types == "categorical")[0]

    if method == FeatureWeightingMethod.INFORMATION_GAIN:
        return compute_information_gain_weights(X, y, categorical_features_indices)
    elif method == FeatureWeightingMethod.RELIEFF:

        return compute_relieff_weights(X, y, categorical_features_indices)


def compute_information_gain_weights(X: np.ndarray, y: np.ndarray, categorical_features_indices: np.ndarray):
    """
    Compute feature weights using Information Gain (mutual information).

    Args:
        X: Training features
        y: Training labels

    Returns:
        np.ndarray: Normalized feature weights (values between 0 and 1)
    """
    mi_scores = mutual_info_classif(
        X, y, discrete_features=categorical_features_indices)

    # Normalize scores to [0, 1] range
    if np.max(mi_scores) > 0:
        weights = mi_scores / np.max(mi_scores)
    else:
        weights = np.ones_like(mi_scores)

    return weights


def compute_relieff_weights(X: np.ndarray, y: np.ndarray, categorical_features_indices: np.ndarray = None):
    """
    Compute feature weights using ReliefF algorithm.

    Args:
        X: Training features
        y: Training labels
        categorical_features: Boolean array indicating which features are categorical

    Returns:
        np.ndarray: Feature weights (values between 0 and 1)
    """
    relieff = Relief(
        categorical=categorical_features_indices)
    relieff.fit(X, y)
    weights = relieff.w_

    # Ensure weights are in [0, 1] range
    if np.max(weights) > 0:
        weights = weights / np.max(weights)
    else:
        weights = np.ones_like(weights)

    return weights

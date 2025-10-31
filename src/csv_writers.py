
import json
import numpy as np


def cm_to_json(cm: np.ndarray, labels: list | None = None) -> str:
    d = {"labels": labels if labels is not None else list(range(cm.shape[0])),
         "matrix": cm.astype(int).tolist()}
    return json.dumps(d)


def create_fw_k_ibl_csv_row(
    fw_method: str,
    metric: str,
    k: int,
    vote: str,
    retention: str,
    fold_id: int,
    num_folds: int,
    n_train: int,
    n_test: int,
    fit_time: float,
    predict_time: float,
    total_time: float,
    accuracy: float,
    precision_macro: float,
    recall_macro: float,
    f1_macro: float,
    precision_weighted: float,
    recall_weighted: float,
    f1_weighted: float,
    confusion_matrix: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Create a CSV row dictionary for feature-weighted k-IBL results."""
    metric_code = {"euclidean": "e", "cosine": "c", "heom": "h"}
    vote_code = {"borda": "b", "modified_plurality": "mp"}
    ret_code = {
        "RetentionPolicy.ALWAYS_RETAIN": "AR",
        "RetentionPolicy.NEVER_RETAIN": "NR",
        "RetentionPolicy.DIFFERENT_CLASS_RETENTION": "DC",
        "RetentionPolicy.DD_RETENTION": "DD",
    }
    feature_weighting_code = {
        "FeatureWeightingMethod.RELIEFF": "rf",
        "FeatureWeightingMethod.INFORMATION_GAIN": "ig",
    }

    m = metric_code.get(str(metric), str(metric)[:2])
    v = vote_code.get(str(vote), str(vote)[:2])
    r = ret_code.get(str(retention), str(retention)[:2])
    fw = feature_weighting_code.get(str(fw_method), str(fw_method)[:2])
    config_id = f"{m}-k{int(k)}-{v}-{r}-{fw}"

    return {
        "config_id": config_id,
        "feature_weighting_method": fw_method,
        "metric": metric,
        "k": k,
        "vote": vote,
        "retention": retention,

        "fold_id": fold_id,
        "num_folds": num_folds,
        "n_train": n_train,
        "n_test": n_test,

        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "total_time_s": total_time,

        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro":    recall_macro,
        "f1_macro":        f1_macro,

        "precision_weighted": precision_weighted,
        "recall_weighted":    recall_weighted,
        "f1_weighted":        f1_weighted,

        "confusion_matrix_json": cm_to_json(confusion_matrix, labels=labels.tolist()),
    }

def create_ir_ibl_csv_row(
    metric: str,
    k: int,
    vote: str,
    retention: str,
    fold_id: int,
    num_folds: int,
    n_train: int,
    n_test: int,
    fit_time: float,
    predict_time: float,
    total_time: float,
    accuracy: float,
    precision_macro: float,
    recall_macro: float,
    f1_macro: float,
    precision_weighted: float,
    recall_weighted: float,
    f1_weighted: float,
    confusion_matrix: np.ndarray,
    labels: np.ndarray,
    instance_reduction_method: str,
    memory_before_ir: float,
    memory_after_ir: float,
    memory_after_training: float,
    percentage_memory_reduction: float
) -> dict:
    """Create a CSV row dictionary for k-IBL results (without fw_method column)."""
    metric_code = {"euclidean": "e", "cosine": "c", "heom": "h"}
    vote_code = {"borda": "b", "modified_plurality": "mp"}
    ret_code = {
        "RetentionPolicy.ALWAYS_RETAIN": "AR",
        "RetentionPolicy.NEVER_RETAIN": "NR",
        "RetentionPolicy.DIFFERENT_CLASS_RETENTION": "DC",
        "RetentionPolicy.DD_RETENTION": "DD",
    }

    m = metric_code.get(str(metric), str(metric)[:2])
    v = vote_code.get(str(vote), str(vote)[:2])
    r = ret_code.get(str(retention), str(retention)[:2])
    config_id = f"{m}-k{int(k)}-{v}-{r}"

    return {
        "config_id": config_id,
        "instance_reduction_method": instance_reduction_method,
        "metric": metric,
        "k": k,
        "vote": vote,
        "retention": retention,


        "fold_id": fold_id,
        "num_folds": num_folds,
        "n_train": n_train,
        "n_test": n_test,

        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "total_time_s": total_time,
        "memory_before_ir": memory_before_ir,
        "memory_after_ir": memory_after_ir,
        "memory_after_training": memory_after_training,
        "percentage_memory_reduction": percentage_memory_reduction,

        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro":    recall_macro,
        "f1_macro":        f1_macro,

        "precision_weighted": precision_weighted,
        "recall_weighted":    recall_weighted,
        "f1_weighted":        f1_weighted,

        "confusion_matrix_json": cm_to_json(confusion_matrix, labels=labels.tolist()),
    }


def create_k_ibl_csv_row(
    metric: str,
    k: int,
    vote: str,
    retention: str,
    fold_id: int,
    num_folds: int,
    n_train: int,
    n_test: int,
    fit_time: float,
    predict_time: float,
    total_time: float,
    accuracy: float,
    precision_macro: float,
    recall_macro: float,
    f1_macro: float,
    precision_weighted: float,
    recall_weighted: float,
    f1_weighted: float,
    confusion_matrix: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Create a CSV row dictionary for k-IBL results (without fw_method column)."""
    metric_code = {"euclidean": "e", "cosine": "c", "heom": "h"}
    vote_code = {"borda": "b", "modified_plurality": "mp"}
    ret_code = {
        "RetentionPolicy.ALWAYS_RETAIN": "AR",
        "RetentionPolicy.NEVER_RETAIN": "NR",
        "RetentionPolicy.DIFFERENT_CLASS_RETENTION": "DC",
        "RetentionPolicy.DD_RETENTION": "DD",
    }

    m = metric_code.get(str(metric), str(metric)[:2])
    v = vote_code.get(str(vote), str(vote)[:2])
    r = ret_code.get(str(retention), str(retention)[:2])
    config_id = f"{m}-k{int(k)}-{v}-{r}"

    return {
        "config_id": config_id,
        "metric": metric,
        "k": k,
        "vote": vote,
        "retention": retention,

        "fold_id": fold_id,
        "num_folds": num_folds,
        "n_train": n_train,
        "n_test": n_test,

        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "total_time_s": total_time,

        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro":    recall_macro,
        "f1_macro":        f1_macro,

        "precision_weighted": precision_weighted,
        "recall_weighted":    recall_weighted,
        "f1_weighted":        f1_weighted,

        "confusion_matrix_json": cm_to_json(confusion_matrix, labels=labels.tolist()),
    }

def create_svm_csv_row(
    kernel: str,
    C: float,
    gamma: float,
    Degree: int,
    fold_id: int,
    num_folds: int,
    n_train: int,
    n_test: int,
    fit_time: float,
    predict_time: float,
    total_time: float,
    accuracy: float,
    precision_macro: float,
    recall_macro: float,
    f1_macro: float,
    precision_weighted: float,
    recall_weighted: float,
    f1_weighted: float,
    confusion_matrix: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """Create a CSV row dictionary for k-IBL results (without fw_method column)."""
    config_id = f"svm-{kernel}-c{C}-gamma{gamma}-degree{Degree}"

    return {
        "config_id": config_id,
        "kernel": kernel,
        "c": C,
        "gamma": gamma,
        "degree": Degree,

        "fold_id": fold_id,
        "num_folds": num_folds,
        "n_train": n_train,
        "n_test": n_test,

        "fit_time_s": fit_time,
        "predict_time_s": predict_time,
        "total_time_s": total_time,

        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro":    recall_macro,
        "f1_macro":        f1_macro,

        "precision_weighted": precision_weighted,
        "recall_weighted":    recall_weighted,
        "f1_weighted":        f1_weighted,

        "confusion_matrix_json": cm_to_json(confusion_matrix, labels=labels.tolist()),
    }

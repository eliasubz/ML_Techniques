import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from IBL import IBL
from Parser import Parser
from argument_parser import parse_arguments
from processing_types import EncodingStrategy, FeatureWeightingMethod, MissingValuesCategoricalStrategy, MissingValuesNumericStrategy, NormalizationStrategy, RetentionPolicy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# Valid options for validation
NUM_SPLITS = 10
BASE_PATH = "datasetsCBR/datasetsCBR/"
RESULTS_PATH = "results/"


def cm_to_json(cm: np.ndarray, labels: list | None = None) -> str:
    d = {"labels": labels if labels is not None else list(range(cm.shape[0])),
         "matrix": cm.astype(int).tolist()}
    return json.dumps(d)


def run_suite(
    dataset_name: str,
    k: int,
    metric: str,
    retention: RetentionPolicy,
    vote: str,
    feature_weighting_method: FeatureWeightingMethod,
    normalization_strategy: NormalizationStrategy,
    encoding_strategy: EncodingStrategy,
    missing_values_numeric_strategy: MissingValuesNumericStrategy,
    missing_values_categorical_strategy: MissingValuesCategoricalStrategy,
    out_csv: Path
):

    parser = Parser(
        base_path=BASE_PATH,
        dataset_name=dataset_name,
        normalization_strategy=normalization_strategy,
        encoding_strategy=encoding_strategy,
        missing_values_numeric_strategy=missing_values_numeric_strategy,
        missing_values_categorical_strategy=missing_values_categorical_strategy,
        num_splits=NUM_SPLITS,
    )
    types = parser.get_types()
    post_encoding_types = parser.get_post_encoding_types()

    splits = [parser.get_split(fold) for fold in range(NUM_SPLITS)]

    all_labels = set()
    for tr, te in splits:
        all_labels.update(np.unique(tr.iloc[:, -1]))
        all_labels.update(np.unique(te.iloc[:, -1]))
    labels = np.array(sorted(all_labels))

    rows = []
    for fold_id, (train_matrix, test_matrix) in enumerate(splits):

        ibl = IBL()

        # Fit + predict
        t0 = time.perf_counter()
        ibl.fit(train_matrix)
        t1 = time.perf_counter()
        retention_policy_str = retention.value if isinstance(
            retention, RetentionPolicy) else retention
        preds = ibl.fw_KIBLAlgorithm(
            test_matrix,
            k=k,
            metric=metric,
            vote=vote,
            retention_policy=retention_policy_str,
            types=types,
            feature_weighting_method=feature_weighting_method,
            post_encoding_types=post_encoding_types
        )
        t2 = time.perf_counter()

        # Times
        fit_time = t1 - t0
        predict_time = t2 - t1
        total_time = t2 - t0

        # Metrics
        y_true = test_matrix.iloc[:, -1].to_numpy()
        y_pred = np.asarray(preds)

        acc = accuracy_score(y_true, y_pred)

        pM, rM, fM, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        pW, rW, fW, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        cm_fold = confusion_matrix(y_true, y_pred, labels=labels).astype(int)

        row = {
            "fw_method": feature_weighting_method.value,
            "metric": metric,
            "k": k,
            "vote": vote,
            "retention": retention.value if isinstance(retention, RetentionPolicy) else retention,

            "fold_id": fold_id,
            "num_folds": NUM_SPLITS,
            "n_train": len(train_matrix),
            "n_test":  len(test_matrix),

            "fit_time_s": fit_time,
            "predict_time_s": predict_time,
            "total_time_s": total_time,

            "accuracy": acc,
            "precision_macro": pM,
            "recall_macro":    rM,
            "f1_macro":        fM,

            "precision_weighted": pW,
            "recall_weighted":    rW,
            "f1_weighted":        fW,

            "confusion_matrix_json": cm_to_json(cm_fold, labels=labels.tolist()),
        }
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_rows = pd.DataFrame(rows)

    write_header = not out_csv.exists()
    df_rows.to_csv(out_csv, mode="a", header=write_header, index=False)

    return df_rows


if __name__ == "__main__":
    try:
        args = parse_arguments()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    for method in FeatureWeightingMethod:
        out_csv = Path(
            RESULTS_PATH + f"{args['dataset_name']}_fw_{method.value}.csv")
        print(
            f"dataset={args['dataset_name']}, metric={args['metric']}, k={args['k']}, "
            f"vote={args['vote']}, retention={args['retention_policy'].value}, fw_method={method.value}")
        run_suite(
            dataset_name=args['dataset_name'],
            k=args['k'],
            metric=args['metric'],
            retention=args['retention_policy'],
            vote=args['vote'],
            feature_weighting_method=method,
            normalization_strategy=args['normalization_strategy'],
            encoding_strategy=args['encoding_strategy'],
            missing_values_numeric_strategy=args['missing_values_numeric_strategy'],
            missing_values_categorical_strategy=args['missing_values_categorical_strategy'],
            out_csv=out_csv
        )

    # print("\n=== Parsing/Preprocessing Data ... ===")

    # parser = Parser(
    #     base_path="datasetsCBR/datasetsCBR",
    #     dataset_name="pen-based",
    #     normalization_strategy=NormalizationStrategy.MINMAX_SCALING,
    #     encoding_strategy=EncodingStrategy.ONE_HOT_ENCODE,
    #     missing_values_numeric_strategy=MissingValuesNumericStrategy.MEDIAN,
    #     missing_values_categorical_strategy=MissingValuesCategoricalStrategy.MODE
    # )

    # train_matrix, test_matrix = parser.get_split(0)
    # types = parser.get_types()
    # post_encoding_types = parser.get_post_encoding_types()

    # print("\n=== Testing Feature-Weighted IBL ===")
    # ibl_fw_relief = IBL()
    # ibl_fw_relief.fit(train_matrix)
    # preds_fw_relief = ibl_fw_relief.fw_KIBLAlgorithm(
    #     test_matrix=test_matrix,
    #     k=5,
    #     metric="euclidean",
    #     vote="modified_plurality",
    #     retention_policy=RetentionPolicy.DD_RETENTION,
    #     types=types,
    #     feature_weighting_method=FeatureWeightingMethod.INFORMATION_GAIN,
    #     post_encoding_types=post_encoding_types
    # )

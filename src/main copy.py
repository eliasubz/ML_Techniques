from pathlib import Path
import time
import numpy as np
import pandas as pd
from IBL import IBL
from Parser import Parser
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from enum import Enum  # Needed for type hinting retention_policy
from typing import Optional, List, Any, Dict
from sklearn.svm import SVC


# Dependencies from your original file structure (assuming these imports exist)
from argument_parser import parse_arguments
from csv_writers import (
    create_fw_k_ibl_csv_row,
    create_k_ibl_csv_row,
    create_ir_ibl_csv_row,
    create_svm_csv_row,
)
from model_types import Models  # Assuming Models is an Enum

# Global Constants (used in the function body)
BASE_PATH = "datasetsCBR/datasetsCBR/"
NUM_SPLITS = 10
RESULTS_PATH = "src/results/"


def run_experiment_portable(
    dataset_name: str,
    model: Models,
    normalization_strategy: str,
    encoding_strategy: str,
    missing_values_numeric_strategy: str,
    missing_values_categorical_strategy: str,
    k: int,
    distance_metric: str,
    voting_strategy: str,
    svm_kernel,
    C: float,
    gamma: Any,
    Degree: Any,
    retention_policy: Enum,  # Assuming this is an Enum type

    instance_reduction_strategy: Optional[str] = None,
    feature_weighting_strategy: Optional[str] = None,
    BASE_PATH = "datasetsCBR/datasetsCBR/"
):
    """
    Core function to run a single machine learning experiment.
    All configuration is passed via explicit parameters, making it fully portable.

    Args:
        All parameters correspond directly to the command-line arguments.
    """

    # 1. --- Configuration and Data Parsing ---

    # Initialize data parser based on explicit specifications
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

    out_csv = Path(RESULTS_PATH + f"{dataset_name}-{model.name.lower()}.csv")

    rows = []

    # 2. --- Cross-Validation Loop and Execution ---

    for fold_id, (train_matrix, test_matrix) in enumerate(splits):

        # --- Fit Time (Instance Reduction/Feature Weighting) ---
        t0 = time.perf_counter()

        if model is not Models.SVM:
            ibl = IBL()


            # Apply Instance Reduction (IR) if specified.
            if instance_reduction_strategy is None:
                print("No instance reduction selected.")
                ibl.fit(train_matrix)

            # The original code's `if/elif` structure for IR is simplified using a direct call.
            # This assumes your IBL.fit handles the direct string strategy names.

            elif instance_reduction_strategy == "IBL3":
                print("Applying IBL3 instance reduction...")
                ibl.fit(train_matrix, instance_red="IBL3")

            elif instance_reduction_strategy == "IBL3_verbose":
                print("Applying IBL3 (verbose) instance reduction...")
                ibl.fit(train_matrix, instance_red="IBL3_verbose")

            elif instance_reduction_strategy == "CNN":
                print("Applying Condensed Nearest Neighbor (CNN) instance reduction...")
                ibl.fit(train_matrix, instance_red="CNN")

            elif instance_reduction_strategy == "MCNN":
                print("Applying Modified Condensed Nearest Neighbor (MCNN) instance reduction...")
                ibl.fit(train_matrix, instance_red="MCNN")

            elif instance_reduction_strategy.lower() == "enn":
                print("Applying Edited Nearest Neighbor (ENN) instance reduction...")
                ibl.fit(train_matrix, instance_red="enn")

            elif instance_reduction_strategy.upper() == "RENN":
                print("Applying Repeated Edited Nearest Neighbor (RENN) instance reduction...")
                ibl.fit(train_matrix, instance_red="RENN")

            else:
                raise ValueError(f"Unknown instance reduction strategy: {instance_reduction_strategy}")

        else:   
            # Set up SVM fit
            np_train_matrix = train_matrix.reset_index(drop=True).to_numpy()
            X_train, y_train = np_train_matrix[:, :-1], np_train_matrix[:, -1]

            svm = SVC(kernel=svm_kernel, C=C, gamma=gamma, degree=Degree)
            svm.fit(X_train, y_train)

        t1 = time.perf_counter()
        
        preds = []
        # --- Predict Time (Model Run) ---
        if model in [Models.K_IBL, Models.IR_K_IBL]:
            preds = ibl.run(
                test_matrix,
                k=k,
                metric=distance_metric,
                vote=voting_strategy,
                retention_policy=retention_policy,
                types=types,
            )
        elif model is Models.FW_K_IBL:
            preds = ibl.fw_KIBLAlgorithm(
                test_matrix,
                k=k,
                metric=distance_metric,
                vote=voting_strategy,
                retention_policy=retention_policy,
                types=types,
                feature_weighting_method=feature_weighting_strategy,
                post_encoding_types=post_encoding_types[fold_id],
            )
        
        elif model is Models.SVM:
            X_test = test_matrix.reset_index(drop=True).to_numpy()[:, :-1]
            preds = svm.predict(X_test)
        

        t2 = time.perf_counter()

        # 3. --- Metric Calculation and Row Creation ---

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

        # Create the CSV row dictionary based on the model type
        if model is Models.FW_K_IBL:
            row = create_fw_k_ibl_csv_row(
                fw_method=feature_weighting_strategy,
                metric=distance_metric,
                k=k,
                vote=voting_strategy,
                retention=retention_policy.value,
                fold_id=fold_id,
                num_folds=NUM_SPLITS,
                n_train=train_matrix.shape[0],
                n_test=test_matrix.shape[0],
                fit_time=fit_time,
                predict_time=predict_time,
                total_time=total_time,
                accuracy=acc,
                precision_macro=pM,
                recall_macro=rM,
                f1_macro=fM,
                precision_weighted=pW,
                recall_weighted=rW,
                f1_weighted=fW,
                confusion_matrix=cm_fold,
                labels=labels,
            )
        elif model is Models.IR_K_IBL:
            row = create_ir_ibl_csv_row(
                metric=distance_metric,
                k=k,
                vote=voting_strategy,
                retention=retention_policy.value,
                fold_id=fold_id,
                num_folds=NUM_SPLITS,
                n_train=train_matrix.shape[0],
                n_test=test_matrix.shape[0],
                fit_time=fit_time,
                predict_time=predict_time,
                total_time=total_time,
                accuracy=acc,
                precision_macro=pM,
                recall_macro=rM,
                f1_macro=fM,
                precision_weighted=pW,
                recall_weighted=rW,
                f1_weighted=fW,
                confusion_matrix=cm_fold,
                labels=labels,
                instance_reduction_method=instance_reduction_strategy,
                memory_before_ir=ibl.cp_before_ir,
                memory_after_ir=ibl.cp_after_ir,
                memory_after_training=ibl.cp_after_training,
                percentage_memory_reduction=(ibl.cp_before_ir - ibl.cp_after_ir) / ibl.cp_before_ir * 100,
            )
        elif model is Models.K_IBL:
            row = create_k_ibl_csv_row(
                metric=distance_metric,
                k=k,
                vote=voting_strategy,
                retention=retention_policy.value,
                fold_id=fold_id,
                num_folds=NUM_SPLITS,
                n_train=train_matrix.shape[0],
                n_test=test_matrix.shape[0],
                fit_time=fit_time,
                predict_time=predict_time,
                total_time=total_time,
                accuracy=acc,
                precision_macro=pM,
                recall_macro=rM,
                f1_macro=fM,
                precision_weighted=pW,
                recall_weighted=rW,
                f1_weighted=fW,
                confusion_matrix=cm_fold,
                labels=labels,
            )
        elif model is Models.SVM:
            row = create_svm_csv_row(
                kernel=svm_kernel,
                C=C,
                gamma=gamma,
                Degree=Degree,
                fold_id=fold_id,
                num_folds=NUM_SPLITS,
                n_train=train_matrix.shape[0],
                n_test=test_matrix.shape[0],
                fit_time=fit_time,
                predict_time=predict_time,
                total_time=total_time,
                accuracy=acc,
                precision_macro=pM,
                recall_macro=rM,
                f1_macro=fM,
                precision_weighted=pW,
                recall_weighted=rW,
                f1_weighted=fW,
                confusion_matrix=cm_fold,
                labels=labels,
            )
        rows.append(row)

    # 4. --- Saving Results ---
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_rows = pd.DataFrame(rows)

    write_header = not out_csv.exists()
    df_rows.to_csv(out_csv, mode="a", header=write_header, index=False)

    print(
        f"\nSuccessfully completed {NUM_SPLITS} folds for {dataset_name}-{model.name.lower()}."
    )
    print(f"Results saved to: {out_csv}")

    # --- Save every 10 folds or at the end ---
    SAVE_EVERY = 10

    if (fold_id + 1) % SAVE_EVERY == 0 or (fold_id + 1) == NUM_SPLITS:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_rows = pd.DataFrame(rows)

        write_header = not out_csv.exists()
        df_rows.to_csv(out_csv, mode="a", header=write_header, index=False)


def main():
    """
    Main function for command-line execution. Parses arguments and calls the
    portable run_experiment function.
    """
    try:
        args = parse_arguments()
    except ValueError as e:
        raise ValueError(f"Argument parsing error: {e}")

    # Map parsed arguments to the portable function's explicit signature
    run_experiment_portable(
        dataset_name=args.dataset_name,
        model=args.model,
        normalization_strategy=args.normalization_strategy,
        encoding_strategy=args.encoding_strategy,
        missing_values_numeric_strategy=args.missing_values_numeric_strategy,
        missing_values_categorical_strategy=args.missing_values_categorical_strategy,
        k=args.k,
        distance_metric=args.distance_metric,
        voting_strategy=args.voting_strategy,
        retention_policy=args.retention_policy,
        svm_kernel=args.svm_kernel,
        C=args.C,
        gamma=args.gamma,
        Degree=args.Degree,
        instance_reduction_strategy=getattr(args, "instance_reduction_strategy", None),
        feature_weighting_strategy=getattr(args, "feature_weighting_strategy", None),
    )
    


if __name__ == "__main__":
    main()

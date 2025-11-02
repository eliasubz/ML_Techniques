from pathlib import Path
import time
import numpy as np
import pandas as pd
from IBL import IBL
from Parser import Parser
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.svm import SVC

from argument_parser import parse_arguments
from csv_writers import create_fw_k_ibl_csv_row, create_k_ibl_csv_row, create_ir_ibl_csv_row, create_svm_csv_row
from model_types import Models


BASE_PATH = "datasetsCBR/datasetsCBR/"
NUM_SPLITS = 10
RESULTS_PATH = "src/results/"


if __name__ == "__main__":
    try:
        args = parse_arguments()
    except ValueError as e:
        raise ValueError(f"Argument parsing error: {e}")

    parser = Parser(
        base_path=BASE_PATH,
        dataset_name=args.dataset_name,
        normalization_strategy=args.normalization_strategy,
        encoding_strategy=args.encoding_strategy,
        missing_values_numeric_strategy=args.missing_values_numeric_strategy,
        missing_values_categorical_strategy=args.missing_values_categorical_strategy,
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

    out_csv = Path(RESULTS_PATH +
                   f"{args.dataset_name}-{args.model}.csv")

    rows = []
    for fold_id, (train_matrix, test_matrix) in enumerate(splits):

        t0 = time.perf_counter()

        if args.model is not Models.SVM:
            ibl = IBL()

            # Fit + predict
            t0 = time.perf_counter()
            
            # --- Instance reduction strategy selection ---
            if args.instance_reduction_strategy is None:
                print("No instance reduction selected.")
                ibl.fit(train_matrix)

            elif args.instance_reduction_strategy == "IBL3":
                print("Applying IBL3 instance reduction...")
                ibl.fit(train_matrix, instance_red="IBL3")

            elif args.instance_reduction_strategy == "IBL3_verbose":
                print("Applying IBL3 (verbose) instance reduction...")
                ibl.fit(train_matrix, instance_red="IBL3_verbose")

            elif args.instance_reduction_strategy == "CNN":
                print("Applying Condensed Nearest Neighbor (CNN) instance reduction...")
                ibl.fit(train_matrix, instance_red="CNN")

            elif args.instance_reduction_strategy == "MCNN":
                print("Applying Modified Condensed Nearest Neighbor (MCNN) instance reduction...")
                ibl.fit(train_matrix, instance_red="MCNN")

            elif args.instance_reduction_strategy.lower() == "enn":
                print("Applying Edited Nearest Neighbor (ENN) instance reduction...")
                ibl.fit(train_matrix, instance_red="enn")

            elif args.instance_reduction_strategy.upper() == "RENN":
                print("Applying Repeated Edited Nearest Neighbor (RENN) instance reduction...")
                ibl.fit(train_matrix, instance_red="RENN")

            else:
                raise ValueError(f"Unknown instance reduction strategy: {args.instance_reduction_strategy}")
        else:   
            # Set up SVM fit
            np_train_matrix = train_matrix.reset_index(drop=True).to_numpy()
            X_train, y_train = np_train_matrix[:, :-1], np_train_matrix[:, -1]

            svm = SVC(kernel=args.svm_kernel, C=args.C, gamma=args.gamma, degree=args.Degree)
            svm.fit(X_train, y_train)

        t1 = time.perf_counter()

        if args.model is Models.K_IBL or args.model is Models.IR_K_IBL:
            preds = ibl.run(
                test_matrix,
                k=args.k,
                metric=args.distance_metric,
                vote=args.voting_strategy,
                retention_policy=args.retention_policy,
                types=types,
            )
        elif args.model is Models.FW_K_IBL:
            preds = ibl.fw_KIBLAlgorithm(
                test_matrix,
                k=args.k,
                metric=args.distance_metric,
                vote=args.voting_strategy,
                retention_policy=args.retention_policy,
                types=types,
                feature_weighting_method=args.feature_weighting_strategy,
                post_encoding_types=post_encoding_types
            )
        elif args.model is Models.SVM:

            X_test = test_matrix.reset_index(drop=True).to_numpy()[:, :-1]
            preds = svm.predict(X_test)
    

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

        if args.model is Models.FW_K_IBL:
            row = create_fw_k_ibl_csv_row(
                fw_method=args.feature_weighting_strategy,
                metric=args.distance_metric,
                k=args.k,
                vote=args.voting_strategy,
                retention=args.retention_policy.value,
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
        elif args.model is Models.IR_K_IBL:
            row = create_ir_ibl_csv_row(
                metric=args.distance_metric,
                k=args.k,
                vote=args.voting_strategy,
                retention=args.retention_policy.value,
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
                instance_reduction_method=args.instance_reduction_strategy,
                memory_before_ir=ibl.cp_before_ir,
                memory_after_ir=ibl.cp_after_ir,
                memory_after_training=ibl.cp_after_training,
                percentage_memory_reduction=(ibl.cp_before_ir - ibl.cp_after_ir) / ibl.cp_before_ir * 100
            )
        elif args.model is Models.K_IBL:
            row = create_k_ibl_csv_row(
                metric=args.distance_metric,
                k=args.k,
                vote=args.voting_strategy,
                retention=args.retention_policy.value,
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
        elif args.model is Models.SVM:
            row = create_svm_csv_row(
                kernel=args.svm_kernel,
                C=args.C,
                gamma=args.gamma,
                Degree=args.Degree,
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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_rows = pd.DataFrame(rows)

    write_header = not out_csv.exists()
    df_rows.to_csv(out_csv, mode="a", header=write_header, index=False)


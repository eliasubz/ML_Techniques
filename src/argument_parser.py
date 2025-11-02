import argparse
from dataclasses import dataclass

from model_types import Models
from processing_types import EncodingStrategy, FeatureWeightingMethod, MissingValuesCategoricalStrategy, MissingValuesNumericStrategy, NormalizationStrategy, RetentionPolicy

VALID_DISTANCE_METRICS = ["euclidean", "cosine", "heom"]
VALID_VOTING_STRATEGIES = ["modified_plurality", "borda"]


@dataclass
class ParsedArguments:
    """Strongly-typed parsed command-line arguments.

    Using a dataclass enables attribute access with IDE autocompletion.
    """
    # Always needed
    dataset_name: str
    model: Models

    # k-IBL
    k: int
    distance_metric: str
    voting_strategy: str
    retention_policy: RetentionPolicy

    feature_weighting_strategy: FeatureWeightingMethod
    instance_reduction_strategy: str

    # SVM
    svm_kernel: str
    C: float
    gamma: float
    degree: int


def parse_arguments() -> ParsedArguments:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="pen-based",
        help="Name of the dataset to use"
    )

    parser.add_argument(
        "--model",
        type=Models,
        default=Models.K_IBL,
        choices=list(Models),
        help=f"Model to use. Valid options"
    )

    parser.add_argument(
        "--k",
        type=int,
        choices=[3, 5, 7],
        help="Number of nearest neighbors (k)"
    )

    parser.add_argument(
        "--distance-metric",
        type=str,
        choices=VALID_DISTANCE_METRICS,
        help="Distance metric to use"
    )

    parser.add_argument(
        "--voting-strategy",
        type=str,
        choices=VALID_VOTING_STRATEGIES,
        help="Voting strategy"
    )

    parser.add_argument(
        "--retention-strategy",
        type=RetentionPolicy,
        choices=list(RetentionPolicy),
        help=f"Retention policy"
    )

    parser.add_argument(
        "--feature-weighting-strategy",
        type=FeatureWeightingMethod,
        choices=list(FeatureWeightingMethod),
        help="Feature weighting strategy"
    )

    parser.add_argument(
        "--instance-reduction-strategy",
        type=str,
        choices=["IBL3", "CNN", "enn"],
        help="Instance Reduction type"
    )

    parser.add_argument(
        "--svm-kernel",
        type=str,
        choices=['rbf', 'poly'],
        help="SVM kernel type"
    )

    parser.add_argument(
        "--C",
        type=float,
        choices=[0.01, 0.1, 1, 10],
        help="SVM kernel type"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        choices=[0.001, 0.01, 0.1],
        help="SVM kernel type"
    )

    parser.add_argument(
        "--degree",
        type=int,
        choices=[2, 3],
        help="SVM kernel type"
    )

    args = parser.parse_args()

    parsed_args = ParsedArguments(
        dataset_name=args.dataset,
        model=args.model,
        k=args.k,
        distance_metric=args.distance_metric,
        voting_strategy=args.voting_strategy,
        retention_policy=args.retention_strategy,
        feature_weighting_strategy=args.feature_weighting_strategy,
        instance_reduction_strategy=args.instance_reduction_strategy,
        svm_kernel=args.svm_kernel,
        C=args.C,
        gamma=args.gamma,
        Degree=args.degree
    )

    if parsed_args.model is Models.SVM:  # Missing parameters for SVM
        if parsed_args.svm_kernel is None:
            parser.error("--svm-kernel is required when --model is 'svm'.")
        if parsed_args.C is None:
            parser.error("--C is required when --model is 'svm'.")
        if parsed_args.gamma is None:
            parser.error("--gamma is required when --model is 'svm'.")
        if parsed_args.degree is None:
            parser.error("--degree is required when --model is 'svm'.")

    # Missing parameters for IBL
    if parsed_args.model in [Models.K_IBL, Models.FW_K_IBL, Models.IR_K_IBL]:
        if parsed_args.k is None:
            parser.error("--k is required when --model is a k-IBL variant.")
        if parsed_args.distance_metric is None:
            parser.error(
                "--distance-metric is required when --model is a k-IBL variant.")
        if parsed_args.voting_strategy is None:
            parser.error(
                "--voting-strategy is required when --model is a k-IBL variant.")
        if parsed_args.retention_policy is None:
            parser.error(
                "--retention-strategy is required when --model is a k-IBL variant.")

        if (parsed_args.model is Models.FW_K_IBL and parsed_args.feature_weighting_strategy is None):
            parser.error(
                "--feature-weighting-strategy is required when --model is 'fw_k_ibl'.")
        if (parsed_args.model is Models.IR_K_IBL and parsed_args.instance_reduction_strategy is None):
            parser.error(
                "--instance-reduction-strategy is required when --model is 'ir_k_ibl'.")

    return parsed_args

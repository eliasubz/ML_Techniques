from IBL import IBL
from Parser import Parser
from processing_types import EncodingStrategy, FeatureWeightingMethod, MissingValuesCategoricalStrategy, MissingValuesNumericStrategy, NormalizationStrategy, RetentionPolicy


if __name__ == "__main__":
    print("\n=== Parsing/Preprocessing Data ... ===")

    parser = Parser(
        base_path="datasetsCBR/datasetsCBR",
        dataset_name="adult",
        normalization_strategy=NormalizationStrategy.MINMAX_SCALING,
        encoding_strategy=EncodingStrategy.ONE_HOT_ENCODE,
        missing_values_numeric_strategy=MissingValuesNumericStrategy.MEDIAN,
        missing_values_categorical_strategy=MissingValuesCategoricalStrategy.MODE
    )

    train_matrix, test_matrix = parser.get_split(0)
    types = parser.get_types()
    post_encoding_types = parser.get_post_encoding_types()

    print("\n=== Testing Feature-Weighted IBL ===")
    ibl_fw_relief = IBL()
    ibl_fw_relief.fit(train_matrix)
    preds_fw_relief = ibl_fw_relief.fw_KIBLAlgorithm(
        test_matrix=test_matrix,
        k=5,
        metric="euclidean",
        vote="modified_plurality",
        retention_policy=RetentionPolicy.DD_RETENTION,
        types=types,
        feature_weighting_method=FeatureWeightingMethod.RELIEFF,
        post_encoding_types=post_encoding_types
    )

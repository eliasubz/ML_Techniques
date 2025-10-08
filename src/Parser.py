
import os
import pandas as pd
from scipy.io import arff

from encoding import encode_data
from missing_values import handle_missing_values
from normalization import normalize_data
from preprocessing_types import (
    NormalizationStrategy, EncodingStrategy, MissingValuesNumericStrategy, MissingValuesCategoricalStrategy
)


class Parser:
    """
    Parser class to load ARFF datasets with multiple splits and apply preprocessing.
    """

    def __init__(self, base_path: str, dataset_name: str,
                 normalization_strategy: NormalizationStrategy = None,
                 encoding_strategy: EncodingStrategy = None,
                 missing_values_numeric_strategy: MissingValuesNumericStrategy = None,
                 missing_values_categorical_strategy: MissingValuesCategoricalStrategy = None,
                 num_splits: int = 10):
        """
        Initialize the Parser with dataset path and preprocessing strategies.

        Parameters:
        - base_path: str, base folder containing datasets
        - dataset_name: str, name of the dataset folder (e.g., 'adult')
        - num_splits: int, number of folds to load
        - normalization_strategy: NormalizationStrategy enum
        - encoding_strategy: EncodingStrategy enum
        - missing_values_numeric_strategy: MissingValuesNumericStrategy enum
        - missing_values_categorical_strategy: MissingValuesCategoricalStrategy enum
        """
        if normalization_strategy is not None and normalization_strategy not in list(NormalizationStrategy):
            raise ValueError(
                f'normalization_strategy must be one of {[s.name for s in NormalizationStrategy]} or None')
        if encoding_strategy is not None and encoding_strategy not in list(EncodingStrategy):
            raise ValueError(
                f'encoding_strategy must be one of {[s.name for s in EncodingStrategy]} or None')
        if missing_values_numeric_strategy is not None and missing_values_numeric_strategy not in list(MissingValuesNumericStrategy):
            raise ValueError(
                f'missing_values_numeric_strategy must be one of {[s.name for s in MissingValuesNumericStrategy]} or None')
        if missing_values_categorical_strategy is not None and missing_values_categorical_strategy not in list(MissingValuesCategoricalStrategy):
            raise ValueError(
                f'missing_values_categorical_strategy must be one of {[s.name for s in MissingValuesCategoricalStrategy]} or None')

        self.base_path = base_path
        self.dataset_name = dataset_name
        self.num_splits = num_splits
        self.normalization_strategy = normalization_strategy
        self.encoding_strategy = encoding_strategy
        self.missing_values_numeric_strategy = missing_values_numeric_strategy
        self.missing_values_categorical_strategy = missing_values_categorical_strategy

        data_splits = self._load_arff_dataset()
        data_splits = self.preprocess(data_splits)
        self.data_splits = data_splits

    def _load_arff_dataset(self):
        data_splits = []
        dataset_path = os.path.join(self.base_path, self.dataset_name)

        for i in range(self.num_splits):
            fold_num = f"{i:06d}"
            train_file = os.path.join(
                dataset_path, f"{self.dataset_name}.fold.{fold_num}.train.arff")
            test_file = os.path.join(
                dataset_path, f"{self.dataset_name}.fold.{fold_num}.test.arff")

            # Load ARFF files
            train_data, _ = arff.loadarff(train_file)
            test_data, _ = arff.loadarff(test_file)

            # Convert to pandas DataFrame
            train_matrix = pd.DataFrame(train_data)
            test_matrix = pd.DataFrame(test_data)

            data_splits.append((train_matrix, test_matrix))

        return data_splits

    def preprocess(self, data_splits):
        """Apply preprocessing steps to all train and test DataFrames using the configured strategies."""
        processed_splits = []
        for train_df, test_df in data_splits:
            # Normalization
            if self.normalization_strategy is not None:
                train_df, test_df = normalize_data(
                    train_df, test_df, self.normalization_strategy)

            # Encoding
            if self.encoding_strategy is not None:
                train_df, test_df, _ = encode_data(
                    train_df, test_df, self.encoding_strategy)

            # Handle Missing Values
            if self.missing_values_numeric_strategy is not None and self.missing_values_categorical_strategy is not None:
                train_df, test_df = handle_missing_values(
                    train_df, test_df, self.missing_values_numeric_strategy.value, self.missing_values_categorical_strategy.value)

            processed_splits.append((train_df, test_df))

        return processed_splits

    def get_split(self, index: int):
        """Return train and test DataFrame for a given split index."""
        return self.data_splits[index]


if __name__ == "__main__":
    # Example usage
    parser = Parser(
        base_path="datasetsCBR/datasetsCBR",
        dataset_name="adult",
        normalization_strategy=NormalizationStrategy.STANDARDIZE,
        encoding_strategy=EncodingStrategy.ONE_HOT_ENCODE,
        missing_values_numeric_strategy=MissingValuesNumericStrategy.DROP,
        missing_values_categorical_strategy=MissingValuesCategoricalStrategy.MODE
    )

    train, test = parser.get_split(0)
    print(train.head())
    print(list(test.columns.values))

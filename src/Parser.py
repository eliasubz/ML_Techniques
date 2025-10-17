
import os
import pandas as pd
from scipy.io import arff
import numpy as np
from encoding import encode_data
from missing_values import handle_missing_values
from normalization import normalize_data
from processing_types import (
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
                 num_splits: int = 10,
                 faster_parser: bool = False):
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
        if faster_parser: 
            data_splits = self._load_arff_dataset(True)
        else: 
            data_splits = self._load_arff_dataset()
        self.types = self._save_feature_types(data_splits)
        data_splits = self.preprocess(data_splits)
        self.data_splits = data_splits

    def _load_arff_dataset(self, faster_parser: bool = False):
        data_splits = []
        dataset_path = os.path.join(self.base_path, self.dataset_name)

        # Return only 1 split and skip the rest
        if faster_parser: 
            i = 0
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

            train_matrix = self._decode_arff_df(train_matrix)
            test_matrix = self._decode_arff_df(test_matrix)

            data_splits.append((train_matrix, test_matrix))
            return data_splits
            

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

            train_matrix = self._decode_arff_df(train_matrix)
            test_matrix = self._decode_arff_df(test_matrix)

            data_splits.append((train_matrix, test_matrix))

        return data_splits

    @staticmethod
    def _decode_arff_df(df: pd.DataFrame) -> pd.DataFrame:
        """Decode byte strings to str and normalize ARFF missing markers to NaN."""
        out = df.copy()

        # decode bytes/bytearray -> str
        for c in out.columns:
            col = out[c]
            if col.dtype == object:
                out[c] = col.apply(
                    lambda v: v.decode(
                        "utf-8") if isinstance(v, (bytes, bytearray)) else v
                )

        # normalize common ARFF missing tokens to NaN
        out.replace({b'?': np.nan, '?': np.nan, ' ?': np.nan,
                    '? ': np.nan}, inplace=True)

        # whitespace around strings
        for c in out.columns:
            if out[c].dtype == object:
                out[c] = out[c].astype(str).str.strip()

        return out

    @staticmethod
    def _save_feature_types(data_splits: pd.DataFrame):
        """
        Save per-column types from TRAIN only, before any encoding.
        Returns (types).
        """
        train_df = data_splits[0][0]  # First train fold
        target_col = train_df.columns[-1]

        X_train = train_df.drop(columns=[target_col]).reset_index(drop=True)

        types = ["numeric" if pd.api.types.is_numeric_dtype(X_train.dtypes[c])
                 else "categorical"
                 for c in X_train.columns]

        return types

    def preprocess(self, data_splits):
        """Apply preprocessing steps to all train and test DataFrames using the configured strategies."""
        processed_splits = []
        for train_df, test_df in data_splits:

            target_col = train_df.columns[-1]

            y_train = train_df[target_col].reset_index(drop=True)
            y_test = test_df[target_col].reset_index(drop=True)

            X_train = train_df.drop(
                columns=[target_col]).reset_index(drop=True)
            X_test = test_df.drop(columns=[target_col]).reset_index(drop=True)

            # Handle Missing Values
            if self.missing_values_numeric_strategy is not None and self.missing_values_categorical_strategy is not None:
                X_train, X_test = handle_missing_values(
                    X_train, X_test, self.missing_values_numeric_strategy.value, self.missing_values_categorical_strategy.value)

            # Normalization
            if self.normalization_strategy is not None:
                X_train, X_test = normalize_data(
                    X_train, X_test, self.normalization_strategy)

            # Encoding
            if self.encoding_strategy is not None:
                X_train, X_test, _ = encode_data(
                    X_train, X_test, self.encoding_strategy)

            train_out = pd.concat([X_train.reset_index(drop=True),
                                   y_train.rename(target_col)], axis=1)
            test_out = pd.concat([X_test.reset_index(drop=True),
                                  y_test.rename(target_col)], axis=1)

            processed_splits.append((train_out, test_out))

        return processed_splits

    def get_split(self, index: int, reduced: float = 1.0):
        """
        Return (train_df, test_df) for a given split index.
        
        If `reduced` < 1, return a random subset of each matrix 
        of size = reduced * len(train/test).
        """
        train_df, test_df = self.data_splits[index]

        if 0 < reduced < 1:
            # Randomly sample a subset
            train_df = train_df.sample(frac=reduced, random_state=42)
            test_df = test_df.sample(frac=reduced, random_state=42)

        return train_df, test_df

    def get_types(self):
        return self.types

if __name__ == "__main__":
    # Example usage
    parser = Parser(
        base_path="datasetsCBR/datasetsCBR",
        dataset_name="adult",
        normalization_strategy=NormalizationStrategy.STANDARDIZE,
        encoding_strategy=EncodingStrategy.LABEL_ENCODE,
        missing_values_numeric_strategy=MissingValuesNumericStrategy.DROP,
        missing_values_categorical_strategy=MissingValuesCategoricalStrategy.MODE
    )

    train, test = parser.get_split(0)
    types = parser.get_types()  # Only used for OHE
    print(types)
    print(train.head())
    print(list(test.columns.values))


    # Testing fast parser
    parser = Parser(
        base_path="datasetsCBR/datasetsCBR",
        dataset_name="adult",
        normalization_strategy=NormalizationStrategy.STANDARDIZE,
        encoding_strategy=EncodingStrategy.LABEL_ENCODE,
        missing_values_numeric_strategy=MissingValuesNumericStrategy.DROP,
        missing_values_categorical_strategy=MissingValuesCategoricalStrategy.MODE,
        faster_parser=True,
    )

    train, test = parser.get_split(0)
    types = parser.get_types()  # Only used for OHE
    print(types)
    print(train.head())
    print(list(test.columns.values))

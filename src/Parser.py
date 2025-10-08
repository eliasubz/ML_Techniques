import os
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import (
    scale, StandardScaler, MinMaxScaler, Normalizer,
    LabelEncoder, OneHotEncoder
)
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor


class Parser:
    def __init__(self, base_path: str, dataset_name: str, num_splits: int = 10):
        """
        Parser class to load ARFF datasets with multiple splits and apply preprocessing.

        Parameters:
        - base_path: str, base folder containing datasets
        - dataset_name: str, name of the dataset folder (e.g., 'adult')
        - num_splits: int, number of folds to load
        """
        self.base_path = base_path
        self.dataset_name = dataset_name
        self.num_splits = num_splits
        self.data_splits = self._load_arff_dataset()

    def _load_arff_dataset(self):
        data_splits = []
        dataset_path = os.path.join(self.base_path, self.dataset_name)

        for i in range(self.num_splits):
            fold_num = f"{i:06d}"
            train_file = os.path.join(dataset_path, f"{self.dataset_name}.fold.{fold_num}.train.arff")
            test_file = os.path.join(dataset_path, f"{self.dataset_name}.fold.{fold_num}.test.arff")

            # Load ARFF files
            train_data, _ = arff.loadarff(train_file)
            test_data, _ = arff.loadarff(test_file)

            # Convert to pandas DataFrame
            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)

            data_splits.append((train_df, test_df))

        return data_splits

    def get_split(self, index: int):
        """Return train and test DataFrame for a given split index."""
        return self.data_splits[index]

    def get_numeric_columns(self, df: pd.DataFrame):
        """Return numeric columns from a DataFrame."""
        return df.select_dtypes(include=["number"]).columns

    def get_categorical_columns(self, df: pd.DataFrame):
        """Return categorical columns from a DataFrame."""
        return df.select_dtypes(include=["object", "category"]).columns

    def standardize(self, train_df, test_df):
        """Z-score standardization."""
        numeric_cols = self.get_numeric_columns(train_df)
        return scale(train_df[numeric_cols]), scale(test_df[numeric_cols])

    def mean_normalize(self, train_df, test_df):
        """Mean normalization (values centered around 0)."""
        numeric_cols = self.get_numeric_columns(train_df)
        scaler = StandardScaler(with_mean=True, with_std=False)
        return scaler.fit_transform(train_df[numeric_cols]), scaler.transform(test_df[numeric_cols])

    def minmax_scale(self, train_df, test_df):
        """Min-Max scaling (0 to 1)."""
        numeric_cols = self.get_numeric_columns(train_df)
        scaler = MinMaxScaler()
        return scaler.fit_transform(train_df[numeric_cols]), scaler.transform(test_df[numeric_cols])

    def unit_normalize(self, train_df, test_df):
        """Unit vector normalization."""
        numeric_cols = self.get_numeric_columns(train_df)
        scaler = Normalizer()
        return scaler.fit_transform(train_df[numeric_cols]), scaler.transform(test_df[numeric_cols])

    def label_encode(self, train_df, test_df):
        """Apply label encoding to categorical columns."""
        cat_cols = self.get_categorical_columns(train_df)
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            train_df[col] = le.fit_transform(train_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))
            encoders[col] = le
        return train_df, test_df, encoders

    def one_hot_encode(self, train_df, test_df):
        """Apply one-hot encoding to categorical columns."""
        cat_cols = self.get_categorical_columns(train_df)
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoder.fit(train_df[cat_cols])

        train_encoded = encoder.transform(train_df[cat_cols])
        test_encoded = encoder.transform(test_df[cat_cols])

        train_encoded_df = pd.DataFrame(train_encoded, columns=encoder.get_feature_names_out(cat_cols))
        test_encoded_df = pd.DataFrame(test_encoded, columns=encoder.get_feature_names_out(cat_cols))

        # Drop categorical columns and concatenate encoded ones
        train_df = train_df.drop(columns=cat_cols).reset_index(drop=True)
        test_df = test_df.drop(columns=cat_cols).reset_index(drop=True)

        train_df = pd.concat([train_df.reset_index(drop=True), train_encoded_df.reset_index(drop=True)], axis=1)
        test_df = pd.concat([test_df.reset_index(drop=True), test_encoded_df.reset_index(drop=True)], axis=1)

        return train_df, test_df, encoder

    def handle_missing_values(self, train_df, test_df, strategy_num="mean", strategy_cat="mode"):
        """
        Handle missing values for numerical and categorical columns.

        Parameters:
        - strategy_num: str, strategy for numerical columns ("mean", "median", "zero", "drop", or "model")
        - strategy_cat: str, strategy for categorical columns ("mode", "constant", or "drop")
        """
        train_df = train_df.copy()
        test_df = test_df.copy()

        # Handle numeric columns
        numeric_cols = self.get_numeric_columns(train_df)
        if strategy_num == "drop":
            train_df = train_df.dropna(subset=numeric_cols)
            test_df = test_df.dropna(subset=numeric_cols)
        elif strategy_num == "model":
            imputer = IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=10), random_state=42)
            train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
            test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])
        else:
            for col in numeric_cols:
                if strategy_num == "mean":
                    fill_value = train_df[col].mean()
                elif strategy_num == "median":
                    fill_value = train_df[col].median()
                elif strategy_num == "zero":
                    fill_value = 0
                else:
                    raise ValueError("Invalid strategy_num. Use 'mean', 'median', 'zero', 'drop', or 'model'.")
                train_df[col] = train_df[col].fillna(fill_value)
                test_df[col] = test_df[col].fillna(fill_value)

        # Handle categorical columns
        cat_cols = self.get_categorical_columns(train_df)
        if strategy_cat == "drop":
            train_df = train_df.dropna(subset=cat_cols)
            test_df = test_df.dropna(subset=cat_cols)
        else:
            for col in cat_cols:
                if strategy_cat == "mode":
                    fill_value = train_df[col].mode()[0]
                elif strategy_cat == "constant":
                    fill_value = "missing"
                else:
                    raise ValueError("Invalid strategy_cat. Use 'mode', 'constant', or 'drop'.")
                train_df[col] = train_df[col].fillna(fill_value)
                test_df[col] = test_df[col].fillna(fill_value)

        return train_df, test_df


# Example usage
if __name__ == "__main__":
    base_path = "datasetsCBR/datasetsCBR"
    dataset_name = "adult"

    parser = Parser(base_path, dataset_name, num_splits=5)

    # Get first split
    train_df, test_df = parser.get_split(0)
    print(train_df.head())

    # Handle missing values using model-based imputation
    train_filled, test_filled = parser.handle_missing_values(train_df, test_df, strategy_num="model", strategy_cat="mode")

    # Apply transformations
    train_std, test_std = parser.standardize(train_filled, test_filled)
    train_mean, test_mean = parser.mean_normalize(train_filled, test_filled)
    train_minmax, test_minmax = parser.minmax_scale(train_filled, test_filled)
    train_unit, test_unit = parser.unit_normalize(train_filled, test_filled)
    train_le, test_le, encoders = parser.label_encode(train_filled.copy(), test_filled.copy())
    train_ohe, test_ohe, ohe = parser.one_hot_encode(train_filled.copy(), test_filled.copy())

    print(train_ohe.head())

# 
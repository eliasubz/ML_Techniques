import pandas as pd
from sklearn.preprocessing import (
    scale, StandardScaler, MinMaxScaler, Normalizer,
)

from processing_types import NormalizationStrategy
from utils import get_numeric_columns


def standardize(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Z-score standardization."""
    numeric_cols = get_numeric_columns(train_df)

    if numeric_cols is None or len(numeric_cols) == 0:
        return train_df, test_df

    train_scaled = pd.DataFrame(
        scale(train_df[numeric_cols]),
        columns=numeric_cols,
        index=train_df.index
    )
    test_scaled = pd.DataFrame(
        scale(test_df[numeric_cols]),
        columns=numeric_cols,
        index=test_df.index
    )
    # Replace only the numeric columns in the original DataFrames
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[numeric_cols] = train_scaled
    test_df_scaled[numeric_cols] = test_scaled
    return train_df_scaled, test_df_scaled


def mean_normalize(train_df, test_df):
    """Mean normalization (values centered around 0)."""
    numeric_cols = get_numeric_columns(train_df)
    if numeric_cols is None or len(numeric_cols) == 0:
        return train_df, test_df

    scaler = StandardScaler(with_mean=True, with_std=False)
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df[numeric_cols]),
        columns=numeric_cols,
        index=train_df.index
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df[numeric_cols]),
        columns=numeric_cols,
        index=test_df.index
    )
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[numeric_cols] = train_scaled
    test_df_scaled[numeric_cols] = test_scaled
    return train_df_scaled, test_df_scaled


def minmax_scale(train_df, test_df):
    """Min-Max scaling (0 to 1)."""
    numeric_cols = get_numeric_columns(train_df)

    if numeric_cols is None or len(numeric_cols) == 0:
        return train_df, test_df

    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df[numeric_cols]),
        columns=numeric_cols,
        index=train_df.index
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df[numeric_cols]),
        columns=numeric_cols,
        index=test_df.index
    )
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[numeric_cols] = train_scaled
    test_df_scaled[numeric_cols] = test_scaled
    return train_df_scaled, test_df_scaled


def unit_normalize(train_df, test_df):
    """Unit vector normalization."""
    numeric_cols = get_numeric_columns(train_df)

    if numeric_cols is None or len(numeric_cols) == 0:
        return train_df, test_df

    scaler = Normalizer()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df[numeric_cols]),
        columns=numeric_cols,
        index=train_df.index
    )
    test_scaled = pd.DataFrame(
        scaler.transform(test_df[numeric_cols]),
        columns=numeric_cols,
        index=test_df.index
    )
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[numeric_cols] = train_scaled
    test_df_scaled[numeric_cols] = test_scaled
    return train_df_scaled, test_df_scaled


def normalize_data(train_df: pd.DataFrame, test_df: pd.DataFrame, strategy: NormalizationStrategy):
    """Normalize numeric columns in train and test DataFrames based on the specified strategy."""
    if strategy == NormalizationStrategy.STANDARDIZE:
        train_df, test_df = standardize(train_df, test_df)
    elif strategy == NormalizationStrategy.MEAN_NORMALIZE:
        train_df, test_df = mean_normalize(train_df, test_df)
    elif strategy == NormalizationStrategy.MINMAX_SCALING:
        train_df, test_df = minmax_scale(train_df, test_df)
    elif strategy == NormalizationStrategy.UNIT_VECTOR:
        train_df, test_df = unit_normalize(train_df, test_df)
    else:
        raise ValueError(
            f"Unknown normalization strategy: {strategy}. Choose from {list(NormalizationStrategy)}")

    return train_df, test_df

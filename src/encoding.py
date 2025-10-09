import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder
)
from preprocessing_types import EncodingStrategy
from utils import get_categorical_columns


def label_encode(train_df, test_df):
    """Apply label encoding to categorical columns."""
    cat_cols = get_categorical_columns(train_df)
    encoders = {}
    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        train_df_encoded[col] = le.fit_transform(
            train_df_encoded[col].astype(str))
        # Map test values: unseen labels become -1
        test_vals = test_df_encoded[col].astype(str)
        test_df_encoded[col] = test_vals.map(
            lambda s: le.transform([s])[0] if s in le.classes_ else -1
        )
        encoders[col] = le
    return train_df_encoded, test_df_encoded, encoders


def one_hot_encode(train_df, test_df):
    """Apply one-hot encoding to categorical columns."""
    cat_cols = get_categorical_columns(train_df)
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(train_df[cat_cols])

    train_encoded = encoder.transform(train_df[cat_cols])
    test_encoded = encoder.transform(test_df[cat_cols])

    train_encoded_df = pd.DataFrame(
        train_encoded, columns=encoder.get_feature_names_out(cat_cols), index=train_df.index)
    test_encoded_df = pd.DataFrame(
        test_encoded, columns=encoder.get_feature_names_out(cat_cols), index=test_df.index)

    # Drop categorical columns and concatenate encoded ones, preserving index
    train_df_encoded = train_df.drop(columns=cat_cols).copy()
    test_df_encoded = test_df.drop(columns=cat_cols).copy()
    train_df_encoded = pd.concat([train_df_encoded, train_encoded_df], axis=1)
    test_df_encoded = pd.concat([test_df_encoded, test_encoded_df], axis=1)

    return train_df_encoded, test_df_encoded, encoder


def encode_data(train_df, test_df, strategy=EncodingStrategy.LABEL_ENCODE):
    """Encode categorical columns in train and test DataFrames based on the specified strategy."""
    if strategy == EncodingStrategy.LABEL_ENCODE:
        train_df, test_df, encoders = label_encode(train_df, test_df)
    elif strategy == EncodingStrategy.ONE_HOT_ENCODE:
        train_df, test_df, encoders = one_hot_encode(train_df, test_df)
    else:
        raise ValueError(
            f"Unknown encoding strategy: {strategy}. Choose from {list(EncodingStrategy)}")

    return train_df, test_df, encoders

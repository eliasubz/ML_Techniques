import pandas as pd


def get_numeric_columns(df: pd.DataFrame):
    """Return numeric columns from a DataFrame."""
    return df.select_dtypes(include=["number"]).columns


def get_categorical_columns(df: pd.DataFrame):
    """Return categorical columns from a DataFrame."""
    return df.select_dtypes(include=["object", "category"]).columns

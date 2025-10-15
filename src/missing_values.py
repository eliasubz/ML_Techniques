from processing_types import MissingValuesCategoricalStrategy, MissingValuesNumericStrategy
from utils import get_categorical_columns, get_numeric_columns
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor


def handle_missing_values(train_df, test_df, strategy_num: str, strategy_cat: str):
    """
    Handle missing values for numerical and categorical columns.

    Parameters:
    - strategy_num: str, strategy for numerical columns ("mean", "median", "zero", "drop", or "model")
    - strategy_cat: str, strategy for categorical columns ("mode", "constant", or "drop")
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Handle numeric columns
    numeric_cols = get_numeric_columns(train_df)
    if strategy_num == "drop":
        train_df = train_df.dropna(subset=numeric_cols)
        test_df = test_df.dropna(subset=numeric_cols)
    elif strategy_num == "model":
        imputer = IterativeImputer(estimator=ExtraTreesRegressor(
            n_estimators=10), random_state=42)
        train_df[numeric_cols] = imputer.fit_transform(
            train_df[numeric_cols])
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
                raise ValueError(
                    "Invalid strategy_num. Use 'mean', 'median', 'zero', 'drop', or 'model'.")
            train_df[col] = train_df[col].fillna(fill_value)
            test_df[col] = test_df[col].fillna(fill_value)

    # Handle categorical columns
    cat_cols = get_categorical_columns(train_df)
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
                raise ValueError(
                    "Invalid strategy_cat. Use 'mode', 'constant', or 'drop'.")
            train_df[col] = train_df[col].fillna(fill_value)
            test_df[col] = test_df[col].fillna(fill_value)

    return train_df, test_df

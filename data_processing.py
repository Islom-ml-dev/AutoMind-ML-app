import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


# ============================================================
# Data loading & summary
# ============================================================

def load_data(file) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a pandas DataFrame.
    """
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def get_data_summary(df: pd.DataFrame):
    """
    Separate numerical and categorical data and provide basic summaries.

    Returns:
        numeric_summary: describe() for numeric columns (transposed)
        categorical_summary: describe() for non-numeric columns (transposed)
        numeric_cols: Index of numeric columns
        categorical_cols: Index of categorical columns
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    numeric_summary = (
        df[numeric_cols].describe().T if len(numeric_cols) > 0 else pd.DataFrame()
    )
    categorical_summary = (
        df[categorical_cols].describe().T
        if len(categorical_cols) > 0
        else pd.DataFrame()
    )

    return numeric_summary, categorical_summary, numeric_cols, categorical_cols


# ============================================================
# Missing values
# ============================================================

def detect_missing_values(df: pd.DataFrame):
    """
    Detect columns in the DataFrame that contain missing values.

    Returns:
        List of column names that have at least one missing value.
    """
    return df.columns[df.isnull().any()].tolist()


def handle_missing_values(df: pd.DataFrame, strategy_dict: dict) -> pd.DataFrame:
    """
    Handle missing values based on a strategy dictionary.

    strategy_dict: {column_name: strategy}
        strategy can be:
            - "Mean"
            - "Median"
            - "Mode"
            - "Drop"
            - any constant value (int/float/str)
    """
    df_copy = df.copy()

    for col, strategy in strategy_dict.items():
        if col not in df_copy.columns:
            continue

        col_is_numeric = pd.api.types.is_numeric_dtype(df_copy[col])

        # Drop strategy (drop rows where this column is NaN)
        if strategy == "Drop":
            df_copy = df_copy.dropna(subset=[col])
            continue

        # Decide imputer for this column
        if strategy == "Mean":
            if col_is_numeric:
                imp = SimpleImputer(strategy="mean")
            else:
                imp = SimpleImputer(strategy="most_frequent")

        elif strategy == "Median":
            if col_is_numeric:
                imp = SimpleImputer(strategy="median")
            else:
                imp = SimpleImputer(strategy="most_frequent")

        elif strategy == "Mode":
            imp = SimpleImputer(strategy="most_frequent")

        elif isinstance(strategy, (int, float, str)):
            imp = SimpleImputer(strategy="constant", fill_value=strategy)

        else:
            # Fallback: median for numeric, most_frequent for categorical
            if col_is_numeric:
                imp = SimpleImputer(strategy="median")
            else:
                imp = SimpleImputer(strategy="most_frequent")

        df_copy[col] = imp.fit_transform(df_copy[[col]]).ravel()

    return df_copy


# ============================================================
# Encoding (helper for correlation / EDA)
# ============================================================

def encode_categorical(df: pd.DataFrame, columns=None):
    """
    Encode categorical columns using LabelEncoder (for correlation or clustering).

    Returns:
        encoded_df: DataFrame with encoded columns
        label_encoder_dict: {column_name: LabelEncoder object}
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.select_dtypes(exclude=[np.number]).columns

    le_dict = {}
    for col in columns:
        le = LabelEncoder()
        df_copy[col] = le.fit_transform(df_copy[col].astype(str))
        le_dict[col] = le

    return df_copy, le_dict


# ============================================================
# Internal helpers for prepare_data
# ============================================================

def _split_num_cat(X: pd.DataFrame):
    """Split feature dataframe into numeric and categorical column lists."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return num_cols, cat_cols


def _impute_features(X: pd.DataFrame, num_cols, cat_cols) -> pd.DataFrame:
    """
    Impute numeric columns with median, categorical with most_frequent.
    Returns a new DataFrame (no in-place modification).
    """
    X = X.copy()

    if num_cols:
        imp_num = SimpleImputer(strategy="median")
        X[num_cols] = imp_num.fit_transform(X[num_cols])

    if cat_cols:
        imp_cat = SimpleImputer(strategy="most_frequent")
        X[cat_cols] = imp_cat.fit_transform(X[cat_cols])

    return X


def _apply_feature_encoding(
    X: pd.DataFrame,
    cat_cols,
    encoding_method: str | None,
) -> pd.DataFrame:
    """
    Apply feature encoding according to encoding_method.

    encoding_method: "onehot", "label", "none"
    """
    method = (encoding_method or "onehot").lower()

    # No categorical columns → nothing to do
    if not cat_cols:
        return X

    # One-Hot Encoding
    if method == "onehot":
        return pd.get_dummies(X, columns=cat_cols, drop_first=False)

    # Label Encoding
    if method == "label":
        X = X.copy()
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        return X

    # Leave as-is
    if method in ("none", "no", "raw"):
        return X

    # Fallback: unknown method → default to one-hot
    return pd.get_dummies(X, columns=cat_cols, drop_first=False)


def _prepare_target(
    df: pd.DataFrame,
    target_col: str,
    task_type: str | None,
):
    """
    Prepare target column:
        - Impute missing values
        - Encode for classification if needed
        - Cast to float for regression if numeric
    """
    y = df[target_col].copy()

    # Impute missing values in target
    if y.isnull().any():
        if pd.api.types.is_numeric_dtype(y):
            imp_y = SimpleImputer(strategy="mean")
        else:
            imp_y = SimpleImputer(strategy="most_frequent")
        y = pd.Series(
            imp_y.fit_transform(y.values.reshape(1, -1)).ravel(),
            index=y.index,
            name=target_col,
        )

    # Classification → encode non-numeric target
    if task_type == "Classification":
        if not pd.api.types.is_numeric_dtype(y):
            le_y = LabelEncoder()
            y = pd.Series(
                le_y.fit_transform(y.astype(str)),
                index=y.index,
                name=target_col,
            )

    # Regression → ensure float dtype where applicable
    if task_type == "Regression":
        if pd.api.types.is_numeric_dtype(y):
            y = y.astype(float)

    return y


# ============================================================
# Main data preparation for ML
# ============================================================

def prepare_data(
    df: pd.DataFrame,
    feature_cols,
    target_col: str | None = None,
    encoding_method: str = "onehot",
    task_type: str | None = None,
):
    """
    Prepare data for modeling.

    Steps:
        - Subset features
        - Impute missing values for X
        - Encode categorical features (One-Hot / Label / None)
        - Optionally impute and encode target y (for supervised tasks)

    Args:
        df: Input DataFrame.
        feature_cols: List of feature column names.
        target_col: Name of target column, or None (for unsupervised).
        encoding_method: "onehot", "label", or "none".
        task_type: "Regression" or "Classification" (affects target encoding).

    Returns:
        If target_col is not None:
            X (DataFrame), y (Series)
        Else:
            X (DataFrame)
    """
    # Work on a copy of the feature subset to avoid modifying original df
    X = df[feature_cols].copy()

    # 1) Split numeric / categorical
    num_cols, cat_cols = _split_num_cat(X)

    # 2) Impute missing values
    X = _impute_features(X, num_cols, cat_cols)

    # 3) Apply encoding
    X = _apply_feature_encoding(X, cat_cols, encoding_method)

    # 4) Prepare target if needed
    if target_col is not None:
        y = _prepare_target(df, target_col, task_type)
        return X, y

    # Unsupervised case: only X
    return X


# ============================================================
# Scaling, splitting, PCA, task detection
# ============================================================

def normalize_data(X, method: str | None = None):
    """
    Apply selected normalization method to the features.

    method: "Standard", "MinMax", "Robust", or None.
    Returns a numpy array.
    """
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X

    if method is None:
        return X_values

    if method == "Standard":
        scaler = StandardScaler()
    elif method == "MinMax":
        scaler = MinMaxScaler()
    elif method == "Robust":
        scaler = RobustScaler()
    else:
        # Unknown method → no scaling
        return X_values

    return scaler.fit_transform(X_values)


def split_and_scale(
    X,
    y=None,
    test_size: float = 0.2,
    normalization: str | None = None,
):
    """
    If y is provided (supervised):
        - Split into train/test
        - Scale X
        - Return X_train, X_test, y_train, y_test

    If y is None (unsupervised):
        - Scale X only
        - Return scaled X
    """
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        X_train_scaled = normalize_data(X_train, normalization)
        X_test_scaled = normalize_data(X_test, normalization)
        return X_train_scaled, X_test_scaled, y_train, y_test

    # Unsupervised case
    X_scaled = normalize_data(X, normalization)
    return X_scaled


def detect_task_type(y: pd.Series, threshold_unique: int = 15) -> str:
    """
    Detect task type (Regression or Classification) from target series.

    Logic:
        - If dtype is object, category, or bool → Classification
        - Else if numeric and unique values > threshold_unique → Regression
        - Else → Classification
    """
    if (
        y.dtype == "O"
        or str(y.dtype).startswith("category")
        or y.dtype == "bool"
    ):
        return "Classification"

    n_unique = y.nunique()

    if pd.api.types.is_numeric_dtype(y) and n_unique > threshold_unique:
        return "Regression"

    return "Classification"


def apply_pca(X, n_components: int):
    """
    Apply PCA to reduce dimensionality.

    Args:
        X: 2D array-like (DataFrame or numpy array)
        n_components: number of principal components

    Returns:
        X_pca: transformed array
        pca: fitted PCA object
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

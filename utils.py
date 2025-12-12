from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    classification_report,
)


def _to_1d_array(y):
    """
    Helper to safely convert targets/predictions to a 1D numpy array.
    This avoids unexpected shape issues (e.g., (n, 1) vs (n,)).
    """
    return np.asarray(y).ravel()


def calculate_metrics(task_type: str, y_true, y_pred) -> Dict[str, object]:
    """
    Calculate and return appropriate metrics for regression or classification.

    Args:
        task_type: "Regression" or "Classification".
        y_true: Ground truth target values.
        y_pred: Model predictions.

    Returns:
        Dictionary of metrics. For example:
            Regression:
                {
                    "Mean Squared Error": float,
                    "Root Mean Squared Error": float,
                    "Mean Absolute Error": float,
                    "R²": float,
                }
            Classification:
                {
                    "Accuracy": float,
                    "Classification Report": str,
                }
    """
    y_true = _to_1d_array(y_true)
    y_pred = _to_1d_array(y_pred)

    if task_type == "Regression":
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        return {
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "Mean Absolute Error": mae,
            "R²": r2,
        }

    # Classification
    acc = accuracy_score(y_true, y_pred)
    cls_report = classification_report(y_true, y_pred, zero_division=0)

    return {
        "Accuracy": acc,
        "Classification Report": cls_report,
    }


def compute_outlier_stats_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute outlier statistics for each numeric column using the IQR method.

    For each numeric feature:
        - Q1, Q3, IQR
        - Lower / upper bounds
        - Outlier count and percentage

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame sorted by "Outlier %" in descending order.
        Columns:
            ["feature", "Q1", "Q3", "IQR",
             "Lower Bound", "Upper Bound",
             "Outlier Count", "Outlier %"]
        If no numeric columns → returns empty DataFrame.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    rows = []
    for col in num_cols:
        s = df[col].dropna()
        if s.empty:
            continue

        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            # All values are (almost) identical → no outliers by IQR definition
            outlier_count = 0
            outlier_pct = 0.0
            lower = q1
            upper = q3
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask_out = (s < lower) | (s > upper)
            outlier_count = int(mask_out.sum())
            outlier_pct = (outlier_count / len(s)) * 100.0

        rows.append(
            {
                "feature": col,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Lower Bound": lower,
                "Upper Bound": upper,
                "Outlier Count": outlier_count,
                "Outlier %": outlier_pct,
            }
        )

    if not rows:
        return pd.DataFrame()

    stats_df = pd.DataFrame(rows).sort_values("Outlier %", ascending=False)
    return stats_df.reset_index(drop=True)


def interpret_regression_metrics(y_true, y_pred) -> List[str]:
    """
    Provide simple, human-readable interpretation for regression metrics.

    Args:
        y_true: Ground truth target values.
        y_pred: Model predictions.

    Returns:
        List of interpretation strings (suitable for display in Streamlit or PDF).
    """
    y_true = _to_1d_array(y_true)
    y_pred = _to_1d_array(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    interpretations: List[str] = []

    # RMSE interpretation
    interpretations.append(
        f"RMSE = {rmse:.2f}: on average, the model's predictions differ from the true values by about {rmse:.2f} units."
    )

    # MAE interpretation
    interpretations.append(
        f"MAE = {mae:.2f}: on average, the absolute error of the predictions is {mae:.2f} units."
    )

    # R² interpretation
    if r2 >= 0.8:
        interpretations.append(
            "R² is very good — the model explains a large portion of the variance in the target."
        )
    elif r2 >= 0.5:
        interpretations.append(
            "R² is moderate — the model explains some of the variance, but there is room for improvement."
        )
    elif r2 >= 0.2:
        interpretations.append(
            "R² is low — the model is not explaining much of the variance; more feature engineering or a different model may be needed."
        )
    else:
        interpretations.append(
            "R² is very low — the model fits the data poorly or the dataset is highly noisy."
        )

    return interpretations

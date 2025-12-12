import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from data_processing import encode_categorical


# =========================================================
# 1. CORRELATION MATRICES
# =========================================================

def plot_correlation_matrix_supervised(df, feature_cols, target_col, st):
    """
    Plot a correlation matrix between selected feature columns and the target.
    Categorical columns are automatically encoded.
    """
    # Ensure valid columns
    corr_cols = [c for c in feature_cols + [target_col] if c in df.columns]
    if len(corr_cols) < 2:
        st.info("At least 2 columns are required to compute a correlation matrix.")
        return

    df_subset = df[corr_cols]

    # Encode categorical features
    categorical_cols = df_subset.select_dtypes(exclude=[np.number]).columns.tolist()
    if categorical_cols:
        df_subset, _ = encode_categorical(df_subset, categorical_cols)

    corr_df = df_subset.corr()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Correlation Matrix (Features + Target)")
    st.pyplot(fig)


def plot_correlation_matrix_eda(df, cols, st):
    """
    Plot a correlation matrix for numeric columns selected in EDA mode.
    """
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        st.info("Select at least 2 numeric columns.")
        return

    df_num = df[cols].select_dtypes(include=[np.number])

    if df_num.shape[1] < 2:
        st.info("Selected columns do not include enough numeric columns.")
        return

    corr = df_num.corr()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Correlation Matrix (Numeric Columns)")
    st.pyplot(fig)


# =========================================================
# 2. SUPERVISED VISUALS (REGRESSION / CLASSIFICATION)
# =========================================================

def plot_regression_results(y_test, y_pred, st):
    """
    Display Actual vs Predicted results in a scatter plot for regression tasks.
    """
    y_test = pd.Series(y_test).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)

    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2)

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)


def plot_classification_results(y_test, y_pred, st):
    """
    Plot a confusion matrix for classification results.
    """
    y_test = pd.Series(y_test)
    y_pred = pd.Series(y_pred)

    cm = pd.crosstab(y_test, y_pred, rownames=["Actual"], colnames=["Predicted"])

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def plot_model_comparison(results, task_type, st):
    """
    Radar chart comparison:
      - Regression → RMSE-based score (lower RMSE → higher score)
      - Classification → Accuracy
    """
    model_names = list(results.keys())

    if task_type == "Regression":
        raw_values = [results[m]["rmse"] for m in model_names]
        max_rmse = max(raw_values)
        scores = [max_rmse / v if v > 0 else max_rmse for v in raw_values]
        metric_label = "RMSE Score (Higher is Better)"
    else:
        scores = [results[m]["accuracy"] for m in model_names]
        metric_label = "Accuracy"

    # Close the polygon
    scores += scores[:1]
    N = len(model_names)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, scores, alpha=0.25)
    ax.plot(angles, scores, linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(model_names)
    ax.set_title(f"Model Comparison ({metric_label})")

    st.pyplot(fig)


def plot_feature_importance(model, X, st):
    """
    Plot feature importance for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        st.info("Model does not provide feature_importances_.")
        return

    importances = model.feature_importances_
    if len(importances) != X.shape[1]:
        st.warning("Feature importance length mismatch. Skipping plot.")
        return

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "importance": importances
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=importance_df, x="importance", y="feature", ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)


# =========================================================
# 3. UNSUPERVISED VISUALS (CLUSTERING)
# =========================================================

def plot_clusters(df, feature_cols, clusters, st, sample_size=1000):
    """
    Visualize clustering using pairplot.
    If dataset is large, a random sample is taken for speed.
    """
    df_copy = df.copy()
    df_copy["Cluster"] = clusters

    viz_cols = feature_cols[:4] if len(feature_cols) > 4 else feature_cols
    viz_cols = [c for c in viz_cols if c in df_copy.columns]
    viz_cols.append("Cluster")

    df_viz = (
        df_copy[viz_cols].sample(sample_size, random_state=42)
        if len(df_copy) > sample_size else df_copy[viz_cols]
    )

    fig = sns.pairplot(df_viz, hue="Cluster", palette="deep")
    st.pyplot(fig.figure)


# =========================================================
# 4. EDA VISUALS
# =========================================================

def plot_distribution(df, col, st):
    """
    Plot distribution for numeric or categorical columns.
    """
    if col not in df.columns:
        st.warning(f"Column '{col}' not found.")
        return

    series = df[col]
    fig, ax = plt.subplots()

    if pd.api.types.is_numeric_dtype(series):
        sns.histplot(series.dropna(), kde=True, ax=ax)
        ax.set_xlabel(col)
        ax.set_title(f"Distribution of {col}")
    else:
        vc = series.value_counts().head(30)
        sns.barplot(x=vc.values, y=vc.index, ax=ax)
        ax.set_xlabel("Count")
        ax.set_ylabel(col)
        ax.set_title(f"Category Counts of {col}")

    st.pyplot(fig)


def plot_pairplot_eda(df, cols, st, sample_size=500):
    """
    Plot pairplot for numeric columns (EDA).
    """
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        st.info("At least 2 numeric columns required for pairplot.")
        return

    df_viz = df[cols].select_dtypes(include=[np.number])
    if df_viz.shape[1] < 2:
        st.info("Selected columns do not contain enough numeric data.")
        return

    if len(df_viz) > sample_size:
        df_viz = df_viz.sample(sample_size, random_state=42)

    fig = sns.pairplot(df_viz)
    st.pyplot(fig.figure)


def plot_boxplot(df, col, st):
    """
    Plot a boxplot for a single numeric column.
    """
    if col not in df.columns:
        st.warning(f"Column '{col}' not found.")
        return

    series = df[col]
    if not pd.api.types.is_numeric_dtype(series):
        st.info("Boxplot is only available for numeric columns.")
        return

    fig, ax = plt.subplots()
    sns.boxplot(x=series, ax=ax)
    ax.set_title(f"Boxplot of {col}")
    st.pyplot(fig)

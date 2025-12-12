from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from data_processing import (
    load_data,
    detect_missing_values,
    handle_missing_values,
    prepare_data,
    split_and_scale,
    detect_task_type,
    get_data_summary,
    apply_pca,
)

from models import (
    get_supervised_models,
    get_unsupervised_models,
    train_supervised,
    train_multiple_models,
    train_unsupervised,
)

from visualization import (
    plot_correlation_matrix_supervised,
    plot_correlation_matrix_eda,
    plot_regression_results,
    plot_classification_results,
    plot_model_comparison,
    plot_feature_importance,
    plot_clusters,
    plot_distribution,
    plot_pairplot_eda,
    plot_boxplot,
)

from utils import (
    calculate_metrics,
    interpret_regression_metrics,
    compute_outlier_stats_iqr,
)
from report import generate_pdf_report


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="AutoMind ML App", layout="wide")

def load_css(file_name):
    with open(Path(__file__).parent / file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


# -----------------------------
# Top Page Title (visual)
# -----------------------------
st.title("AutoMind Machine Learning 🚀")
st.write("Upload your dataset and build ML models easily.")

# -----------------------------
# Sidebar Info
# -----------------------------
st.sidebar.header("📝 Instructions")
st.sidebar.write(
    """
1. Upload your CSV/Excel file 📁  
2. Explore the dataset with EDA 🔍  
3. Handle missing values if needed ⚠️  
4. Choose normalization & encoding 🛠️  
5. Use **Supervised** tab to train models 🎓  
6. Use **Unsupervised** tab for clustering 🎨  
7. Export PDF report after training 📄  
"""
)

st.sidebar.header("🛠 Requirements")
st.sidebar.code(
    "pip install streamlit pandas numpy sklearn seaborn matplotlib xgboost fpdf"
)

# -----------------------------
# File Upload
# -----------------------------
st.markdown("### 📂 1. Upload Dataset")

with st.container():
    st.info("Supported formats: CSV, Excel (.xlsx)!")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if not uploaded_file:
    st.stop()  # Show nothing else until file is uploaded

# -----------------------------
# Load and Preview Data
# -----------------------------
df = load_data(uploaded_file)

st.markdown("### 👀 2. Dataset Preview")
st.dataframe(df.head())

# -----------------------------
# Dataset Summary
# -----------------------------
st.markdown("### 📈 3. Dataset Summary")

numeric_summary, categorical_summary, numeric_cols, categorical_cols = get_data_summary(
    df
)

col_left, col_right = st.columns(2)

with col_left:
    st.write("**Numerical Features:**")
    if not numeric_summary.empty:
        st.dataframe(numeric_summary.reset_index(drop=False))
    else:
        st.info("No numeric columns detected.")

with col_right:
    st.write("**Categorical Features:**")
    if not categorical_summary.empty:
        st.dataframe(categorical_summary.reset_index(drop=False))
    else:
        st.info("No categorical columns detected.")

# -----------------------------
# EDA Explorer
# -----------------------------
st.markdown("### 🔍 4. Exploratory Data Analysis (EDA)")

tab_dist, tab_corr, tab_pair, tab_outliers = st.tabs(
    ["📊 Distributions", "🔗 Correlation", "🔁 Pairplot", "⚠️ Outliers"]
)

# ---------- 1) Distributions ----------
with tab_dist:
    st.write("Select a column to explore its distribution:")
    col_selected = st.selectbox("Column", df.columns, key="eda_dist_col")
    plot_distribution(df, col_selected, st)

# ---------- 2) Correlation ----------
with tab_corr:
    st.write("Select numeric columns for the correlation matrix (at least 2):")
    numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()

    selected_corr_cols = st.multiselect(
        "Columns:",
        numeric_cols_all,
        default=numeric_cols_all[:2] if len(numeric_cols_all) >= 2 else [],
        key="eda_corr_cols",
    )

    if len(selected_corr_cols) < 2:
        st.info("Please select at least 2 numeric columns for the correlation matrix 🙂")
    else:
        plot_correlation_matrix_eda(df, selected_corr_cols, st)

# ---------- 3) Pairplot ----------
with tab_pair:
    st.write("Select numeric columns for the pairplot (at least 2; 3–4 recommended):")

    numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()

    selected_pair_cols = st.multiselect(
        "Columns:",
        numeric_cols_all,
        default=numeric_cols_all[:2] if len(numeric_cols_all) >= 2 else [],
        key="eda_pair_cols_v2",
    )

    if len(selected_pair_cols) < 2:
        st.info("Please select at least 2 numeric columns for the pairplot 🙂")
    else:
        plot_pairplot_eda(df, selected_pair_cols, st)

# ---------- 4) Outliers ----------
with tab_outliers:
    st.write("Outlier statistics based on the IQR method:")

    outlier_stats = compute_outlier_stats_iqr(df)
    if outlier_stats.empty:
        st.info("No suitable numeric columns found for outlier statistics.")
    else:
        st.dataframe(outlier_stats.reset_index(drop=True))

        numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols_all:
            box_col = st.selectbox(
                "Select a numeric column for boxplot:",
                numeric_cols_all,
                key="eda_box_col",
            )
            plot_boxplot(df, box_col, st)

# -----------------------------
# Handle Missing Values
# -----------------------------
missing_cols = detect_missing_values(df)
if missing_cols:
    st.markdown("### ⚠️ 5. Missing Values Handling")
    st.write(f"Columns with missing values: **{list(missing_cols)}**")

    missing_strategies = {}
    for col in missing_cols:
        strategy = st.selectbox(
            f"How to handle missing values in '{col}'? 🤔",
            ["Mean", "Median", "Mode", "Drop", "Custom Value"],
            key=f"missing_{col}",
        )

        if strategy == "Custom Value":
            custom_value = st.text_input(f"Enter value for {col}", key=f"custom_{col}")
            missing_strategies[col] = custom_value if custom_value != "" else 0
        else:
            missing_strategies[col] = strategy

    if st.button("Apply Missing Value Handling ✅"):
        df = handle_missing_values(df, missing_strategies)
        st.success("Missing values handled. Updated dataset preview:")
        st.dataframe(df.head())

# -----------------------------
# Normalization & Encoding
# -----------------------------
st.markdown("### 🛠 6. Preprocessing Options")

pcol1, pcol2 = st.columns(2)

with pcol1:
    normalization_method = st.selectbox(
        "Select Normalization Method",
        ["Standard (Z-score)", "MinMax (0-1)", "Robust", "None"],
        index=0,
    )
    norm_dict = {
        "Standard (Z-score)": "Standard",
        "MinMax (0-1)": "MinMax",
        "Robust": "Robust",
        "None": None,
    }

with pcol2:
    encoding_method = st.selectbox(
        "Select Encoding Method",
        ["One-Hot (get_dummies)", "Label Encoding", "None"],
        index=0,
    )
    encoding_dict = {
        "One-Hot (get_dummies)": "onehot",
        "Label Encoding": "label",
        "None": "none",
    }

columns = df.columns.tolist()

# -----------------------------
# Modeling Tabs: Supervised / Unsupervised
# -----------------------------
st.markdown("### 🎓 7. Modeling")

tab_supervised, tab_unsupervised = st.tabs(
    ["🧠 Supervised Learning", "🎨 Unsupervised (Clustering)"]
)

# =============================
#       SUPERVISED PART
# =============================
with tab_supervised:
    st.markdown("#### 🧠 Supervised ML (Regression / Classification)")

    # Target selection
    target_col = st.selectbox("Select Target Column 🎯", columns)
    detected_task = detect_task_type(df[target_col])
    st.write(f"Detected Task Type: **{detected_task}** 🧠")

    task_type = st.radio(
        "Confirm Task Type ✅",
        ["Regression", "Classification"],
        index=0 if detected_task == "Regression" else 1,
        horizontal=True,
    )

    # Feature columns
    feature_cols = st.multiselect(
        "Select Feature Columns 📋",
        [col for col in columns if col != target_col],
        default=[col for col in columns if col != target_col],
    )

    # Correlation visualization (features vs target)
    if st.button("Show Feature–Target Correlation 🌐"):
        if not feature_cols:
            st.error("Please select at least one feature column ❌")
        else:
            st.subheader("Feature–Target Correlation 🌐")
            plot_correlation_matrix_supervised(df, feature_cols, target_col, st)

    # Train/test split
    test_size = st.slider("Test Split Ratio ⚖️", 0.1, 0.5, 0.2, 0.05)

    # Model options
    model_options = get_supervised_models(task_type)
    selected_models = st.multiselect(
        "Select Models to Train 🏋️",
        list(model_options.keys()),
        default=list(model_options.keys()),
    )

    train_mode = st.radio(
        "Training Mode", ["Single Model", "Compare Multiple Models"], horizontal=True
    )

    st.markdown("### 🧪 Train & Evaluate")

    tcol1, tcol2 = st.columns([3, 2])
    with tcol1:
        train_clicked = st.button(
            "🚀 Train Model(s)",
            type="primary",
            use_container_width=True,
        )
    with tcol2:
        st.info(
            "Tip: Select target, features, preprocessing options and models first, "
            "then click **Train Model(s)**."
        )

    if train_clicked:
        try:
            # Prepare data
            X, y = prepare_data(
                df,
                feature_cols,
                target_col=target_col,
                encoding_method=encoding_dict[encoding_method],
                task_type=task_type,  # "Regression" or "Classification"
            )
            X_train, X_test, y_train, y_test = split_and_scale(
                X, y, test_size, norm_dict[normalization_method]
            )

            # =======================
            # Single model training
            # =======================
            if train_mode == "Single Model":
                if len(selected_models) != 1:
                    st.error("Please select exactly one model for Single Model mode ❌")
                else:
                    model_name = selected_models[0]
                    model = model_options[model_name]

                    # Train
                    y_pred = train_supervised(model, X_train, y_train, X_test)

                    st.subheader(f"Results for {model_name} 🎉")

                    # Metrics
                    metrics = calculate_metrics(task_type, y_test, y_pred)

                    st.subheader("Metrics 📏")
                    for metric, value in metrics.items():
                        if metric == "Classification Report":
                            st.subheader("Classification Report 📄")
                            st.text(value)
                        else:
                            st.write(f"**{metric}:** {value}")

                    # Plots + interpretation
                    if task_type == "Regression":
                        plot_regression_results(y_test, y_pred, st)
                        interpretation = interpret_regression_metrics(y_test, y_pred)

                        st.subheader("Interpretation 💡")
                        for line in interpretation:
                            st.write(f"- {line}")
                    else:
                        plot_classification_results(y_test, y_pred, st)
                        interpretation = None  # classification interpretation optional

                    # Feature importance (if supported)
                    plot_feature_importance(model, X, st)

                    # PDF Report Export
                    st.markdown("### 📄 Export Report")

                    project_name = "AutoMind Machine Learning"

                    pdf_bytes = generate_pdf_report(
                        project_name=project_name,
                        task_type=task_type,
                        model_name=model_name,
                        metrics=metrics,
                        target_col=target_col,
                        feature_cols=feature_cols,
                        interpretation_text=interpretation,
                        data_filename=uploaded_file.name
                        if uploaded_file is not None
                        else None,
                    )

                    st.success(
                        "✅ Report generated successfully. Click below to download the PDF."
                    )
                    st.download_button(
                        label="📥 Download Report PDF",
                        data=pdf_bytes,
                        file_name="ml_report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

            # =======================
            # Multiple model comparison
            # =======================
            else:
                if not selected_models:
                    st.error("Please select at least one model to compare ❌")
                else:
                    models_to_train = {
                        name: model_options[name] for name in selected_models
                    }

                    results = train_multiple_models(
                        models_to_train,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        task_type,
                    )

                    st.subheader("Model Comparison 🎉")
                    plot_model_comparison(results, task_type, st)

                    # Show per-model metrics + plots
                    for name, result in results.items():
                        st.markdown(f"### 🔹 {name}")
                        if task_type == "Regression":
                            st.write(f"RMSE: {result['rmse']:.4f}")
                            plot_regression_results(y_test, result["y_pred"], st)
                        else:
                            st.write(f"Accuracy: {result['accuracy']:.4f}")
                            plot_classification_results(
                                y_test, result["y_pred"], st
                            )

                        plot_feature_importance(result["model"], X, st)

        except ValueError as e:
            st.error(f"Error: {str(e)} ❌")
            st.write("Please ensure all missing values are handled properly.")

# =============================
#       UNSUPERVISED PART
# =============================
with tab_unsupervised:
    st.markdown("#### 🎨 Unsupervised Learning (Clustering)")

    feature_cols_unsup = st.multiselect(
        "Select Features for Clustering 📋", columns, default=columns
    )

    use_pca = st.checkbox("Apply PCA for dimensionality reduction 🌟")
    if use_pca:
        n_components = st.slider(
            "Number of PCA Components",
            2,
            min(len(feature_cols_unsup), 10),
            2,
        )

    # Clustering model options
    model_options_unsup = get_unsupervised_models()
    selected_model_unsup = st.selectbox(
        "Select Clustering Model 🏋️", list(model_options_unsup.keys())
    )

    # Model-specific parameters
    if selected_model_unsup == "KMeans":
        n_clusters = st.slider("Number of Clusters 🎨", 2, 10, 3)
        params = {"n_clusters": n_clusters}
    else:  # DBSCAN
        eps = st.slider("Epsilon (eps) 📏", 0.1, 5.0, 0.5, 0.1)
        min_samples = st.slider("Min Samples per Cluster 👥", 3, 20, 5)
        params = {"eps": eps, "min_samples": min_samples}

    if st.button("Run Clustering 🚀"):
        try:
            # Prepare data for clustering (features only, no target)
            X_unsup = prepare_data(
                df,
                feature_cols_unsup,
                target_col=None,
                encoding_method=encoding_dict[encoding_method],
                task_type=None,
            )

            # Scale data
            X_scaled_unsup = split_and_scale(
                X_unsup, normalization=norm_dict[normalization_method]
            )

            # PCA for visualization if selected
            if use_pca:
                X_scaled_unsup, pca = apply_pca(X_scaled_unsup, n_components)
                st.write(
                    f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_} 📊"
                )

            # Train clustering model
            cluster_model, clusters = train_unsupervised(
                selected_model_unsup, X_scaled_unsup, params
            )

            st.subheader("Clustering Results 🎉")
            plot_clusters(df, feature_cols_unsup, clusters, st)

            st.write("Number of samples per cluster:")
            cluster_counts = pd.Series(clusters).value_counts()
            st.write(cluster_counts)

            if -1 in cluster_counts.index:
                st.write("Note: -1 indicates noise points in DBSCAN (-1 = noise)")

        except ValueError as e:
            st.error(f"Error: {str(e)} ❌")
            st.write("Please ensure all missing values are handled properly.")

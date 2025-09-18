# clustering_app.py
"""
Streamlit app for clustering analysis and comparative evaluation.
Features:
- Upload dataset (CSV or XLSX)
- Basic cleaning (currency/percent stripping on common columns)
- Select features for clustering
- Scaling, PCA (2 components) for visualization
- Run multiple clustering algorithms: KMeans, Agglomerative, DBSCAN, GaussianMixture
- Evaluate with Silhouette and Calinski-Harabasz scores
- Show comparison table, cluster assignments, PCA scatter and 3D PCA (if available)
- Download clustered dataset

Run:
    streamlit run clustering_app.py
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

st.set_page_config(page_title="Clustering Explorer", layout="wide")
st.title("🔎 Clustering Explorer — Compare clustering models interactively")

st.markdown(
    """
Upload your dataset (CSV or XLSX). Then choose numeric features to use for clustering,
adjust parameters, and run multiple algorithms to compare results.
"""
)

# -------------------- File upload --------------------
uploaded = st.sidebar.file_uploader("Upload CSV or XLSX file", type=["csv", "xlsx"])
sample_data_btn = st.sidebar.button("Load sample dataset")

if sample_data_btn and not uploaded:
    # create a small synthetic sample dataset
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
    sample_df = pd.DataFrame(X, columns=["feature_1", "feature_2"])
    sample_df["feature_3"] = sample_df["feature_1"] * 0.5 + np.random.normal(scale=0.5, size=len(sample_df))
    df = sample_df.copy()
    st.sidebar.success("Sample dataset loaded")
elif uploaded:
    try:
        if uploaded.name.lower().endswith(".xlsx"):
            df = pd.read_excel(uploaded, engine="openpyxl")
        else:
            df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()
else:
    st.info("Upload a dataset (CSV/XLSX) or click 'Load sample dataset' in the sidebar.")
    st.stop()

st.write("Dataset shape:", df.shape)
st.dataframe(df.head())

# -------------------- Basic cleaning --------------------
st.sidebar.header("Preprocessing options")
auto_clean = st.sidebar.checkbox("Auto-clean common currency/percent columns", value=True)
if auto_clean:
    # common currency/percent column name patterns
    currency_patterns = ["gdp", "gdp_usd", "gdp_$", "gdp_usd", "gdp_per_capita", "health_exp", "tourism"]
    for col in df.columns:
        if any(p in col.lower() for p in currency_patterns) or any(sym in col for sym in ["$", ","]):
            # strip $ and , and convert to numeric
            df[col] = df[col].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # percent columns
    for col in df.columns:
        if "%" in col or "percent" in col.lower() or "rate" in col.lower():
            df[col] = df[col].astype(str).str.replace("%", "", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")

# show columns types
st.sidebar.subheader("Column types (detected)")
with st.sidebar.expander("Columns"):
    st.write(df.dtypes)

# -------------------- Feature selection --------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if not numeric_cols:
    st.error("No numeric columns found in dataset. Clustering needs numeric features.")
    st.stop()

st.sidebar.header("Features & Scaling")
features = st.sidebar.multiselect("Select numeric features to use for clustering", numeric_cols, default=numeric_cols[:5])
if not features:
    st.error("Select at least one feature.")
    st.stop()

fill_method = st.sidebar.selectbox("Missing value handling (numeric cols)", ["mean", "median", "drop_rows"], index=0)

# Impute / drop missing according to selection
X = df[features].copy()
X = X.replace([np.inf, -np.inf], np.nan)
if fill_method == "mean":
    X = X.fillna(X.mean())
elif fill_method == "median":
    X = X.fillna(X.median())
else:
    X = X.dropna()

if X.shape[0] == 0:
    st.error("No rows left after missing value handling. Try a different option.")
    st.stop()

scale_data = st.sidebar.checkbox("Scale features (StandardScaler)", value=True)
if scale_data:
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
else:
    X_scaled = X.copy()

# PCA for visualization
do_pca = st.sidebar.checkbox("Use PCA for 2D/3D visualization", value=True)
if do_pca and X_scaled.shape[1] >= 2:
    pca = PCA(n_components=min(10, X_scaled.shape[1]))
    X_pca_full = pca.fit_transform(X_scaled)
    # 2-component for scatter and 3-component if available
    X_pca_2 = X_pca_full[:, :2]
    X_pca_3 = X_pca_full[:, :3] if X_pca_full.shape[1] >= 3 else None
    st.sidebar.write(f"PCA explained variance (first 5): {np.round(pca.explained_variance_ratio_[:5],3)}")
else:
    X_pca_2 = None
    X_pca_3 = None

# -------------------- Clustering options --------------------
st.sidebar.header("Clustering models & parameters")
run_kmeans = st.sidebar.checkbox("KMeans", value=True)
kmeans_k = st.sidebar.slider("KMeans: n_clusters", 2, 15, 4)
run_agglom = st.sidebar.checkbox("Agglomerative (Ward)", value=True)
agg_k = st.sidebar.slider("Agglomerative: n_clusters", 2, 15, 4)
run_dbscan = st.sidebar.checkbox("DBSCAN", value=False)
db_eps = st.sidebar.number_input("DBSCAN: eps", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
db_min_samples = st.sidebar.number_input("DBSCAN: min_samples", min_value=1, max_value=50, value=5, step=1)
run_gmm = st.sidebar.checkbox("GaussianMixture", value=False)
gmm_k = st.sidebar.slider("GMM: n_components", 2, 15, 4)

st.sidebar.markdown("---")
st.sidebar.write("Evaluation metrics will be calculated for models that produce >=2 clusters.")

# Button to run clustering
if st.sidebar.button("Run clustering and compare"):
    results = {}
    labels_store = pd.DataFrame(index=X_scaled.index)

    # Helper to compute scores safely
    def safe_scores(Xarr, labels):
        scores = {"n_clusters": len(set(labels)) - (1 if -1 in labels else 0)}
        try:
            if scores["n_clusters"] >= 2:
                scores["silhouette"] = float(silhouette_score(Xarr, labels))
                scores["calinski_harabasz"] = float(calinski_harabasz_score(Xarr, labels))
            else:
                scores["silhouette"] = np.nan
                scores["calinski_harabasz"] = np.nan
        except Exception as e:
            scores["silhouette"] = np.nan
            scores["calinski_harabasz"] = np.nan
        return scores

    # KMeans
    if run_kmeans:
        km = KMeans(n_clusters=kmeans_k, random_state=42, n_init=10)
        km_labels = km.fit_predict(X_scaled)
        labels_store["KMeans"] = km_labels
        results["KMeans"] = safe_scores(X_scaled, km_labels)
        # store cluster centers
        kmeans_centers = pd.DataFrame(km.cluster_centers_, columns=X_scaled.columns)

    # Agglomerative
    if run_agglom:
        ag = AgglomerativeClustering(n_clusters=agg_k)
        ag_labels = ag.fit_predict(X_scaled)
        labels_store["Agglomerative"] = ag_labels
        results["Agglomerative"] = safe_scores(X_scaled, ag_labels)

    # DBSCAN
    if run_dbscan:
        db = DBSCAN(eps=db_eps, min_samples=int(db_min_samples))
        db_labels = db.fit_predict(X_scaled)
        labels_store["DBSCAN"] = db_labels
        results["DBSCAN"] = safe_scores(X_scaled, db_labels)

    # GMM
    if run_gmm:
        gm = GaussianMixture(n_components=gmm_k, random_state=42)
        gm_labels = gm.fit_predict(X_scaled)
        labels_store["GMM"] = gm_labels
        results["GMM"] = safe_scores(X_scaled, gm_labels)

    # Results display
    st.subheader("📋 Model Comparison")
    res_df = pd.DataFrame(results).T
    st.dataframe(res_df.style.format({"silhouette": "{:.4f}", "calinski_harabasz": "{:.2f}"}))

    # Show counts per cluster for each model
    st.subheader("🔢 Cluster counts per model")
    st.write(labels_store.apply(lambda col: pd.Series(col).value_counts()).fillna(0).astype(int))

    # Merge labels back to original df for download and inspection
    out_df = df.copy().loc[labels_store.index]
    for col in labels_store.columns:
        out_df[f"cluster_{col}"] = labels_store[col].values

    st.subheader("📥 Download clustered dataset")
    to_download = out_df.copy()
    csv = to_download.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", data=csv, file_name="clustered_dataset.csv", mime="text/csv")

    # Visualization: PCA 2D scatter colored by selected model
    st.subheader("📊 PCA Scatter (2 components)")
    model_choice = st.selectbox("Color clusters by model", labels_store.columns.tolist(), index=0)
    if model_choice not in labels_store.columns:
        st.warning(f"{model_choice} was not run. Choose a run model.")
    else:
        labels = labels_store[model_choice].values
        if X_pca_2 is None:
            st.warning("PCA not available (requires at least 2 numeric features).")
        else:
            fig, ax = plt.subplots(figsize=(8,6))
            # create palette with enough colors
            unique_labels = sorted(set(labels))
            palette = sns.color_palette("tab10", n_colors=max(3, len(unique_labels)))
            sns.scatterplot(x=X_pca_2[:,0], y=X_pca_2[:,1], hue=labels, palette=palette, legend='brief', ax=ax, s=50)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA 2D - colored by {model_choice}")
            st.pyplot(fig)

    # 3D PCA plot if available
    if X_pca_3 is not None:
        st.subheader("🌐 PCA Scatter (3 components)")
        fig = plt.figure(figsize=(9,7))
        ax = fig.add_subplot(111, projection='3d')
        labels = labels_store.iloc[:,0].values  # default to first model's labels
        sc = ax.scatter(X_pca_3[:,0], X_pca_3[:,1], X_pca_3[:,2], c=labels, cmap='tab10', s=30)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")
        st.pyplot(fig)

    # Show cluster centers for KMeans if available
    if run_kmeans:
        st.subheader("🏷️ KMeans cluster centers (scaled feature space)")
        st.dataframe(kmeans_centers.round(4))
        # also show centers in original scale if scaled
        if scale_data:
            inv_centers = pd.DataFrame(scaler.inverse_transform(kmeans_centers), columns=kmeans_centers.columns)
            st.write("KMeans centers (original feature scale):")
            st.dataframe(inv_centers.round(4))

    st.success("Clustering run complete!")
else:
    st.info("Adjust options in the sidebar and click 'Run clustering and compare'.")

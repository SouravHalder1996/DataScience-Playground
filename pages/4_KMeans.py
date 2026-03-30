"""
pages/4_KMeans.py
Interactive K-Means Clustering explorer.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.decomposition import PCA

from utils.helpers import (
    inject_page_css, page_header, info_box, metric_row, sidebar_header,
    CLUSTERING_DATASETS, load_clustering_data,
    PLOTLY_LAYOUT, PALETTE, BORDER, TEXT, MUTED,
)

st.set_page_config(page_title="K-Means · ML Playground",
                   page_icon="⭕", layout="wide")
inject_page_css()

page_header(
    "K-Means Clustering", "Voronoi partitions · centroid convergence · silhouette analysis · elbow method",
    emoji="⭕",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
sidebar_header()
with st.sidebar:
    with st.expander("⚙️ Dataset", expanded=True):
        dataset_label = st.selectbox("Dataset", list(CLUSTERING_DATASETS.keys()))
        dataset_key   = CLUSTERING_DATASETS[dataset_label]
        n_samples     = st.slider("n_samples", 100, 1500, 400, 50)
        noise         = st.slider("Noise level", 0.0, 0.5, 0.1, 0.01)
        rand_state    = st.slider("Random seed", 0, 99, 42)

        if dataset_key == "blobs":
            true_k = st.slider("True n_clusters (blobs)", 2, 8, 3)
        else:
            true_k = 2

    with st.expander("🎛️ Hyperparameters", expanded=True):

        k           = st.slider("n_clusters  (k)", 2, 10, true_k if dataset_key == "blobs" else 2)
        init        = st.selectbox("init", ["k-means++", "random"])
        n_init      = st.slider("n_init", 1, 20, 10)
        max_iter    = st.slider("max_iter", 10, 1000, 300, 10)
        tol         = st.select_slider("tol", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], value=1e-4)

    with st.expander("🖼️ Visualisation", expanded=True):
        show_voronoi = st.checkbox("Show Voronoi regions", value=True)
        show_elbow   = st.checkbox("Elbow / silhouette curve", value=True)
        elbow_max_k  = st.slider("Max k for elbow curve", 3, 15, 10)

# ── Load data ──────────────────────────────────────────────────────────────────
X, y_true = load_clustering_data(
    dataset_key, n_samples=n_samples,
    n_clusters=true_k, noise=noise, random_state=rand_state,
)

# Reduce to 2D if needed
if X.shape[1] > 2:
    pca = PCA(n_components=2, random_state=rand_state)
    X2 = pca.fit_transform(X)
else:
    X2 = X

# ── Fit model ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="⭕ Running K-Means …")
def fit_kmeans(k_, init_, ni, mi, tol_, rs, _X):
    return KMeans(n_clusters=k_, init=init_, n_init=ni,
                  max_iter=mi, tol=tol_, random_state=rs).fit(_X)

km = fit_kmeans(k, init, n_init, max_iter, tol, rand_state, X2)
labels = km.labels_
centroids = km.cluster_centers_
inertia   = km.inertia_

sil_score = silhouette_score(X2, labels) if k > 1 else 0.0
db_score  = davies_bouldin_score(X2, labels) if k > 1 else 0.0

# ── Metrics ────────────────────────────────────────────────────────────────────
metric_row({
    "Inertia (WCSS)":   round(inertia, 2),
    "Silhouette Score": round(sil_score, 4),
    "Davies-Bouldin":   round(db_score, 4),
    "n_iter":           km.n_iter_,
    "k":                k,
})
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️ Cluster Map", "📉 Elbow / Silhouette", "🌸 Silhouette Plot", "📊 Cluster Stats",
])

# ── Tab 1 : Cluster map ────────────────────────────────────────────────────────
with tab1:
    df_plot = pd.DataFrame(X2, columns=["x", "y"])
    df_plot["cluster"] = [f"Cluster {l}" for l in labels]

    fig_map = go.Figure()

    if show_voronoi:
        # Mesh for Voronoi regions
        x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
        y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
        res = 150
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, res),
                              np.linspace(y_min, y_max, res))
        Z = km.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        for cls_idx in range(k):
            color_hex = PALETTE[cls_idx % len(PALETTE)]
            r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
            region_color = f"rgba({r},{g},{b},0.18)"
            fig_map.add_trace(go.Contour(
                x=np.linspace(x_min, x_max, res),
                y=np.linspace(y_min, y_max, res),
                z=(Z == cls_idx).astype(float),
                showscale=False,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, region_color]],
                contours=dict(start=0.5, end=1.5, size=1),
                hoverinfo="skip", name="",
            ))

    # Data points
    for cls_idx in range(k):
        mask = labels == cls_idx
        fig_map.add_trace(go.Scatter(
            x=X2[mask, 0], y=X2[mask, 1],
            mode="markers",
            name=f"Cluster {cls_idx}",
            marker=dict(
                color=PALETTE[cls_idx % len(PALETTE)],
                size=7, opacity=0.85,
                line=dict(width=0.8, color="#0a0a0f"),
            ),
        ))

    # Centroids
    fig_map.add_trace(go.Scatter(
        x=centroids[:, 0], y=centroids[:, 1],
        mode="markers+text",
        text=[f"C{i}" for i in range(k)],
        textposition="top center",
        textfont=dict(color="#fff", size=10),
        name="Centroids",
        marker=dict(
            symbol="star", size=20,
            color=[PALETTE[i % len(PALETTE)] for i in range(k)],
            line=dict(color="#ffffff", width=1.5),
        ),
    ))

    fig_map.update_layout(**PLOTLY_LAYOUT,
                          title=f"K-Means Clustering — k={k}, init={init}, inertia={inertia:.1f}")
    st.plotly_chart(fig_map, use_container_width=True)

# ── Tab 2 : Elbow / Silhouette curve ──────────────────────────────────────────
with tab2:
    if show_elbow:
        @st.cache_data(show_spinner="📉 Computing elbow curve …")
        def compute_elbow(_X, max_k, init_, ni, mi, tol_, rs):
            inertias, sils, dbs = [], [], []
            for ki in range(2, max_k + 1):
                m = KMeans(n_clusters=ki, init=init_, n_init=ni,
                           max_iter=mi, tol=tol_, random_state=rs).fit(_X)
                inertias.append(m.inertia_)
                sils.append(silhouette_score(_X, m.labels_))
                dbs.append(davies_bouldin_score(_X, m.labels_))
            return list(range(2, max_k + 1)), inertias, sils, dbs

        ks, inertias, sils, dbs = compute_elbow(
            X2, elbow_max_k, init, n_init, max_iter, tol, rand_state)

        from plotly.subplots import make_subplots
        fig_elbow = make_subplots(rows=1, cols=2,
                                   subplot_titles=["Inertia (Elbow)", "Silhouette Score"])

        fig_elbow.add_trace(go.Scatter(
            x=ks, y=inertias, mode="lines+markers",
            line=dict(color=PALETTE[0], width=2.5),
            marker=dict(size=7, color=[PALETTE[1] if ki == k else PALETTE[0] for ki in ks]),
            name="Inertia",
        ), row=1, col=1)
        fig_elbow.add_trace(go.Scatter(
            x=ks, y=sils, mode="lines+markers",
            line=dict(color=PALETTE[1], width=2.5),
            marker=dict(size=7, color=[PALETTE[0] if ki == k else PALETTE[1] for ki in ks]),
            name="Silhouette",
        ), row=1, col=2)

        fig_elbow.add_vline(x=k, line=dict(color="#f0a050", dash="dash", width=1.5),
                             annotation_text=f"k={k}", row=1, col=1)
        fig_elbow.add_vline(x=k, line=dict(color="#f0a050", dash="dash", width=1.5),
                             annotation_text=f"k={k}", row=1, col=2)

        fig_elbow.update_layout(
            **PLOTLY_LAYOUT,
            title="Elbow & Silhouette Analysis",
            showlegend=False,
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
    else:
        st.info("Enable 'Elbow / silhouette curve' in the sidebar.")

# ── Tab 3 : Silhouette plot ────────────────────────────────────────────────────
with tab3:
    sil_vals = silhouette_samples(X2, labels)
    fig_sil  = go.Figure()
    y_lower  = 0

    for cls_idx in range(k):
        cls_sil = np.sort(sil_vals[labels == cls_idx])
        size    = cls_sil.shape[0]
        y_upper = y_lower + size

        color_hex = PALETTE[cls_idx % len(PALETTE)]
        fig_sil.add_trace(go.Bar(
            x=cls_sil,
            y=list(range(y_lower, y_upper)),
            orientation="h",
            marker_color=color_hex,
            marker_opacity=0.8,
            name=f"Cluster {cls_idx}",
            showlegend=True,
        ))
        y_lower = y_upper + 5  # gap

    fig_sil.add_vline(x=sil_score, line=dict(color="#f0a050", dash="dash", width=2),
                      annotation_text=f"Mean={sil_score:.3f}")
    fig_sil.update_layout(
        **PLOTLY_LAYOUT,
        title="Silhouette Plot (per sample)",
        xaxis_title="Silhouette Coefficient",
        barmode="overlay",
    )
    fig_sil.update_yaxes(visible=False)
    st.plotly_chart(fig_sil, use_container_width=True)

# ── Tab 4 : Cluster stats ──────────────────────────────────────────────────────
with tab4:
    df_stats = pd.DataFrame({
        "Cluster": [f"Cluster {i}" for i in range(k)],
        "Size":    [int((labels == i).sum()) for i in range(k)],
        "% of data": [f"{(labels == i).mean()*100:.1f}%" for i in range(k)],
        "Centroid x": [f"{centroids[i, 0]:.3f}" for i in range(k)],
        "Centroid y": [f"{centroids[i, 1]:.3f}" for i in range(k)],
        "Avg Silhouette": [f"{sil_vals[labels == i].mean():.4f}" for i in range(k)],
    })
    st.dataframe(df_stats, use_container_width=True, hide_index=True)

    # Cluster size bar
    fig_size = go.Figure(go.Bar(
        x=[f"Cluster {i}" for i in range(k)],
        y=[(labels == i).sum() for i in range(k)],
        marker_color=[PALETTE[i % len(PALETTE)] for i in range(k)],
        marker_opacity=0.85,
    ))
    fig_size.update_layout(**PLOTLY_LAYOUT, title="Cluster Sizes",
                           xaxis_title="Cluster", yaxis_title="Count")
    st.plotly_chart(fig_size, use_container_width=True)

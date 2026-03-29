"""
pages/2_Decision_Tree.py
Interactive Decision Tree explorer.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, export_text

from utils.helpers import (
    inject_page_css, page_header, info_box, metric_row, sidebar_header,
    DATASETS, load_classification_data,
    fig_decision_boundary, fig_confusion_matrix,
    fig_roc_curves, fig_feature_importance,
    metrics_dict, PLOTLY_LAYOUT, PALETTE, BORDER, TEXT, MUTED,
)

st.set_page_config(page_title="Decision Tree · ML Playground",
                   page_icon="🌳", layout="wide")
inject_page_css()

page_header(
    "Decision Tree", "Recursive binary splitting · Gini / Entropy · depth & leaf controls",
    emoji="🌳",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
sidebar_header()
with st.sidebar:
    with st.expander("⚙️ Dataset", expanded=False):
        dataset_label = st.selectbox("Dataset", list(DATASETS.keys()))
        dataset_key   = DATASETS[dataset_label]
        test_size     = st.slider("Test split", 0.10, 0.50, 0.25, 0.05)
        rand_state    = st.slider("Random seed", 0, 99, 42)

        if dataset_key in ("moons", "circles", "synthetic"):
            n_samples = st.slider("n_samples", 100, 1000, 300, 50)
            noise     = st.slider("Noise level", 0.0, 0.5, 0.15, 0.01)
        else:
            n_samples, noise = 300, 0.15

    with st.expander("🎛️ Hyperparameters", expanded=False):

        criterion     = st.selectbox("Criterion", ["gini", "entropy", "log_loss"])
        max_depth     = st.slider("max_depth  (None = unlimited)", 1, 20, 4)
        min_samples_split = st.slider("min_samples_split", 2, 50, 2)
        min_samples_leaf  = st.slider("min_samples_leaf", 1, 50, 1)
        max_features  = st.selectbox("max_features", ["None (all)", "sqrt", "log2", "0.5"])
        ccp_alpha     = st.slider("ccp_alpha  (pruning)", 0.0, 0.05, 0.0, 0.001,
                                format="%.3f")

    with st.expander("🖼️ Visualisation", expanded=False):
        show_boundary = st.checkbox("Decision boundary (2-D)", value=True)
        decision_res  = st.select_slider("Boundary resolution", [50, 100, 150, 200], value=100)

# ── Load & fit ─────────────────────────────────────────────────────────────────
data = load_classification_data(
    dataset_key, test_size=test_size, random_state=rand_state,
    n_samples=n_samples, noise=noise,
)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
class_names     = data["class_names"]
feature_names   = data["feature_names"]

mf = None if max_features == "None (all)" else (0.5 if max_features == "0.5" else max_features)

model = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    max_features=mf,
    ccp_alpha=ccp_alpha,
    random_state=rand_state,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── Metrics ────────────────────────────────────────────────────────────────────
mets = metrics_dict(y_test, y_pred)
depth_actual = model.get_depth()
n_leaves = model.get_n_leaves()
metric_row({**mets, "Tree depth": depth_actual, "Leaves": n_leaves})
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Decision Boundary", "🔢 Confusion Matrix", "📈 ROC Curve",
    "🌿 Feature Importance", "🌳 Tree Structure",
])

with tab1:
    if X_train.shape[1] == 2 and show_boundary:
        fig = fig_decision_boundary(model, data["X"], data["y"], class_names,
                                    resolution=decision_res,
                                    title=f"Decision Boundary — depth={depth_actual}, {criterion}")
        st.plotly_chart(fig, use_container_width=True)
    elif not X_train.shape[1] == 2:
        info_box("Decision boundary requires 2-D data. Showing PCA projection below.")
        from sklearn.decomposition import PCA
        import plotly.express as px
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(data["X"])
        df2 = pd.DataFrame(X2, columns=["PC1","PC2"])
        df2["label"] = [class_names[i] for i in data["y"]]
        fig = px.scatter(df2, x="PC1", y="PC2", color="label",
                         color_discrete_sequence=PALETTE, title="PCA projection (2-D)")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Enable 'Decision boundary' in the sidebar.")

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_confusion_matrix(y_test, y_pred, class_names),
                        use_container_width=True)
    with col2:
        from sklearn.metrics import classification_report
        report = classification_report(y_test, y_pred,
                                       target_names=class_names, output_dict=True)
        df_rep = pd.DataFrame(report).transpose()
        st.markdown("##### Classification Report")
        st.dataframe(df_rep.style.background_gradient(cmap="Purples", axis=None)
                     .format("{:.3f}"), use_container_width=True)

with tab3:
    fig_roc = fig_roc_curves(model, X_test, y_test, class_names)
    st.plotly_chart(fig_roc, use_container_width=True)

with tab4:
    importances = model.feature_importances_
    fig_fi = fig_feature_importance(importances, feature_names)
    st.plotly_chart(fig_fi, use_container_width=True)

    df_imp = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df_imp = df_imp.sort_values("Importance", ascending=False)
    st.dataframe(df_imp.style.bar(subset=["Importance"], color="#6c63ff")
                 .format({"Importance": "{:.4f}"}), use_container_width=True)

with tab5:
    # Plotly tree visualisation using sklearn's tree_ attributes
    tree = model.tree_
    n_nodes = tree.node_count
    children_left  = tree.children_left
    children_right = tree.children_right
    feature_arr    = tree.feature
    threshold_arr  = tree.threshold
    values         = tree.value
    impurity       = tree.impurity

    # BFS to get positions
    from collections import deque
    pos_x = np.zeros(n_nodes)
    pos_y = np.zeros(n_nodes)
    layer = np.zeros(n_nodes, dtype=int)
    layer_counts = {}

    queue = deque([(0, 0, 0.5)])  # node, depth, x_center
    x_offsets = {}

    def assign_positions(node, depth, x, x_step):
        pos_x[node] = x
        pos_y[node] = -depth
        if children_left[node] != -1:
            assign_positions(children_left[node],  depth+1, x - x_step/2, x_step/2)
            assign_positions(children_right[node], depth+1, x + x_step/2, x_step/2)

    assign_positions(0, 0, 0.5, 0.25)

    # Build edges
    edge_x, edge_y = [], []
    for i in range(n_nodes):
        if children_left[i] != -1:
            edge_x += [pos_x[i], pos_x[children_left[i]],  None]
            edge_y += [pos_y[i], pos_y[children_left[i]],  None]
            edge_x += [pos_x[i], pos_x[children_right[i]], None]
            edge_y += [pos_y[i], pos_y[children_right[i]], None]

    # Node labels
    def node_label(i):
        if children_left[i] == -1:  # leaf
            pred = np.argmax(values[i][0])
            return f"Leaf<br>{class_names[pred]}<br>imp={impurity[i]:.2f}"
        fn = feature_names[feature_arr[i]] if feature_arr[i] < len(feature_names) else "?"
        return f"{fn}<br>≤ {threshold_arr[i]:.2f}<br>imp={impurity[i]:.2f}"

    node_color = [PALETTE[int(np.argmax(values[i][0])) % len(PALETTE)] for i in range(n_nodes)]
    node_text  = [node_label(i) for i in range(n_nodes)]
    is_leaf    = [children_left[i] == -1 for i in range(n_nodes)]

    fig_tree = go.Figure()
    fig_tree.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color=BORDER, width=1.2), hoverinfo="skip",
    ))
    fig_tree.add_trace(go.Scatter(
        x=pos_x, y=pos_y, mode="markers+text",
        text=[("●" if l else "◆") for l in is_leaf],
        textfont=dict(color="#0a0a0f", size=8),
        hovertext=node_text, hoverinfo="text",
        marker=dict(size=[14 if l else 16 for l in is_leaf],
                    color=node_color, opacity=0.9,
                    line=dict(color="#0a0a0f", width=1)),
    ))
    fig_tree.update_layout(
        **PLOTLY_LAYOUT, title=f"Tree Structure (depth={depth_actual}, leaves={n_leaves})",
        showlegend=False, height=max(400, depth_actual * 120),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    with st.expander("🖨️ Text representation"):
        st.code(export_text(model, feature_names=feature_names), language="text")

"""
pages/_TEMPLATE.py
─────────────────────────────────────────────────────────────────
Copy this file, rename it, and fill in the TODOs to add a new algorithm.
Pattern followed by all pages in this playground.
─────────────────────────────────────────────────────────────────
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd

# TODO: import your sklearn model
# from sklearn.??? import ???

from utils.helpers import (
    inject_page_css, page_header, info_box, metric_row, sidebar_header,
    DATASETS, load_classification_data,   # or load_clustering_data
    fig_decision_boundary, fig_confusion_matrix,
    fig_roc_curves, fig_feature_importance,
    metrics_dict, PLOTLY_LAYOUT, PALETTE, sidebar_header,
)

# ── 1. Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Algorithm Name · ML Playground",
    page_icon="🔷",          # TODO: choose an emoji
    layout="wide",
)
inject_page_css()

page_header(
    "Algorithm Name",         # TODO
    "short tagline here",     # TODO
    emoji="🔷",               # TODO
)

# ── 2. Sidebar — dataset controls ─────────────────────────────────────────────
sidebar_header()
with st.sidebar:
    with st.expander("⚙️ Dataset", expanded=True):
        dataset_label = st.selectbox("Dataset", list(DATASETS.keys()))
        dataset_key   = DATASETS[dataset_label]
        test_size     = st.slider("Test split", 0.10, 0.50, 0.25, 0.05)
        rand_state    = st.slider("Random seed", 0, 99, 42)

        if dataset_key in ("moons", "circles", "synthetic"):
            n_samples = st.slider("n_samples", 100, 1000, 300, 50)
            noise     = st.slider("Noise level", 0.0, 0.5, 0.15, 0.01)
        else:
            n_samples, noise = 300, 0.15

    with st.expander("🎛️ Hyperparameters", expanded=True):
        # TODO: add your hyperparameter sliders / selectboxes here
        # example:
        # param1 = st.slider("param1", 1, 100, 10)
        # param2 = st.selectbox("param2", ["option_a", "option_b"])

# ── 3. Load data ───────────────────────────────────────────────────────────────
        data = load_classification_data(
    dataset_key, test_size=test_size, random_state=rand_state,
    n_samples=n_samples, noise=noise,
)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
class_names     = data["class_names"]
feature_names   = data["feature_names"]

# ── 4. Fit model ───────────────────────────────────────────────────────────────
# TODO: instantiate and fit your model
# model = YourModel(param1=param1, param2=param2, random_state=rand_state)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# ── 5. Metrics strip ───────────────────────────────────────────────────────────
# mets = metrics_dict(y_test, y_pred)
# metric_row(mets)
# st.markdown("<br>", unsafe_allow_html=True)

# ── 6. Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📊 Decision Boundary",
    "🔢 Confusion Matrix",
    "📈 ROC Curve",
    # TODO: add more tabs as needed
])

with tab1:
    # TODO: use fig_decision_boundary() for 2-D data
    st.info("Implement Tab 1")

with tab2:
    # TODO: use fig_confusion_matrix()
    st.info("Implement Tab 2")

with tab3:
    # TODO: use fig_roc_curves()
    st.info("Implement Tab 3")

# ── 7. Optional algorithm-specific section ────────────────────────────────────
# For example: coefficients, support vectors, tree, etc.
# Add more tabs or expanders as needed.

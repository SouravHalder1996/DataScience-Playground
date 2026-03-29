"""
pages/1_Logistic_Regression.py
Interactive Logistic Regression explorer.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

from utils.helpers import (
    inject_page_css, page_header, info_box, metric_row, sidebar_header,
    DATASETS, load_classification_data,
    fig_decision_boundary, fig_confusion_matrix,
    fig_roc_curves, metrics_dict, PLOTLY_LAYOUT, PALETTE,
)

st.set_page_config(page_title="Logistic Regression · ML Playground",
                   page_icon="🔵", layout="wide")
inject_page_css()

# ── Header ─────────────────────────────────────────────────────────────────────
page_header(
    "Logistic Regression", "Linear decision boundary · probabilistic output · L1 / L2 regularisation",
    emoji="🔵",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
sidebar_header()
with st.sidebar:
    with st.expander("⚙️ Dataset", expanded=False):
        dataset_label = st.selectbox("Dataset", list(DATASETS.keys()))
        dataset_key   = DATASETS[dataset_label]

        test_size  = st.slider("Test split", 0.10, 0.50, 0.25, 0.05)
        rand_state = st.slider("Random seed", 0, 99, 42)

        if dataset_key in ("moons", "circles", "synthetic"):
            n_samples = st.slider("n_samples", 100, 1000, 300, 50)
            noise     = st.slider("Noise level", 0.0, 0.5, 0.15, 0.01)
        else:
            n_samples, noise = 300, 0.15

    with st.expander("🎛️ Hyperparameters", expanded=False):

        C        = st.select_slider("C  (inverse regularisation)",
                                    options=[0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0], value=1.0)
        penalty  = st.selectbox("Penalty", ["l2", "l1", "elasticnet", "none"])
        solver   = st.selectbox("Solver",
                                ["lbfgs", "liblinear", "saga", "newton-cg", "sag"],
                                help="solver must be compatible with penalty")
        max_iter = st.slider("max_iter", 50, 2000, 200, 50)

        if penalty == "elasticnet":
            l1_ratio = st.slider("l1_ratio (Elastic Net)", 0.0, 1.0, 0.5, 0.05)
        else:
            l1_ratio = None

    with st.expander("🖼️ Visualisation", expanded=False):
        show_boundary = st.checkbox("Decision boundary (2-D)", value=True)
        decision_res  = st.select_slider("Boundary resolution", [50, 100, 150, 200], value=100)

# ── Load data ──────────────────────────────────────────────────────────────────
data = load_classification_data(
    dataset_key, test_size=test_size, random_state=rand_state,
    n_samples=n_samples, noise=noise,
)
X_train, X_test = data["X_train"], data["X_test"]
y_train, y_test = data["y_train"], data["y_test"]
class_names     = data["class_names"]
feature_names   = data["feature_names"]

# ── Fit model ──────────────────────────────────────────────────────────────────
penalty_arg = None if penalty == "none" else penalty

try:
    model = LogisticRegression(
        C=C if penalty_arg else 1e9,
        penalty=penalty_arg,
        solver=solver,
        max_iter=max_iter,
        l1_ratio=l1_ratio,
        random_state=rand_state,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    fit_error = None
except Exception as e:
    fit_error = str(e)

# ── Main area ──────────────────────────────────────────────────────────────────
if fit_error:
    st.error(f"⚠️ Solver/penalty combination failed: `{fit_error}`\n\nTry changing the solver or penalty in the sidebar.")
    st.stop()

# Metrics strip
mets = metrics_dict(y_test, y_pred)
metric_row({**mets, "Train samples": int(len(y_train)), "Test samples": int(len(y_test))})
st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Decision Boundary", "🔢 Confusion Matrix", "📈 ROC Curve", "🧮 Coefficients"])

# ── Tab 1 : Decision boundary ──────────────────────────────────────────────────
with tab1:
    is_2d = X_train.shape[1] == 2

    if is_2d and show_boundary:
        fig = fig_decision_boundary(model, data["X"], data["y"], class_names,
                                    resolution=decision_res,
                                    title=f"Decision Boundary — C={C}, {penalty} penalty")
        st.plotly_chart(fig, use_container_width=True)
    elif not is_2d:
        info_box("Decision boundary visualisation requires 2-D data. "
                 "Select <b>Moons</b>, <b>Circles</b>, or <b>Synthetic (2 features)</b> in the sidebar, "
                 "or the first two features are projected below.")
        # Project to 2-D for display
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X2 = pca.fit_transform(data["X"])
        import plotly.express as px
        import pandas as pd
        df2 = pd.DataFrame(X2, columns=["PC1", "PC2"])
        df2["label"] = [class_names[i] for i in data["y"]]
        fig = px.scatter(df2, x="PC1", y="PC2", color="label",
                         color_discrete_sequence=PALETTE,
                         title="PCA projection of data (2-D)")
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Enable 'Decision boundary' in the sidebar to see this plot.")

# ── Tab 2 : Confusion matrix ───────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns([1, 1])
    with col1:
        fig_cm = fig_confusion_matrix(y_test, y_pred, class_names)
        st.plotly_chart(fig_cm, use_container_width=True)
    with col2:
        from sklearn.metrics import classification_report
        import pandas as pd
        report = classification_report(y_test, y_pred,
                                       target_names=class_names,
                                       output_dict=True)
        df_rep = pd.DataFrame(report).transpose()
        st.markdown("##### Classification Report")
        st.dataframe(df_rep.style.background_gradient(cmap="Purples", axis=None)
                     .format("{:.3f}"),
                     use_container_width=True)

# ── Tab 3 : ROC ───────────────────────────────────────────────────────────────
with tab3:
    fig_roc = fig_roc_curves(model, X_test, y_test, class_names)
    if fig_roc.data:
        st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.info("ROC curve requires `predict_proba`. All solvers for Logistic Regression support this.")

# ── Tab 4 : Coefficients ───────────────────────────────────────────────────────
with tab4:
    coef = model.coef_   # shape (n_classes, n_features) or (1, n_features)
    import plotly.express as px, pandas as pd

    if coef.shape[0] == 1:
        # Binary
        df_coef = pd.DataFrame({"Feature": feature_names, "Coefficient": coef[0]})
        df_coef = df_coef.sort_values("Coefficient")
        colors  = [PALETTE[0] if v >= 0 else PALETTE[3] for v in df_coef["Coefficient"]]
        fig_c = go.Figure(go.Bar(
            x=df_coef["Coefficient"], y=df_coef["Feature"],
            orientation="h",
            marker_color=colors,
        ))
        fig_c.update_layout(**PLOTLY_LAYOUT, title="Learned Coefficients",
                            xaxis_title="Coefficient", yaxis_title="")
    else:
        # Multi-class — heatmap
        df_coef = pd.DataFrame(coef, columns=feature_names,
                               index=[f"{c} (vs rest)" for c in class_names])
        fig_c = go.Figure(go.Heatmap(
            z=coef,
            x=feature_names,
            y=[f"{c} (OvR)" for c in class_names],
            colorscale=[[0, "#f07070"], [0.5, "#0d0d14"], [1, "#6c63ff"]],
            zmid=0,
            colorbar=dict(title="Coefficient"),
        ))
        fig_c.update_layout(**PLOTLY_LAYOUT, title="Coefficient Heatmap (multi-class OvR)")

    st.plotly_chart(fig_c, use_container_width=True)

    info_box(
        f"<b>Intercept:</b> {model.intercept_} &nbsp;·&nbsp; "
        f"<b>n_iter:</b> {model.n_iter_} &nbsp;·&nbsp; "
        f"<b>Classes:</b> {list(model.classes_)}"
    )

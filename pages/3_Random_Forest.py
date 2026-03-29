"""
pages/3_Random_Forest.py
Interactive Random Forest explorer.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier

from utils.helpers import (
    inject_page_css, page_header, info_box, metric_row, sidebar_header,
    DATASETS, load_classification_data,
    fig_decision_boundary, fig_confusion_matrix,
    fig_roc_curves, fig_feature_importance,
    metrics_dict, PLOTLY_LAYOUT, PALETTE, BORDER,
)

st.set_page_config(page_title="Random Forest · ML Playground",
                   page_icon="🌲", layout="wide")
inject_page_css()

page_header(
    "Random Forest", "Bagged ensemble of decision trees · OOB error · feature importances",
    emoji="🌲",
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

        n_estimators      = st.slider("n_estimators", 1, 500, 100, 5)
        criterion         = st.selectbox("Criterion", ["gini", "entropy", "log_loss"])
        max_depth         = st.slider("max_depth", 1, 20, 5)
        max_features      = st.selectbox("max_features", ["sqrt", "log2", "None (all)", "0.5"])
        min_samples_split = st.slider("min_samples_split", 2, 30, 2)
        min_samples_leaf  = st.slider("min_samples_leaf", 1, 30, 1)
        bootstrap         = st.checkbox("Bootstrap sampling", value=True)
        oob_score         = st.checkbox("OOB score", value=True) if bootstrap else False

    with st.expander("🖼️ Visualisation", expanded=False):
        show_boundary     = st.checkbox("Decision boundary (2-D)", value=True)
        decision_res      = st.select_slider("Boundary resolution", [50, 100, 150, 200], value=100)
        show_oob_curve    = st.checkbox("Show OOB error vs n_estimators", value=True)

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

@st.cache_data(show_spinner="🌲 Growing forest …")
def fit_rf(n_est, crit, depth, mf, mss, msl, boot, oob, rstate,
           _X_train, _y_train):
    return RandomForestClassifier(
        n_estimators=n_est, criterion=crit, max_depth=depth,
        max_features=mf, min_samples_split=mss, min_samples_leaf=msl,
        bootstrap=boot, oob_score=oob, random_state=rstate, n_jobs=-1,
    ).fit(_X_train, _y_train)

model = fit_rf(
    n_estimators, criterion, max_depth, mf,
    min_samples_split, min_samples_leaf,
    bootstrap, oob_score, rand_state,
    X_train, y_train,
)
y_pred = model.predict(X_test)

# ── Metrics ────────────────────────────────────────────────────────────────────
mets = metrics_dict(y_test, y_pred)
extra = {"OOB Score": round(model.oob_score_, 4)} if oob_score else {}
metric_row({**mets, **extra, "n_estimators": n_estimators})
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Decision Boundary", "🔢 Confusion Matrix", "📈 ROC Curve",
    "🌿 Feature Importance", "📉 OOB Error Curve",
])

with tab1:
    if X_train.shape[1] == 2 and show_boundary:
        fig = fig_decision_boundary(model, data["X"], data["y"], class_names,
                                    resolution=decision_res,
                                    title=f"Decision Boundary — {n_estimators} trees, depth≤{max_depth}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        info_box("Select a 2-D dataset (Moons / Circles / Synthetic 2 features) to see decision boundary.")

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
    std_imp = np.std([t.feature_importances_ for t in model.estimators_], axis=0)

    idx = np.argsort(importances)[-15:]
    fig_fi = go.Figure()
    fig_fi.add_trace(go.Bar(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        error_x=dict(type="data", array=std_imp[idx], color="#555568"),
        marker=dict(
            color=importances[idx],
            colorscale=[[0, "#1e1e2e"], [1, "#6c63ff"]],
            showscale=False,
        ),
    ))
    fig_fi.update_layout(**PLOTLY_LAYOUT, title="Feature Importances (mean ± std across trees)",
                         xaxis_title="Mean Decrease in Impurity")
    st.plotly_chart(fig_fi, use_container_width=True)

with tab5:
    if not bootstrap:
        st.info("OOB error requires bootstrap=True.")
    elif show_oob_curve:
        with st.spinner("Computing OOB curve …"):
            step = max(1, n_estimators // 40)
            ns_range = list(range(1, n_estimators + 1, step))
            if ns_range[-1] != n_estimators:
                ns_range.append(n_estimators)

            @st.cache_data(show_spinner=False)
            def oob_curve(_X_train, _y_train, max_n, crit, depth, mf_, mss, msl, rstate):
                oob_errors = []
                estimator = RandomForestClassifier(
                    n_estimators=max_n, criterion=crit, max_depth=depth,
                    max_features=mf_, min_samples_split=mss, min_samples_leaf=msl,
                    bootstrap=True, oob_score=True, warm_start=False,
                    random_state=rstate, n_jobs=-1,
                )
                # Build incrementally
                results = {}
                for n in ns_range:
                    m = RandomForestClassifier(
                        n_estimators=n, criterion=crit, max_depth=depth,
                        max_features=mf_, min_samples_split=mss, min_samples_leaf=msl,
                        bootstrap=True, oob_score=True, random_state=rstate, n_jobs=-1,
                    ).fit(_X_train, _y_train)
                    results[n] = 1 - m.oob_score_
                return results

            oob_dict = oob_curve(
                X_train, y_train, n_estimators,
                criterion, max_depth, mf,
                min_samples_split, min_samples_leaf, rand_state,
            )
            ns_vals   = list(oob_dict.keys())
            oob_vals  = list(oob_dict.values())

        fig_oob = go.Figure()
        fig_oob.add_trace(go.Scatter(
            x=ns_vals, y=oob_vals, mode="lines+markers",
            line=dict(color=PALETTE[0], width=2.5),
            marker=dict(size=5, color=PALETTE[0]),
            name="OOB Error",
        ))
        fig_oob.add_vline(x=n_estimators, line=dict(color=PALETTE[1], dash="dash", width=1.5),
                          annotation_text=f"n={n_estimators}", annotation_position="top right")
        fig_oob.update_layout(**PLOTLY_LAYOUT,
                              title="OOB Error vs n_estimators",
                              xaxis_title="n_estimators",
                              yaxis_title="OOB Error")
        st.plotly_chart(fig_oob, use_container_width=True)
    else:
        st.info("Enable 'Show OOB error vs n_estimators' in the sidebar.")

"""
utils/helpers.py
Shared helpers for the ML Playground pages.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.datasets import (
    load_iris, load_breast_cancer, load_wine,
    make_classification, make_moons, make_circles, make_blobs,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    classification_report, accuracy_score,
    precision_score, recall_score, f1_score,
)

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = ["#6c63ff", "#52c4a0", "#f0a050", "#f07070", "#50b0f0", "#c47aff"]
DARK_BG  = "#0a0a0f"
CARD_BG  = "#111118"
BORDER   = "#1e1e2e"
TEXT     = "#e0e0ef"
MUTED    = "#606078"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0d0d14",
    font=dict(family="Space Grotesk, sans-serif", color=TEXT, size=12),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
)


# ── Dataset loader ─────────────────────────────────────────────────────────────
DATASETS = {
    "Iris (4 features, 3 classes)":         "iris",
    "Breast Cancer (30 features, 2 cls)":   "breast_cancer",
    "Wine (13 features, 3 classes)":         "wine",
    "Moons (2 features, 2 classes)":         "moons",
    "Circles (2 features, 2 classes)":       "circles",
    "Synthetic (configurable)":              "synthetic",
}

CLUSTERING_DATASETS = {
    "Blobs (isotropic)":    "blobs",
    "Moons":                "moons",
    "Circles":              "circles",
    "Anisotropic blobs":    "aniso",
}


def load_classification_data(
    name: str,
    test_size: float = 0.25,
    random_state: int = 42,
    n_samples: int = 300,
    n_features: int = 2,
    n_classes: int = 2,
    noise: float = 0.15,
) -> dict:
    """Return a dict with X_train/test, y_train/test, feature_names, class_names, df."""

    if name == "iris":
        raw = load_iris()
        X, y = raw.data, raw.target
        feature_names = list(raw.feature_names)
        class_names   = list(raw.target_names)

    elif name == "breast_cancer":
        raw = load_breast_cancer()
        X, y = raw.data, raw.target
        feature_names = list(raw.feature_names)
        class_names   = list(raw.target_names)

    elif name == "wine":
        raw = load_wine()
        X, y = raw.data, raw.target
        feature_names = list(raw.feature_names)
        class_names   = [f"Class {i}" for i in raw.target_names]

    elif name == "moons":
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        feature_names = ["Feature 1", "Feature 2"]
        class_names   = ["Class 0", "Class 1"]

    elif name == "circles":
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
        feature_names = ["Feature 1", "Feature 2"]
        class_names   = ["Class 0", "Class 1"]

    else:  # synthetic
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features - 1),
            n_redundant=0,
            n_classes=n_classes,
            random_state=random_state,
        )
        feature_names = [f"Feature {i+1}" for i in range(n_features)]
        class_names   = [f"Class {i}" for i in range(n_classes)]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = [class_names[i] for i in y]

    return dict(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        X=X, y=y,
        feature_names=feature_names,
        class_names=class_names,
        df=df,
        scaler=scaler,
    )


def load_clustering_data(
    name: str,
    n_samples: int = 300,
    n_clusters: int = 3,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (X, true_labels_or_None)."""
    if name == "blobs":
        X, y = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state)
        return X, y
    elif name == "moons":
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        return X, y
    elif name == "circles":
        X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.4, random_state=random_state)
        return X, y
    elif name == "aniso":
        X, y = make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state)
        rng = np.random.RandomState(random_state)
        T = rng.randn(2, 2)
        X = X @ T
        return X, y
    return make_blobs(n_samples=n_samples, centers=n_clusters, random_state=random_state)


# ── Plot helpers ───────────────────────────────────────────────────────────────

def fig_scatter_2d(X, y, class_names: list[str], title="Data") -> go.Figure:
    """2-D scatter coloured by class (uses first 2 features)."""
    df = pd.DataFrame(X[:, :2], columns=["x", "y"])
    df["label"] = [class_names[i] for i in y]

    fig = px.scatter(df, x="x", y="y", color="label",
                     color_discrete_sequence=PALETTE,
                     title=title)
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=0)))
    return fig


def fig_decision_boundary(
    model, X, y, class_names: list[str],
    resolution: int = 200,
    title: str = "Decision Boundary",
) -> go.Figure:
    """Mesh-based decision boundary for 2-D data."""
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Soft region colours
    region_colors = [
        f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.15)"
        for c in PALETTE
    ]

    fig = go.Figure()
    unique = np.unique(Z)
    for cls_idx in unique:
        mask = Z == cls_idx
        fig.add_trace(go.Contour(
            x=np.linspace(x_min, x_max, resolution),
            y=np.linspace(y_min, y_max, resolution),
            z=(Z == cls_idx).astype(float),
            showscale=False,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, region_colors[cls_idx % len(region_colors)]]],
            contours=dict(start=0.5, end=1.5, size=1),
            hoverinfo="skip",
            name="",
        ))

    # Data points
    for cls_idx, cls_name in enumerate(class_names):
        mask = y == cls_idx
        fig.add_trace(go.Scatter(
            x=X[mask, 0], y=X[mask, 1],
            mode="markers",
            name=cls_name,
            marker=dict(
                color=PALETTE[cls_idx % len(PALETTE)],
                size=7, opacity=0.9,
                line=dict(width=1, color="#0a0a0f"),
            ),
        ))

    fig.update_layout(**PLOTLY_LAYOUT, title=title)
    return fig


def fig_confusion_matrix(y_true, y_pred, class_names: list[str]) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    text = [[f"{cm[i,j]}<br>({cm_norm[i,j]:.0%})"
             for j in range(len(class_names))]
            for i in range(len(class_names))]

    fig = go.Figure(go.Heatmap(
        z=cm_norm,
        x=class_names, y=class_names,
        colorscale=[[0, "#0d0d14"], [1, "#6c63ff"]],
        showscale=True,
        text=text, texttemplate="%{text}",
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{text}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="Actual",
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def fig_roc_curves(model, X_test, y_test, class_names: list[str]) -> go.Figure:
    """Multi-class ROC curves (OvR)."""
    n_classes = len(class_names)
    fig = go.Figure()

    if n_classes == 2:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            proba = model.decision_function(X_test)
        else:
            return fig
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"AUC = {roc_auc:.3f}",
            line=dict(color=PALETTE[0], width=2.5),
        ))
    else:
        if not hasattr(model, "predict_proba"):
            return fig
        proba = model.predict_proba(X_test)
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_test, classes=list(range(n_classes)))
        for i, cls in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            roc_auc = auc(fpr, tpr)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{cls} (AUC={roc_auc:.2f})",
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
            ))

    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random", line=dict(color=MUTED, width=1.5, dash="dash"),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return fig


def fig_feature_importance(importances: np.ndarray, feature_names: list[str], top_n: int = 15) -> go.Figure:
    idx = np.argsort(importances)[-top_n:]
    fig = go.Figure(go.Bar(
        x=importances[idx],
        y=[feature_names[i] for i in idx],
        orientation="h",
        marker=dict(
            color=importances[idx],
            colorscale=[[0, "#1e1e2e"], [1, "#6c63ff"]],
            showscale=False,
        ),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Feature Importances",
                      xaxis_title="Importance", yaxis_title="")
    return fig


def metrics_dict(y_true, y_pred, average="weighted") -> dict:
    return dict(
        Accuracy  = accuracy_score(y_true, y_pred),
        Precision = precision_score(y_true, y_pred, average=average, zero_division=0),
        Recall    = recall_score(y_true, y_pred, average=average, zero_division=0),
        F1        = f1_score(y_true, y_pred, average=average, zero_division=0),
    )


# ── Streamlit UI helpers ───────────────────────────────────────────────────────

def inject_page_css():
    """Inject dark-theme CSS into any page."""
    import streamlit as st
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background: #0a0a0f; color: #e8e8f0; }
    .block-container { padding: 1.5rem 2.5rem !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d0d16 !important;
        border-right: 1px solid #1e1e2e !important;
    }
    [data-testid="stSidebar"] .stSlider > div > div > div { background: #6c63ff !important; }
                
    /* Hide sidebar page navigation */
    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    /* Fix content hidden under navbar */
    .block-container {
        padding-top: 3rem !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: #111118;
        border: 1px solid #1e1e2e;
        border-radius: 10px;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; color: #6c63ff !important; }

    /* Buttons */
    .stButton > button {
        background: #111118; border: 1px solid #2e2e45;
        color: #e0e0ef; border-radius: 8px;
        font-family: 'Space Grotesk', sans-serif;
        transition: all 0.2s;
    }
    .stButton > button:hover { border-color: #6c63ff; color: #6c63ff; }

    /* Selectbox */
    [data-testid="stSelectbox"] > div > div {
        background: #111118 !important;
        border-color: #1e1e2e !important;
    }

    /* Tab styling */
    [data-testid="stTabs"] button {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #606078 !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #6c63ff !important;
        border-bottom-color: #6c63ff !important;
    }

    /* Page title style */
    .page-header { padding: 0.5rem 0 1.5rem; border-bottom: 1px solid #1e1e2e; margin-bottom: 1.5rem; }
    .page-title { font-size: 1.8rem; font-weight: 700; color: #e0e0ef; margin: 0; }
    .page-subtitle { font-size: 0.88rem; color: #606078; margin: 0.3rem 0 0; font-family: 'JetBrains Mono', monospace; }

    /* Info box */
    .info-box {
        background: #111118; border: 1px solid #1e1e2e;
        border-left: 3px solid #6c63ff;
        border-radius: 8px; padding: 0.9rem 1.1rem;
        font-size: 0.85rem; color: #9090a8;
        line-height: 1.6; margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, emoji: str = ""):
    import streamlit as st
    st.markdown(f"""
    <div class="page-header">
        <p class="page-title">{emoji} {title}</p>
        <p class="page-subtitle">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def info_box(text: str):
    import streamlit as st
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def metric_row(metrics: dict):
    import streamlit as st
    cols = st.columns(len(metrics))
    for col, (name, val) in zip(cols, metrics.items()):
        col.metric(name, f"{val:.4f}" if isinstance(val, float) else str(val))

def sidebar_header():
    import streamlit as st
    with st.sidebar:
        st.markdown("""
        <style>
        div[data-testid="stSidebar"] .home-btn button {
            background: linear-gradient(135deg, #1a1a2e, #16162a) !important;
            border: 1px solid #6c63ff !important;
            color: #6c63ff !important;
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.78rem !important;
            letter-spacing: 0.08em !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
        }
        div[data-testid="stSidebar"] .home-btn button:hover {
            background: #6c63ff !important;
            color: #fff !important;
            box-shadow: 0 0 20px rgba(108,99,255,0.4) !important;
            transform: translateX(-3px) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="home-btn">', unsafe_allow_html=True)
        if st.button("← Back to Home", use_container_width=True):
            st.markdown("""
            <style>
            .block-container {
                animation: fadeOut 0.3s ease forwards;
            }
            @keyframes fadeOut {
                from { opacity: 1; transform: translateY(0); }
                to   { opacity: 0; transform: translateY(10px); }
            }
            </style>
            """, unsafe_allow_html=True)
            import time
            time.sleep(0.3)
            st.switch_page("app.py")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<hr style='border-color:#1e1e2e; margin: 0.8rem 0'>", unsafe_allow_html=True)

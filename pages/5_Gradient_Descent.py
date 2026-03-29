"""
pages/5_Gradient_Descent.py
Interactive Gradient Descent explorer — Batch, Stochastic, Mini-Batch.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils.helpers import (
    inject_page_css, page_header, info_box, sidebar_header,
    PLOTLY_LAYOUT, PALETTE, TEXT, BORDER,
)

st.set_page_config(
    page_title="Gradient Descent · ML Playground",
    page_icon="📉",
    layout="wide",
)
inject_page_css()
sidebar_header()

page_header(
    "Gradient Descent",
    "Batch · Stochastic · Mini-Batch — watch every step descend the loss surface",
    emoji="📉",
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    with st.expander("⚙️ Data", expanded=True):
        n_samples  = st.slider("n_samples", 50, 500, 150, 50)
        noise      = st.slider("Noise", 0.1, 3.0, 1.0, 0.1)
        rand_state = st.slider("Random seed", 0, 99, 42)

    with st.expander("🎛️ Hyperparameters", expanded=True):
        lr         = st.select_slider(
                        "Learning rate",
                        options=[0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5],
                        value=0.05,
                     )
        epochs     = st.slider("Epochs", 5, 200, 50, 5)
        batch_size = st.slider("Mini-Batch size", 2, 64, 16, 2)
        w0_init    = st.slider("Initial w₀ (intercept)", -5.0, 5.0, 4.0, 0.5)
        w1_init    = st.slider("Initial w₁ (slope)",    -5.0, 5.0, -4.0, 0.5)
        momentum   = st.slider("Momentum (β)", 0.0, 0.99, 0.0, 0.01)

    with st.expander("🖼️ Visualisation", expanded=True):
        contour_res  = st.select_slider("Contour resolution", [30, 50, 80, 100], value=50)
        anim_speed   = st.select_slider(
                          "Animation speed (ms/step)",
                          options=[20, 50, 100, 200, 500],
                          value=100,
                       )

# ── Helper functions ───────────────────────────────────────────────────────────
def mse(w0, w1, X, y):
    return float(np.mean((w0 + w1 * X - y) ** 2))

def grads(w0, w1, X, y):
    e = w0 + w1 * X - y
    return 2 * np.mean(e), 2 * np.mean(e * X)

# ── Generate & normalise data ──────────────────────────────────────────────────
rng  = np.random.RandomState(rand_state)
Xraw = rng.uniform(-3, 3, n_samples)
yraw = 2.0 * Xraw + 1.0 + rng.randn(n_samples) * noise
X    = (Xraw - Xraw.mean()) / (Xraw.std() + 1e-8)
y    = (yraw - yraw.mean()) / (yraw.std() + 1e-8)

# ── Run all 3 variants ─────────────────────────────────────────────────────────
tol = 1e-5  # convergence threshold

def run_variant(variant: str):
    w0, w1 = float(w0_init), float(w1_init)
    v0, v1 = 0.0, 0.0
    rng2   = np.random.RandomState(rand_state)
    pw0, pw1, pl = [w0], [w1], [mse(w0, w1, X, y)]
    converged_at  = None

    for epoch in range(epochs):
        if variant == "batch":
            g0, g1 = grads(w0, w1, X, y)
            v0 = momentum * v0 + (1 - momentum) * g0
            v1 = momentum * v1 + (1 - momentum) * g1
            w0 -= lr * v0;  w1 -= lr * v1

        elif variant == "sgd":
            for i in rng2.permutation(n_samples):
                g0, g1 = grads(w0, w1, X[i:i+1], y[i:i+1])
                v0 = momentum * v0 + (1 - momentum) * g0
                v1 = momentum * v1 + (1 - momentum) * g1
                w0 -= lr * v0;  w1 -= lr * v1

        elif variant == "minibatch":
            for s in range(0, n_samples, batch_size):
                b  = rng2.permutation(n_samples)[s:s+batch_size]
                g0, g1 = grads(w0, w1, X[b], y[b])
                v0 = momentum * v0 + (1 - momentum) * g0
                v1 = momentum * v1 + (1 - momentum) * g1
                w0 -= lr * v0;  w1 -= lr * v1

        pw0.append(w0);  pw1.append(w1)
        pl.append(mse(w0, w1, X, y))

        if len(pl) > 2 and abs(pl[-1] - pl[-2]) < tol:
            converged_at = epoch + 1
            break

    return np.array(pw0), np.array(pw1), np.array(pl), converged_at

bw0, bw1, bl, b_conv = run_variant("batch")
sw0, sw1, sl, s_conv = run_variant("sgd")
mw0, mw1, ml, m_conv = run_variant("minibatch")

VARIANTS = {
    "Batch GD":      (bw0, bw1, bl, PALETTE[0], b_conv),
    "Stochastic GD": (sw0, sw1, sl, PALETTE[1], s_conv),
    "Mini-Batch GD": (mw0, mw1, ml, PALETTE[2], m_conv),
}

# ── Contour loss surface ───────────────────────────────────────────────────────
all_w0 = np.concatenate([bw0, sw0, mw0, [w0_init]])
all_w1 = np.concatenate([bw1, sw1, mw1, [w1_init]])
w0r = np.linspace(all_w0.min() - 1, all_w0.max() + 1, contour_res)
w1r = np.linspace(all_w1.min() - 1, all_w1.max() + 1, contour_res)
Zg  = np.array([[mse(a, b, X, y) for a in w0r] for b in w1r])

def base_contour(title=""):
    fig = go.Figure()
    fig.add_trace(go.Contour(
        z=Zg, x=w0r, y=w1r,
        colorscale=[[0,"#0d0d14"],[0.4,"#1a1a3a"],[0.7,"#3a2a5a"],[1,"#6c63ff"]],
        showscale=True, opacity=0.85,
        contours=dict(showlabels=False, coloring="heatmap"),
        colorbar=dict(title="MSE", tickfont=dict(color=TEXT)),
        hovertemplate="w₀=%{x:.2f}<br>w₁=%{y:.2f}<br>Loss=%{z:.4f}<extra></extra>",
        name="Loss surface",
    ))
    mi = np.unravel_index(np.argmin(Zg), Zg.shape)
    fig.add_trace(go.Scatter(
        x=[w0r[mi[1]]], y=[w1r[mi[0]]],
        mode="markers", name="Minimum",
        marker=dict(symbol="star", size=16, color="#f0a050",
                    line=dict(color="#fff", width=1)),
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT, title=title,
        xaxis_title="w₀ (intercept)", yaxis_title="w₁ (slope)",
        height=440,
    )
    return fig

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ All Paths",
    "🎬 Animate",
    "📉 Loss Curves",
    "📊 Side-by-Side",
    "📋 Stats",
])

# ── Tab 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    fig = base_contour("All 3 Gradient Descent Paths")
    for name, (pw0, pw1, pl, color, conv) in VARIANTS.items():
        fig.add_trace(go.Scatter(
            x=pw0, y=pw1, mode="lines+markers", name=name,
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
        ))
        fig.add_trace(go.Scatter(
            x=[pw0[0]], y=[pw1[0]], mode="markers", showlegend=False,
            marker=dict(symbol="circle-open", size=13, color=color,
                        line=dict(width=2.5, color=color)),
        ))
        fig.add_trace(go.Scatter(
            x=[pw0[-1]], y=[pw1[-1]], mode="markers", showlegend=False,
            marker=dict(symbol="x", size=11, color=color,
                        line=dict(width=2.5, color=color)),
        ))
    st.plotly_chart(fig, use_container_width=True)
    conv_msgs = []
    for name, (*_, color, conv) in VARIANTS.items():
        if conv:
            conv_msgs.append(f"<b>{name}</b>: converged at epoch {conv}")
        else:
            conv_msgs.append(f"<b>{name}</b>: did not fully converge in {epochs} epochs")
    info_box(
        "⭐ <b>Gold star</b> = minimum &nbsp;·&nbsp; "
        "<b>Open circle</b> = start &nbsp;·&nbsp; "
        "<b>✕</b> = final position<br><br>" +
        " &nbsp;·&nbsp; ".join(conv_msgs)
    )

# ── Tab 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns([1, 2])
    with col1:
        anim_var = st.selectbox(
            "Which variant",
            ["All 3", "Batch GD", "Stochastic GD", "Mini-Batch GD"],
            key="anim_var",
        )

    to_animate = (
        list(VARIANTS.items())
        if anim_var == "All 3"
        else [(anim_var, VARIANTS[anim_var])]
    )

    max_steps = max(len(v[0]) for _, v in to_animate)

    # Build Plotly animation frames
    frames = []
    for step in range(0, max_steps + 1):
        frame_traces = []
        # contour + star (always present)
        frame_traces.append(go.Contour(
            z=Zg, x=w0r, y=w1r,
            colorscale=[[0,"#0d0d14"],[0.4,"#1a1a3a"],[0.7,"#3a2a5a"],[1,"#6c63ff"]],
            showscale=True, opacity=0.85,
            contours=dict(showlabels=False, coloring="heatmap"),
            colorbar=dict(title="MSE", tickfont=dict(color=TEXT)),
            name="Loss surface", showlegend=False,
        ))
        mi = np.unravel_index(np.argmin(Zg), Zg.shape)
        frame_traces.append(go.Scatter(
            x=[w0r[mi[1]]], y=[w1r[mi[0]]],
            mode="markers", name="Minimum", showlegend=False,
            marker=dict(symbol="star", size=16, color="#f0a050",
                        line=dict(color="#fff", width=1)),
        ))
        for name, (pw0, pw1, pl, color, conv) in to_animate:
            s = min(step, len(pw0) - 1)
            # path so far
            frame_traces.append(go.Scatter(
                x=pw0[:s+1] if step > 0 else [pw0[0]],
                y=pw1[:s+1] if step > 0 else [pw1[0]],
                mode="lines+markers" if step > 0 else "markers",
                name=name, showlegend=True,
                line=dict(color=color, width=2),
                marker=dict(size=4, color=color),
            ))
            # current position dot
            frame_traces.append(go.Scatter(
                x=[pw0[s]], y=[pw1[s]],
                mode="markers", showlegend=False,
                marker=dict(size=14, color=color,
                            line=dict(color="#fff", width=2)),
            ))
        frames.append(go.Frame(data=frame_traces, name=str(step)))

    # Initial figure (step 0 — just start point, no path)
    fig_anim = base_contour("Gradient Descent Animation")
    for name, (pw0, pw1, pl, color, conv) in to_animate:
        fig_anim.add_trace(go.Scatter(
            x=[pw0[0]], y=[pw1[0]],
            mode="markers", name=name,
            marker=dict(size=10, color=color,
                        symbol="circle-open",
                        line=dict(color=color, width=2.5)),
        ))
        fig_anim.add_trace(go.Scatter(
            x=[pw0[0]], y=[pw1[0]],
            mode="markers", showlegend=False,
            marker=dict(size=14, color=color,
                        line=dict(color="#fff", width=2)),
        ))

    fig_anim.frames = frames

    fig_anim.update_layout(
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=-0.18, x=0.5, xanchor="center", yanchor="top",
            buttons=[
                dict(
                    label="▶  Play",
                    method="animate",
                    args=[None, dict(
                        frame=dict(duration=anim_speed, redraw=True),
                        fromcurrent=True,
                        transition=dict(duration=0),
                        mode="immediate",
                    )],
                ),
                dict(
                    label="⏹  Stop",
                    method="animate",
                    args=[[None], dict(
                        frame=dict(duration=0, redraw=False),
                        mode="immediate",
                        transition=dict(duration=0),
                    )],
                ),
            ],
        )],
        sliders=[dict(
            steps=[
                dict(method="animate",
                     args=[[str(k)], dict(mode="immediate",
                                          frame=dict(duration=anim_speed, redraw=True),
                                          transition=dict(duration=0))],
                     label=str(k))
                for k in range(max_steps + 1)
            ],
            x=0, y=0, len=1.0,
            currentvalue=dict(prefix="Epoch: ", font=dict(color=TEXT)),
            pad=dict(t=50),
            bgcolor="#1e1e2e",
            activebgcolor="#6c63ff",
            bordercolor=BORDER,
            font=dict(color=TEXT),
        )],
        height=520,
    )

    st.plotly_chart(fig_anim, use_container_width=True)
    info_box(
        "Hit <b>▶ Play</b> to animate · drag the <b>epoch slider</b> to scrub manually · "
        "select variant in the dropdown above"
    )

# ── Tab 3 ──────────────────────────────────────────────────────────────────────
with tab3:
    fig_l = go.Figure()
    for name, (pw0, pw1, pl, color, conv) in VARIANTS.items():
        fig_l.add_trace(go.Scatter(
            x=list(range(len(pl))), y=pl,
            mode="lines", name=name,
            line=dict(color=color, width=2.5),
        ))
    fig_l.update_layout(
        **PLOTLY_LAYOUT,
        title="Loss vs Epochs",
        xaxis_title="Epoch", yaxis_title="MSE Loss",
        height=400,
    )
    st.plotly_chart(fig_l, use_container_width=True)

    fig_g = go.Figure()
    for name, (pw0, pw1, pl, color, conv) in VARIANTS.items():
        step_sizes = np.sqrt(np.diff(pw0)**2 + np.diff(pw1)**2)
        fig_g.add_trace(go.Scatter(
            x=list(range(1, len(step_sizes)+1)), y=step_sizes,
            mode="lines", name=name,
            line=dict(color=color, width=2),
        ))
    fig_g.update_layout(
        **PLOTLY_LAYOUT,
        title="Step Size ‖Δw‖ vs Epochs",
        xaxis_title="Epoch", yaxis_title="‖Δw‖",
        height=350,
    )
    st.plotly_chart(fig_g, use_container_width=True)

# ── Tab 4 ──────────────────────────────────────────────────────────────────────
with tab4:
    cols = st.columns(3)
    for idx, (name, (pw0, pw1, pl, color, conv)) in enumerate(VARIANTS.items()):
        with cols[idx]:
            fig_s = base_contour(name)
            fig_s.add_trace(go.Scatter(
                x=pw0, y=pw1, mode="lines+markers", name=name,
                line=dict(color=color, width=2.5),
                marker=dict(size=5, color=color),
            ))
            fig_s.add_trace(go.Scatter(
                x=[pw0[0]], y=[pw1[0]], mode="markers", name="Start",
                marker=dict(symbol="circle-open", size=14, color="#f0a050",
                            line=dict(color="#f0a050", width=2)),
            ))
            fig_s.update_layout(height=360, showlegend=False,
                                margin=dict(l=20, r=10, t=40, b=20))
            st.plotly_chart(fig_s, use_container_width=True)
            st.metric("Final Loss", f"{pl[-1]:.5f}")
            converge = int(np.argmax(np.abs(np.diff(pl)) < 1e-4) or epochs)
            st.metric("~Converged at epoch", converge)

# ── Tab 5 ──────────────────────────────────────────────────────────────────────
with tab5:
    rows = []
    for name, (pw0, pw1, pl, color, conv) in VARIANTS.items():
        path_len = float(np.sum(np.sqrt(np.diff(pw0)**2 + np.diff(pw1)**2)))
        rows.append({
            "Variant":        name,
            "Initial Loss":   round(float(pl[0]),  5),
            "Final Loss":     round(float(pl[-1]), 5),
            "Loss Reduction": f"{(pl[0]-pl[-1])/pl[0]*100:.1f}%",
            "Final w₀":       round(float(pw0[-1]), 4),
            "Final w₁":       round(float(pw1[-1]), 4),
            "Path Length":    round(path_len, 4),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    info_box(
        "<b>Batch GD</b> — smooth path, uses all data per step. Slow on large datasets.<br>"
        "<b>Stochastic GD</b> — noisy path, one sample per step. Can escape local minima.<br>"
        "<b>Mini-Batch GD</b> — best of both worlds. Foundation of Adam / SGD in PyTorch."
    )

    with st.expander("📊 Raw training data (first 50 rows)"):
        st.dataframe(
            pd.DataFrame({"X (normalised)": X[:50], "y (normalised)": y[:50]}),
            use_container_width=True,
        )
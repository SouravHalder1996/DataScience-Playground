import streamlit as st
import re

st.set_page_config(
    page_title="ML Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}
.block-container { padding: 2rem 3rem !important; }

/* Hide sidebar on landing page */
section[data-testid="stSidebar"] {
    display: none !important;
}

div[data-testid="column"] {
    position: relative;
}
div[data-testid="column"] a[data-testid="stPageLink-NavLink"] {
    position: absolute;
    inset: 0;
    opacity: 0;
    z-index: 9;
    height: 100%;
    cursor: pointer;
}
            
a:has(.card) {
    text-decoration: none !important;
    color: inherit !important;
    display: block;
    height: 100%;
}

/* Hero */
.hero {
    padding: 4rem 0 2.5rem;
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.25em;
    color: #6c63ff;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 700;
    line-height: 1.05;
    -webkit-text-fill-color: #6c63ff;
    margin: 0 0 1.2rem;
}
.hero-sub {
    font-size: 1.1rem;
    color: #9090a8;
    max-width: 540px;
    margin: 0 0 2.5rem;
    line-height: 1.7;
    text-align: center !important;
    display: block !important;
    float: none !important;
}
.hero-stats {
    display: flex;
    justify-content: center;
    gap: 3rem;
    margin-bottom: 3.5rem;
}
.stat { text-align: center; }
.stat-num {
    font-size: 1.8rem;
    font-weight: 700;
    color: #6c63ff;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label { font-size: 0.78rem; color: #666680; letter-spacing: 0.05em; }

/* Section header */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #555570;
    text-transform: uppercase;
    margin: 0 0 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #1e1e2e;
}

/* Cards */
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1.2rem;
    margin-bottom: 2rem;
    align-items: stretch;
}
.card {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 14px;
    padding: 1.5rem;
    transition: all 0.25s ease;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    min-height: 200px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-sizing: border-box;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #6c63ff);
    opacity: 0;
    transition: opacity 0.25s;
}
.card:hover { border-color: #2e2e45; transform: translateY(-2px); box-shadow: 0 8px 32px rgba(108,99,255,0.12); }
.card:hover::before { opacity: 1; }

.card-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
.card-icon { font-size: 1.8rem; }
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.06em;
    padding: 3px 8px;
    border-radius: 20px;
    font-weight: 600;
    text-transform: uppercase;
}
.badge-classification { background: #1a1a3a; color: #7a72ff; border: 1px solid #2e2e55; }
.badge-clustering     { background: #1a2e2a; color: #52c4a0; border: 1px solid #1e3d35; }
.badge-regression     { background: #2e2010; color: #f0a050; border: 1px solid #3d2a10; }
.badge-optimization   { background: #2a1a2e; color: #c47aff; border: 1px solid #3d1e45; }
.badge-dim-reduction  { background: #10202e; color: #50b0f0; border: 1px solid #103040; }

.card-name {
    font-size: 1.05rem;
    font-weight: 600;
    color: #e0e0ef;
    margin-bottom: 0.4rem;
}
.card-desc {
    font-size: 0.82rem;
    color: #6060780;
    line-height: 1.55;
    color: #787890;
    margin-bottom: 1rem;
}
.card-footer { display: flex; justify-content: space-between; align-items: center; }
.difficulty {
    display: flex; gap: 3px; align-items: center;
}
.dot { width: 6px; height: 6px; border-radius: 50%; }
.dot-filled  { background: var(--accent, #6c63ff); }
.dot-empty   { background: #2a2a3a; }
.difficulty-label { font-size: 0.7rem; color: #555568; margin-left: 5px; font-family: 'JetBrains Mono', monospace; }

.status-live {
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    color: #52c4a0;
    display: flex; align-items: center; gap: 4px;
}
.status-live::before { content: '●'; font-size: 0.55rem; }
.status-soon {
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    color: #444458;
}
.status-soon::before { content: '○ '; }

/* Divider */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e1e2e 20%, #1e1e2e 80%, transparent);
    margin: 3rem 0;
}

/* Footer */
.footer {
    text-align: center;
    padding: 2rem 0;
    font-size: 0.78rem;
    color: #404055;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Interactive Machine Learning</div>
    <h1 class="hero-title">ML Playground</h1>
    <p class="hero-sub">
        Explore every algorithm hands-on. Tune hyperparameters, watch
        decision boundaries shift, and build intuition through real-time feedback.
    </p>
    <div class="hero-stats">
        <div class="stat"><div class="stat-num">12</div><div class="stat-label">Algorithms</div></div>
        <div class="stat"><div class="stat-num">5</div><div class="stat-label">Categories</div></div>
        <div class="stat"><div class="stat-num">∞</div><div class="stat-label">Experiments</div></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Algorithm catalogue ────────────────────────────────────────────────────────
ALGORITHMS = {
    "Classification": [
        dict(
            icon="🔵", name="Logistic Regression",
            desc="Linear decision boundary with probabilistic output. Great for understanding odds ratios and regularisation.",
            difficulty=1, status="live", page="pages/1_Logistic_Regression.py",
            accent="#6c63ff", badge="badge-classification",
        ),
        dict(
            icon="🌳", name="Decision Tree",
            desc="Recursive binary splits on features. Visualise the full tree and watch it over-fit in real time.",
            difficulty=1, status="live", page="pages/2_Decision_Tree.py",
            accent="#7a72ff", badge="badge-classification",
        ),
        dict(
            icon="🌲", name="Random Forest",
            desc="Bagged ensemble of decision trees. Explore feature importance and OOB error as n_estimators grows.",
            difficulty=2, status="live", page="pages/3_Random_Forest.py",
            accent="#c47aff", badge="badge-classification",
        ),
        dict(
            icon="⚡", name="Support Vector Machine",
            desc="Maximum-margin hyperplane with kernel tricks. Switch kernels and C to reshape the decision surface.",
            difficulty=2, status="soon", page=None,
            accent="#f070a0", badge="badge-classification",
        ),
        dict(
            icon="🧮", name="Naive Bayes",
            desc="Probabilistic classifier assuming feature independence. Fast, interpretable, and surprisingly robust.",
            difficulty=1, status="soon", page=None,
            accent="#70b0f0", badge="badge-classification",
        ),
        dict(
            icon="🏘️", name="K-Nearest Neighbours",
            desc="Instance-based learning. Watch how k and the distance metric reshape class regions.",
            difficulty=1, status="soon", page=None,
            accent="#6c63ff", badge="badge-classification",
        ),
    ],
    "Clustering": [
        dict(
            icon="⭕", name="K-Means",
            desc="Partition data into k Voronoi cells. Animate centroid convergence and evaluate with silhouette scores.",
            difficulty=1, status="live", page="pages/4_KMeans.py",
            accent="#52c4a0", badge="badge-clustering",
        ),
        dict(
            icon="🫧", name="DBSCAN",
            desc="Density-based clusters of arbitrary shape. Tune eps & min_samples to control what counts as noise.",
            difficulty=2, status="soon", page=None,
            accent="#52c4a0", badge="badge-clustering",
        ),
    ],
    "Regression": [
        dict(
            icon="📈", name="Linear Regression",
            desc="OLS, Ridge, and Lasso in one page. See how regularisation shrinks coefficients toward zero.",
            difficulty=1, status="soon", page=None,
            accent="#f0a050", badge="badge-regression",
        ),
        dict(
            icon="🌀", name="Polynomial Regression",
            desc="Raise the degree and watch the model interpolate then wildly extrapolate.",
            difficulty=1, status="soon", page=None,
            accent="#f0a050", badge="badge-regression",
        ),
    ],
    "Optimization": [
        dict(
            icon="📉", name="Gradient Descent",
            desc="Batch, Stochastic & Mini-Batch side by side. Watch every step descend the loss surface in real time.",
            difficulty=2, status="live", page="pages/5_Gradient_Descent.py",
            accent="#52c4a0", badge="badge-optimization",
        ),
    ],
    "Dimensionality Reduction": [
        dict(
            icon="🗜️", name="PCA",
            desc="Project high-dimensional data onto principal components. Explained variance bar chart included.",
            difficulty=2, status="soon", page=None,
            accent="#50b0f0", badge="badge-dim-reduction",
        ),
        dict(
            icon="🌌", name="t-SNE",
            desc="Non-linear manifold embedding. Play with perplexity and see cluster structure emerge.",
            difficulty=3, status="soon", page=None,
            accent="#50b0f0", badge="badge-dim-reduction",
        ),
    ],
}


def difficulty_dots(level: int, accent: str) -> str:
    labels = {1: "Beginner", 2: "Intermediate", 3: "Advanced"}
    dots = "".join(
        f'<div class="dot dot-filled" style="background:{accent}"></div>' if i < level
        else '<div class="dot dot-empty"></div>'
        for i in range(3)
    )
    return f'<div class="difficulty">{dots}<span class="difficulty-label">{labels[level]}</span></div>'


for category, algos in ALGORITHMS.items():
    st.markdown(f'<div class="section-header">// {category}</div>', unsafe_allow_html=True)

    cards_html = '<div class="card-grid">'
    for a in algos:
        status_html = (
            '<span class="status-live">Live</span>'
            if a["status"] == "live"
            else '<span class="status-soon">Coming soon</span>'
        )
        card_inner = f"""
        <div class="card {'card-live' if a['status'] == 'live' else ''}" style="--accent:{a['accent']}">
            <div class="card-top">
                <div class="card-icon">{a['icon']}</div>
                <span class="badge {a['badge']}">{category}</span>
            </div>
            <div class="card-name">{a['name']}</div>
            <div class="card-desc">{a['desc']}</div>
            <div class="card-footer">
                {difficulty_dots(a['difficulty'], a['accent'])}
                {status_html}
            </div>
        </div>"""

        if a["status"] == "live":
            page_url = "/" + re.sub(r"^\d+_", "", a["page"].replace("pages/", "").replace(".py", ""))
            cards_html += f'<a href="{page_url}" target="_self" style="text-decoration:none;">{card_inner}</a>'
        else:
            cards_html += card_inner

    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


st.markdown("""
<div class="footer">
    Made with curiosity &amp; persistence by Sourav Halder &nbsp;·&nbsp; A personal journey through machine learning
</div>
""", unsafe_allow_html=True)

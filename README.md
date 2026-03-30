# 🧠 Data Science Playground

An interactive Streamlit app for exploring Machine Learning algorithms hands-on.
Tune hyperparameters and watch decision boundaries, metrics, and visualisations
update in real time.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## 📁 Project Structure

```
ml_playground/
├── app.py                          ← Landing page (algorithm catalogue)
│
├── pages/
│   ├── 1_Logistic_Regression.py   ✅ Implemented
│   ├── 2_Decision_Tree.py         ✅ Implemented
│   ├── 3_Random_Forest.py         ✅ Implemented
│   ├── 4_KMeans.py                ✅ Implemented
│   └── _TEMPLATE.py               📋 Copy to add a new algorithm
│
├── utils/
│   ├── __init__.py
│   └── helpers.py                 ← Shared datasets, plots, CSS helpers
│
├── requirements.txt
└── README.md
```

---

## 🗺️ Pages & Features

### 🔵 Logistic Regression
- **Hyperparams:** C, penalty (L1/L2/Elastic Net/none), solver, max_iter, multi_class
- **Tabs:** Decision Boundary · Confusion Matrix · ROC Curve · Coefficients

### 🌳 Decision Tree
- **Hyperparams:** criterion (gini/entropy), max_depth, min_samples_split/leaf, max_features, ccp_alpha
- **Tabs:** Decision Boundary · Confusion Matrix · ROC Curve · Feature Importance · **Tree Structure** (interactive Plotly tree)

### 🌲 Random Forest
- **Hyperparams:** n_estimators, criterion, max_depth, max_features, bootstrap, oob_score
- **Tabs:** Decision Boundary · Confusion Matrix · ROC Curve · Feature Importance (with std bars) · **OOB Error Curve**

### ⭕ K-Means Clustering
- **Hyperparams:** k, init (k-means++ / random), n_init, max_iter, tol
- **Tabs:** Cluster Map (with Voronoi) · **Elbow + Silhouette Curve** · **Silhouette Plot** · Cluster Stats

---

## ➕ Adding a New Algorithm

1. Copy `pages/_TEMPLATE.py` → `pages/5_Your_Algorithm.py`
2. Fill in the `# TODO` sections (model, hyperparams, tabs)
3. Add an entry to the `ALGORITHMS` dict in `app.py`
4. Restart Streamlit — your page appears automatically in the sidebar

---

## 🎨 Design System

All pages share a dark theme via `inject_page_css()` from `utils/helpers.py`.
Colours, fonts, and Plotly layout are defined in one place for consistency:

| Token         | Value      | Usage              |
|---------------|------------|--------------------|
| `PALETTE[0]`  | `#6c63ff`  | Primary / Class 0  |
| `PALETTE[1]`  | `#52c4a0`  | Secondary / Class 1|
| `PALETTE[2]`  | `#f0a050`  | Tertiary / Class 2 |
| `DARK_BG`     | `#0a0a0f`  | Page background    |
| `CARD_BG`     | `#111118`  | Card background    |
| `BORDER`      | `#1e1e2e`  | Borders            |

---

## 📦 Datasets Available

| Key            | Description                            |
|----------------|----------------------------------------|
| `iris`         | 4 features, 3 classes (sklearn)        |
| `breast_cancer`| 30 features, 2 classes (sklearn)       |
| `wine`         | 13 features, 3 classes (sklearn)       |
| `moons`        | 2-D non-linear, 2 classes             |
| `circles`      | 2-D concentric rings, 2 classes       |
| `synthetic`    | Configurable n_features / n_classes   |

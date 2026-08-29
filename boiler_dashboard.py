"""
Industrial Boiler ML Dashboard
================================
Author: Sujal Jaiswal | 230107079

WHAT THIS PROJECT ACTUALLY IS
------------------------------
The raw dataset (`boiler_data.csv`) is pure, UNLABELED multivariate time-series
sensor data from an industrial boiler's SCADA system, sampled roughly every 5
seconds. There is:
    - a single timestamp column ("date")
    - 30 numeric sensor channels (pressures, temperatures, flows, fan current
      and vibration, etc.)
    - NO failure labels
    - NO efficiency labels
    - NO maintenance-event labels
    - NO Remaining-Useful-Life (RUL) labels

This dashboard builds three tasks the data supports:

    1. Anomaly Detection          -> Isolation Forest        (unsupervised)
    2. Operating Mode Discovery   -> K-Means + PCA            (unsupervised)
    3. Sensor Forecasting         -> Random Forest / Linear   (SUPERVISED)
       Regression on lag features

Task 3 is a real supervised-learning problem built from the data itself:
historical (lagged) sensor readings are used as features (X) and a future
reading of a chosen sensor is the label (y) - no labels were invented..
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score,
    mean_absolute_error, mean_squared_error, r2_score
)

import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# PAGE CONFIG & STYLE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Boiler ML Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { color: #FF6B6B; }
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "boiler_data.csv")

# Best-effort, human-readable labels for the raw SCADA tag names, inferred
# from common industrial tag-naming conventions (TE = temperature element,
# PT = pressure transmitter, FT = flow transmitter, YFJ = induced-draft fan,
# ZD = vibration, AI = current, etc.). These are NOT verified against plant
# documentation - they exist purely to make the dashboard readable, not as a
# scientific claim about the exact physical meaning of every tag. Each entry
# below is deliberately kept unique so it can double as a UI label.
FRIENDLY_NAMES = {
    "PT_8313A.AV_0#": "Furnace Pressure A",
    "PT_8313B.AV_0#": "Furnace Pressure B",
    "PT_8313C.AV_0#": "Furnace Pressure C",
    "PT_8313D.AV_0#": "Furnace Pressure D",
    "PT_8313E.AV_0#": "Furnace Pressure E",
    "PT_8313F.AV_0#": "Furnace Pressure F",
    "PTCA_8322A.AV_0#": "Calibrated Pressure A",
    "PTCA_8324.AV_0#": "Calibrated Pressure B",
    "TE_8319A.AV_0#": "Flue Gas Temperature A",
    "TE_8319B.AV_0#": "Flue Gas Temperature B",
    "TE_8313B.AV_0#": "Furnace Temperature",
    "TE_8303.AV_0#": "Process Temperature 1",
    "TE_8304.AV_0#": "Process Temperature 2",
    "TV_8329ZC.AV_0#": "Control Valve Position",
    "FT_8301.AV_0#": "Flow Rate 1",
    "FT_8302.AV_0#": "Flow Rate 2",
    "FT_8306A.AV_0#": "Flow Rate 3A",
    "AIR_8301A.AV_0#": "Flue Gas O2 A",
    "FT_8306B.AV_0#": "Flow Rate 3B",
    "AIR_8301B.AV_0#": "Flue Gas O2 B",
    "YFJ3_AI.AV_0#": "ID Fan Motor Current",
    "YFJ3_ZD1.AV_0#": "ID Fan Vibration 1",
    "YFJ3_ZD2.AV_0#": "ID Fan Vibration 2",
    "SXLTCYZ.AV_0#": "Unlabeled Process Tag 1 (SXLTCYZ)",
    "SXLTCYY.AV_0#": "Unlabeled Process Tag 2 (SXLTCYY)",
    "ZCLCCY.AV_0#": "Unlabeled Process Tag 3 (ZCLCCY)",
    "YCLCCY.AV_0#": "Unlabeled Process Tag 4 (YCLCCY)",
    "YJJWSLL.AV_0#": "Unlabeled Process Tag 5 (YJJWSLL)",
    "ZZQBCHLL.AV_0#": "Main Steam Flow",
    "TE_8332A.AV_0#": "Main Steam Temperature",
}

def friendly(col: str) -> str:
    """Human-readable label for a raw sensor column (falls back to raw name)."""
    return FRIENDLY_NAMES.get(col, col)

def friendly_feature(feat: str) -> str:
    """Turn a lag-feature column name like 'YFJ3_ZD1.AV_0# (t-2)' into a
    readable label like 'ID Fan Vibration 1 (t-2)'."""
    if " (t-" in feat:
        raw_col, suffix = feat.split(" (t-", 1)
        return f"{friendly(raw_col)} (t-{suffix}"
    return friendly(feat)

# Sensors that are physically meaningful forecasting targets (vibration,
# current, temperature) - matches the assignment's suggested target types,
# and YFJ3_ZD1 is the honest replacement for the old "fan health" heuristic.
TARGET_CANDIDATES = {
    "ID Fan Vibration 1 (YFJ3_ZD1)": "YFJ3_ZD1.AV_0#",
    "ID Fan Vibration 2 (YFJ3_ZD2)": "YFJ3_ZD2.AV_0#",
    "ID Fan Motor Current (YFJ3_AI)": "YFJ3_AI.AV_0#",
    "Flue Gas Temperature A (TE_8319A)": "TE_8319A.AV_0#",
    "Main Steam Temperature (TE_8332A)": "TE_8332A.AV_0#",
}

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    time_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if time_cols:
        df[time_cols[0]] = pd.to_datetime(df[time_cols[0]], errors="coerce")
        df = df.set_index(time_cols[0]).sort_index()
    df = df.select_dtypes(include=[np.number])
    # Sensor dropouts in SCADA logs are typically short and rare - linear
    # interpolation plus edge-fill closes small gaps without inventing values
    # that don't reflect the underlying process.
    df = df.interpolate(method="linear").bfill().ffill()
    return df

# ----------------------------------------------------------------------------
# MODEL 1: ANOMALY DETECTION (Isolation Forest, unsupervised)
# ----------------------------------------------------------------------------
@st.cache_data
def scale_features(df):
    scaler = StandardScaler()
    return scaler.fit_transform(df.values)

@st.cache_data
def run_anomaly_detection(X, contamination):
    iso = IsolationForest(
        n_estimators=200, contamination=contamination,
        random_state=42, n_jobs=-1
    )
    labels = iso.fit_predict(X)          # -1 = anomaly, 1 = normal
    scores = iso.decision_function(X)    # higher = more "normal"
    return (labels == -1), scores

# ----------------------------------------------------------------------------
# MODEL 2: OPERATING MODE DISCOVERY
# (K-Means, unsupervised, K chosen by Silhouette Score)
# ----------------------------------------------------------------------------
@st.cache_data
def evaluate_k_range(X, k_min=2, k_max=7, sample_size=5000, random_state=42):
    """Fit K-Means for K = k_min..k_max and score each with Silhouette and
    Davies-Bouldin. Silhouette is O(n^2), so it's computed on a random
    subsample for speed; Davies-Bouldin is computed on the full data."""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    sample_idx = rng.choice(n, min(sample_size, n), replace=False)

    sil_scores, db_scores = {}, {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil_scores[k] = silhouette_score(X[sample_idx], labels[sample_idx])
        db_scores[k] = davies_bouldin_score(X, labels)

    best_k = max(sil_scores, key=sil_scores.get)
    return sil_scores, db_scores, best_k

@st.cache_data
def fit_final_clusters(X, k, random_state=42):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=15)
    labels = km.fit_predict(X)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X)
    return labels, coords, pca.explained_variance_ratio_

# ----------------------------------------------------------------------------
# MODEL 3: SENSOR FORECASTING (real supervised learning problem)
# ----------------------------------------------------------------------------
def build_lag_features(df, target_col, n_lags, horizon):
    """
    Build a supervised table from raw time-series:
      X: lagged values (t-1 ... t-n_lags) of EVERY sensor
      y: the target sensor's value `horizon` steps in the FUTURE

    This is the standard way to turn a forecasting problem into a
    regression problem an off-the-shelf ML model can learn from.
    Raw column names (not friendly labels) are used as feature keys here
    to guarantee uniqueness; friendly labels are applied only for display.
    """
    feature_data = {}
    for col in df.columns:
        for lag in range(1, n_lags + 1):
            feature_data[f"{col} (t-{lag})"] = df[col].shift(lag)

    X = pd.DataFrame(feature_data, index=df.index)
    y = df[target_col].shift(-horizon)

    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]

@st.cache_data
def run_forecasting(df, target_col, n_lags, horizon, test_frac):
    X, y = build_lag_features(df, target_col, n_lags, horizon)

    # Chronological split - NEVER shuffle time-series data, or the model
    # would effectively "see the future" during training (data leakage).
    split_idx = int(len(X) * (1 - test_frac))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Scaler is fit on TRAIN ONLY, then applied to test - this avoids
    # leaking test-set statistics into training.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # max_features='sqrt' keeps training fast even with many lag features
    # (RandomForestRegressor considers ALL features at every split by default,
    # which gets slow quickly as n_lags * n_sensors grows).
    rf = RandomForestRegressor(n_estimators=150, max_depth=10, max_features="sqrt",
                                random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)

    lr = LinearRegression()
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)

    metrics = {}
    for name, pred in [("Random Forest", rf_pred), ("Linear Regression", lr_pred)]:
        metrics[name] = {
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2": r2_score(y_test, pred),
        }

    importances = pd.Series(rf.feature_importances_, index=X_train.columns) \
                    .sort_values(ascending=False)

    return {
        "y_test": y_test,
        "rf_pred": rf_pred,
        "lr_pred": lr_pred,
        "metrics": metrics,
        "importances": importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
    }

# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------
if not os.path.exists(DATA_PATH):
    st.error(f"Could not find `boiler_data.csv` next to app.py at: {DATA_PATH}")
    st.stop()

df_full = load_data(DATA_PATH)

if df_full.empty or df_full.shape[1] < 2:
    st.error("boiler_data.csv did not load any usable numeric sensor columns.")
    st.stop()

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
st.sidebar.caption(f"Dataset: {len(df_full):,} rows × {df_full.shape[1]} sensors")
st.sidebar.markdown("---")

# Guard against tiny datasets: only show a "rows used" slider when there's
# actually a meaningful range to pick from.
if len(df_full) > 5000:
    max_rows = st.sidebar.slider(
        "Rows used for modeling (most recent)",
        min_value=5000,
        max_value=len(df_full),
        value=min(30000, len(df_full)),
        step=5000,
        help="Keeps Isolation Forest / K-Means / Random Forest fast. Uses the most recent rows chronologically."
    )
else:
    max_rows = len(df_full)
    st.sidebar.caption(f"Using all {max_rows:,} rows (dataset is small).")
df = df_full.tail(max_rows).copy()

if len(df) > 500:
    n_show = st.sidebar.slider(
        "Chart window (last N samples)",
        min_value=500,
        max_value=len(df),
        value=min(5000, len(df)),
        step=500
    )
else:
    n_show = len(df)
df_view = df.tail(n_show)

st.sidebar.markdown("---")
st.sidebar.subheader("1️⃣ Anomaly Detection")
contamination = st.sidebar.slider("Expected anomaly fraction", 0.01, 0.15, 0.05, 0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("2️⃣ Operating Modes")
manual_override = st.sidebar.checkbox("Manually set K", value=False)
manual_k = st.sidebar.slider("K (clusters)", 2, 7, 4) if manual_override else None

st.sidebar.markdown("---")
st.sidebar.subheader("3️⃣ Sensor Forecasting")
available_targets = {label: col for label, col in TARGET_CANDIDATES.items() if col in df.columns}
if not available_targets:
    available_targets = {friendly(c): c for c in df.columns[:1]}
target_label = st.sidebar.selectbox("Sensor to forecast", list(available_targets.keys()))
target_col = available_targets[target_label]

n_lags = st.sidebar.slider("Lag steps per sensor", 2, 10, 3)
horizon = st.sidebar.slider("Forecast horizon (steps ahead)", 1, 60, 12)
test_frac = st.sidebar.slider("Test set size (%)", 10, 40, 20) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("**Sujal Jaiswal | 230107079**")

# ----------------------------------------------------------------------------
# RUN MODELS
# ----------------------------------------------------------------------------
with st.spinner("Training models..."):
    X_scaled = scale_features(df)

    anomaly_flags, anomaly_scores = run_anomaly_detection(X_scaled, contamination)

    sil_scores, db_scores, best_k = evaluate_k_range(X_scaled)
    chosen_k = manual_k if manual_k else best_k
    cluster_labels, pca_coords, pca_var = fit_final_clusters(X_scaled, chosen_k)

    forecast_results = run_forecasting(df, target_col, n_lags, horizon, test_frac)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("🔥 Industrial Boiler — ML Analysis Dashboard")
st.caption("Anomaly Detection · Operating Mode Discovery · Sensor Forecasting")

with st.expander("📖 About this project & methodology", expanded=False):
    st.markdown(f"""
This dataset is **unlabeled multivariate sensor time-series** ({df_full.shape[1]} sensors,
sampled roughly every 5 seconds). It has **no efficiency, failure, maintenance, or RUL
labels**, so this project only makes claims the data can support:

| Task | Method | Learning type | What it actually tells you |
|---|---|---|---|
| Anomaly Detection | Isolation Forest | Unsupervised | Which time steps look statistically abnormal across all sensors together |
| Operating Mode Discovery | K-Means + PCA | Unsupervised | Distinct operating regimes the boiler cycles through |
| Sensor Forecasting | Random Forest vs. Linear Regression | **Supervised** | Predicts a real, measured future sensor value from lagged history |

Isolation Forest does **not** predict failure - it flags statistical outliers.
The forecasting model does **not** predict efficiency or RUL - it forecasts a real
sensor reading (currently **{target_label}**) {horizon} step(s) ahead, using genuine
lag-based supervised learning with a chronological train/test split (no shuffling,
no leakage). The dashboard's old "combustion efficiency" and "fan health" scores
have been removed because no ground-truth efficiency or health label exists in
the raw data to train or validate such a model against.
""")

# ----------------------------------------------------------------------------
# OVERVIEW METRICS
# ----------------------------------------------------------------------------
rf_metrics = forecast_results["metrics"]["Random Forest"]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📊 Data Points", f"{len(df):,}")
k2.metric("📡 Sensors", f"{df.shape[1]}")
k3.metric("🚨 Anomalies", f"{anomaly_flags.sum():,}", f"{anomaly_flags.mean()*100:.1f}% of data")
k4.metric("🔁 Operating Modes", f"{chosen_k}", "manual" if manual_k else "auto (Silhouette)")
k5.metric("📈 Forecast MAE / RMSE", f"{rf_metrics['MAE']:.3f} / {rf_metrics['RMSE']:.3f}", f"R² {rf_metrics['R2']:.2f}")

st.markdown("---")

# ----------------------------------------------------------------------------
# SECTION 1: ANOMALY DETECTION
# ----------------------------------------------------------------------------
st.subheader("🚨 Anomaly Detection — Isolation Forest")
st.caption(
    "Unsupervised outlier detection across all sensors at once. Isolation Forest "
    "isolates points that are statistically rare given the rest of the data - it does "
    "not know what a 'failure' looks like, and this panel does not claim to predict one."
)

col_a, col_b = st.columns([3, 1])
with col_a:
    scores_view = anomaly_scores[-n_show:]
    anomaly_view = anomaly_flags[-n_show:]
    overlay_col = target_col

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Isolation Forest Anomaly Score (< 0 ⇒ flagged)",
                         f"{friendly(overlay_col)} with Flagged Anomalies"],
        vertical_spacing=0.14
    )
    fig.add_trace(go.Scatter(y=scores_view, mode="lines",
                              line=dict(color="#5BC0EB", width=0.8), name="Score"), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)

    fig.add_trace(go.Scatter(y=df_view[overlay_col].values, mode="lines",
                              line=dict(color="#4A90D9", width=0.6), name=friendly(overlay_col),
                              opacity=0.7), row=2, col=1)
    anom_idx = np.where(anomaly_view)[0]
    fig.add_trace(go.Scatter(x=anom_idx, y=df_view[overlay_col].values[anom_idx],
                              mode="markers", marker=dict(color="red", size=4), name="Anomaly"),
                  row=2, col=1)
    fig.update_layout(height=420, showlegend=True, margin=dict(l=0, r=0, t=40, b=0),
                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    n_anom = int(anomaly_flags.sum())
    n_norm = len(anomaly_flags) - n_anom
    fig_pie = go.Figure(go.Pie(labels=["Normal", "Anomaly"], values=[n_norm, n_anom],
                                hole=0.55, marker_colors=["#4A90D9", "#E74C3C"]))
    fig_pie.update_layout(height=230, margin=dict(l=0, r=0, t=10, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
                           annotations=[dict(text=f"{anomaly_flags.mean()*100:.1f}%<br>anomaly",
                                              x=0.5, y=0.5, font_size=12, showarrow=False)])
    st.plotly_chart(fig_pie, use_container_width=True)
    st.info(f"**{n_anom:,}** anomalies out of **{len(anomaly_flags):,}** modeled samples")

st.markdown("**Sensor behavior during anomalous vs. normal periods**")
compare_df = df.copy()
compare_df["status"] = np.where(anomaly_flags, "Anomaly", "Normal")
summary = compare_df.groupby("status").mean(numeric_only=True).T
summary["abs_diff"] = (summary["Anomaly"] - summary["Normal"]).abs()
summary = summary.sort_values("abs_diff", ascending=False).drop(columns="abs_diff").head(10)
summary.index = [friendly(c) for c in summary.index]
st.dataframe(summary.round(2), use_container_width=True)
st.caption("Top 10 sensors with the largest mean difference between anomalous and normal periods.")

with st.expander("🕓 Anomaly locations (timestamps in current chart window)"):
    anomaly_times = df_view.index[anomaly_view]
    if len(anomaly_times) > 0:
        st.dataframe(pd.DataFrame({"timestamp": anomaly_times[:50]}), use_container_width=True)
    else:
        st.write("No anomalies flagged in the current chart window.")

st.markdown("---")

# ----------------------------------------------------------------------------
# SECTION 2: OPERATING MODE CLUSTERING
# ----------------------------------------------------------------------------
st.subheader("🔁 Operating Mode Discovery — K-Means Clustering")
st.caption(
    "K-Means groups similar sensor states into distinct 'operating modes'. Rather than "
    "picking K arbitrarily, K = 2 to 7 are all evaluated and the best K is recommended "
    "using the Silhouette Score (higher = better-separated clusters)."
)

col_k1, col_k2 = st.columns(2)
k_vals = list(sil_scores.keys())
with col_k1:
    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(x=k_vals, y=[sil_scores[k] for k in k_vals],
                                  mode="lines+markers", line=dict(color="#2ECC71"), name="Silhouette"))
    fig_sil.add_vline(x=best_k, line_dash="dash", line_color="orange",
                       annotation_text=f"Best K={best_k}")
    fig_sil.update_layout(title="Silhouette Score by K (higher is better)",
                           xaxis_title="K", yaxis_title="Silhouette Score", height=300,
                           margin=dict(l=0, r=0, t=40, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig_sil, use_container_width=True)
with col_k2:
    fig_db = go.Figure()
    fig_db.add_trace(go.Scatter(x=k_vals, y=[db_scores[k] for k in k_vals],
                                 mode="lines+markers", line=dict(color="#E67E22"), name="Davies-Bouldin"))
    fig_db.update_layout(title="Davies-Bouldin Score by K (lower is better)",
                          xaxis_title="K", yaxis_title="DB Score", height=300,
                          margin=dict(l=0, r=0, t=40, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig_db, use_container_width=True)

st.success(
    f"✅ Recommended K = **{best_k}** (Silhouette Score = {sil_scores[best_k]:.3f}). "
    + (f"Currently showing manual override K = **{manual_k}**." if manual_k else "Currently showing the recommended K.")
)

col_c, col_d = st.columns(2)
COLORS = px.colors.qualitative.Bold
with col_c:
    fig_pca = go.Figure()
    for m in range(chosen_k):
        mask = cluster_labels == m
        fig_pca.add_trace(go.Scatter(x=pca_coords[mask, 0], y=pca_coords[mask, 1],
                                      mode="markers", name=f"Mode {m+1}",
                                      marker=dict(size=3, opacity=0.5, color=COLORS[m % len(COLORS)])))
    fig_pca.update_layout(title=f"PCA View (PC1={pca_var[0]*100:.1f}%, PC2={pca_var[1]*100:.1f}%) — for visualization only",
                           height=350, margin=dict(l=0, r=0, t=40, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig_pca, use_container_width=True)
with col_d:
    mode_counts = pd.Series(cluster_labels).value_counts().sort_index()
    fig_bar = go.Figure(go.Bar(x=[f"Mode {i+1}" for i in mode_counts.index], y=mode_counts.values,
                                marker_color=COLORS[:chosen_k], text=mode_counts.values, textposition="outside"))
    fig_bar.update_layout(title="Mode Distribution", height=350, margin=dict(l=0, r=0, t=40, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig_bar, use_container_width=True)

mode_view = cluster_labels[-n_show:]
fig_timeline = go.Figure(go.Scatter(y=mode_view, mode="lines", line=dict(color="#A8DADC", width=0.8),
                                     fill="tozeroy", fillcolor="rgba(168,218,220,0.2)"))
fig_timeline.update_layout(title="Operating Mode Over Time",
                            yaxis=dict(tickvals=list(range(chosen_k)),
                                       ticktext=[f"Mode {i+1}" for i in range(chosen_k)]),
                            height=220, margin=dict(l=0, r=0, t=40, b=0),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
st.plotly_chart(fig_timeline, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------------------
# SECTION 3: SENSOR FORECASTING
# ----------------------------------------------------------------------------
st.subheader("📈 Sensor Forecasting — Supervised Predictive Analytics")
st.caption(
    f"Genuine supervised regression: using the last {n_lags} lagged readings of every "
    f"sensor as features, predict **{target_label}** {horizon} step(s) ahead "
    f"(~{horizon*5}s at ~5s sampling). This is NOT efficiency, failure, or RUL prediction - "
    f"it forecasts a real, measured sensor value."
)

st.write(
    f"Training samples: **{forecast_results['n_train']:,}** · "
    f"Test samples: **{forecast_results['n_test']:,}** (chronological split, "
    f"test = most recent {int(test_frac*100)}%) · "
    f"Features: **{forecast_results['n_features']}** ({n_lags} lags × {df.shape[1]} sensors)"
)

if forecast_results["n_test"] < 20:
    st.warning("Very few test samples for a reliable evaluation — try increasing "
               "'Rows used for modeling' or reducing the lag/horizon sliders.")

metrics_df = pd.DataFrame(forecast_results["metrics"]).T
st.dataframe(metrics_df.round(4), use_container_width=True)

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(y=forecast_results["y_test"].values, mode="lines",
                             name="Actual", line=dict(color="#4A90D9", width=1)))
fig_fc.add_trace(go.Scatter(y=forecast_results["rf_pred"], mode="lines",
                             name="Random Forest", line=dict(color="#E74C3C", width=1, dash="dot")))
fig_fc.add_trace(go.Scatter(y=forecast_results["lr_pred"], mode="lines",
                             name="Linear Regression", line=dict(color="#F39C12", width=1, dash="dash")))
fig_fc.update_layout(title=f"Actual vs. Predicted — {target_label} (Test Set)",
                      height=380, margin=dict(l=0, r=0, t=40, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
st.plotly_chart(fig_fc, use_container_width=True)

col_e, col_f = st.columns(2)
with col_e:
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(x=forecast_results["y_test"].values, y=forecast_results["rf_pred"],
                                      mode="markers", marker=dict(size=3, color="#E74C3C", opacity=0.4),
                                      name="RF predictions"))
    lims = [forecast_results["y_test"].min(), forecast_results["y_test"].max()]
    fig_scatter.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                                      line=dict(color="white", dash="dash"), name="Perfect fit"))
    fig_scatter.update_layout(title="Predicted vs. Actual (Random Forest)",
                               xaxis_title="Actual", yaxis_title="Predicted", height=340,
                               margin=dict(l=0, r=0, t=40, b=0),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig_scatter, use_container_width=True)
with col_f:
    top_imp = forecast_results["importances"].head(10)[::-1]
    top_imp.index = [friendly_feature(i) for i in top_imp.index]
    fig_imp = go.Figure(go.Bar(x=top_imp.values, y=top_imp.index, orientation="h",
                                marker_color="#5BC0EB"))
    fig_imp.update_layout(title="Top 10 Most Important Lag Features (Random Forest)",
                           height=340, margin=dict(l=0, r=0, t=40, b=0),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
    st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")

# ----------------------------------------------------------------------------
# SECTION 4: RAW DATA EXPLORER
# ----------------------------------------------------------------------------
with st.expander("📋 Raw Sensor Data Explorer", expanded=False):
    friendly_options = {friendly(c): c for c in df.columns}
    default_sel = list(friendly_options.keys())[:3]
    selected_friendly = st.multiselect("Select sensors to plot", list(friendly_options.keys()), default=default_sel)
    selected_cols = [friendly_options[f] for f in selected_friendly]

    if selected_cols:
        fig_raw = go.Figure()
        for col in selected_cols:
            fig_raw.add_trace(go.Scatter(y=df_view[col].values, mode="lines",
                                          name=friendly(col), line=dict(width=0.8)))
        fig_raw.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.05)")
        st.plotly_chart(fig_raw, use_container_width=True)

    display_df = df_view.tail(100).copy()
    display_df.columns = [friendly(c) for c in display_df.columns]
    st.dataframe(display_df.round(3), use_container_width=True)

st.caption("Industrial Boiler ML Dashboard · Sujal Jaiswal 230107079 · AIML Project Stage 2")
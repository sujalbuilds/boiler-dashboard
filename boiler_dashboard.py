"""
Industrial Boiler ML Dashboard
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import os
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="🔥 Boiler ML Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #1e1e2e; border-radius: 10px;
        padding: 18px; text-align: center;
    }
    h1 { color: #FF6B6B; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("⚙️ Controls")
DATA_PATH = os.path.join(os.path.dirname(__file__), "boiler_data.csv")
uploaded = "boiler_data.csv"
st.sidebar.markdown("---")
contamination = st.sidebar.slider("Isolation Forest contamination", 0.01, 0.15, 0.05, 0.01)
n_clusters    = st.sidebar.slider("Operating Modes (K)", 2, 7, 4)
window_size   = st.sidebar.slider("Rolling window (steps)", 100, 2000, 500, 100)
failure_thresh= st.sidebar.slider("Fan failure threshold (%)", 5, 40, 20)
st.sidebar.markdown("---")
st.sidebar.markdown("**Sujal Jaiswal | 230107079**")

@st.cache_data
def load_data(file):
    df = pd.read_csv(file) if not isinstance(file, str) else pd.read_csv(file)
    time_col = [c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()]
    if time_col:
        df[time_col[0]] = pd.to_datetime(df[time_col[0]], errors='coerce')
        df = df.set_index(time_col[0]).sort_index()
    df = df.select_dtypes(include=[np.number])
    df = df.interpolate(method='linear').bfill().ffill()
    return df

@st.cache_data
def run_models(df, contamination, n_clusters, window_size, failure_thresh):
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=42, n_jobs=-1)
    iso_labels  = iso.fit_predict(X)
    iso_scores  = iso.decision_function(X)
    iso_anomaly = iso_labels == -1

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
    cluster_labels = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    mm = MinMaxScaler()
    def safe_norm(series, invert=False):
        arr = mm.fit_transform(series.values.reshape(-1,1)).flatten()
        return 1 - arr if invert else arr

    def find_col(df, keywords):
        for kw in keywords:
            matches = [c for c in df.columns if kw.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    col_flue   = 'TE_8319A.AV_0'  if 'TE_8319A.AV_0'  in df.columns else None
    col_o2     = 'AIR_8301A.AV_0' if 'AIR_8301A.AV_0' in df.columns else None
    col_steam  = 'ZZQBCHLL.AV_0'  if 'ZZQBCHLL.AV_0'  in df.columns else None
    col_steamt = 'TE_8332A.AV_0'  if 'TE_8332A.AV_0'  in df.columns else None

    components = []
    if col_flue:   components.append(safe_norm(df[col_flue], invert=True))
    if col_o2:     components.append(safe_norm(df[col_o2],   invert=True))
    if col_steam:  components.append(safe_norm(df[col_steam], invert=False))
    if col_steamt: components.append(safe_norm(df[col_steamt], invert=False))
    if not components:
        components = [safe_norm(df[df.columns[0]], invert=False)]

    eff_raw = np.mean(components, axis=0)
    efficiency = mm.fit_transform(eff_raw.reshape(-1,1)).flatten() * 100

    col_vib  = 'YFJ3_ZD1.AV_0' if 'YFJ3_ZD1.AV_0' in df.columns else None
    col_curr = 'YFJ3_AI.AV_0'  if 'YFJ3_AI.AV_0'  in df.columns else None

    health_parts = []
    if col_vib:  health_parts.append(safe_norm(df[col_vib], invert=True))
    if col_curr: health_parts.append(safe_norm(df[col_curr], invert=True))
    if not health_parts:
        health_parts = [safe_norm(df[df.columns[1]], invert=True)]

    health_raw = np.mean(health_parts, axis=0)
    health_smooth = pd.Series(health_raw).rolling(window_size, min_periods=1).mean().values * 100

    t_idx = np.arange(len(health_smooth)).reshape(-1,1)
    lr = LinearRegression().fit(t_idx, health_smooth)
    trend = lr.predict(t_idx)

    if lr.coef_[0] < 0:
        steps_left = max(0, (health_smooth[-1] - failure_thresh) / abs(lr.coef_[0]))
        rul_label = f"~{int(steps_left):,} steps"
    else:
        steps_left = None
        rul_label = "Stable / improving"

    return {
        "iso_scores":     iso_scores.tolist(),
        "iso_anomaly":    iso_anomaly.tolist(),
        "cluster_labels": cluster_labels.tolist(),
        "pca":            X_pca.tolist(),
        "pca_var":        pca.explained_variance_ratio_.tolist(),
        "efficiency":     efficiency.tolist(),
        "health":         health_smooth.tolist(),
        "trend":          trend.tolist(),
        "rul_label":      rul_label,
        "col_vib":        col_vib,
        "col_curr":       col_curr,
        "col_vib_data":   df[col_vib].tolist() if col_vib else [],
    }

st.title("🔥 Industrial Boiler — ML Analysis Dashboard")
st.caption("Anomaly Detection · Operating Mode Clustering · Combustion Efficiency · Fan Health & RUL")

if uploaded is None:
    st.info("👈 Upload your boiler CSV file from the sidebar to begin.")
    st.markdown("""
    **What this dashboard shows:**
    | Panel | Method | What it tells you |
    |---|---|---|
    | Anomaly Detection | Isolation Forest | Which time steps are abnormal |
    | Operating Modes | K-Means Clustering | What regime the boiler is running in |
    | Combustion Efficiency | Physics-ML hybrid | How efficiently fuel is being burned |
    | Fan Health & RUL | Trend Analysis + Linear Regression | When the fan needs maintenance |
    """)
    st.stop()

df = load_data(uploaded)
n_show = st.sidebar.slider("Show last N samples", 1000, len(df), min(20000, len(df)), 1000)
df_view = df.iloc[-n_show:]

with st.spinner("Running ML models..."):
    results = run_models(df, contamination, n_clusters, window_size, failure_thresh)

iso_scores     = np.array(results["iso_scores"])
iso_anomaly    = np.array(results["iso_anomaly"])
cluster_labels = np.array(results["cluster_labels"])
pca_pts        = np.array(results["pca"])
pca_var        = results["pca_var"]
efficiency     = np.array(results["efficiency"])
health         = np.array(results["health"])
trend          = np.array(results["trend"])
rul_label      = results["rul_label"]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("📊 Data Points",      f"{len(df):,}")
k2.metric("🚨 Anomalies Found",  f"{iso_anomaly.sum():,}",
          f"{iso_anomaly.mean()*100:.1f}% of data")
k3.metric("🔁 Operating Modes",  str(n_clusters))
k4.metric("🔥 Mean Efficiency",  f"{efficiency.mean():.1f}%",
          f"Current: {efficiency[-100:].mean():.1f}%")
k5.metric("🔧 Fan Health",       f"{health[-1]:.1f}%",
          f"RUL: {rul_label}")

st.markdown("---")

st.subheader("🚨 Anomaly Detection — Isolation Forest")
col_a, col_b = st.columns([3, 1])

with col_a:
    idx = np.arange(len(iso_scores[-n_show:]))
    scores_view = iso_scores[-n_show:]
    anomaly_view = iso_anomaly[-n_show:]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Anomaly Score Over Time", "First Sensor Signal"],
                        vertical_spacing=0.1)
    fig.add_trace(go.Scatter(y=scores_view, mode='lines',
                             line=dict(color='#5BC0EB', width=0.8), name='Score'), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    fig.add_trace(go.Scatter(
        y=df_view[df_view.columns[0]].values, mode='lines',
        line=dict(color='#4A90D9', width=0.5), name='Sensor', opacity=0.7), row=2, col=1)
    anom_idx = np.where(anomaly_view)[0]
    fig.add_trace(go.Scatter(
        x=anom_idx, y=df_view[df_view.columns[0]].values[anom_idx],
        mode='markers', marker=dict(color='red', size=4), name='Anomaly'), row=2, col=1)
    fig.update_layout(height=400, showlegend=True, margin=dict(l=0,r=0,t=30,b=0),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.05)')
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    n_anom = int(iso_anomaly.sum())
    n_norm = len(iso_anomaly) - n_anom
    fig_pie = go.Figure(go.Pie(
        labels=['Normal', 'Anomaly'],
        values=[n_norm, n_anom],
        hole=0.5,
        marker_colors=['#4A90D9', '#E74C3C'],
        textfont_size=11
    ))
    fig_pie.update_layout(height=250, margin=dict(l=0,r=0,t=10,b=0),
                          paper_bgcolor='rgba(0,0,0,0)', showlegend=True,
                          annotations=[dict(text=f'{iso_anomaly.mean()*100:.1f}%<br>anomaly',
                                           x=0.5, y=0.5, font_size=12, showarrow=False)])
    st.plotly_chart(fig_pie, use_container_width=True)
    st.info(f"**{n_anom:,}** anomalies in **{len(iso_anomaly):,}** points")

st.markdown("---")

st.subheader("🔁 Operating Mode Clustering")
col_c, col_d = st.columns(2)

COLORS = px.colors.qualitative.Bold

with col_c:
    fig_pca = go.Figure()
    for m in range(n_clusters):
        mask = cluster_labels == m
        fig_pca.add_trace(go.Scatter(
            x=pca_pts[mask, 0], y=pca_pts[mask, 1],
            mode='markers', name=f'Mode {m+1}',
            marker=dict(size=3, opacity=0.5, color=COLORS[m % len(COLORS)])
        ))
    fig_pca.update_layout(
        title=f"PCA View (PC1={pca_var[0]*100:.1f}%,  PC2={pca_var[1]*100:.1f}%)",
        height=350, margin=dict(l=0,r=0,t=40,b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.05)'
    )
    st.plotly_chart(fig_pca, use_container_width=True)

with col_d:
    mode_counts = pd.Series(cluster_labels).value_counts().sort_index()
    fig_bar = go.Figure(go.Bar(
        x=[f'Mode {i+1}' for i in mode_counts.index],
        y=mode_counts.values,
        marker_color=COLORS[:n_clusters],
        text=mode_counts.values, textposition='outside'
    ))
    fig_bar.update_layout(
        title="Mode Distribution", height=350,
        margin=dict(l=0,r=0,t=40,b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.05)'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

mode_view = cluster_labels[-n_show:]
fig_timeline = go.Figure(go.Scatter(
    y=mode_view, mode='lines',
    line=dict(color='#A8DADC', width=0.8),
    fill='tozeroy', fillcolor='rgba(168,218,220,0.2)'
))
fig_timeline.update_layout(
    title="Operating Mode Over Time",
    yaxis=dict(tickvals=list(range(n_clusters)),
               ticktext=[f'Mode {i+1}' for i in range(n_clusters)]),
    height=200, margin=dict(l=0,r=0,t=40,b=0),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.05)'
)
st.plotly_chart(fig_timeline, use_container_width=True)

st.markdown("---")

st.subheader("🔥 Combustion Efficiency Index")
eff_view = efficiency[-n_show:]
low_eff_threshold = np.percentile(efficiency, 10)

fig_eff = go.Figure()
fig_eff.add_trace(go.Scatter(
    y=eff_view, mode='lines', name='Efficiency',
    line=dict(color='#2ECC71', width=0.8),
    fill='tozeroy', fillcolor='rgba(46,204,113,0.15)'
))
fig_eff.add_hline(y=efficiency.mean(), line_dash="dash", line_color="orange",
                  annotation_text=f"Mean: {efficiency.mean():.1f}%")
fig_eff.add_hline(y=low_eff_threshold, line_dash="dot", line_color="red",
                  annotation_text=f"Low efficiency threshold: {low_eff_threshold:.1f}%")
fig_eff.update_layout(
    yaxis=dict(range=[0, 105], title="Efficiency (%)"),
    height=280, margin=dict(l=0,r=0,t=20,b=0),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.05)'
)
st.plotly_chart(fig_eff, use_container_width=True)

current_eff = float(efficiency[-200:].mean())
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=current_eff,
    delta={'reference': efficiency.mean(), 'valueformat': '.1f'},
    title={'text': "Current Combustion Efficiency (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#2ECC71" if current_eff > 66 else "#F39C12" if current_eff > 33 else "#E74C3C"},
        'steps': [
            {'range': [0, 33], 'color': "rgba(231,76,60,0.2)"},
            {'range': [33, 66], 'color': "rgba(243,156,18,0.2)"},
            {'range': [66, 100], 'color': "rgba(46,204,113,0.2)"}
        ],
        'threshold': {'line': {'color': "red", 'width': 3}, 'thickness': 0.75, 'value': 33}
    }
))
fig_gauge.update_layout(height=280, margin=dict(l=30,r=30,t=30,b=0),
                         paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_gauge, use_container_width=True)

st.markdown("---")

st.subheader("🔧 Induced Draft Fan — Health & Remaining Useful Life")
health_view = health[-n_show:]
trend_view  = trend[-n_show:]

fig_health = go.Figure()
fig_health.add_trace(go.Scatter(
    y=health_view, mode='lines', name='Health Score',
    line=dict(color='#E67E22', width=0.8),
    fill='tozeroy', fillcolor='rgba(230,126,34,0.15)'
))
fig_health.add_trace(go.Scatter(
    y=trend_view, mode='lines', name='Degradation Trend',
    line=dict(color='red', width=2, dash='dash')
))
fig_health.add_hline(y=failure_thresh, line_dash="dot", line_color="darkred",
                     annotation_text=f"Failure threshold: {failure_thresh}%")
fig_health.update_layout(
    yaxis=dict(range=[-5, 110], title="Health Score (%)"),
    height=300, margin=dict(l=0,r=0,t=20,b=0),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.05)'
)
st.plotly_chart(fig_health, use_container_width=True)

current_health = float(health[-1])
fig_fan_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=current_health,
    title={'text': f"Fan Health — RUL: {rul_label}"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#27AE60" if current_health > 60 else "#F39C12" if current_health > 30 else "#E74C3C"},
        'steps': [
            {'range': [0, 30],  'color': "rgba(231,76,60,0.25)"},
            {'range': [30, 60], 'color': "rgba(243,156,18,0.25)"},
            {'range': [60, 100],'color': "rgba(39,174,96,0.25)"}
        ],
    }
))
fig_fan_gauge.update_layout(height=280, margin=dict(l=30,r=30,t=30,b=0),
                              paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_fan_gauge, use_container_width=True)

st.markdown("---")

with st.expander("📋 Raw Sensor Data Explorer"):
    sensor_sel = st.multiselect("Select sensors to plot", df.columns.tolist(), default=df.columns[:3].tolist())
    if sensor_sel:
        fig_raw = go.Figure()
        for col in sensor_sel:
            fig_raw.add_trace(go.Scatter(y=df_view[col].values, mode='lines', name=col[:40],
                                          line=dict(width=0.8)))
        fig_raw.update_layout(height=350, margin=dict(l=0,r=0,t=10,b=0),
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.05)')
        st.plotly_chart(fig_raw, use_container_width=True)
    st.dataframe(df_view.tail(100).round(3), use_container_width=True)

st.caption("Industrial Boiler ML Dashboard · Sujal Jaiswal 230107079 · AIML Project Stage 2")

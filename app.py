import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Reactor Predictor",
    page_icon="⚗️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

.stApp { background: #0d0f12; color: #e2e8f0; }

section[data-testid="stSidebar"] {
    background: #111318;
    border-right: 1px solid #1e2330;
}

.metric-card {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 10px;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #4a5568;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 4px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px;
    font-weight: 500;
    color: #e2e8f0;
    line-height: 1;
}
.metric-unit {
    font-size: 13px;
    color: #4a5568;
    margin-left: 4px;
}
.metric-delta-pos {
    font-size: 12px;
    color: #48bb78;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-delta-neg {
    font-size: 12px;
    color: #f56565;
    margin-top: 4px;
    font-family: 'IBM Plex Mono', monospace;
}
.stop-banner {
    background: linear-gradient(135deg, #0f2027, #1a2a1a);
    border: 1px solid #2d6a2d;
    border-left: 4px solid #48bb78;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 16px 0;
}
.stop-banner-warn {
    background: linear-gradient(135deg, #1a1505, #1a1a05);
    border: 1px solid #6a5d2d;
    border-left: 4px solid #ecc94b;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 16px 0;
}
.tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    padding: 2px 8px;
    border-radius: 3px;
    background: #1e2330;
    color: #4a5568;
    margin-right: 6px;
}
div[data-testid="stSlider"] > div { padding-top: 0; }
.stSlider [data-baseweb="slider"] { margin-top: -8px; }
label[data-testid="stWidgetLabel"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important;
    color: #4a5568 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
div[data-testid="stNumberInput"] input {
    background: #111318 !important;
    border: 1px solid #1e2330 !important;
    color: #e2e8f0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
.stButton button {
    background: #48bb78 !important;
    color: #0d0f12 !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em;
    width: 100%;
    padding: 12px !important;
    border-radius: 6px !important;
}
.stButton button:hover { background: #68d391 !important; }
h1, h2, h3 { color: #e2e8f0 !important; font-weight: 300 !important; }
hr { border-color: #1e2330 !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants (must match training) ──────────────────────────────────────────
R, T_NOM, T_AMP = 8.314, 300.0, 25.0
Ea1, Ea2, Ea3   = 50_000, 70_000, 60_000
A1, A2, A3      = 5.1e7, 2.0e6, 1.5e7
K_DEACT         = 0.05
FEATURES        = ['time', 'temp', 'temp_sq', 'ideal_B', 'B_lag1', 'B_lag2', 'activity']
PRICE, OP_COST, CATALYST_COST = 1_000, 20, 50


# ── ODE functions ─────────────────────────────────────────────────────────────
def ideal_kinetics(y, t):
    A, B, C, D = y
    k1 = A1 * np.exp(-Ea1 / (R * T_NOM))
    k2 = A2 * np.exp(-Ea2 / (R * T_NOM))
    k3 = A3 * np.exp(-Ea3 / (R * T_NOM))
    return [-(k1+k3)*A, k1*A - k2*B, k2*B, k3*A]

def real_system(y, t):
    A, B, C, D, act = y
    T  = T_NOM + T_AMP * np.sin(2 * np.pi * t / 24)
    k1 = A1 * np.exp(-Ea1 / (R * T)) * act
    k2 = A2 * np.exp(-Ea2 / (R * T)) * act
    k3 = A3 * np.exp(-Ea3 / (R * T)) * act
    return [-(k1+k3)*A, k1*A - k2*B, k2*B, k3*A, -K_DEACT*act*A]


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model  = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    meta   = json.load(open("models/metadata.json"))
    return model, scaler, meta

try:
    model, scaler, meta = load_artefacts()
    model_ok = True
except Exception as e:
    model_ok = False
    load_error = str(e)


# ── Predict helpers ───────────────────────────────────────────────────────────
def _predict_raw(X_raw):
    X_s = scaler.transform(np.atleast_2d(X_raw))
    if meta["model_name"] == "LightGBM":
        return model.predict(pd.DataFrame(X_s, columns=FEATURES))
    return model.predict(X_s)

def run_trajectory(price, op_cost, cat_cost, n=600):
    t  = np.linspace(0, 48, n)
    id = odeint(ideal_kinetics, [1,0,0,0], t)
    rl = odeint(real_system,    [1,0,0,0,1], t)
    ideal_B  = id[:, 1]
    activity = rl[:, 4]
    temp     = T_NOM + T_AMP * np.sin(2*np.pi*t/24)
    lag1 = np.concatenate([[ideal_B[0]], ideal_B[:-1]])
    lag2 = np.concatenate([[ideal_B[0], ideal_B[0]], ideal_B[:-2]])
    X = np.column_stack([t, temp, temp**2, ideal_B, lag1, lag2, activity])
    hybrid_B = ideal_B + _predict_raw(X)
    profit_h = (hybrid_B * price) - (t * op_cost) - cat_cost
    profit_i = (ideal_B  * price) - (t * op_cost) - cat_cost
    best = int(np.argmax(profit_h))
    return t, ideal_B, hybrid_B, activity, profit_h, profit_i, best


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚗️ Reactor Predictor")
    st.markdown('<hr>', unsafe_allow_html=True)

    if model_ok:
        st.markdown(f"""
        <div style="margin-bottom:20px">
            <div class="metric-label">Model</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:14px;color:#48bb78">
                {meta['model_name']}
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#4a5568;margin-top:4px">
                R² = {meta['metrics']['r2']}  ·  MAE = {meta['metrics']['mae']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**Mode**")
    mode = st.radio("", ["Point prediction", "Full trajectory"], label_visibility="collapsed")

    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown("**Profit parameters**")
    price     = st.slider("Price  ($/unit yield)", 500, 2000, PRICE, 50)
    op_cost   = st.slider("Operating cost  ($/h)",   5,   80, OP_COST, 5)
    cat_cost  = st.slider("Catalyst cost  ($)",       0,  200, CATALYST_COST, 10)

    if model_ok:
        trained = meta.get("trained_at", "")[:10]
        st.markdown(f'<div style="margin-top:24px"><span class="tag">trained {trained}</span></div>',
                    unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if not model_ok:
    st.error(f"Could not load model files. Make sure `models/best_model.pkl` and `models/scaler.pkl` exist.\n\n`{load_error}`")
    st.stop()

PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#111318",
    font=dict(family="IBM Plex Mono", color="#4a5568", size=11),
    xaxis=dict(gridcolor="#1e2330", linecolor="#1e2330", zerolinecolor="#1e2330"),
    yaxis=dict(gridcolor="#1e2330", linecolor="#1e2330", zerolinecolor="#1e2330"),
    margin=dict(l=10, r=10, t=30, b=10),
)


# ── MODE 1: POINT PREDICTION ─────────────────────────────────────────────────
if mode == "Point prediction":
    st.markdown("## Point prediction")
    st.markdown("Enter current reactor readings to get an instant hybrid B estimate.")
    st.markdown("")

    c1, c2, c3 = st.columns(3)
    with c1:
        time_h   = st.slider("Time elapsed  (h)",       0.0, 48.0, 10.0, 0.1)
        temp_K   = st.slider("Temperature  (K)",       275.0, 325.0, 302.0, 0.5)
    with c2:
        ideal_B  = st.slider("Ideal B  (ODE output)",  0.0, 1.0, 0.50, 0.01)
        activity = st.slider("Catalyst activity",       0.0, 1.0, 0.85, 0.01)
    with c3:
        B_lag1   = st.slider("B  (1 step ago)",         0.0, 1.0, 0.49, 0.01)
        B_lag2   = st.slider("B  (2 steps ago)",        0.0, 1.0, 0.48, 0.01)

    feats     = [time_h, temp_K, temp_K**2, ideal_B, B_lag1, B_lag2, activity]
    correction = float(_predict_raw([feats])[0])
    hybrid_B  = ideal_B + correction
    profit_val = (hybrid_B * price) - (time_h * op_cost) - cat_cost

    st.markdown('<hr>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Hybrid B</div>
            <div class="metric-value">{hybrid_B:.4f}</div>
            <div class="metric-delta-pos">↑ {correction:+.4f} vs ideal</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Ideal B (ODE only)</div>
            <div class="metric-value">{ideal_B:.4f}</div>
            <div style="margin-top:4px;font-size:11px;color:#4a5568;font-family:'IBM Plex Mono',monospace">physics baseline</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">ML Correction</div>
            <div class="metric-value">{correction:+.4f}</div>
            <div style="margin-top:4px;font-size:11px;color:#4a5568;font-family:'IBM Plex Mono',monospace">residual learned</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        pcolor = "metric-delta-pos" if profit_val > 0 else "metric-delta-neg"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Profit at this state</div>
            <div class="metric-value">{profit_val:,.0f}</div>
            <div class="{pcolor}">{'▲' if profit_val > 0 else '▼'} ${profit_val:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    # Mini gauge — how far through the optimal run are we?
    t_g, id_B, hy_B, act_t, pr_h, pr_i, best_idx = run_trajectory(price, op_cost, cat_cost)
    opt_t = t_g[best_idx]
    pct   = min(time_h / opt_t * 100, 100) if opt_t > 0 else 0
    bar_color = "#48bb78" if pct < 80 else "#ecc94b" if pct < 100 else "#f56565"

    if time_h < opt_t:
        banner_cls = "stop-banner-warn" if pct > 80 else "stop-banner"
        msg = f"{'⚠️  Approaching optimal stop' if pct > 80 else '✓  Within optimal window'} — optimal stop at <b>{opt_t:.1f} h</b> ({pct:.0f}% elapsed)"
    else:
        banner_cls = "stop-banner"
        msg = f"🛑  Past optimal stop time ({opt_t:.1f} h) — consider stopping the reactor"

    st.markdown(f'<div class="{banner_cls}">{msg}</div>', unsafe_allow_html=True)

    # Progress bar
    st.markdown(f"""
    <div style="margin:6px 0 18px">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#4a5568;margin-bottom:4px">
            RUN PROGRESS  {time_h:.1f}h / {opt_t:.1f}h optimal
        </div>
        <div style="background:#1e2330;border-radius:3px;height:6px">
            <div style="background:{bar_color};width:{min(pct,100):.1f}%;height:6px;border-radius:3px;transition:width 0.3s"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── MODE 2: FULL TRAJECTORY ───────────────────────────────────────────────────
else:
    st.markdown("## Full trajectory")
    st.markdown("Simulates the complete 48-hour run and finds the profit-optimal stop time.")
    st.markdown("")

    with st.spinner("Running simulation..."):
        t_g, id_B, hy_B, act_t, pr_h, pr_i, best_idx = run_trajectory(price, op_cost, cat_cost)

    opt_t      = t_g[best_idx]
    ideal_opt  = int(np.argmax(pr_i))
    ideal_t    = t_g[ideal_opt]
    saving_h   = ideal_t - opt_t
    max_profit = pr_h[best_idx]

    # ── Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Hybrid stop time</div>
            <div class="metric-value">{opt_t:.1f}<span class="metric-unit">h</span></div>
            <div class="metric-delta-pos">↑ {saving_h:.1f}h earlier than ideal</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Ideal stop time</div>
            <div class="metric-value">{ideal_t:.1f}<span class="metric-unit">h</span></div>
            <div style="margin-top:4px;font-size:11px;color:#4a5568;font-family:'IBM Plex Mono',monospace">physics-only decision</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Max profit</div>
            <div class="metric-value">${max_profit:,.0f}</div>
            <div class="metric-delta-pos">at {opt_t:.1f}h</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">B yield at stop</div>
            <div class="metric-value">{hy_B[best_idx]:.3f}</div>
            <div style="margin-top:4px;font-size:11px;color:#4a5568;font-family:'IBM Plex Mono',monospace">hybrid prediction</div>
        </div>""", unsafe_allow_html=True)

    # ── Charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=["B concentration", "Profit curve", "Catalyst activity", "ML correction"],
        vertical_spacing=0.14, horizontal_spacing=0.08,
    )

    # 1. Concentration
    fig.add_trace(go.Scatter(x=t_g, y=id_B, name="Ideal (ODE)",
        line=dict(color="#4a5568", dash="dash", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_g, y=hy_B, name="Hybrid",
        line=dict(color="#48bb78", width=2)), row=1, col=1)
    fig.add_vline(x=opt_t,  line_color="#48bb78", line_dash="dot", line_width=1, row=1, col=1)
    fig.add_vline(x=ideal_t, line_color="#718096", line_dash="dot", line_width=1, row=1, col=1)

    # 2. Profit
    fig.add_trace(go.Scatter(x=t_g, y=pr_i, name="Ideal profit",
        line=dict(color="#4a5568", dash="dash", width=1.5), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=t_g, y=pr_h, name="Hybrid profit",
        line=dict(color="#48bb78", width=2), showlegend=False), row=1, col=2)
    fig.add_vline(x=opt_t, line_color="#48bb78", line_dash="dot", line_width=1, row=1, col=2)
    fig.add_annotation(x=opt_t, y=max_profit, text=f" stop<br> {opt_t:.1f}h",
        font=dict(color="#48bb78", size=10, family="IBM Plex Mono"),
        showarrow=False, xanchor="left", row=1, col=2)

    # 3. Activity
    fig.add_trace(go.Scatter(x=t_g, y=act_t, name="Activity",
        line=dict(color="#ecc94b", width=1.5),
        fill="tozeroy", fillcolor="rgba(236,201,75,0.06)", showlegend=False), row=2, col=1)

    # 4. ML correction
    correction = hy_B - id_B
    fig.add_trace(go.Scatter(x=t_g, y=correction, name="ML correction",
        line=dict(color="#63b3ed", width=1.5), showlegend=False), row=2, col=2)
    fig.add_hline(y=0, line_color="#1e2330", line_width=1, row=2, col=2)

    fig.update_layout(
        height=540,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(family="IBM Plex Mono", size=11), bgcolor="rgba(0,0,0,0)"),
        **PLOT_THEME,
    )
    for ax in ["xaxis", "xaxis2", "xaxis3", "xaxis4"]:
        fig.update_layout(**{ax: dict(gridcolor="#1e2330", linecolor="#1e2330")})
    for ax in ["yaxis", "yaxis2", "yaxis3", "yaxis4"]:
        fig.update_layout(**{ax: dict(gridcolor="#1e2330", linecolor="#1e2330")})
    for ann in fig.layout.annotations:
        ann.font.color = "#4a5568"
        ann.font.family = "IBM Plex Mono"
        ann.font.size = 11

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""<div class="stop-banner">
        <b style="color:#48bb78">Recommendation:</b>
        Stop the reactor at <b style="color:#e2e8f0">{opt_t:.1f} h</b> —
        that is <b style="color:#e2e8f0">{saving_h:.1f} h earlier</b> than the ideal-model decision,
        saving operating costs without sacrificing yield.
        Max projected profit: <b style="color:#e2e8f0">${max_profit:,.0f}</b>.
    </div>""", unsafe_allow_html=True)
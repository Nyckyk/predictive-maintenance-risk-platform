from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================================================
# CONFIGURATION
# =========================================================

OUTPUT_DIR = Path("outputs")
FINANCIAL_RESULTS_PATH = OUTPUT_DIR / "financial_risk_results.csv"
SENSITIVITY_PATH = OUTPUT_DIR / "cost_sensitivity_analysis.csv"
THRESHOLD_PATH = OUTPUT_DIR / "threshold_optimization.csv"
CALIBRATION_PATH = OUTPUT_DIR / "probability_calibration.csv"

RUL_CAP = 125
ACTUAL_HIGH_THRESHOLD = 30
MEDIUM_THRESHOLD = 60

RISK_COLORS = {
    "HIGH": "#ff5a5f",
    "MEDIUM": "#f6c34a",
    "LOW": "#63d98a",
}

BLUE = "#4f8cff"
CYAN = "#55c7ff"
PURPLE = "#9b6dff"
GOLD = "#f6c34a"
RED = "#ff5a5f"
TEXT = "#eaf2ff"
MUTED = "#8ea1be"
GRID = "rgba(136,155,190,0.13)"
TRANSPARENT = "rgba(0,0,0,0)"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Predictive Maintenance & Financial Risk Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLOBAL CSS
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --text:#eef5ff;
        --muted:#8ea1be;
        --blue:#4f8cff;
        --green:#63d98a;
        --amber:#f6c34a;
        --red:#ff5a5f;
    }

    html {
        font-size: 11.5px;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                     BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 73% -5%, rgba(56,115,255,.14), transparent 30rem),
            linear-gradient(180deg, #07111f 0%, #08111d 55%, #07101c 100%);
        color:var(--text);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        top: .35rem;
    }

    .block-container {
        max-width:2040px;
        padding-top:1.7rem;
        padding-bottom:.85rem;
        padding-left:.75rem;
        padding-right:.75rem;
    }

    section[data-testid="stSidebar"] {
        background:linear-gradient(180deg, #0b1830 0%, #081425 100%);
        border-right:1px solid rgba(96,140,202,.20);
        box-shadow:12px 0 38px rgba(0,0,0,.22);
        min-width:285px !important;
        max-width:285px !important;
    }

    section[data-testid="stSidebar"] > div { padding-top:.6rem; }

    section[data-testid="stSidebar"] .stRadio label {
        padding:.34rem .52rem;
        border-radius:10px;
        transition:.15s ease;
        border:1px solid transparent;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background:rgba(79,140,255,.10);
        border-color:rgba(79,140,255,.18);
    }

    section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background:linear-gradient(90deg, rgba(79,140,255,.26), rgba(79,140,255,.08));
        border-color:rgba(79,140,255,.55);
        box-shadow:inset 3px 0 0 #4f8cff;
    }

    .sidebar-brand {
        display:flex;
        gap:.75rem;
        align-items:center;
        margin-bottom:1.15rem;
    }

    .brand-mark {
        width:38px;
        height:38px;
        border-radius:12px;
        background:linear-gradient(135deg, #55c7ff 0%, #4f8cff 52%, #9b6dff 100%);
        box-shadow:0 0 22px rgba(79,140,255,.35), inset 0 0 12px rgba(255,255,255,.18);
        position:relative;
    }

    .brand-mark:before, .brand-mark:after {
        content:"";
        position:absolute;
        background:rgba(255,255,255,.78);
        border-radius:10px;
        transform:rotate(45deg);
    }
    .brand-mark:before { width:9px; height:28px; left:14px; top:5px; }
    .brand-mark:after  { width:28px; height:9px; left:5px; top:14px; }

    .brand-title { font-size:1.02rem; font-weight:800; letter-spacing:.02em; }
    .brand-subtitle {
        font-size:.68rem;
        color:var(--muted);
        letter-spacing:.12em;
        text-transform:uppercase;
        margin-top:1px;
    }

    .hero-shell {
        position:relative;
        overflow:hidden;
        border:1px solid rgba(90,139,211,.28);
        border-radius:20px;
        background:linear-gradient(90deg, rgba(11,29,53,.98) 0%, rgba(10,31,60,.96) 46%, rgba(8,25,51,.82) 68%, rgba(8,20,38,.97) 100%);
        padding:.8rem 1rem .7rem;
        margin-bottom:1.15rem;
        box-shadow:0 18px 52px rgba(0,0,0,.22);
        min-height:112px;
    }

    .hero-shell:before {
        content:"";
        position:absolute;
        inset:0;
        background:
            radial-gradient(circle at 72% 48%, rgba(69,139,255,.22), transparent 17rem),
            linear-gradient(110deg, transparent 50%, rgba(68,134,255,.12), transparent 72%);
        pointer-events:none;
    }

    .hero-copy-wrap { position:relative; z-index:3; width:66%; }
    .hero-kicker {
        color:#9fb3d2;
        font-size:.73rem;
        letter-spacing:.18em;
        text-transform:uppercase;
        font-weight:700;
        margin-bottom:.35rem;
    }
    .hero-title {
        font-size:1.42rem;
        line-height:1.12;
        font-weight:800;
        letter-spacing:-.02em;
        margin-bottom:.42rem;
    }
    .hero-subtitle {
        color:#a8bbd6;
        font-size:.74rem;
        max-width:840px;
        line-height:1.55;
    }

    .value-row { display:flex; gap:.75rem; margin-top:.45rem; flex-wrap:wrap; }
    .value-pill {
        display:flex;
        align-items:center;
        gap:.55rem;
        padding:.26rem .46rem;
        border:1px solid rgba(92,142,214,.24);
        background:rgba(10,31,61,.62);
        border-radius:999px;
        font-size:.72rem;
        color:#b8c9df;
        letter-spacing:.04em;
    }
    .value-dot {
        width:24px;
        height:24px;
        border-radius:999px;
        display:grid;
        place-items:center;
        background:rgba(79,140,255,.16);
        color:#77b3ff;
        font-size:.78rem;
        font-weight:800;
        border:1px solid rgba(79,140,255,.35);
    }

    .turbine {
        position:absolute;
        width:225px;
        height:225px;
        right:-22px;
        top:-50px;
        border-radius:50%;
        background:
            radial-gradient(circle at center, #132b4b 0 12%, #08192e 13% 22%, transparent 23%),
            repeating-conic-gradient(from 9deg, rgba(86,151,255,.58) 0deg 5deg, rgba(17,52,95,.28) 5deg 13deg, rgba(89,147,240,.38) 13deg 18deg, rgba(8,30,58,.26) 18deg 24deg);
        box-shadow:0 0 85px rgba(67,128,255,.28), inset 0 0 38px rgba(110,174,255,.20);
        opacity:.72;
        transform:rotate(-8deg);
    }
    .turbine:after {
        content:"";
        position:absolute;
        inset:39%;
        border-radius:50%;
        background:radial-gradient(circle at 40% 35%, #5e8ecc 0%, #1d416d 40%, #071323 78%);
        box-shadow:0 0 14px rgba(95,160,255,.25), inset 0 0 10px rgba(255,255,255,.08);
    }

    .hero-status {
        position:absolute;
        z-index:4;
        right:1.1rem;
        bottom:.9rem;
        display:flex;
        align-items:center;
        gap:.5rem;
        color:#90a6c4;
        font-size:.70rem;
    }
    .status-dot {
        width:8px;
        height:8px;
        border-radius:999px;
        background:#63d98a;
        box-shadow:0 0 10px rgba(99,217,138,.75);
    }

    .section-title {
        display:flex;
        align-items:center;
        gap:.55rem;
        font-size:1.02rem;
        font-weight:800;
        letter-spacing:-.01em;
        margin:.2rem 0 .15rem;
    }
    .section-subtitle {
        color:#8193af;
        font-size:.83rem;
        margin-bottom:.38rem;
    }

    .kpi-card {
        min-height:78px;
        border-radius:16px;
        border:1px solid rgba(103,145,207,.24);
        background:linear-gradient(145deg, rgba(14,33,59,.92), rgba(9,23,42,.88));
        padding:.56rem .68rem;
        box-shadow:0 12px 34px rgba(0,0,0,.16);
        position:relative;
        overflow:hidden;
    }
    .kpi-card:after {
        content:"";
        position:absolute;
        inset:auto -18px -28px auto;
        width:86px;
        height:86px;
        border-radius:50%;
        background:radial-gradient(circle, var(--glow) 0%, transparent 70%);
        opacity:.26;
    }
    .kpi-head { display:flex; align-items:center; gap:.58rem; margin-bottom:.35rem; }
    .kpi-icon {
        width:32px;
        height:32px;
        border-radius:10px;
        display:grid;
        place-items:center;
        font-size:.72rem;
        font-weight:800;
        color:white;
        background:var(--icon-bg);
        border:1px solid var(--icon-border);
        box-shadow:0 0 18px var(--icon-glow);
    }
    .kpi-label { color:#a3b5cf; font-size:.73rem; }
    .kpi-value { font-size:1.08rem; font-weight:800; margin-top:.05rem; }
    .kpi-note { color:#7286a5; font-size:.68rem; margin-top:.22rem; }

    .fleet-card {
        min-height:74px;
        border-radius:16px;
        border:1px solid rgba(105,146,206,.22);
        background:linear-gradient(145deg, rgba(13,31,55,.96), rgba(8,22,40,.92));
        padding:.72rem .8rem;
        position:relative;
        overflow:hidden;
    }
    .fleet-card .bar {
        position:absolute;
        left:0;
        top:0;
        bottom:0;
        width:4px;
        background:var(--risk);
        box-shadow:0 0 15px var(--risk);
    }
    .fleet-label { color:#a5b6ce; font-size:.74rem; }
    .fleet-value { font-size:1.1rem; font-weight:800; margin-top:.15rem; }
    .fleet-note { color:#7286a5; font-size:.70rem; margin-top:.15rem; }

    .insight-banner {
        display:flex;
        align-items:center;
        gap:.95rem;
        border:1px solid rgba(78,140,255,.46);
        background:linear-gradient(90deg, rgba(41,94,190,.27), rgba(19,53,108,.24));
        border-radius:15px;
        padding:.52rem .7rem;
        box-shadow:inset 3px 0 0 #4f8cff;
        margin:1.05rem 0 .75rem;
    }
    .insight-icon {
        width:38px;
        height:38px;
        border-radius:12px;
        display:grid;
        place-items:center;
        background:rgba(79,140,255,.16);
        border:1px solid rgba(79,140,255,.30);
        color:#8eb9ff;
        font-size:1rem;
        font-weight:800;
        flex:0 0 auto;
    }
    .insight-title { font-weight:800; margin-bottom:.1rem; }
    .insight-copy { color:#95aac7; font-size:.78rem; line-height:1.45; }

    .panel-title {
        font-size:1rem;
        font-weight:800;
        margin-top:.18rem;
        margin-bottom:.28rem;
        line-height:1.25;
    }
    .panel-subtitle { color:#7f92ad; font-size:.72rem; margin-bottom:.55rem; }

    .status-card {
        border:1px solid rgba(102,145,207,.22);
        border-radius:15px;
        padding:.95rem 1rem;
        background:linear-gradient(145deg, rgba(13,31,55,.92), rgba(8,22,40,.90));
        margin-bottom:.75rem;
    }

    .small-muted { color:#7f92ad; font-size:.73rem; }

    div[data-testid="stDataFrame"] {
        border:1px solid rgba(96,137,195,.18);
        border-radius:14px;
        overflow:hidden;
        background:rgba(7,18,33,.35);
    }

    div[data-testid="stMetric"] {
        background:linear-gradient(145deg, rgba(13,31,55,.96), rgba(8,22,40,.92));
        border:1px solid rgba(102,145,207,.22);
        border-radius:15px;
        padding:.95rem 1rem;
        min-height:72px;
    }
    div[data-testid="stMetric"] label { color:#9bb0ca !important; font-size:.73rem !important; }
    div[data-testid="stMetricValue"] { font-size:1.45rem !important; font-weight:800 !important; }

    .stButton > button, .stDownloadButton > button {
        border-radius:10px;
        border:1px solid rgba(91,138,207,.28);
        background:rgba(12,29,51,.84);
        color:#dbeafe;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color:rgba(79,140,255,.60);
        background:rgba(29,67,119,.50);
    }


    div[data-testid="stVerticalBlock"] {
        gap: .42rem;
    }

    div[data-testid="stHorizontalBlock"] {
        gap: .55rem;
    }

    .stSelectbox, .stMultiSelect, .stSlider {
        margin-bottom: -.15rem;
    }


    h1, h2, h3, h4 {
        margin-top: .25rem !important;
        margin-bottom: .35rem !important;
    }

    p {
        margin-bottom: .35rem;
    }

    .footer {
        margin-top:.9rem;
        border-top:1px solid rgba(120,150,190,.13);
        padding-top:.55rem;
        text-align:center;
        color:#61738f;
        font-size:.66rem;
        letter-spacing:.04em;
    }

    @media (max-width:1100px) {
        .hero-copy-wrap { width:78%; }
        .turbine { opacity:.34; right:-95px; }
    }

    @media (max-width:850px) {
        .block-container { padding-top:2.35rem; }
        .hero-copy-wrap { width:100%; }
        .turbine { opacity:.12; right:-135px; }
        .hero-status { position:relative; right:auto; bottom:auto; margin-top:.8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

@st.cache_data
def load_dashboard_data():
    """Load outputs generated by src/pipeline.py."""

    return (
        pd.read_csv(FINANCIAL_RESULTS_PATH),
        pd.read_csv(SENSITIVITY_PATH),
        pd.read_csv(THRESHOLD_PATH),
        pd.read_csv(CALIBRATION_PATH),
    )


def get_optimised_threshold(thresholds: pd.DataFrame) -> int:
    if "selected" not in thresholds.columns:
        return ACTUAL_HIGH_THRESHOLD

    selected_mask = (
        thresholds["selected"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
    )

    selected = thresholds[selected_mask]

    if selected.empty:
        return ACTUAL_HIGH_THRESHOLD

    return int(selected.iloc[0]["threshold"])


def assign_predicted_risk(predicted_rul: float, high_threshold: int) -> str:
    if predicted_rul <= high_threshold:
        return "HIGH"
    if predicted_rul <= MEDIUM_THRESHOLD:
        return "MEDIUM"
    return "LOW"


def recommendation_for(risk: str) -> str:
    if risk == "HIGH":
        return "Schedule preventative maintenance"
    if risk == "MEDIUM":
        return "Increase monitoring"
    return "Continue operating"


def short_action(action: str) -> str:
    return {
        "Preventative maintenance": "Maintain",
        "Continue / monitor": "Monitor",
    }.get(action, action)


def money(value: float) -> str:
    value = float(value)
    return f"-£{abs(value):,.0f}" if value < 0 else f"£{value:,.0f}"


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def configure_plot(fig, *, height: int | None = None):
    fig.update_layout(
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        font={"color": TEXT},
        margin={"l": 20, "r": 20, "t": 45, "b": 35},
        legend={"title_text": ""},
    )

    if height is not None:
        fig.update_layout(height=height)

    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def render_kpi_card(icon: str, label: str, value: str, note: str, accent: str):
    st.markdown(
        f"""
        <div class="kpi-card" style="--glow:{accent};--icon-bg:{accent}22;--icon-border:{accent}66;--icon-glow:{accent}55;">
            <div class="kpi-head">
                <div class="kpi-icon">{icon}</div>
                <div class="kpi-label">{label}</div>
            </div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_fleet_card(label: str, value: int, note: str, risk_color: str):
    st.markdown(
        f"""
        <div class="fleet-card" style="--risk:{risk_color};">
            <div class="bar"></div>
            <div class="fleet-label">{label}</div>
            <div class="fleet-value">{value}</div>
            <div class="fleet-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# LOAD DATA
# =========================================================

required_paths = [
    FINANCIAL_RESULTS_PATH,
    SENSITIVITY_PATH,
    THRESHOLD_PATH,
    CALIBRATION_PATH,
]

missing_paths = [path for path in required_paths if not path.exists()]

if missing_paths:
    st.error(
        "Required pipeline outputs are missing. Run `python src/pipeline.py` first."
    )
    st.code("\n".join(str(path) for path in missing_paths))
    st.stop()

try:
    (
        financial_results,
        sensitivity_results,
        threshold_results,
        calibration_results,
    ) = load_dashboard_data()
except Exception as error:
    st.error("The dashboard could not load the pipeline outputs.")
    st.exception(error)
    st.stop()

optimised_threshold = get_optimised_threshold(threshold_results)
financial_results = financial_results.copy()

financial_results["actual_high"] = (
    financial_results["actual_rul_capped"] <= ACTUAL_HIGH_THRESHOLD
).astype(int)

financial_results["predicted_high"] = (
    financial_results["predicted_rul_capped"] <= optimised_threshold
).astype(int)

financial_results["risk_level"] = (
    financial_results["predicted_rul_capped"]
    .apply(lambda x: assign_predicted_risk(x, optimised_threshold))
)

financial_results["recommendation"] = (
    financial_results["risk_level"].apply(recommendation_for)
)

capped_mae = mean_absolute_error(
    financial_results["actual_rul_capped"],
    financial_results["predicted_rul_capped_raw"],
)

high_recall = recall_score(
    financial_results["actual_high"],
    financial_results["predicted_high"],
)

high_precision = precision_score(
    financial_results["actual_high"],
    financial_results["predicted_high"],
    zero_division=0,
)

roc_auc = roc_auc_score(
    financial_results["actual_high"],
    financial_results["high_risk_probability"],
)

brier_score = brier_score_loss(
    financial_results["actual_high"],
    financial_results["high_risk_probability"],
)

# Honest comparison against the default 30-cycle threshold.
default_high = (
    financial_results["predicted_rul_capped"] <= ACTUAL_HIGH_THRESHOLD
).astype(int)

default_recall = recall_score(
    financial_results["actual_high"],
    default_high,
)

default_precision = precision_score(
    financial_results["actual_high"],
    default_high,
    zero_division=0,
)

high_count = int((financial_results["risk_level"] == "HIGH").sum())
medium_count = int((financial_results["risk_level"] == "MEDIUM").sum())
low_count = int((financial_results["risk_level"] == "LOW").sum())
financially_selected = int(
    financial_results["maintenance_economically_justified"].sum()
)

base_maintenance_cost = float(financial_results["maintenance_cost"].iloc[0])
base_failure_cost = float(financial_results["failure_cost"].iloc[0])
base_break_even = float(financial_results["break_even_probability"].iloc[0])

last_updated = datetime.fromtimestamp(
    FINANCIAL_RESULTS_PATH.stat().st_mtime
).strftime("%d %b %Y · %H:%M")


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-mark"></div>
            <div>
                <div class="brand-title">Predictive Maintenance</div>
                <div class="brand-subtitle">Reliability analytics</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Decision-support dashboard")

    selected_page = st.radio(
        "Dashboard section",
        [
            "▣  Executive Overview",
            "▥  Fleet Risk",
            "⚙  Individual Engine",
            "⌁  Model Performance",
            "▤  Financial Scenarios",
        ],
    )

    page = (
        selected_page
        .replace("▣  ", "")
        .replace("▥  ", "")
        .replace("⚙  ", "")
        .replace("⌁  ", "")
        .replace("▤  ", "")
    )

    st.divider()
    st.caption("MODEL")
    st.write("**Gradient Boosting**")
    st.caption("Capped RUL target")
    st.write(f"**{RUL_CAP} cycles**")
    st.caption("Operational HIGH threshold")
    st.write(f"**{optimised_threshold} cycles**")

    st.divider()

    with st.expander("Methodology & assumptions"):
        st.markdown(
            f"""
            **Dataset**
            - NASA C-MAPSS FD001 simulated turbofan degradation data

            **Predictive modelling**
            - Gradient Boosting selected with engine-level grouped cross-validation
            - RUL capped at {RUL_CAP} cycles
            - Reference HIGH risk: actual RUL ≤ {ACTUAL_HIGH_THRESHOLD} cycles

            **Operational threshold**
            - HIGH alert when predicted RUL ≤ {optimised_threshold} cycles

            **Probability layer**
            - Logistic regression estimates the probability that actual RUL is within {ACTUAL_HIGH_THRESHOLD} cycles

            **Financial assumptions**
            - Preventative maintenance: {money(base_maintenance_cost)}
            - Unplanned failure: {money(base_failure_cost)}
            - Costs are illustrative, not NASA dataset values
            """
        )


# =========================================================
# HERO
# =========================================================

hero_html = (
    f'<div class="hero-shell">'
    f'<div class="hero-copy-wrap">'
    f'<div class="hero-kicker">Industrial reliability analytics</div>'
    f'<div class="hero-title">Predictive Maintenance &amp; Financial Risk Platform</div>'
    f'<div class="hero-subtitle">'
    f'Sensor-driven Remaining Useful Life prediction, maintenance prioritisation, '
    f'calibrated short-horizon risk estimation and scenario-based financial decision support.'
    f'</div>'
    f'<div class="value-row">'
    f'<div class="value-pill"><div class="value-dot">R</div>Predict remaining life</div>'
    f'<div class="value-pill"><div class="value-dot">!</div>Prioritise maintenance risk</div>'
    f'<div class="value-pill"><div class="value-dot">£</div>Test financial scenarios</div>'
    f'</div>'
    f'</div>'
    f'<div class="turbine"></div>'
    f'<div class="hero-status">'
    f'<span class="status-dot"></span>'
    f'Model outputs loaded · {last_updated}'
    f'</div>'
    f'</div>'
)

st.markdown(
    hero_html,
    unsafe_allow_html=True,
)


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":
    st.markdown(
        """
        <div class="section-title">▥ Executive Overview</div>
        <div class="section-subtitle">Predictive performance, fleet condition and current maintenance priorities.</div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5, gap="small")

    with k1:
        render_kpi_card(
            "MAE",
            "Capped Test MAE",
            f"{capped_mae:.2f} cycles",
            "Lower is better",
            BLUE,
        )
    with k2:
        render_kpi_card(
            "R",
            "HIGH-Risk Recall",
            f"{high_recall:.1%}",
            f"{default_recall:.1%} → {high_recall:.1%} vs default threshold",
            "#43c6a0",
        )
    with k3:
        render_kpi_card(
            "P",
            "HIGH-Risk Precision",
            f"{high_precision:.1%}",
            f"Default threshold: {default_precision:.1%}",
            PURPLE,
        )
    with k4:
        render_kpi_card(
            "AUC",
            "ROC AUC",
            f"{roc_auc:.3f}",
            "HIGH-risk discrimination",
            GOLD,
        )
    with k5:
        render_kpi_card(
            "B",
            "Brier Score",
            f"{brier_score:.4f}",
            "Lower is better",
            "#e75d7d",
        )

    st.markdown(
        '<div class="section-title" style="font-size:1.05rem;margin-top:.9rem;margin-bottom:.22rem;">◉ Fleet summary</div>',
        unsafe_allow_html=True,
    )

    f1, f2, f3, f4 = st.columns(4, gap="small")

    with f1:
        render_fleet_card(
            "HIGH Risk",
            high_count,
            "Requires near-term attention",
            RISK_COLORS["HIGH"],
        )
    with f2:
        render_fleet_card(
            "MEDIUM Risk",
            medium_count,
            "Monitor closely",
            RISK_COLORS["MEDIUM"],
        )
    with f3:
        render_fleet_card(
            "LOW Risk",
            low_count,
            "Operating normally",
            RISK_COLORS["LOW"],
        )
    with f4:
        render_fleet_card(
            "Financially Selected",
            financially_selected,
            "Cross current economic threshold",
            BLUE,
        )

    st.markdown(
        f"""
        <div class="insight-banner">
            <div class="insight-icon">i</div>
            <div>
                <div class="insight-title">Key decision insight</div>
                <div class="insight-copy">
                    Under the current illustrative cost assumptions, {financially_selected} of
                    {len(financial_results)} engines cross the economic maintenance threshold.
                    The operational HIGH-risk threshold is {optimised_threshold} cycles, with
                    {high_recall:.1%} recall and {high_precision:.1%} precision on the test fleet.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.95, 1.25], gap="medium")

    with left:
        st.markdown(
            """
            <div class="panel-title">◔ Predicted fleet risk</div>
            <div class="panel-subtitle">Fleet Risk Distribution</div>
            """,
            unsafe_allow_html=True,
        )

        risk_counts = (
            financial_results["risk_level"]
            .value_counts()
            .reindex(["HIGH", "MEDIUM", "LOW"], fill_value=0)
            .rename_axis("Risk Level")
            .reset_index(name="Engines")
        )

        risk_fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Engines",
            hole=0.61,
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
        )
        risk_fig.update_traces(
            textposition="inside",
            textinfo="percent",
            marker={"line": {"color": "#0b1625", "width": 2}},
        )
        risk_fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"<b>{len(financial_results)}</b><br><span style='font-size:11px;color:#8ea1be'>Engines</span>",
            showarrow=False,
            font={"size": 22, "color": TEXT},
        )
        risk_fig = configure_plot(risk_fig, height=325)
        st.plotly_chart(risk_fig, use_container_width=True)

    with right:
        st.markdown(
            """
            <div class="panel-title">☷ Top priority engines</div>
            <div class="panel-subtitle">Highest estimated short-horizon risk</div>
            """,
            unsafe_allow_html=True,
        )

        top_priority = (
            financial_results
            .sort_values(
                ["high_risk_probability", "predicted_rul_capped"],
                ascending=[False, True],
            )
            .head(10)
            .copy()
        )

        top_priority["Action"] = top_priority["economic_action"].apply(short_action)
        top_priority = top_priority[
            [
                "engine_id",
                "cycle",
                "predicted_rul_capped",
                "high_risk_probability",
                "risk_level",
                "Action",
            ]
        ].rename(
            columns={
                "engine_id": "Engine",
                "cycle": "Cycle",
                "predicted_rul_capped": "Predicted RUL",
                "high_risk_probability": "HIGH-Risk Probability",
                "risk_level": "Risk",
            }
        )

        st.dataframe(
            top_priority,
            hide_index=True,
            use_container_width=True,
            height=250,
            column_config={
                "Predicted RUL": st.column_config.NumberColumn(format="%.1f"),
                "HIGH-Risk Probability": st.column_config.ProgressColumn(
                    format="percent",
                    min_value=0,
                    max_value=1,
                ),
            },
        )

    with st.expander("How to interpret this dashboard"):
        st.markdown(
            f"""
            - **RUL** means Remaining Useful Life in engine cycles.
            - **HIGH risk** means predicted RUL ≤ **{optimised_threshold} cycles**.
            - **HIGH-risk probability** estimates whether actual RUL is within **{ACTUAL_HIGH_THRESHOLD} cycles**.
            - **Financially selected** means risk-adjusted exposure exceeds the assumed maintenance cost.
            - Actual future RUL is intentionally excluded from operational views and shown only for evaluation.
            - Financial values are illustrative scenario outputs, not demonstrated real-world savings.
            """
        )


# =========================================================
# FLEET RISK
# =========================================================

elif page == "Fleet Risk":
    st.markdown(
        """
        <div class="section-title">▥ Fleet Risk</div>
        <div class="section-subtitle">Operational fleet view using predicted RUL, estimated short-horizon risk and economic action.</div>
        """,
        unsafe_allow_html=True,
    )

    filter_col1, filter_col2, filter_col3 = st.columns([1.35, 1, 1])

    risk_filter = filter_col1.multiselect(
        "Risk level",
        options=["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"],
    )

    minimum_probability = (
        filter_col2.slider(
            "Minimum HIGH-risk probability",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
        )
        / 100
    )

    financially_selected_only = filter_col3.toggle(
        "Financially selected only",
        value=False,
    )

    fleet = financial_results[
        financial_results["risk_level"].isin(risk_filter)
    ].copy()

    fleet = fleet[
        fleet["high_risk_probability"] >= minimum_probability
    ]

    if financially_selected_only:
        fleet = fleet[fleet["maintenance_economically_justified"]]

    fleet = fleet.sort_values(
        ["high_risk_probability", "predicted_rul_capped"],
        ascending=[False, True],
    )

    c1, c2, c3 = st.columns(3, gap="small")
    c1.metric("Engines Shown", len(fleet))
    c2.metric(
        "Mean HIGH-Risk Probability",
        f"{fleet['high_risk_probability'].mean():.1%}" if not fleet.empty else "—",
    )
    c3.metric(
        "Financially Selected",
        int(fleet["maintenance_economically_justified"].sum()) if not fleet.empty else 0,
    )

    fleet_display = fleet[
        [
            "engine_id",
            "cycle",
            "predicted_rul_capped",
            "risk_level",
            "high_risk_probability",
            "risk_adjusted_failure_exposure",
            "risk_adjusted_net_benefit",
            "economic_action",
        ]
    ].copy()

    fleet_display["risk_adjusted_failure_exposure"] = (
        fleet_display["risk_adjusted_failure_exposure"].apply(money)
    )
    fleet_display["risk_adjusted_net_benefit"] = (
        fleet_display["risk_adjusted_net_benefit"].apply(money)
    )
    fleet_display["economic_action"] = (
        fleet_display["economic_action"].apply(short_action)
    )

    fleet_display = fleet_display.rename(
        columns={
            "engine_id": "Engine",
            "cycle": "Current Cycle",
            "predicted_rul_capped": "Predicted RUL",
            "risk_level": "Risk",
            "high_risk_probability": "HIGH-Risk Probability",
            "risk_adjusted_failure_exposure": "Risk-Adjusted Exposure",
            "risk_adjusted_net_benefit": "Risk-Adjusted Net Benefit",
            "economic_action": "Action",
        }
    )

    st.dataframe(
        fleet_display,
        hide_index=True,
        use_container_width=True,
        height=390,
        column_config={
            "Predicted RUL": st.column_config.NumberColumn(format="%.1f"),
            "HIGH-Risk Probability": st.column_config.ProgressColumn(
                format="percent",
                min_value=0,
                max_value=1,
            ),
        },
    )

    st.download_button(
        "Download filtered fleet CSV",
        data=csv_bytes(fleet),
        file_name="fleet_risk_filtered.csv",
        mime="text/csv",
    )

    st.caption(
        "Actual RUL is intentionally hidden because it would not be known in a live deployment."
    )


# =========================================================
# INDIVIDUAL ENGINE
# =========================================================

elif page == "Individual Engine":
    st.markdown(
        """
        <div class="section-title">⚙ Individual Engine Analysis</div>
        <div class="section-subtitle">Operational decision view for one engine. Ground-truth RUL is available separately for evaluation only.</div>
        """,
        unsafe_allow_html=True,
    )

    engine_ids = sorted(financial_results["engine_id"].unique())
    selected_engine = st.selectbox("Select engine", engine_ids)

    engine = financial_results[
        financial_results["engine_id"] == selected_engine
    ].iloc[0]

    risk = engine["risk_level"]
    recommendation = recommendation_for(risk)
    risk_color = RISK_COLORS[risk]

    st.markdown(
        f"""
        <div class="status-card" style="border-left:4px solid {risk_color};box-shadow:inset 0 0 30px {risk_color}08;">
            <div class="small-muted">Current operational status</div>
            <div style="font-size:1.26rem;font-weight:800;margin-top:.18rem;">
                Engine {int(engine['engine_id'])} · <span style="color:{risk_color};">{risk}</span>
            </div>
            <div class="small-muted" style="margin-top:.3rem;">{recommendation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="small")
    c1.metric("Current Cycle", int(engine["cycle"]))
    c2.metric("Predicted RUL", f"{engine['predicted_rul_capped']:.1f} cycles")
    c3.metric("Operational Risk", risk)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("#### Engineering risk")

        probability = float(engine["high_risk_probability"])

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "valueformat": ".1f"},
                title={"text": f"Probability actual RUL ≤ {ACTUAL_HIGH_THRESHOLD} cycles"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": MUTED},
                    "bar": {"color": risk_color},
                    "bgcolor": "rgba(11,24,42,.82)",
                    "bordercolor": "rgba(108,150,208,.22)",
                    "steps": [
                        {
                            "range": [0, base_break_even * 100],
                            "color": "rgba(99,217,138,.08)",
                        },
                        {
                            "range": [base_break_even * 100, 100],
                            "color": "rgba(255,90,95,.06)",
                        },
                    ],
                    "threshold": {
                        "line": {"color": "#eaf2ff", "width": 4},
                        "thickness": 0.8,
                        "value": base_break_even * 100,
                    },
                },
            )
        )

        gauge.update_layout(
            height=210,
            paper_bgcolor=TRANSPARENT,
            font={"color": TEXT},
            margin={"l": 30, "r": 30, "t": 75, "b": 10},
        )

        st.plotly_chart(gauge, use_container_width=True)
        st.caption(
            f"White marker = financial break-even probability ({base_break_even:.1%})."
        )
        st.write(
            f"Operational HIGH threshold: **{optimised_threshold} cycles**"
        )
        st.write(f"Recommendation: **{recommendation}**")

    with right:
        st.markdown("#### Financial decision")

        st.metric(
            "Risk-Adjusted Exposure",
            money(engine["risk_adjusted_failure_exposure"]),
        )
        st.metric(
            "Illustrative Net Benefit",
            money(engine["risk_adjusted_net_benefit"]),
        )
        st.metric(
            "Break-Even Probability",
            f"{base_break_even:.1%}",
        )

        st.write(f"Economic action: **{engine['economic_action']}**")

        probability_gap = probability - base_break_even

        if probability_gap >= 0:
            st.success(
                "Estimated HIGH-risk probability is "
                f"{probability_gap:.1%} above the current break-even threshold."
            )
        else:
            st.info(
                "Estimated HIGH-risk probability is "
                f"{abs(probability_gap):.1%} below the current break-even threshold."
            )

    with st.expander("Evaluation-only ground truth"):
        prediction_error = (
            engine["predicted_rul_capped"] - engine["actual_rul_capped"]
        )

        e1, e2 = st.columns(2)
        e1.metric("Actual RUL", f"{engine['actual_rul_capped']:.0f} cycles")
        e2.metric("Prediction Error", f"{prediction_error:+.1f} cycles")

        st.caption(
            "These values are available because this is an evaluation dataset. "
            "A live system would not know future actual RUL."
        )

    st.warning(
        "Financial values use illustrative assumptions and are not demonstrated real-world savings."
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":
    st.markdown(
        """
        <div class="section-title">⌁ Model Performance</div>
        <div class="section-subtitle">Evaluation-only view of RUL accuracy, threshold behaviour and probability calibration.</div>
        """,
        unsafe_allow_html=True,
    )

    p1, p2, p3, p4 = st.columns(4, gap="small")

    with p1:
        render_kpi_card(
            "MAE",
            "Capped Test MAE",
            f"{capped_mae:.2f} cycles",
            "Lower is better",
            BLUE,
        )
    with p2:
        render_kpi_card(
            "R",
            "HIGH-Risk Recall",
            f"{high_recall:.1%}",
            f"Default threshold: {default_recall:.1%}",
            "#43c6a0",
        )
    with p3:
        render_kpi_card(
            "AUC",
            "ROC AUC",
            f"{roc_auc:.3f}",
            "HIGH-risk discrimination",
            GOLD,
        )
    with p4:
        render_kpi_card(
            "B",
            "Brier Score",
            f"{brier_score:.4f}",
            "Lower is better",
            "#e75d7d",
        )

    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        st.markdown(
            '<div class="panel-title" style="padding-top:.9rem;">RUL prediction performance</div>',
            unsafe_allow_html=True,
        )

        rul_fig = px.scatter(
            financial_results,
            x="actual_rul_capped",
            y="predicted_rul_capped",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            hover_data={
                "engine_id": True,
                "actual_rul_capped": ":.0f",
                "predicted_rul_capped": ":.1f",
                "high_risk_probability": ":.1%",
            },
            labels={
                "actual_rul_capped": "Actual RUL (cycles)",
                "predicted_rul_capped": "Predicted RUL (cycles)",
                "engine_id": "Engine",
                "risk_level": "Risk",
            },
        )

        max_rul = max(
            float(financial_results["actual_rul_capped"].max()),
            float(financial_results["predicted_rul_capped"].max()),
        )

        rul_fig.add_trace(
            go.Scatter(
                x=[0, max_rul],
                y=[0, max_rul],
                mode="lines",
                name="Perfect prediction",
                line={"dash": "dash", "color": MUTED},
            )
        )

        rul_fig = configure_plot(rul_fig, height=330)
        st.plotly_chart(rul_fig, use_container_width=True)

    with chart_col2:
        st.markdown(
            '<div class="panel-title" style="padding-top:.9rem;">Probability calibration</div>',
            unsafe_allow_html=True,
        )

        calibration_fig = go.Figure()
        calibration_fig.add_trace(
            go.Scatter(
                x=calibration_results["mean_predicted_probability"],
                y=calibration_results["actual_high_rate"],
                mode="lines+markers",
                name="Observed calibration",
                line={"color": BLUE, "width": 3},
                marker={"size": 8},
            )
        )
        calibration_fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Perfect calibration",
                line={"dash": "dash", "color": MUTED},
            )
        )
        calibration_fig.update_layout(
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Observed HIGH Rate",
        )
        calibration_fig = configure_plot(calibration_fig, height=330)
        st.plotly_chart(calibration_fig, use_container_width=True)

    st.markdown("#### Default vs optimised HIGH-risk threshold")

    comparison = pd.DataFrame(
        {
            "Threshold": [
                f"Default ({ACTUAL_HIGH_THRESHOLD} cycles)",
                f"Optimised ({optimised_threshold} cycles)",
            ],
            "Recall": [
                default_recall,
                high_recall,
            ],
            "Precision": [
                default_precision,
                high_precision,
            ],
            "HIGH-Risk Engines Flagged": [
                int(default_high.sum()),
                int(financial_results["predicted_high"].sum()),
            ],
        }
    )

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Recall": st.column_config.NumberColumn(format="percent"),
            "Precision": st.column_config.NumberColumn(format="percent"),
        },
    )

    st.markdown("#### Calibration data")

    st.dataframe(
        calibration_results,
        hide_index=True,
        use_container_width=True,
        column_config={
            "mean_predicted_probability": st.column_config.NumberColumn(
                "Mean Predicted Probability",
                format="percent",
            ),
            "actual_high_rate": st.column_config.NumberColumn(
                "Observed HIGH Rate",
                format="percent",
            ),
        },
    )

    with st.expander("Metric interpretation"):
        st.markdown(
            """
            - **MAE:** average RUL prediction error in cycles.
            - **Recall:** share of truly HIGH-risk engines detected.
            - **Precision:** share of HIGH alerts that are genuinely HIGH risk.
            - **ROC AUC:** ranking discrimination between HIGH and non-HIGH cases.
            - **Brier score:** probability forecast error; lower is better.
            """
        )


# =========================================================
# FINANCIAL SCENARIOS
# =========================================================

elif page == "Financial Scenarios":
    st.markdown(
        """
        <div class="section-title">▤ Financial Scenario Analysis</div>
        <div class="section-subtitle">Test how preventative maintenance and failure-cost assumptions change the illustrative economic decision threshold.</div>
        """,
        unsafe_allow_html=True,
    )

    maintenance_options = sorted(
        sensitivity_results["maintenance_cost"].unique()
    )
    failure_options = sorted(
        sensitivity_results["failure_cost"].unique()
    )

    c1, c2 = st.columns(2)

    selected_maintenance_cost = c1.selectbox(
        "Preventative maintenance cost",
        maintenance_options,
        index=(
            maintenance_options.index(8000)
            if 8000 in maintenance_options
            else 0
        ),
        format_func=money,
    )

    selected_failure_cost = c2.selectbox(
        "Unplanned failure cost",
        failure_options,
        index=(
            failure_options.index(40000)
            if 40000 in failure_options
            else 0
        ),
        format_func=money,
    )

    scenario = sensitivity_results[
        (
            sensitivity_results["maintenance_cost"]
            == selected_maintenance_cost
        )
        &
        (
            sensitivity_results["failure_cost"]
            == selected_failure_cost
        )
    ].iloc[0]

    st.markdown("#### Selected scenario")

    s1, s2, s3, s4 = st.columns(4, gap="small")
    s1.metric(
        "Break-Even Probability",
        f"{scenario['break_even_probability']:.1%}",
    )
    s2.metric(
        "Engines Selected",
        int(scenario["engines_selected"]),
    )
    s3.metric(
        "Maintenance Outlay",
        money(scenario["maintenance_outlay"]),
    )
    s4.metric(
        "Illustrative Net Benefit",
        money(scenario["risk_adjusted_net_benefit"]),
    )

    st.info(
        "The break-even probability is the estimated HIGH-risk probability "
        "at which risk-adjusted exposure equals the assumed preventative maintenance cost."
    )

    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        st.markdown('<div class="panel-title">Net benefit vs failure cost</div>', unsafe_allow_html=True)

        line_data = sensitivity_results[
            sensitivity_results["maintenance_cost"]
            == selected_maintenance_cost
        ].copy()

        line_fig = px.line(
            line_data,
            x="failure_cost",
            y="risk_adjusted_net_benefit",
            markers=True,
            labels={
                "failure_cost": "Failure Cost (£)",
                "risk_adjusted_net_benefit": "Illustrative Net Benefit (£)",
            },
        )

        line_fig.update_traces(
            line={"color": BLUE, "width": 3},
            marker={"size": 8},
        )

        line_fig.add_trace(
            go.Scatter(
                x=[selected_failure_cost],
                y=[float(scenario["risk_adjusted_net_benefit"])],
                mode="markers",
                name="Selected scenario",
                marker={
                    "size": 14,
                    "symbol": "diamond",
                    "color": GOLD,
                    "line": {"width": 2, "color": "#f8fafc"},
                },
            )
        )

        line_fig = configure_plot(line_fig, height=330)
        st.plotly_chart(line_fig, use_container_width=True)

    with chart_col2:
        st.markdown('<div class="panel-title">Scenario heatmap</div>', unsafe_allow_html=True)

        heatmap_data = (
            sensitivity_results
            .pivot(
                index="maintenance_cost",
                columns="failure_cost",
                values="risk_adjusted_net_benefit",
            )
            .sort_index()
        )

        heatmap_fig = px.imshow(
            heatmap_data,
            aspect="auto",
            labels={
                "x": "Failure Cost (£)",
                "y": "Maintenance Cost (£)",
                "color": "Illustrative Net Benefit (£)",
            },
            color_continuous_scale=[
                [0.0, "#10284b"],
                [0.5, "#2e6db8"],
                [1.0, "#8ed1ff"],
            ],
        )

        heatmap_fig.add_trace(
            go.Scatter(
                x=[selected_failure_cost],
                y=[selected_maintenance_cost],
                mode="markers",
                name="Selected scenario",
                marker={
                    "size": 17,
                    "symbol": "x",
                    "color": GOLD,
                    "line": {"width": 2},
                },
            )
        )

        heatmap_fig = configure_plot(heatmap_fig, height=330)
        st.plotly_chart(heatmap_fig, use_container_width=True)

    st.markdown("#### All cost scenarios")

    scenarios_display = sensitivity_results.copy()
    scenarios_display["maintenance_cost"] = scenarios_display["maintenance_cost"].apply(money)
    scenarios_display["failure_cost"] = scenarios_display["failure_cost"].apply(money)
    scenarios_display["break_even_probability"] = scenarios_display["break_even_probability"].apply(
        lambda value: f"{value:.1%}"
    )
    scenarios_display["maintenance_outlay"] = scenarios_display["maintenance_outlay"].apply(money)
    scenarios_display["risk_adjusted_exposure"] = scenarios_display["risk_adjusted_exposure"].apply(money)
    scenarios_display["risk_adjusted_net_benefit"] = scenarios_display["risk_adjusted_net_benefit"].apply(money)

    scenarios_display = scenarios_display.rename(
        columns={
            "maintenance_cost": "Maintenance Cost",
            "failure_cost": "Failure Cost",
            "break_even_probability": "Break-Even Probability",
            "engines_selected": "Engines Selected",
            "maintenance_outlay": "Maintenance Outlay",
            "risk_adjusted_exposure": "Risk-Adjusted Exposure",
            "risk_adjusted_net_benefit": "Illustrative Net Benefit",
        }
    )

    st.dataframe(
        scenarios_display,
        hide_index=True,
        use_container_width=True,
        height=340,
    )

    d1, d2 = st.columns(2)
    d1.download_button(
        "Download sensitivity analysis",
        data=csv_bytes(sensitivity_results),
        file_name="cost_sensitivity_analysis.csv",
        mime="text/csv",
    )
    d2.download_button(
        "Download financial risk results",
        data=csv_bytes(financial_results),
        file_name="financial_risk_results.csv",
        mime="text/csv",
    )

    st.warning(
        "Financial outputs are scenario-based and use illustrative cost assumptions "
        "rather than observed NASA maintenance or failure costs."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        NASA C-MAPSS FD001 · Portfolio project · Financial assumptions are illustrative
    </div>
    """,
    unsafe_allow_html=True,
)

from datetime import datetime
from pathlib import Path
import math

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
        --bg:#050d19;
        --bg2:#071323;
        --panel:#0b1a2e;
        --panel2:#0e2139;
        --panel3:#102846;
        --line:rgba(95,145,214,.22);
        --line-strong:rgba(91,151,239,.42);
        --text:#f3f7ff;
        --muted:#8ea3c2;
        --blue:#4f8cff;
        --cyan:#55c7ff;
        --green:#67d99a;
        --amber:#f5c451;
        --red:#f5656b;
        --purple:#8e68ff;
    }

    html {
        font-size: 11.5px;
    }

    html, body, [class*="css"] {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                     BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    body {
        background:#040b15;
    }

    .stApp {
        background:
            radial-gradient(circle at 74% -8%, rgba(44,103,211,.12), transparent 28rem),
            linear-gradient(180deg, #06101d 0%, #07111e 42%, #050d18 100%);
        color:var(--text);
    }

    header[data-testid="stHeader"] {
        background:transparent;
        height:1.25rem;
    }

    [data-testid="stToolbar"],
    #MainMenu,
    [data-testid="stStatusWidget"] {
        visibility:hidden;
    }

    .block-container {
        max-width:2100px;
        padding-top:.78rem;
        padding-bottom:1rem;
        padding-left:1.05rem;
        padding-right:1.05rem;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 15% 0%, rgba(66,123,229,.16), transparent 18rem),
            linear-gradient(180deg, #0a1830 0%, #081527 52%, #071321 100%);
        border-right:1px solid rgba(105,150,214,.17);
        box-shadow:14px 0 44px rgba(0,0,0,.28);
        min-width:258px !important;
        max-width:258px !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top:.85rem;
        padding-left:.7rem;
        padding-right:.7rem;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
        display:none;
    }

    .sidebar-brand {
        display:flex;
        gap:.72rem;
        align-items:center;
        margin:.15rem .25rem 1.1rem;
    }

    .brand-mark {
        width:40px;
        height:40px;
        border-radius:12px;
        background:
            radial-gradient(circle at 38% 32%, rgba(255,255,255,.30), transparent 24%),
            linear-gradient(135deg, #5ad1ff 0%, #4e87ff 48%, #7f63ff 100%);
        box-shadow:0 0 24px rgba(79,140,255,.32), inset 0 0 12px rgba(255,255,255,.18);
        position:relative;
        transform:rotate(0deg);
        flex:0 0 auto;
    }

    .brand-mark:before,
    .brand-mark:after {
        content:"";
        position:absolute;
        background:rgba(226,240,255,.90);
        border-radius:9px;
        transform:rotate(45deg);
    }

    .brand-mark:before {
        width:9px;
        height:29px;
        left:15px;
        top:5px;
    }

    .brand-mark:after {
        width:29px;
        height:9px;
        left:5px;
        top:15px;
    }

    .brand-title {
        font-size:1.25rem;
        font-weight:850;
        letter-spacing:.04em;
        line-height:1;
    }

    .brand-subtitle {
        margin-top:.28rem;
        font-size:.60rem;
        color:#8196b5;
        letter-spacing:.18em;
        text-transform:uppercase;
        font-weight:700;
    }

    .sidebar-caption {
        color:#8ba0be;
        font-size:.72rem;
        margin:.2rem .25rem .5rem;
    }

    section[data-testid="stSidebar"] .stRadio > label {
        color:#7f94b2 !important;
        font-size:.67rem !important;
        text-transform:uppercase;
        letter-spacing:.14em;
        margin-bottom:.2rem;
    }

    section[data-testid="stSidebar"] .stRadio label {
        padding:.48rem .58rem;
        border-radius:10px;
        transition:.16s ease;
        border:1px solid transparent;
        color:#afc0d8;
    }

    section[data-testid="stSidebar"] .stRadio label:hover {
        background:rgba(72,126,221,.10);
        border-color:rgba(89,145,231,.18);
    }

    section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background:linear-gradient(90deg, rgba(55,112,213,.35), rgba(50,96,180,.18));
        border-color:rgba(79,140,255,.62);
        box-shadow:
            inset 3px 0 0 #55a3ff,
            0 8px 18px rgba(0,0,0,.15);
        color:#eef5ff;
    }

    section[data-testid="stSidebar"] hr {
        border-color:rgba(113,151,205,.14);
        margin:.9rem 0;
    }

    .sidebar-model-label {
        color:#6e84a4;
        font-size:.63rem;
        text-transform:uppercase;
        letter-spacing:.15em;
        font-weight:750;
        margin-bottom:.2rem;
    }

    .sidebar-model-value {
        color:#ecf4ff;
        font-size:.82rem;
        font-weight:750;
        margin-bottom:.52rem;
    }

    .sidebar-footer {
        margin-top:1.2rem;
        padding:1rem .25rem .2rem;
        color:#7890b2;
        font-size:.67rem;
        letter-spacing:.03em;
        display:flex;
        align-items:center;
        gap:.48rem;
    }

    .sidebar-footer .plane {
        font-size:1rem;
        color:#8fb6ff;
    }

    /* HERO */
    .hero-shell {
        position:relative;
        overflow:hidden;
        border:0;
        border-bottom:1px solid rgba(89,139,213,.24);
        border-radius:0;
        background:
            radial-gradient(circle at 64% 46%, rgba(40,105,210,.15), transparent 24rem),
            linear-gradient(90deg, #081a33 0%, #081a33 39%, #07172d 72%, #061425 100%);
        min-height:218px;
        padding:1.45rem 1.7rem 1.18rem;
        margin:-.15rem -.55rem 1.15rem;
        box-shadow:0 14px 38px rgba(0,0,0,.18);
    }

    /* Soft overlays make the turbine feel like part of the same hero rather than a separate panel. */
    .hero-shell:before {
        content:"";
        position:absolute;
        inset:0;
        background:
            linear-gradient(90deg,
                rgba(8,26,51,.99) 0%,
                rgba(8,26,51,.97) 34%,
                rgba(8,26,51,.82) 45%,
                rgba(8,26,51,.46) 57%,
                rgba(8,26,51,.14) 68%,
                transparent 80%),
            radial-gradient(circle at 66% 48%, rgba(66,133,238,.15), transparent 22rem);
        pointer-events:none;
        z-index:3;
    }

    .hero-shell:after {
        content:"";
        position:absolute;
        inset:0;
        background:
            linear-gradient(180deg, rgba(6,18,35,.04), transparent 55%, rgba(4,14,28,.22)),
            linear-gradient(90deg, transparent 80%, rgba(4,14,28,.52) 100%);
        pointer-events:none;
        z-index:4;
    }

    .hero-copy-wrap {
        position:relative;
        z-index:5;
        width:58%;
        padding-top:.10rem;
    }

    .hero-kicker {
        color:#91a7c7;
        font-size:.68rem;
        letter-spacing:.25em;
        text-transform:uppercase;
        font-weight:800;
        margin-bottom:.58rem;
    }

    .hero-title {
        color:#f5f8ff;
        font-size:1.78rem;
        line-height:1.10;
        font-weight:850;
        letter-spacing:-.028em;
        margin-bottom:.58rem;
        text-shadow:0 2px 16px rgba(0,0,0,.2);
    }

    .hero-subtitle {
        color:#a8bdd8;
        font-size:.80rem;
        max-width:860px;
        line-height:1.55;
    }

    .value-row {
        display:flex;
        gap:1.7rem;
        margin-top:1.05rem;
        flex-wrap:wrap;
    }

    .value-pill {
        display:flex;
        align-items:center;
        gap:.58rem;
        font-size:.63rem;
        color:#aac0dc;
        letter-spacing:.12em;
        text-transform:uppercase;
        font-weight:750;
    }

    .value-dot {
        width:35px;
        height:35px;
        border-radius:50%;
        display:grid;
        place-items:center;
        background:rgba(42,105,209,.23);
        border:1px solid rgba(86,151,255,.44);
        color:#8ec5ff;
        font-size:.85rem;
        font-weight:850;
        box-shadow:0 0 20px rgba(70,133,255,.18), inset 0 0 12px rgba(75,137,255,.08);
    }

    .turbine {
        position:absolute;
        z-index:2;
        right:4.6rem;
        top:0;
        bottom:0;
        width:62%;
        height:100%;
        border-radius:0;
        background-image:
            url("data:image/webp;base64,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");
        background-size:cover;
        background-position:center right;
        background-repeat:no-repeat;
        opacity:.98;
        filter:saturate(1.05) contrast(1.04) brightness(.94);
        -webkit-mask-image:linear-gradient(90deg, transparent 0%, rgba(0,0,0,.06) 6%, rgba(0,0,0,.35) 18%, #000 38%, #000 100%);
        mask-image:linear-gradient(90deg, transparent 0%, rgba(0,0,0,.06) 6%, rgba(0,0,0,.35) 18%, #000 38%, #000 100%);
        pointer-events:none;
    }

    .turbine:before,
    .turbine:after {
        display:none;
    }

    .hero-right-copy {
        position:absolute;
        z-index:5;
        right:1.15rem;
        top:1.45rem;
        width:78px;
        color:#8096b6;
        font-size:.60rem;
        line-height:1.68;
        letter-spacing:.22em;
        text-transform:uppercase;
        font-weight:750;
    }

    .hero-right-copy:after {
        content:"";
        display:block;
        width:22px;
        height:2px;
        margin-top:.55rem;
        background:#5aa8ff;
        box-shadow:0 0 9px rgba(90,168,255,.55);
    }

    .hero-status {
        display:none;
    }


    /* FINAL HERO REFINEMENT
       Matches the supplied reference more closely while keeping the content truthful. */
    .hero-shell {
        min-height:190px;
        box-sizing:border-box;
        padding:1.05rem 1.55rem .82rem;
        margin:-.48rem -.62rem 1.02rem;
        border-top:1px solid rgba(91,145,218,.08);
        border-bottom:1px solid rgba(91,145,218,.28);
        background:
            radial-gradient(circle at 57% 48%, rgba(42,108,220,.16), transparent 24rem),
            linear-gradient(90deg, #08182f 0%, #08182f 42%, #07172c 70%, #061322 100%);
        box-shadow:none;
    }

    .hero-shell:before {
        background:
            linear-gradient(
                90deg,
                rgba(7,24,47,1) 0%,
                rgba(7,24,47,.98) 35%,
                rgba(7,24,47,.88) 45%,
                rgba(7,24,47,.56) 55%,
                rgba(7,24,47,.15) 68%,
                transparent 79%
            ),
            radial-gradient(circle at 63% 56%, rgba(60,134,255,.16), transparent 19rem);
        z-index:3;
    }

    .hero-shell:after {
        background:
            linear-gradient(180deg, rgba(4,13,25,.02), transparent 70%, rgba(4,13,25,.28)),
            linear-gradient(90deg, transparent 75%, rgba(4,13,25,.12) 88%, rgba(4,13,25,.55) 100%);
        z-index:4;
    }

    .hero-copy-wrap {
        width:62%;
        padding-top:.02rem;
        z-index:6;
    }

    .hero-kicker {
        font-size:.66rem;
        letter-spacing:.25em;
        margin-bottom:.42rem;
        color:#91a9ca;
    }

    .hero-title {
        font-size:1.72rem;
        line-height:1.08;
        margin-bottom:.48rem;
        letter-spacing:-.032em;
        white-space:nowrap;
    }

    .hero-title-accent {
        color:#7ab8ff;
        text-shadow:0 0 18px rgba(92,164,255,.18);
    }

    .hero-subtitle {
        max-width:790px;
        font-size:.77rem;
        line-height:1.45;
        color:#a9bad3;
    }

    .value-row {
        gap:1.55rem;
        margin-top:.80rem;
        align-items:center;
    }

    .value-pill {
        display:grid;
        grid-template-columns:34px auto;
        align-items:center;
        column-gap:.60rem;
        min-width:150px;
        padding:0;
        color:#a9c0de;
        font-size:.59rem;
        line-height:1.38;
        letter-spacing:.15em;
        text-transform:uppercase;
        font-weight:800;
    }

    .value-dot {
        width:32px;
        height:32px;
        display:grid;
        place-items:center;
        border-radius:50%;
        background:
            radial-gradient(circle at 35% 30%, rgba(110,184,255,.18), transparent 35%),
            rgba(37,91,178,.25);
        border:1px solid rgba(84,151,255,.50);
        color:#8dc4ff;
        box-shadow:
            0 0 18px rgba(69,133,255,.18),
            inset 0 0 14px rgba(84,151,255,.08);
    }

    .value-dot svg {
        width:17px;
        height:17px;
        stroke:#8fc5ff;
        fill:none;
        stroke-width:1.7;
        stroke-linecap:round;
        stroke-linejoin:round;
    }

    .feature-copy {
        display:block;
        white-space:normal;
    }

    .turbine {
        right:5.1rem;
        top:0;
        bottom:0;
        width:58%;
        height:100%;
        opacity:1;
        filter:saturate(1.14) contrast(1.08) brightness(1.03);
        background-size:cover;
        background-position:center right;
        -webkit-mask-image:linear-gradient(
            90deg,
            transparent 0%,
            rgba(0,0,0,.10) 5%,
            rgba(0,0,0,.42) 14%,
            rgba(0,0,0,.82) 25%,
            #000 36%,
            #000 100%
        );
        mask-image:linear-gradient(
            90deg,
            transparent 0%,
            rgba(0,0,0,.10) 5%,
            rgba(0,0,0,.42) 14%,
            rgba(0,0,0,.82) 25%,
            #000 36%,
            #000 100%
        );
    }

    .hero-operator {
        position:absolute;
        z-index:7;
        top:.88rem;
        right:1.20rem;
        display:flex;
        align-items:center;
        gap:.52rem;
        color:#b9c9dd;
    }

    .hero-operator-dot {
        width:7px;
        height:7px;
        border-radius:50%;
        background:#68d99a;
        box-shadow:0 0 10px rgba(104,217,154,.60);
    }

    .hero-operator-avatar {
        width:27px;
        height:27px;
        border-radius:50%;
        display:grid;
        place-items:center;
        border:1px solid rgba(102,160,238,.48);
        background:linear-gradient(145deg, rgba(57,111,196,.48), rgba(17,43,78,.92));
        color:#dcecff;
        font-size:.58rem;
        font-weight:850;
        box-shadow:0 0 14px rgba(54,121,230,.18);
    }

    .hero-operator-copy {
        line-height:1.15;
        min-width:72px;
    }

    .hero-operator-copy strong {
        display:block;
        color:#dfeaff;
        font-size:.62rem;
        font-weight:800;
    }

    .hero-operator-copy span {
        display:block;
        margin-top:.16rem;
        color:#7e94b3;
        font-size:.53rem;
    }

    .hero-right-copy {
        right:1.25rem;
        top:4.25rem;
        width:86px;
        font-size:.57rem;
        line-height:1.75;
        color:#8298b8;
    }

    .hero-right-copy:after {
        width:24px;
        height:2px;
        margin-top:.48rem;
    }



    /* HERO MATCH REFINEMENT
       Tuned to the supplied reference composition. */
    .hero-shell {
        min-height:252px;
        height:252px;
        box-sizing:border-box;
        position:relative;
        overflow:hidden;
        padding:1.28rem 1.72rem 1.02rem;
        margin:-.72rem -.62rem 1.05rem;
        border:0;
        border-bottom:1px solid rgba(83,137,211,.24);
        border-radius:0;
        background:
            radial-gradient(circle at 56% 56%, rgba(42,103,196,.14), transparent 25rem),
            linear-gradient(90deg, #07162b 0%, #07162b 48%, #071528 72%, #061320 100%);
        box-shadow:none;
    }

    .hero-shell:before {
        content:"";
        position:absolute;
        inset:0;
        z-index:3;
        pointer-events:none;
        background:
            linear-gradient(
                90deg,
                rgba(7,22,43,1) 0%,
                rgba(7,22,43,.99) 34%,
                rgba(7,22,43,.96) 42%,
                rgba(7,22,43,.74) 49%,
                rgba(7,22,43,.31) 58%,
                rgba(7,22,43,.06) 67%,
                transparent 75%
            ),
            linear-gradient(
                180deg,
                rgba(3,10,20,.04) 0%,
                transparent 72%,
                rgba(3,10,20,.22) 100%
            );
    }

    .hero-shell:after {
        content:"";
        position:absolute;
        inset:0 0 0 auto;
        width:13.2%;
        z-index:5;
        pointer-events:none;
        border-left:1px solid rgba(106,151,212,.12);
        background:
            linear-gradient(180deg, rgba(5,16,31,.18), rgba(5,16,31,.48)),
            linear-gradient(90deg, transparent, rgba(4,13,25,.48));
    }

    .hero-copy-wrap {
        position:relative;
        z-index:7;
        width:58%;
        padding-top:.02rem;
    }

    .hero-kicker {
        margin:0 0 .52rem;
        color:#91a7c7;
        font-size:.70rem;
        line-height:1;
        font-weight:800;
        letter-spacing:.28em;
        text-transform:uppercase;
    }

    .hero-title {
        margin:0 0 .55rem;
        color:#f4f7fb;
        font-size:2.05rem;
        line-height:1.04;
        font-weight:850;
        letter-spacing:-.035em;
        white-space:nowrap;
    }

    .hero-title-accent {
        color:#7db9ff;
        text-shadow:0 0 18px rgba(77,145,255,.20);
    }

    .hero-subtitle {
        max-width:820px;
        color:#b1bfd4;
        font-size:.88rem;
        line-height:1.48;
    }

    .value-row {
        display:flex;
        align-items:center;
        gap:2.55rem;
        margin-top:1.02rem;
        flex-wrap:nowrap;
    }

    .value-pill {
        display:grid;
        grid-template-columns:38px auto;
        align-items:center;
        column-gap:.72rem;
        min-width:0;
        padding:0;
        border:0;
        border-radius:0;
        background:transparent;
        color:#a9bedb;
        font-size:.66rem;
        line-height:1.42;
        font-weight:850;
        letter-spacing:.15em;
        text-transform:uppercase;
    }

    .value-dot {
        width:38px;
        height:38px;
        display:grid;
        place-items:center;
        border-radius:50%;
        background:
            radial-gradient(circle at 35% 30%, rgba(123,192,255,.18), transparent 38%),
            linear-gradient(145deg, rgba(35,87,169,.42), rgba(16,42,82,.72));
        border:1px solid rgba(86,151,255,.50);
        color:#8fc7ff;
        box-shadow:
            0 0 18px rgba(65,127,235,.18),
            inset 0 0 15px rgba(89,151,255,.08);
    }

    .value-dot svg {
        width:20px;
        height:20px;
        stroke:#91c6ff;
        fill:none;
        stroke-width:1.8;
        stroke-linecap:round;
        stroke-linejoin:round;
    }

    .feature-copy {
        display:block;
        white-space:nowrap;
    }

    .turbine {
        position:absolute;
        z-index:2;
        left:51.8%;
        right:auto;
        top:0;
        bottom:0;
        width:36.5%;
        height:100%;
        opacity:1;
        border-radius:0;
        background-repeat:no-repeat;
        background-size:cover;
        background-position:center center;
        filter:saturate(1.08) contrast(1.05) brightness(1.06);
        -webkit-mask-image:linear-gradient(
            90deg,
            transparent 0%,
            rgba(0,0,0,.16) 6%,
            rgba(0,0,0,.52) 15%,
            rgba(0,0,0,.92) 27%,
            #000 37%,
            #000 100%
        );
        mask-image:linear-gradient(
            90deg,
            transparent 0%,
            rgba(0,0,0,.16) 6%,
            rgba(0,0,0,.52) 15%,
            rgba(0,0,0,.92) 27%,
            #000 37%,
            #000 100%
        );
    }

    .hero-operator {
        position:absolute;
        z-index:8;
        top:1.02rem;
        right:1.15rem;
        width:10.6%;
        display:grid;
        grid-template-columns:28px 1fr;
        align-items:center;
        column-gap:.58rem;
        color:#b9c8dc;
    }

    .hero-operator-dot {
        display:none;
    }

    .hero-operator-avatar {
        width:28px;
        height:28px;
        border-radius:50%;
        display:grid;
        place-items:center;
        border:1px solid rgba(103,158,231,.48);
        background:linear-gradient(145deg, rgba(57,112,198,.50), rgba(16,42,78,.92));
        color:#e0edff;
        font-size:.58rem;
        font-weight:850;
        box-shadow:0 0 15px rgba(55,121,230,.18);
    }

    .hero-operator-copy {
        min-width:0;
        line-height:1.15;
    }

    .hero-operator-copy strong {
        display:block;
        color:#e6eefb;
        font-size:.64rem;
        font-weight:800;
        white-space:nowrap;
    }

    .hero-operator-copy span {
        display:block;
        margin-top:.18rem;
        color:#7f94b2;
        font-size:.54rem;
        white-space:nowrap;
    }

    .hero-right-copy {
        position:absolute;
        z-index:8;
        right:1.42rem;
        top:5.7rem;
        width:8.7%;
        color:#8fa5c4;
        font-size:.66rem;
        line-height:1.65;
        font-weight:800;
        letter-spacing:.24em;
        text-transform:uppercase;
    }

    .hero-right-copy:after {
        content:"";
        display:block;
        width:30px;
        height:2px;
        margin-top:.58rem;
        background:#76adf4;
        box-shadow:0 0 8px rgba(118,173,244,.25);
    }

    @media (max-width:1250px) {
        .hero-shell {
            height:auto;
            min-height:230px;
        }
        .hero-copy-wrap { width:66%; }
        .hero-title {
            white-space:normal;
            font-size:1.72rem;
        }
        .value-row { gap:1.35rem; }
        .value-pill { font-size:.58rem; }
        .turbine {
            left:55%;
            width:40%;
            opacity:.88;
        }
        .hero-operator,
        .hero-right-copy { display:none; }
    }

    @media (max-width:900px) {
        .hero-shell {
            min-height:235px;
            padding:1.05rem 1rem .95rem;
        }
        .hero-copy-wrap { width:100%; }
        .hero-title { font-size:1.50rem; }
        .hero-subtitle { max-width:100%; font-size:.76rem; }
        .turbine {
            left:47%;
            width:60%;
            opacity:.13;
        }
        .value-row {
            gap:.85rem;
            flex-wrap:wrap;
        }
        .value-pill {
            grid-template-columns:31px auto;
            font-size:.53rem;
        }
        .value-dot {
            width:31px;
            height:31px;
        }
    }


    /* PAGE TITLES */
    .overview-head {
        display:flex;
        align-items:flex-start;
        justify-content:space-between;
        gap:1.2rem;
        margin:.10rem 0 .72rem;
    }

    .overview-status {
        display:flex;
        align-items:center;
        gap:1rem;
        color:#7f95b4;
        font-size:.68rem;
        white-space:nowrap;
        padding-top:.2rem;
    }

    .overview-status strong {
        color:#76d9a2;
        font-weight:750;
    }

    .section-title {
        display:flex;
        align-items:center;
        gap:.5rem;
        color:#f3f7ff;
        font-size:1.38rem;
        font-weight:850;
        letter-spacing:-.025em;
        margin:.12rem 0 .12rem;
        line-height:1.16;
    }

    .section-subtitle {
        color:#8196b4;
        font-size:.74rem;
        margin-bottom:.52rem;
    }

    .fleet-summary-heading {
        display:flex;
        align-items:center;
        gap:.5rem;
        color:#f3f7ff;
        font-size:1.05rem;
        font-weight:850;
        line-height:1.25;
        margin:1.28rem 0 .62rem;
        padding-top:.10rem;
    }

    .after-fleet-space {
        height:.72rem;
    }

    /* KPI CARDS */
    .kpi-card {
        min-height:92px;
        border-radius:13px;
        border:1px solid var(--icon-border);
        background:
            radial-gradient(circle at 100% 0%, var(--glow-soft), transparent 7rem),
            linear-gradient(145deg, rgba(12,30,54,.98), rgba(8,22,41,.96));
        padding:.62rem .72rem;
        box-shadow:
            0 9px 28px rgba(0,0,0,.20),
            inset 0 1px 0 rgba(255,255,255,.018);
        position:relative;
        overflow:hidden;
    }

    .kpi-card:before {
        display:none;
    }

    .kpi-head {
        display:flex;
        align-items:center;
        gap:.58rem;
        margin-bottom:.28rem;
    }

    .kpi-icon {
        width:32px;
        height:32px;
        border-radius:9px;
        display:grid;
        place-items:center;
        font-size:.69rem;
        font-weight:850;
        color:white;
        background:var(--icon-bg);
        border:1px solid var(--icon-border);
        box-shadow:0 0 16px var(--icon-glow);
        flex:0 0 auto;
    }

    .kpi-label {
        color:#9fb1ca;
        font-size:.67rem;
        line-height:1.2;
    }

    .kpi-value {
        font-size:1.2rem;
        font-weight:850;
        letter-spacing:-.02em;
        margin:.03rem 0 0 2.57rem;
        color:#f3f7ff;
    }

    .kpi-note {
        color:#6f86a7;
        font-size:.62rem;
        margin:.18rem 0 0 2.57rem;
        line-height:1.2;
    }

    /* FLEET CARDS */
    .fleet-card {
        min-height:88px;
        border-radius:13px;
        border:1px solid rgba(96,145,211,.24);
        background:
            radial-gradient(circle at 94% 24%, color-mix(in srgb, var(--risk) 12%, transparent), transparent 5rem),
            linear-gradient(145deg, rgba(12,30,54,.98), rgba(8,22,41,.96));
        padding:.72rem .78rem .62rem 3.4rem;
        position:relative;
        overflow:hidden;
        box-shadow:0 8px 24px rgba(0,0,0,.16);
    }

    .fleet-card .fleet-icon {
        position:absolute;
        left:.75rem;
        top:.85rem;
        width:31px;
        height:31px;
        border-radius:50%;
        display:grid;
        place-items:center;
        color:#f6f9ff;
        font-size:.85rem;
        font-weight:850;
        background:color-mix(in srgb, var(--risk) 25%, #10213a);
        border:1px solid color-mix(in srgb, var(--risk) 60%, transparent);
        box-shadow:0 0 15px color-mix(in srgb, var(--risk) 35%, transparent);
    }

    .fleet-card .ghost {
        position:absolute;
        right:.85rem;
        top:1.02rem;
        width:46px;
        height:46px;
        border:1px solid color-mix(in srgb, var(--risk) 18%, transparent);
        border-radius:50%;
        opacity:.34;
        background:
            radial-gradient(circle at center, transparent 0 20%, color-mix(in srgb, var(--risk) 18%, transparent) 21% 23%, transparent 24%),
            repeating-conic-gradient(from 0deg, color-mix(in srgb, var(--risk) 22%, transparent) 0deg 5deg, transparent 5deg 15deg);
    }

    .fleet-card .ghost:before,
    .fleet-card .ghost:after {
        display:none;
    }

    .fleet-label {
        color:#9eafc8;
        font-size:.66rem;
    }

    .fleet-value {
        font-size:1.15rem;
        font-weight:850;
        margin-top:.12rem;
        color:#f5f8ff;
    }

    .fleet-note {
        color:#7288a8;
        font-size:.61rem;
        margin-top:.10rem;
    }

    /* INSIGHT */
    .insight-banner {
        display:grid;
        grid-template-columns:42px 1fr auto;
        align-items:center;
        gap:.9rem;
        border:1px solid rgba(79,140,255,.56);
        background:
            radial-gradient(circle at 0 50%, rgba(75,137,255,.16), transparent 17rem),
            linear-gradient(90deg, rgba(25,63,124,.42), rgba(13,39,83,.36));
        border-radius:10px;
        padding:.64rem .78rem;
        box-shadow:
            inset 2px 0 0 #58a0ff,
            0 8px 26px rgba(0,0,0,.15);
        margin:.34rem 0 1rem;
    }

    .insight-icon {
        width:38px;
        height:38px;
        border-radius:50%;
        display:grid;
        place-items:center;
        background:rgba(79,140,255,.19);
        border:1px solid rgba(96,158,255,.37);
        color:#9bc6ff;
        font-size:1rem;
        font-weight:850;
        box-shadow:0 0 16px rgba(79,140,255,.18);
    }

    .insight-title {
        font-weight:850;
        margin-bottom:.08rem;
        color:#f2f7ff;
    }

    .insight-copy {
        color:#99afcc;
        font-size:.70rem;
        line-height:1.42;
    }

    .insight-action {
        padding-left:1rem;
        border-left:1px solid rgba(103,151,219,.25);
        color:#8fb5e9;
        font-size:.60rem;
        line-height:1.55;
        letter-spacing:.17em;
        text-transform:uppercase;
        font-weight:750;
        white-space:nowrap;
    }

    /* PANELS */
    .panel-title {
        font-size:1.03rem;
        font-weight:850;
        margin-top:.06rem;
        margin-bottom:.16rem;
        line-height:1.2;
        color:#f2f6fc;
    }

    .panel-subtitle {
        color:#7288a8;
        font-size:.65rem;
        margin-bottom:.35rem;
    }

    .status-card {
        border:1px solid rgba(102,145,207,.24);
        border-radius:11px;
        padding:.85rem .95rem;
        background:
            radial-gradient(circle at 98% 20%, rgba(79,140,255,.08), transparent 8rem),
            linear-gradient(145deg, rgba(13,31,55,.96), rgba(8,22,40,.94));
        margin-bottom:.55rem;
        box-shadow:0 8px 22px rgba(0,0,0,.14);
    }

    .small-muted {
        color:#7f94b2;
        font-size:.68rem;
    }

    /* INDIVIDUAL ENGINE SPACING */
    .engine-gap-xs { height:.34rem; }
    .engine-gap-sm { height:.68rem; }
    .engine-gap-md { height:1rem; }
    .engine-card-gap { height:.38rem; }

    .engine-section-heading {
        margin-top:.12rem;
        margin-bottom:.48rem;
    }

    .engine-section-heading h4 {
        margin:0 !important;
        font-size:1.05rem;
    }

    /* STREAMLIT COMPONENTS */
    div[data-testid="stMetric"] {
        background:
            radial-gradient(circle at 100% 0%, rgba(79,140,255,.08), transparent 6rem),
            linear-gradient(145deg, rgba(13,31,55,.98), rgba(8,22,40,.96));
        border:1px solid rgba(102,145,207,.24);
        border-radius:11px;
        padding:.78rem .9rem;
        min-height:76px;
        box-shadow:0 7px 22px rgba(0,0,0,.14);
    }

    div[data-testid="stMetric"] label {
        color:#93a8c4 !important;
        font-size:.67rem !important;
    }

    div[data-testid="stMetricValue"] {
        font-size:1.25rem !important;
        font-weight:850 !important;
        color:#f2f7ff !important;
    }

    div[data-testid="stDataFrame"] {
        border:1px solid rgba(96,137,195,.20);
        border-radius:10px;
        overflow:hidden;
        background:#071321;
        box-shadow:0 8px 24px rgba(0,0,0,.16);
    }

    div[data-testid="stDataFrame"] [role="columnheader"] {
        background:#0c1b30 !important;
    }


    div[data-testid="stVerticalBlockBorderWrapper"] {
        border:1px solid rgba(96,145,211,.22) !important;
        border-radius:13px !important;
        background:linear-gradient(145deg, rgba(8,22,41,.72), rgba(6,17,31,.78)) !important;
        box-shadow:0 10px 28px rgba(0,0,0,.14);
        overflow:hidden;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding:.68rem .78rem .42rem !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius:9px;
        border:1px solid rgba(91,138,207,.30);
        background:linear-gradient(180deg, rgba(19,45,79,.96), rgba(10,29,53,.96));
        color:#dbeafe;
        min-height:2.2rem;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color:rgba(79,140,255,.70);
        background:rgba(31,72,126,.72);
        color:white;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stMultiSelect"] > div > div {
        background:#0b1a2e !important;
        border-color:rgba(96,145,211,.28) !important;
        border-radius:9px !important;
    }

    div[data-testid="stExpander"] {
        border:1px solid rgba(96,145,211,.20);
        border-radius:9px;
        background:rgba(8,20,36,.58);
    }

    div[data-testid="stAlert"] {
        border-radius:9px;
    }

    div[data-testid="stVerticalBlock"] {
        gap:.50rem;
    }

    div[data-testid="stHorizontalBlock"] {
        gap:.62rem;
    }

    .stSelectbox,
    .stMultiSelect,
    .stSlider {
        margin-bottom:-.12rem;
    }

    h1, h2, h3, h4 {
        color:#f2f6fc;
        margin-top:.20rem !important;
        margin-bottom:.28rem !important;
    }

    p {
        margin-bottom:.32rem;
    }

    .footer {
        margin-top:.9rem;
        border-top:1px solid rgba(120,150,190,.12);
        padding:.65rem .1rem .2rem;
        color:#5f7697;
        font-size:.62rem;
        letter-spacing:.04em;
        display:flex;
        justify-content:space-between;
        gap:1rem;
        flex-wrap:wrap;
    }

    .footer-right {
        color:#6d84a5;
    }

    @media (max-width:1250px) {
        .hero-copy-wrap { width:68%; }
        .hero-title { white-space:normal; }
        .turbine { right:0; width:46%; opacity:.86; }
        .hero-right-copy,
        .hero-operator { display:none; }
        .overview-status { display:none; }
    }

    @media (max-width:900px) {
        .block-container { padding-top:1.2rem; }
        .hero-shell { margin-top:-.5rem; min-height:205px; padding:1rem 1rem .9rem; }
        .hero-copy-wrap { width:100%; }
        .hero-title { white-space:normal; font-size:1.45rem; }
        .hero-subtitle { max-width:100%; }
        .turbine { opacity:.12; right:-10%; width:62%; }
        .value-row { gap:.85rem; }
        .value-pill { min-width:auto; grid-template-columns:30px auto; font-size:.54rem; }
        .value-dot { width:29px; height:29px; }
        .insight-banner { grid-template-columns:42px 1fr; }
        .insight-action { display:none; }
    }

    /* FINAL SEAMLESS HERO BLEND
       The turbine is rendered as part of the hero background so there is no
       visible rectangular image or right-hand panel boundary. */
    .hero-shell {
        min-height:252px !important;
        height:252px !important;
        position:relative !important;
        overflow:hidden !important;
        box-sizing:border-box !important;
        padding:1.28rem 1.72rem 1.02rem !important;
        margin:-.72rem -.62rem 1.05rem !important;
        border:0 !important;
        border-bottom:1px solid rgba(83,137,211,.22) !important;
        border-radius:0 !important;
        box-shadow:none !important;

        background:
            linear-gradient(
                90deg,
                #07162b 0%,
                #07162b 37%,
                rgba(7,22,43,.98) 44%,
                rgba(7,22,43,.88) 50%,
                rgba(7,22,43,.56) 58%,
                rgba(7,22,43,.22) 67%,
                rgba(6,19,36,.10) 78%,
                rgba(5,15,29,.32) 91%,
                rgba(5,15,29,.62) 100%
            ),
            radial-gradient(
                circle at 61% 53%,
                rgba(50,115,220,.17) 0%,
                rgba(31,72,137,.08) 19%,
                transparent 40%
            ),
            url("data:image/webp;base64,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") right center / 53% 100% no-repeat,
            linear-gradient(90deg, #07162b 0%, #07162b 55%, #061526 100%) !important;
    }

    .hero-shell:before,
    .hero-shell:after {
        display:none !important;
        content:none !important;
    }

    .turbine {
        display:none !important;
    }

    .hero-copy-wrap {
        position:relative !important;
        z-index:4 !important;
        width:58% !important;
    }

    .hero-operator {
        position:absolute !important;
        z-index:6 !important;
        top:1.02rem !important;
        right:1.20rem !important;
        width:auto !important;
        max-width:155px !important;
        padding:0 !important;
        margin:0 !important;
        border:0 !important;
        background:transparent !important;
        box-shadow:none !important;
    }

    .hero-right-copy {
        position:absolute !important;
        z-index:6 !important;
        right:1.40rem !important;
        top:5.70rem !important;
        width:112px !important;
        padding:0 !important;
        margin:0 !important;
        border:0 !important;
        background:transparent !important;
        box-shadow:none !important;
    }

    .hero-right-copy:before {
        content:"" !important;
        position:absolute !important;
        left:-1.15rem !important;
        top:-1.45rem !important;
        width:1px !important;
        height:8.8rem !important;
        background:linear-gradient(
            180deg,
            transparent,
            rgba(113,157,218,.14) 24%,
            rgba(113,157,218,.14) 76%,
            transparent
        ) !important;
    }

    .hero-right-copy:after {
        display:block !important;
        content:"" !important;
        width:30px !important;
        height:2px !important;
        margin-top:.58rem !important;
        background:#76adf4 !important;
        box-shadow:0 0 8px rgba(118,173,244,.25) !important;
    }

    @media (max-width:1250px) {
        .hero-shell {
            min-height:230px !important;
            height:auto !important;
            background:
                linear-gradient(
                    90deg,
                    #07162b 0%,
                    #07162b 43%,
                    rgba(7,22,43,.91) 53%,
                    rgba(7,22,43,.42) 67%,
                    rgba(5,15,29,.42) 100%
                ),
                url("data:image/webp;base64,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") right center / 52% 100% no-repeat,
                #07162b !important;
        }

        .hero-copy-wrap { width:67% !important; }
        .hero-title { white-space:normal !important; }
        .hero-operator,
        .hero-right-copy { display:none !important; }
    }

    @media (max-width:900px) {
        .hero-shell {
            min-height:235px !important;
            height:auto !important;
            padding:1.05rem 1rem .95rem !important;
            background:
                linear-gradient(90deg, rgba(7,22,43,.97), rgba(7,22,43,.78)),
                url("data:image/webp;base64,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") right center / 70% 100% no-repeat,
                #07162b !important;
        }

        .hero-copy-wrap { width:100% !important; }
    }


    /* COMPACT DESKTOP DENSITY
       Tuned for a 1080p-class desktop so the primary content on each page
       fits without routine vertical scrolling. */
    @media (min-width:1100px) {
        html {
            font-size:9.25px !important;
        }

        .block-container {
            max-width:2600px !important;
            padding-top:.36rem !important;
            padding-bottom:.42rem !important;
            padding-left:.58rem !important;
            padding-right:.58rem !important;
        }

        section[data-testid="stSidebar"] {
            min-width:218px !important;
            max-width:218px !important;
        }

        section[data-testid="stSidebar"] > div {
            padding-top:.48rem !important;
            padding-left:.46rem !important;
            padding-right:.46rem !important;
        }

        .sidebar-brand {
            margin-bottom:.60rem !important;
        }

        .brand-mark {
            width:32px !important;
            height:32px !important;
        }

        .brand-title {
            font-size:.92rem !important;
        }

        .brand-subtitle {
            font-size:.55rem !important;
        }

        section[data-testid="stSidebar"] .stRadio label {
            padding:.26rem .42rem !important;
            min-height:1.88rem !important;
        }

        /* Hero */
        .hero-shell {
            min-height:180px !important;
            height:180px !important;
            padding:.82rem 1.22rem .66rem !important;
            margin:-.38rem -.30rem .56rem !important;
        }

        .hero-copy-wrap {
            width:58% !important;
        }

        .hero-kicker {
            margin-bottom:.30rem !important;
            font-size:.58rem !important;
            letter-spacing:.24em !important;
        }

        .hero-title {
            font-size:1.55rem !important;
            margin-bottom:.30rem !important;
            line-height:1.04 !important;
        }

        .hero-subtitle {
            max-width:760px !important;
            font-size:.67rem !important;
            line-height:1.34 !important;
        }

        .value-row {
            margin-top:.56rem !important;
            gap:1.55rem !important;
        }

        .value-dot {
            width:29px !important;
            height:29px !important;
        }

        .value-dot svg {
            width:15px !important;
            height:15px !important;
        }

        .value-pill {
            grid-template-columns:29px auto !important;
            column-gap:.48rem !important;
            font-size:.51rem !important;
            line-height:1.32 !important;
            letter-spacing:.13em !important;
        }

        .hero-operator {
            top:.58rem !important;
            right:.80rem !important;
            transform:scale(.88);
            transform-origin:top right;
        }

        .hero-right-copy {
            top:3.90rem !important;
            right:.94rem !important;
            width:94px !important;
            font-size:.50rem !important;
            line-height:1.55 !important;
        }

        /* Page headings */
        .section-title {
            margin-top:0 !important;
            margin-bottom:.02rem !important;
            font-size:1.02rem !important;
            line-height:1.14 !important;
        }

        .section-subtitle {
            margin-bottom:.16rem !important;
            font-size:.65rem !important;
            line-height:1.20 !important;
        }

        .panel-title {
            margin-top:0 !important;
            margin-bottom:.10rem !important;
            font-size:.94rem !important;
            line-height:1.18 !important;
        }

        .panel-subtitle {
            margin-bottom:.24rem !important;
            font-size:.60rem !important;
            line-height:1.18 !important;
        }

        h4 {
            margin-top:.06rem !important;
            margin-bottom:.18rem !important;
            font-size:.98rem !important;
        }

        /* KPI cards: icon left, text stack right, no overlap */
        .kpi-card {
            min-height:66px !important;
            padding:.43rem .52rem !important;
            display:grid !important;
            grid-template-columns:27px minmax(0, 1fr) !important;
            grid-template-rows:auto auto auto !important;
            column-gap:.48rem !important;
            row-gap:.03rem !important;
            align-content:center !important;
        }

        .kpi-head {
            display:contents !important;
        }

        .kpi-icon {
            grid-column:1 !important;
            grid-row:1 / 4 !important;
            align-self:start !important;
            width:27px !important;
            height:27px !important;
            margin:0 !important;
        }

        .kpi-label {
            grid-column:2 !important;
            grid-row:1 !important;
            margin:0 !important;
            font-size:.59rem !important;
            line-height:1.14 !important;
            white-space:normal !important;
        }

        .kpi-value {
            grid-column:2 !important;
            grid-row:2 !important;
            margin:.06rem 0 0 !important;
            font-size:1.02rem !important;
            line-height:1.05 !important;
        }

        .kpi-note {
            grid-column:2 !important;
            grid-row:3 !important;
            margin:.10rem 0 0 !important;
            font-size:.51rem !important;
            line-height:1.12 !important;
        }

        /* Fleet cards */
        .fleet-card {
            min-height:62px !important;
            padding:.44rem 2.85rem .42rem 2.80rem !important;
        }

        .fleet-card .fleet-icon {
            left:.56rem !important;
            top:.56rem !important;
            width:27px !important;
            height:27px !important;
            font-size:.69rem !important;
        }

        .fleet-card .ghost {
            right:.56rem !important;
            top:.62rem !important;
            width:34px !important;
            height:34px !important;
        }

        .fleet-label {
            font-size:.58rem !important;
            line-height:1.10 !important;
            margin:0 !important;
        }

        .fleet-value {
            margin-top:.05rem !important;
            font-size:1.00rem !important;
            line-height:1.02 !important;
        }

        .fleet-note {
            margin-top:.08rem !important;
            font-size:.50rem !important;
            line-height:1.10 !important;
        }

        .fleet-summary-title {
            margin-top:.26rem !important;
            margin-bottom:.20rem !important;
        }

        /* Insight and lower panels */
        .insight-banner {
            padding:.38rem .54rem !important;
            margin:.22rem 0 .48rem !important;
            min-height:44px !important;
        }

        .insight-icon {
            width:30px !important;
            height:30px !important;
        }

        .insight-title {
            font-size:.88rem !important;
            line-height:1.10 !important;
        }

        .insight-copy {
            font-size:.56rem !important;
            line-height:1.22 !important;
        }

        .panel-shell {
            padding:.44rem .52rem .36rem !important;
        }

        /* Standard Streamlit metrics */
        div[data-testid="stMetric"] {
            min-height:54px !important;
            padding:.52rem .64rem !important;
        }

        div[data-testid="stMetric"] label {
            font-size:.58rem !important;
            line-height:1.10 !important;
        }

        div[data-testid="stMetricValue"] {
            font-size:1.10rem !important;
            line-height:1.04 !important;
        }

        div[data-testid="stDataFrame"] {
            font-size:.68rem !important;
        }

        /* Engine page spacing */
        .engine-gap-xs { height:.12rem !important; }
        .engine-gap-sm { height:.28rem !important; }
        .engine-gap-md { height:.42rem !important; }
        .engine-card-gap { height:.16rem !important; }

        .engine-section-heading {
            margin-top:.02rem !important;
            margin-bottom:.20rem !important;
        }

        .status-card {
            padding:.58rem .68rem !important;
            margin-bottom:.20rem !important;
        }

        .small-muted {
            font-size:.58rem !important;
            line-height:1.15 !important;
        }

        /* Inputs, alerts, expanders */
        div[data-baseweb="select"] > div {
            min-height:2.05rem !important;
        }

        div[data-testid="stAlert"] {
            padding:.42rem .56rem !important;
            min-height:0 !important;
        }

        div[data-testid="stExpander"] summary {
            min-height:2rem !important;
            padding:.28rem .52rem !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            min-height:1.95rem !important;
            padding:.22rem .56rem !important;
        }

        div[data-testid="stVerticalBlock"] {
            gap:.24rem !important;
        }

        div[data-testid="stHorizontalBlock"] {
            gap:.38rem !important;
        }

        .stSelectbox,
        .stMultiSelect,
        .stSlider {
            margin-bottom:-.18rem !important;
        }

        p {
            margin-bottom:.18rem !important;
            line-height:1.22 !important;
        }

        .footer {
            margin-top:.28rem !important;
            padding-top:.24rem !important;
            font-size:.52rem !important;
        }
    }


    /* PRESENTATION SPACING PASS
       Restores breathing room after the compact layout so headings, labels,
       cards and charts do not visually collide. */
    @media (min-width:1100px) {
        /* Page headers */
        .section-title {
            margin-top:.08rem !important;
            margin-bottom:.14rem !important;
            line-height:1.18 !important;
        }

        .section-subtitle {
            margin-top:0 !important;
            margin-bottom:.62rem !important;
            line-height:1.28 !important;
        }

        /* Streamlit widget labels need their own line and clearance */
        [data-testid="stWidgetLabel"] {
            margin-bottom:.22rem !important;
            line-height:1.22 !important;
        }

        [data-testid="stWidgetLabel"] p {
            margin:0 !important;
            line-height:1.22 !important;
        }

        .stSelectbox,
        .stMultiSelect,
        .stSlider {
            margin-top:0 !important;
            margin-bottom:.16rem !important;
        }

        div[data-baseweb="select"] {
            margin-top:.02rem !important;
        }

        /* Section headings above groups of cards */
        h4 {
            margin-top:.32rem !important;
            margin-bottom:.34rem !important;
            line-height:1.18 !important;
        }

        .fleet-summary-title {
            margin-top:.48rem !important;
            margin-bottom:.34rem !important;
            line-height:1.18 !important;
        }

        .panel-title {
            margin-top:.16rem !important;
            margin-bottom:.16rem !important;
            line-height:1.20 !important;
        }

        .panel-subtitle {
            margin-top:0 !important;
            margin-bottom:.36rem !important;
            line-height:1.22 !important;
        }

        /* Keep cards compact, but give text enough internal space */
        .kpi-card {
            min-height:70px !important;
            padding:.50rem .58rem !important;
            row-gap:.07rem !important;
        }

        .kpi-label {
            line-height:1.18 !important;
        }

        .kpi-value {
            margin-top:.10rem !important;
            line-height:1.08 !important;
        }

        .kpi-note {
            margin-top:.13rem !important;
            line-height:1.16 !important;
        }

        .fleet-card {
            min-height:66px !important;
            padding:.49rem 2.85rem .48rem 2.80rem !important;
        }

        .fleet-label {
            line-height:1.16 !important;
        }

        .fleet-value {
            margin-top:.08rem !important;
            line-height:1.08 !important;
        }

        .fleet-note {
            margin-top:.11rem !important;
            line-height:1.16 !important;
        }

        /* Make horizontal card rows visibly separate from nearby headings */
        div[data-testid="stHorizontalBlock"] {
            gap:.44rem !important;
            margin-bottom:.12rem !important;
        }

        div[data-testid="stVerticalBlock"] {
            gap:.34rem !important;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            min-height:58px !important;
            padding:.58rem .68rem !important;
        }

        div[data-testid="stMetric"] label {
            line-height:1.18 !important;
            margin-bottom:.12rem !important;
        }

        div[data-testid="stMetricValue"] {
            line-height:1.08 !important;
        }

        /* Individual Engine page */
        .status-card {
            padding:.66rem .74rem !important;
            margin-top:.12rem !important;
            margin-bottom:.36rem !important;
        }

        .engine-gap-xs { height:.22rem !important; }
        .engine-gap-sm { height:.42rem !important; }
        .engine-gap-md { height:.58rem !important; }
        .engine-card-gap { height:.28rem !important; }

        .engine-section-heading {
            margin-top:.22rem !important;
            margin-bottom:.34rem !important;
        }

        /* Alerts and expanders need visible separation */
        div[data-testid="stAlert"] {
            margin-top:.18rem !important;
            margin-bottom:.24rem !important;
            padding:.50rem .62rem !important;
        }

        div[data-testid="stExpander"] {
            margin-top:.18rem !important;
            margin-bottom:.18rem !important;
        }

        div[data-testid="stExpander"] summary {
            min-height:2.12rem !important;
            padding:.34rem .58rem !important;
        }

        /* Tables */
        div[data-testid="stDataFrame"] {
            margin-top:.14rem !important;
            margin-bottom:.18rem !important;
        }

        /* Footer should not crowd preceding content */
        .footer {
            margin-top:.48rem !important;
            padding-top:.34rem !important;
        }
    }


    /* OVERLAP FIX
       Structural spacing for compact desktop layouts. */
    @media (min-width:1100px) {
        .page-head-gap {
            height:.58rem !important;
            min-height:.58rem !important;
        }

        .fleet-pre-gap {
            height:.42rem !important;
            min-height:.42rem !important;
        }

        .scenario-gap {
            height:.26rem !important;
            min-height:.26rem !important;
        }

        /* Fleet cards use an explicit three-column grid:
           icon | copy | decoration. */
        .fleet-card {
            display:grid !important;
            grid-template-columns:31px minmax(0, 1fr) 42px !important;
            grid-template-rows:1fr !important;
            align-items:center !important;
            column-gap:.62rem !important;
            min-height:68px !important;
            padding:.52rem .58rem !important;
            box-sizing:border-box !important;
        }

        .fleet-card .fleet-icon {
            position:static !important;
            grid-column:1 !important;
            grid-row:1 !important;
            width:29px !important;
            height:29px !important;
            align-self:center !important;
            justify-self:center !important;
            margin:0 !important;
        }

        .fleet-card .fleet-copy {
            grid-column:2 !important;
            grid-row:1 !important;
            min-width:0 !important;
            display:flex !important;
            flex-direction:column !important;
            justify-content:center !important;
            gap:.08rem !important;
            padding:0 !important;
            margin:0 !important;
        }

        .fleet-card .fleet-label,
        .fleet-card .fleet-value,
        .fleet-card .fleet-note {
            position:static !important;
            margin:0 !important;
            padding:0 !important;
            white-space:normal !important;
            overflow:visible !important;
        }

        .fleet-card .fleet-label {
            font-size:.58rem !important;
            line-height:1.15 !important;
        }

        .fleet-card .fleet-value {
            font-size:1.00rem !important;
            line-height:1.06 !important;
            font-weight:850 !important;
        }

        .fleet-card .fleet-note {
            font-size:.49rem !important;
            line-height:1.15 !important;
            color:#7288a8 !important;
        }

        .fleet-card .ghost {
            position:static !important;
            grid-column:3 !important;
            grid-row:1 !important;
            width:36px !important;
            height:36px !important;
            align-self:center !important;
            justify-self:end !important;
            margin:0 !important;
        }

        .fleet-summary-heading {
            display:block !important;
            margin:.02rem 0 .34rem !important;
            padding:0 !important;
            line-height:1.20 !important;
        }

        /* Keep the insight banner visually detached from the cards. */
        .insight-banner {
            margin:.48rem 0 .58rem !important;
        }

        /* Give page headings a real block of space before the next component. */
        .section-title {
            display:block !important;
            margin:0 0 .12rem !important;
            padding:0 !important;
            line-height:1.22 !important;
        }

        .section-subtitle {
            display:block !important;
            margin:0 !important;
            padding:0 !important;
            line-height:1.28 !important;
        }

        /* Model Performance chart row needs space below the KPI cards. */
        .panel-title {
            padding-top:.18rem !important;
            margin-top:.24rem !important;
            margin-bottom:.24rem !important;
        }

        /* Individual Engine: keep plot and explanatory text separate. */
        .engine-risk-notes {
            margin-top:.34rem !important;
            padding-top:.32rem !important;
            border-top:1px solid rgba(111,151,205,.13) !important;
        }

        .engine-risk-caption {
            color:#7f94b2 !important;
            font-size:.54rem !important;
            line-height:1.22 !important;
            margin-bottom:.22rem !important;
        }

        .engine-risk-facts {
            display:flex !important;
            flex-direction:column !important;
            gap:.16rem !important;
            color:#d7e4f4 !important;
            font-size:.63rem !important;
            line-height:1.24 !important;
        }

        .engine-risk-facts strong {
            color:#f0f5fc !important;
        }

        /* Slightly more clearance around the financial decision metrics. */
        .engine-card-gap {
            height:.34rem !important;
            min-height:.34rem !important;
        }

        .engine-section-heading {
            margin-top:.24rem !important;
            margin-bottom:.42rem !important;
        }

        /* Prevent widget label/value collision on Financial Scenarios. */
        [data-testid="stWidgetLabel"] {
            display:block !important;
            min-height:1.05rem !important;
            margin-bottom:.20rem !important;
        }

        [data-testid="stWidgetLabel"] p {
            line-height:1.20 !important;
        }
    }


    /* FINAL NO-OVERLAP LAYOUT PASS
       Uses structural padding and minimum heights so compact typography
       cannot collide with the next Streamlit element. */
    @media (min-width:1100px) {
        .page-header-shell {
            display:block !important;
            box-sizing:border-box !important;
            min-height:3.35rem !important;
            padding:.12rem 0 .86rem !important;
            margin:0 !important;
            overflow:visible !important;
        }

        .page-header-shell .section-title {
            display:block !important;
            min-height:1.28rem !important;
            margin:0 0 .24rem !important;
            padding:0 !important;
            line-height:1.22 !important;
        }

        .page-header-shell .section-subtitle {
            display:block !important;
            min-height:1.02rem !important;
            margin:0 !important;
            padding:0 !important;
            line-height:1.30 !important;
        }

        /* Always leave space between the Executive KPI row and Fleet summary. */
        .fleet-pre-gap {
            display:block !important;
            height:.72rem !important;
            min-height:.72rem !important;
            line-height:.72rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }

        .fleet-summary-heading {
            display:block !important;
            box-sizing:border-box !important;
            min-height:1.55rem !important;
            margin:0 !important;
            padding:0 0 .56rem !important;
            line-height:1.22 !important;
        }

        .after-fleet-space {
            display:block !important;
            height:.64rem !important;
            min-height:.64rem !important;
            line-height:.64rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }

        .insight-banner {
            margin:0 0 .62rem !important;
        }

        /* Model Performance needs a physical row between KPI cards and chart headings. */
        .chart-row-gap {
            display:block !important;
            height:.78rem !important;
            min-height:.78rem !important;
            line-height:.78rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }

        .panel-title {
            display:block !important;
            min-height:1.28rem !important;
            box-sizing:border-box !important;
            margin:0 0 .34rem !important;
            padding:0 !important;
            line-height:1.22 !important;
        }

        .panel-subtitle {
            display:block !important;
            min-height:.88rem !important;
            margin:0 0 .30rem !important;
            padding:0 !important;
            line-height:1.25 !important;
        }

        /* Widget labels get their own reserved height. */
        [data-testid="stWidgetLabel"] {
            display:block !important;
            box-sizing:border-box !important;
            min-height:1.32rem !important;
            margin:0 0 .30rem !important;
            padding:0 !important;
            overflow:visible !important;
        }

        [data-testid="stWidgetLabel"] p {
            display:block !important;
            margin:0 !important;
            padding:0 !important;
            line-height:1.28 !important;
        }

        .stSelectbox,
        .stMultiSelect,
        .stSlider {
            margin-top:0 !important;
            margin-bottom:.34rem !important;
        }

        /* Selected scenario heading must not sit on top of its cards. */
        h4 {
            display:block !important;
            min-height:1.30rem !important;
            margin:.24rem 0 .34rem !important;
            padding:0 !important;
            line-height:1.22 !important;
        }

        .scenario-gap {
            display:block !important;
            height:.44rem !important;
            min-height:.44rem !important;
            line-height:.44rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }

        /* Individual Engine spacing. */
        .engine-gap-xs {
            display:block !important;
            height:.30rem !important;
            min-height:.30rem !important;
            line-height:.30rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }

        .engine-gap-sm {
            display:block !important;
            height:.46rem !important;
            min-height:.46rem !important;
            line-height:.46rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }

        .engine-gap-md {
            display:block !important;
            height:.66rem !important;
            min-height:.66rem !important;
            line-height:.66rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }

        .engine-card-gap {
            display:block !important;
            height:.38rem !important;
            min-height:.38rem !important;
            line-height:.38rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }

        .engine-section-heading {
            display:block !important;
            min-height:1.45rem !important;
            margin:.16rem 0 .42rem !important;
            padding:0 !important;
            line-height:1.22 !important;
        }

        /* Do not let Streamlit's compact vertical block gap undo the explicit spacing. */
        div[data-testid="stVerticalBlock"] {
            gap:.38rem !important;
        }

        /* Keep the compact cards, but preserve their internal content height. */
        .kpi-card {
            min-height:72px !important;
        }

        .fleet-card {
            min-height:70px !important;
        }
    }


    /* FINAL TARGETED SPACING FIX
       Addresses the last visible collisions while preserving the one-screen layout. */
    @media (min-width:1100px) {
        /* Executive Overview: clear separation from KPI row to Fleet summary. */
        .fleet-pre-gap {
            display:block !important;
            height:1.02rem !important;
            min-height:1.02rem !important;
            line-height:1.02rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }

        .fleet-summary-heading {
            display:block !important;
            min-height:1.72rem !important;
            margin:0 !important;
            padding:0 0 .72rem !important;
            line-height:1.22 !important;
            box-sizing:border-box !important;
        }

        /* Fleet cards should not visually touch the insight banner. */
        .after-fleet-space {
            display:block !important;
            height:.88rem !important;
            min-height:.88rem !important;
            line-height:.88rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }

        .insight-banner {
            margin:0 0 .64rem !important;
        }

        /* Individual Engine: status card and metric row need a distinct gap. */
        .engine-status-gap {
            display:block !important;
            height:.78rem !important;
            min-height:.78rem !important;
            line-height:.78rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }

        .status-card {
            margin-bottom:0 !important;
        }

        /* Give the three metric cards a little breathing room below them too. */
        .engine-gap-md {
            height:.78rem !important;
            min-height:.78rem !important;
            line-height:.78rem !important;
        }

        /* Model Performance: separate KPI cards from chart headings. */
        .chart-row-gap {
            display:block !important;
            height:1.20rem !important;
            min-height:1.20rem !important;
            line-height:1.20rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }

        .panel-title {
            min-height:1.38rem !important;
            margin:0 0 .44rem !important;
            padding:0 !important;
            line-height:1.22 !important;
            box-sizing:border-box !important;
        }

        /* Keep the KPI row self-contained so the chart row starts below it. */
        .kpi-card {
            margin-bottom:0 !important;
        }

        /* Slightly increase the inter-column row clearance without enlarging cards. */
        div[data-testid="stHorizontalBlock"] {
            margin-bottom:.18rem !important;
        }
    }


    /* FINAL CARD GAP FIX
       Adds clear separation between the two card groups the user highlighted. */
    @media (min-width:1100px) {
        /* Executive Overview: Fleet summary cards -> Key decision insight */
        div[data-testid="stElementContainer"]:has(.fleet-card) {
            margin-bottom:.85rem !important;
        }

        .after-fleet-space {
            display:block !important;
            height:.55rem !important;
            min-height:.55rem !important;
            line-height:.55rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }

        .insight-banner {
            margin-top:.18rem !important;
        }

        /* Individual Engine: Engine status card -> metric boxes */
        div[data-testid="stElementContainer"]:has(.status-card) {
            margin-bottom:.95rem !important;
        }

        .engine-status-gap {
            display:block !important;
            height:.45rem !important;
            min-height:.45rem !important;
            line-height:.45rem !important;
            font-size:1px !important;
            overflow:hidden !important;
        }
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


def compact_money(value: float) -> str:
    value = float(value)
    sign = "-" if value < 0 else ""
    absolute = abs(value)

    if absolute >= 1_000_000:
        scaled = absolute / 1_000_000
        formatted = f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{sign}£{formatted}m"

    if absolute >= 1_000:
        scaled = absolute / 1_000
        formatted = f"{scaled:.0f}" if scaled >= 10 else f"{scaled:.1f}".rstrip("0").rstrip(".")
        return f"{sign}£{formatted}k"

    return f"{sign}£{absolute:,.0f}"


def currency_axis_ticks(values, target_ticks: int = 4):
    clean = [float(value) for value in values if pd.notna(value)]
    if not clean:
        return [], []

    low = min(clean)
    high = max(clean)
    if low == high:
        return [low], [compact_money(low)]

    raw_step = (high - low) / max(target_ticks - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalised = raw_step / magnitude

    if normalised <= 1:
        nice = 1
    elif normalised <= 2:
        nice = 2
    elif normalised <= 2.5:
        nice = 2.5
    elif normalised <= 5:
        nice = 5
    else:
        nice = 10

    step = nice * magnitude
    start = math.ceil(low / step) * step
    end = math.floor(high / step) * step

    ticks = []
    current = start
    while current <= end + step * 1e-9:
        ticks.append(float(current))
        current += step

    if len(ticks) < 2:
        ticks = [low, high]

    return ticks, [compact_money(value) for value in ticks]


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
        <div class="kpi-card"
             style="--accent:{accent};--glow-soft:{accent}18;--icon-bg:{accent}22;--icon-border:{accent}66;--icon-glow:{accent}45;">
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
    icon = {
        "HIGH Risk": "!",
        "MEDIUM Risk": "!",
        "LOW Risk": "✓",
        "Financially Selected": "£",
    }.get(label, "•")

    st.markdown(
        f"""
        <div class="fleet-card" style="--risk:{risk_color};">
            <div class="fleet-icon">{icon}</div>
            <div class="fleet-copy">
                <div class="fleet-label">{label}</div>
                <div class="fleet-value">{value}</div>
                <div class="fleet-note">{note}</div>
            </div>
            <div class="ghost"></div>
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
                <div class="brand-title">IRIS</div>
                <div class="brand-subtitle">Predictive Maintenance</div>
            </div>
        </div>
        <div class="sidebar-caption">Decision-support dashboard</div>
        """,
        unsafe_allow_html=True,
    )

    selected_page = st.radio(
        "Dashboard section",
        [
            "⌂  Executive Overview",
            "▥  Fleet Risk",
            "⚙  Individual Engine",
            "⌁  Model Performance",
            "▤  Financial Scenarios",
        ],
    )

    page = (
        selected_page
        .replace("⌂  ", "")
        .replace("▥  ", "")
        .replace("⚙  ", "")
        .replace("⌁  ", "")
        .replace("▤  ", "")
    )

    st.divider()

    st.markdown('<div class="sidebar-model-label">Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-model-value">⚙ &nbsp; Gradient Boosting</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-model-label">Capped RUL target</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-model-value">{RUL_CAP} cycles</div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-model-label">Operational HIGH threshold</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sidebar-model-value">{optimised_threshold} cycles</div>', unsafe_allow_html=True)

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

    st.markdown(
        """
        <div class="sidebar-footer">
            <span class="plane">✈</span>
            <span>Engineering reliability through data</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# HERO
# =========================================================

hero_html = (
    f'<div class="hero-shell">'
    f'<div class="hero-copy-wrap">'
    f'<div class="hero-kicker">Industrial reliability analytics</div>'
    f'<div class="hero-title">Predictive Maintenance &amp; <span class="hero-title-accent">Financial Risk Platform</span></div>'
    f'<div class="hero-subtitle">'
    f'Sensor-driven Remaining Useful Life prediction, maintenance prioritisation, calibrated short-horizon risk estimation '
    f'and scenario-based financial decision support.'
    f'</div>'
    f'<div class="value-row">'
    f'<div class="value-pill">'
    f'<div class="value-dot"><svg viewBox="0 0 24 24"><path d="M2 12h4l2.2-5 3.4 10 2.7-7 2 4H22"/></svg></div>'
    f'<span class="feature-copy">Turn data<br>into reliability</span>'
    f'</div>'
    f'<div class="value-pill">'
    f'<div class="value-dot"><svg viewBox="0 0 24 24"><path d="M12 3l7 3v5c0 4.8-2.8 8.1-7 10-4.2-1.9-7-5.2-7-10V6l7-3z"/><path d="M9 12l2 2 4-4"/></svg></div>'
    f'<span class="feature-copy">Reduce risk<br>improve availability</span>'
    f'</div>'
    f'<div class="value-pill">'
    f'<div class="value-dot"><svg viewBox="0 0 24 24"><path d="M5 19V11M12 19V6M19 19V9"/><path d="M3 19h18"/></svg></div>'
    f'<span class="feature-copy">Smarter maintenance<br>stronger business</span>'
    f'</div>'
    f'</div>'
    f'</div>'
    f'<div class="turbine"></div>'
    f'<div class="hero-operator">'
    f'<div class="hero-operator-avatar">PM</div>'
    f'<div class="hero-operator-copy"><strong>Live dashboard</strong><span>Portfolio demo</span></div>'
    f'</div>'
    f'<div class="hero-right-copy">Insight today<br>Reliability tomorrow</div>'
    f'</div>'
)

st.markdown(hero_html, unsafe_allow_html=True)


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":
    st.markdown(
        f"""
        <div class="overview-head">
            <div>
                <div class="section-title">▥ Executive Overview</div>
                <div class="section-subtitle">Predictive performance, fleet condition and current maintenance priorities.</div>
            </div>
            <div class="overview-status">
                <span>▣ &nbsp; Last updated<br><strong>{last_updated}</strong></span>
                <span style="height:28px;border-left:1px solid rgba(110,151,207,.18);"></span>
                <span><span class="status-dot" style="display:inline-block;margin-right:.35rem;"></span><strong>Dashboard ready</strong><br>Outputs loaded successfully</span>
            </div>
        </div>
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

    st.markdown('<div class="fleet-pre-gap">&nbsp;</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="fleet-summary-heading">◉ Fleet summary</div>',
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

    st.markdown('<div class="after-fleet-space">&nbsp;</div>', unsafe_allow_html=True)

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
            <div class="insight-action">Drive action<br>with confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([0.82, 1.48], gap="medium")

    with left:
        with st.container(border=True):
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
                domain={"x": [0.00, 0.62]},
            )
            risk_fig.add_annotation(
                x=0.31,
                y=0.5,
                text=f"<b>{len(financial_results)}</b><br><span style='font-size:11px;color:#8ea1be'>Engines</span>",
                showarrow=False,
                font={"size": 22, "color": TEXT},
            )
            risk_fig = configure_plot(risk_fig, height=225)
            risk_fig.update_layout(
                margin={"l": 4, "r": 4, "t": 4, "b": 4},
                legend={"x": 0.69, "y": 0.50, "xanchor": "left", "yanchor": "middle", "font": {"size": 11}},
            )
            st.plotly_chart(risk_fig, use_container_width=True)

    with right:
        with st.container(border=True):
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
                height=230,
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
        <div class="page-header-shell">
            <div class="section-title">▥ Fleet Risk</div>
            <div class="section-subtitle">Operational fleet view using predicted RUL, estimated short-horizon risk and economic action.</div>
        </div>
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
        height=275,
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
        <div class="page-header-shell">
            <div class="section-title">⚙ Individual Engine Analysis</div>
            <div class="section-subtitle">Operational decision view for one engine. Ground-truth RUL is available separately for evaluation only.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    engine_ids = sorted(financial_results["engine_id"].unique())
    selected_engine = st.selectbox("Select engine", engine_ids)
    st.markdown('<div class="engine-gap-xs">&nbsp;</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="engine-status-gap">&nbsp;</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="small")
    c1.metric("Current Cycle", int(engine["cycle"]))
    c2.metric("Predicted RUL", f"{engine['predicted_rul_capped']:.1f} cycles")
    c3.metric("Operational Risk", risk)
    st.markdown('<div class="engine-gap-md">&nbsp;</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="engine-section-heading"><h4>Engineering risk</h4></div>', unsafe_allow_html=True)

        probability = float(engine["high_risk_probability"])
        probability_percent = probability * 100

        # Preserve meaningful precision for very small probabilities.
        probability_format = ".3f" if probability_percent < 0.1 else ".1f"

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability_percent,
                number={"suffix": "%", "valueformat": probability_format},
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
            height=196,
            paper_bgcolor=TRANSPARENT,
            font={"color": TEXT},
            margin={"l": 18, "r": 18, "t": 52, "b": 24},
        )

        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(
            f"""
            <div class="engine-risk-notes">
                <div class="engine-risk-caption">
                    White marker = financial break-even probability ({base_break_even:.1%}).
                </div>
                <div class="engine-risk-facts">
                    <span>Operational HIGH threshold: <strong>{optimised_threshold} cycles</strong></span>
                    <span>Recommendation: <strong>{recommendation}</strong></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="engine-section-heading"><h4>Financial decision</h4></div>', unsafe_allow_html=True)

        st.metric(
            "Risk-Adjusted Exposure",
            money(engine["risk_adjusted_failure_exposure"]),
        )
        st.markdown('<div class="engine-card-gap">&nbsp;</div>', unsafe_allow_html=True)
        st.metric(
            "Illustrative Net Benefit",
            money(engine["risk_adjusted_net_benefit"]),
        )
        st.markdown('<div class="engine-card-gap">&nbsp;</div>', unsafe_allow_html=True)
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

    st.markdown('<div class="engine-gap-sm">&nbsp;</div>', unsafe_allow_html=True)
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
        <div class="page-header-shell">
            <div class="section-title">⌁ Model Performance</div>
            <div class="section-subtitle">Evaluation-only view of RUL accuracy, threshold behaviour and probability calibration.</div>
        </div>
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

    st.markdown('<div class="chart-row-gap">&nbsp;</div>', unsafe_allow_html=True)
    chart_col1, chart_col2 = st.columns(2, gap="large")

    with chart_col1:
        st.markdown(
            '<div class="panel-title">RUL prediction performance</div>',
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

        rul_fig.add_trace(
            go.Scatter(
                x=[0, RUL_CAP],
                y=[0, RUL_CAP],
                mode="lines",
                name="Perfect prediction",
                line={"dash": "dash", "color": MUTED},
                hoverinfo="skip",
            )
        )

        axis_ticks = list(range(0, RUL_CAP + 1, 25))
        rul_fig.update_traces(cliponaxis=False, selector=dict(mode="markers"))
        rul_fig.update_xaxes(
            range=[-3, RUL_CAP + 7],
            tickmode="array",
            tickvals=axis_ticks,
            title_text="Actual RUL (cycles)",
        )
        rul_fig.update_yaxes(
            range=[-3, RUL_CAP + 7],
            tickmode="array",
            tickvals=axis_ticks,
            title_text="Predicted RUL (cycles)",
        )
        rul_fig = configure_plot(rul_fig, height=238)
        rul_fig.update_layout(margin={"l": 42, "r": 26, "t": 24, "b": 42})
        st.plotly_chart(rul_fig, use_container_width=True)

    with chart_col2:
        st.markdown(
            '<div class="panel-title">Probability calibration</div>',
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
        probability_ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        calibration_fig.update_traces(cliponaxis=False, selector=dict(mode="lines+markers"))
        calibration_fig.update_xaxes(
            range=[-0.025, 1.055],
            tickmode="array",
            tickvals=probability_ticks,
            tickformat=".0%",
        )
        calibration_fig.update_yaxes(
            range=[-0.025, 1.055],
            tickmode="array",
            tickvals=probability_ticks,
            tickformat=".0%",
        )
        calibration_fig = configure_plot(calibration_fig, height=238)
        calibration_fig.update_layout(margin={"l": 48, "r": 26, "t": 24, "b": 42})
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

    with st.expander("Calibration data"):
        st.dataframe(
            calibration_results,
            hide_index=True,
            use_container_width=True,
            height=150,
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
        <div class="page-header-shell">
            <div class="section-title">▤ Financial Scenario Analysis</div>
            <div class="section-subtitle">Test how preventative maintenance and failure-cost assumptions change the illustrative economic decision threshold.</div>
        </div>
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
    st.markdown('<div class="scenario-gap">&nbsp;</div>', unsafe_allow_html=True)

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
            hovertemplate=(
                "Failure cost: £%{x:,.0f}<br>"
                "Illustrative net benefit: £%{y:,.0f}"
                "<extra></extra>"
            ),
        )

        failure_ticks = sorted(line_data["failure_cost"].astype(float).unique().tolist())
        benefit_values = line_data["risk_adjusted_net_benefit"].astype(float).tolist()
        benefit_ticks, benefit_tick_labels = currency_axis_ticks(benefit_values, target_ticks=5)

        x_span = max(failure_ticks) - min(failure_ticks)
        y_min = min(benefit_values)
        y_max = max(benefit_values)
        y_span = max(y_max - y_min, 1.0)

        line_fig.update_xaxes(
            range=[min(failure_ticks) - x_span * 0.045, max(failure_ticks) + x_span * 0.045],
            tickmode="array",
            tickvals=failure_ticks,
            ticktext=[compact_money(value) for value in failure_ticks],
            title_text="Failure Cost",
        )
        line_fig.update_yaxes(
            range=[max(0, y_min - y_span * 0.10), y_max + y_span * 0.12],
            tickmode="array",
            tickvals=benefit_ticks,
            ticktext=benefit_tick_labels,
            title_text="Illustrative Net Benefit",
        )

        selected_y = float(scenario["risk_adjusted_net_benefit"])
        line_fig.add_trace(
            go.Scatter(
                x=[selected_failure_cost],
                y=[selected_y],
                mode="markers",
                name="Selected scenario",
                showlegend=False,
                marker={
                    "size": 13,
                    "symbol": "diamond",
                    "color": GOLD,
                    "line": {"width": 2, "color": "#f8fafc"},
                },
                hovertemplate=(
                    "Selected scenario<br>"
                    "Failure cost: £%{x:,.0f}<br>"
                    "Illustrative net benefit: £%{y:,.0f}"
                    "<extra></extra>"
                ),
            )
        )
        line_fig.add_annotation(
            x=selected_failure_cost,
            y=selected_y,
            text="Selected",
            showarrow=True,
            arrowhead=0,
            ax=34,
            ay=-24,
            font={"size": 9, "color": "#f7d36b"},
            arrowcolor="rgba(246,195,74,.65)",
            bgcolor="rgba(7,19,34,.82)",
            bordercolor="rgba(246,195,74,.35)",
            borderpad=3,
        )

        line_fig = configure_plot(line_fig, height=232)
        line_fig.update_layout(
            showlegend=False,
            margin={"l": 48, "r": 20, "t": 22, "b": 44},
        )
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

        heatmap_x_labels = [compact_money(value) for value in heatmap_data.columns]
        heatmap_y_labels = [compact_money(value) for value in heatmap_data.index]
        heatmap_values = heatmap_data.to_numpy(dtype=float)
        heatmap_text = [
            [compact_money(value) for value in row]
            for row in heatmap_values
        ]
        color_ticks, color_tick_labels = currency_axis_ticks(
            heatmap_values.ravel().tolist(), target_ticks=4
        )

        heatmap_fig = go.Figure(
            data=go.Heatmap(
                x=heatmap_x_labels,
                y=heatmap_y_labels,
                z=heatmap_values,
                colorscale=[
                    [0.0, "#10284b"],
                    [0.45, "#2d5f9f"],
                    [0.75, "#5f94d3"],
                    [1.0, "#9bcdf4"],
                ],
                xgap=3,
                ygap=3,
                text=heatmap_text,
                texttemplate="%{text}",
                textfont={"size": 9, "color": "#f8fbff"},
                customdata=[
                    [
                        [float(failure_cost), float(maintenance_cost)]
                        for failure_cost in heatmap_data.columns
                    ]
                    for maintenance_cost in heatmap_data.index
                ],
                hovertemplate=(
                    "Failure cost: £%{customdata[0]:,.0f}<br>"
                    "Maintenance cost: £%{customdata[1]:,.0f}<br>"
                    "Illustrative net benefit: £%{z:,.0f}"
                    "<extra></extra>"
                ),
                colorbar={
                    "title": {"text": "Net Benefit"},
                    "tickmode": "array",
                    "tickvals": color_ticks,
                    "ticktext": color_tick_labels,
                    "thickness": 10,
                    "len": 0.78,
                    "outlinewidth": 0,
                },
                hoverongaps=False,
            )
        )

        selected_x_label = compact_money(selected_failure_cost)
        selected_y_label = compact_money(selected_maintenance_cost)
        heatmap_fig.add_trace(
            go.Scatter(
                x=[selected_x_label],
                y=[selected_y_label],
                mode="markers",
                showlegend=False,
                marker={
                    "size": 30,
                    "symbol": "square-open",
                    "color": GOLD,
                    "line": {"width": 3, "color": GOLD},
                },
                hovertemplate=(
                    "Selected scenario<br>"
                    f"Failure cost: {money(selected_failure_cost)}<br>"
                    f"Maintenance cost: {money(selected_maintenance_cost)}"
                    "<extra></extra>"
                ),
            )
        )

        heatmap_fig.update_xaxes(
            title_text="Failure Cost",
            categoryorder="array",
            categoryarray=heatmap_x_labels,
            showgrid=False,
            zeroline=False,
        )
        heatmap_fig.update_yaxes(
            title_text="Maintenance Cost",
            categoryorder="array",
            categoryarray=heatmap_y_labels,
            autorange="reversed",
            showgrid=False,
            zeroline=False,
        )

        heatmap_fig = configure_plot(heatmap_fig, height=232)
        heatmap_fig.update_layout(
            showlegend=False,
            margin={"l": 54, "r": 74, "t": 22, "b": 44},
        )
        st.plotly_chart(heatmap_fig, use_container_width=True)

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

    with st.expander("All cost scenarios and downloads"):
        st.dataframe(
            scenarios_display,
            hide_index=True,
            use_container_width=True,
            height=190,
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
        <span>NASA C-MAPSS FD001 · Portfolio project · Financial assumptions are illustrative</span>
        <span class="footer-right">From data to decisions &nbsp; ✈ &nbsp; For a more reliable tomorrow</span>
    </div>
    """,
    unsafe_allow_html=True,
)

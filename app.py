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

FINANCIAL_RESULTS_PATH = (
    OUTPUT_DIR / "financial_risk_results.csv"
)

SENSITIVITY_PATH = (
    OUTPUT_DIR / "cost_sensitivity_analysis.csv"
)

THRESHOLD_PATH = (
    OUTPUT_DIR / "threshold_optimization.csv"
)

CALIBRATION_PATH = (
    OUTPUT_DIR / "probability_calibration.csv"
)

RUL_CAP = 125
ACTUAL_HIGH_THRESHOLD = 30
MEDIUM_THRESHOLD = 60

RISK_COLORS = {
    "HIGH": "#ef4444",
    "MEDIUM": "#f59e0b",
    "LOW": "#22c55e",
}

PLOT_BG = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(148,163,184,0.14)"
TEXT_COLOR = "#e5e7eb"
MUTED_TEXT = "#94a3b8"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Predictive Maintenance Risk Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# GLOBAL STYLING
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1480px;
        padding-top: 1.15rem;
        padding-bottom: 2.5rem;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(148, 163, 184, 0.16);
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.44);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 16px;
        padding: 1rem 1.05rem;
        min-height: 110px;
    }

    div[data-testid="stMetric"] label {
        opacity: 0.72;
        font-size: 0.78rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.65rem;
        font-weight: 750;
    }

    .hero {
        padding: 1.45rem 1.6rem;
        border-radius: 20px;
        border: 1px solid rgba(148, 163, 184, 0.17);
        background:
            linear-gradient(
                135deg,
                rgba(30, 41, 59, 0.82),
                rgba(15, 23, 42, 0.50)
            );
        margin-bottom: 1.15rem;
    }

    .hero-kicker {
        opacity: 0.60;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.13em;
        margin-bottom: 0.35rem;
    }

    .hero-title {
        font-size: 2rem;
        line-height: 1.15;
        font-weight: 760;
        margin-bottom: 0.45rem;
    }

    .hero-copy {
        opacity: 0.76;
        max-width: 970px;
        line-height: 1.55;
    }

    .status-card {
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(15, 23, 42, 0.35);
        margin-bottom: 0.9rem;
    }

    .small-muted {
        opacity: 0.67;
        font-size: 0.86rem;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 14px;
        overflow: hidden;
    }

    .stDownloadButton > button,
    .stButton > button {
        border-radius: 10px;
    }

    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(148, 163, 184, 0.14);
        font-size: 0.78rem;
        opacity: 0.62;
        text-align: center;
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
    """Load the files created by src/pipeline.py."""

    return (
        pd.read_csv(
            FINANCIAL_RESULTS_PATH
        ),
        pd.read_csv(
            SENSITIVITY_PATH
        ),
        pd.read_csv(
            THRESHOLD_PATH
        ),
        pd.read_csv(
            CALIBRATION_PATH
        ),
    )


def get_optimised_threshold(
    thresholds: pd.DataFrame,
) -> int:
    """Return the selected HIGH-risk RUL threshold."""

    if "selected" not in thresholds.columns:
        return ACTUAL_HIGH_THRESHOLD

    selected_mask = (
        thresholds["selected"]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
            ]
        )
    )

    selected = thresholds[
        selected_mask
    ]

    if selected.empty:
        return ACTUAL_HIGH_THRESHOLD

    return int(
        selected.iloc[0]["threshold"]
    )


def assign_predicted_risk(
    predicted_rul: float,
    high_threshold: int,
) -> str:
    """Convert predicted RUL into an operational risk level."""

    if predicted_rul <= high_threshold:
        return "HIGH"

    if predicted_rul <= MEDIUM_THRESHOLD:
        return "MEDIUM"

    return "LOW"


def recommendation_for(
    risk: str,
) -> str:
    """Map a risk level to an operational recommendation."""

    if risk == "HIGH":
        return "Schedule preventative maintenance"

    if risk == "MEDIUM":
        return "Increase monitoring"

    return "Continue operating"


def short_action(
    action: str,
) -> str:
    """Shorten action text for compact tables."""

    mapping = {
        "Preventative maintenance":
            "Maintain",

        "Continue / monitor":
            "Monitor",
    }

    return mapping.get(
        action,
        action,
    )


def money(
    value: float,
) -> str:
    """Format currency naturally, including negatives."""

    value = float(
        value
    )

    if value < 0:
        return f"-£{abs(value):,.0f}"

    return f"£{value:,.0f}"


def percent(
    value: float,
    decimals: int = 1,
) -> str:
    """Format a proportion as a percentage."""

    return (
        f"{value:.{decimals}%}"
    )


def csv_bytes(
    df: pd.DataFrame,
) -> bytes:
    """Convert a dataframe to downloadable CSV bytes."""

    return df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


def configure_plot(
    fig,
    *,
    height: int | None = None,
):
    """Apply a consistent dark, transparent Plotly theme."""

    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font={
            "color":
                TEXT_COLOR,
        },
        margin={
            "l": 20,
            "r": 20,
            "t": 50,
            "b": 35,
        },
        legend={
            "title_text":
                "",
        },
    )

    if height is not None:
        fig.update_layout(
            height=height
        )

    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
    )

    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
    )

    return fig


# =========================================================
# LOAD OUTPUTS
# =========================================================

required_paths = [
    FINANCIAL_RESULTS_PATH,
    SENSITIVITY_PATH,
    THRESHOLD_PATH,
    CALIBRATION_PATH,
]

missing_paths = [
    path
    for path in required_paths
    if not path.exists()
]

if missing_paths:

    st.error(
        "Required pipeline outputs are missing. "
        "Run `python src/pipeline.py` first."
    )

    st.code(
        "\n".join(
            str(path)
            for path in missing_paths
        )
    )

    st.stop()


try:

    (
        financial_results,
        sensitivity_results,
        threshold_results,
        calibration_results,
    ) = load_dashboard_data()

except Exception as error:

    st.error(
        "The dashboard could not load the pipeline outputs."
    )

    st.exception(
        error
    )

    st.stop()


optimised_threshold = (
    get_optimised_threshold(
        threshold_results
    )
)


financial_results = (
    financial_results.copy()
)


financial_results[
    "actual_high"
] = (
    financial_results[
        "actual_rul_capped"
    ]
    <= ACTUAL_HIGH_THRESHOLD
).astype(
    int
)


financial_results[
    "predicted_high"
] = (
    financial_results[
        "predicted_rul_capped"
    ]
    <= optimised_threshold
).astype(
    int
)


financial_results[
    "risk_level"
] = (
    financial_results[
        "predicted_rul_capped"
    ]
    .apply(
        lambda value:
        assign_predicted_risk(
            value,
            optimised_threshold,
        )
    )
)


financial_results[
    "recommendation"
] = (
    financial_results[
        "risk_level"
    ]
    .apply(
        recommendation_for
    )
)


capped_mae = mean_absolute_error(
    financial_results[
        "actual_rul_capped"
    ],
    financial_results[
        "predicted_rul_capped_raw"
    ],
)


high_recall = recall_score(
    financial_results[
        "actual_high"
    ],
    financial_results[
        "predicted_high"
    ],
)


high_precision = precision_score(
    financial_results[
        "actual_high"
    ],
    financial_results[
        "predicted_high"
    ],
    zero_division=0,
)


roc_auc = roc_auc_score(
    financial_results[
        "actual_high"
    ],
    financial_results[
        "high_risk_probability"
    ],
)


brier_score = brier_score_loss(
    financial_results[
        "actual_high"
    ],
    financial_results[
        "high_risk_probability"
    ],
)


high_count = int(
    (
        financial_results[
            "risk_level"
        ]
        == "HIGH"
    ).sum()
)


medium_count = int(
    (
        financial_results[
            "risk_level"
        ]
        == "MEDIUM"
    ).sum()
)


low_count = int(
    (
        financial_results[
            "risk_level"
        ]
        == "LOW"
    ).sum()
)


financially_selected = int(
    financial_results[
        "maintenance_economically_justified"
    ].sum()
)


base_maintenance_cost = float(
    financial_results[
        "maintenance_cost"
    ].iloc[0]
)


base_failure_cost = float(
    financial_results[
        "failure_cost"
    ].iloc[0]
)


base_break_even = float(
    financial_results[
        "break_even_probability"
    ].iloc[0]
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "### Predictive Maintenance"
    )

    st.caption(
        "Decision-support dashboard"
    )

    page = st.radio(
        "Dashboard section",
        [
            "Executive Overview",
            "Fleet Risk",
            "Individual Engine",
            "Model Performance",
            "Financial Scenarios",
        ],
    )

    st.divider()

    st.caption(
        "MODEL"
    )

    st.write(
        "**Gradient Boosting**"
    )

    st.caption(
        "Capped RUL target"
    )

    st.write(
        f"**{RUL_CAP} cycles**"
    )

    st.caption(
        "Operational HIGH threshold"
    )

    st.write(
        f"**{optimised_threshold} cycles**"
    )

    st.divider()

    with st.expander(
        "Methodology & assumptions"
    ):

        st.markdown(
            f"""
            **Dataset**
            - NASA C-MAPSS FD001 simulated turbofan degradation data

            **Predictive modelling**
            - Gradient Boosting selected with engine-level grouped cross-validation
            - RUL target capped at {RUL_CAP} cycles
            - Reference HIGH risk: actual RUL ≤ {ACTUAL_HIGH_THRESHOLD} cycles

            **Operational threshold**
            - HIGH alert when predicted RUL ≤ {optimised_threshold} cycles

            **Probability layer**
            - Logistic regression estimates the probability that actual RUL is within {ACTUAL_HIGH_THRESHOLD} cycles

            **Financial assumptions**
            - Preventative maintenance: {money(base_maintenance_cost)}
            - Unplanned failure: {money(base_failure_cost)}
            - Costs are illustrative and are not NASA dataset values
            """
        )


# =========================================================
# GLOBAL HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">
            Industrial reliability analytics
        </div>
        <div class="hero-title">
            Predictive Maintenance & Financial Risk Platform
        </div>
        <div class="hero-copy">
            Sensor-driven Remaining Useful Life prediction,
            maintenance prioritisation, calibrated short-horizon
            risk estimation and scenario-based financial decision support.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":

    st.subheader(
        "Executive Overview"
    )

    st.caption(
        "Predictive performance, fleet condition and "
        "current maintenance priorities."
    )

    col1, col2, col3, col4, col5 = st.columns(
        5
    )

    col1.metric(
        "Capped Test MAE",
        f"{capped_mae:.2f} cycles",
    )

    col2.metric(
        "HIGH-Risk Recall",
        f"{high_recall:.1%}",
    )

    col3.metric(
        "HIGH-Risk Precision",
        f"{high_precision:.1%}",
    )

    col4.metric(
        "ROC AUC",
        f"{roc_auc:.3f}",
    )

    col5.metric(
        "Brier Score",
        f"{brier_score:.4f}",
    )

    st.markdown(
        "#### Fleet summary"
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    col1.metric(
        "HIGH Risk",
        high_count,
    )

    col2.metric(
        "MEDIUM Risk",
        medium_count,
    )

    col3.metric(
        "LOW Risk",
        low_count,
    )

    col4.metric(
        "Financially Selected",
        financially_selected,
    )

    st.markdown(
        "#### Key decision insight"
    )

    st.info(
        f"Under the current illustrative cost assumptions, "
        f"{financially_selected} of {len(financial_results)} engines "
        f"cross the economic maintenance threshold. "
        f"The operational HIGH-risk threshold is "
        f"{optimised_threshold} cycles, with "
        f"{high_recall:.1%} recall and "
        f"{high_precision:.1%} precision on the test fleet."
    )

    left, right = st.columns(
        [
            0.9,
            1.1,
        ]
    )

    with left:

        st.markdown(
            "#### Predicted fleet risk"
        )

        risk_counts = (
            financial_results[
                "risk_level"
            ]
            .value_counts()
            .reindex(
                [
                    "HIGH",
                    "MEDIUM",
                    "LOW",
                ],
                fill_value=0,
            )
            .rename_axis(
                "Risk Level"
            )
            .reset_index(
                name="Engines"
            )
        )

        risk_fig = px.pie(
            risk_counts,
            names="Risk Level",
            values="Engines",
            hole=0.60,
            color="Risk Level",
            color_discrete_map=
                RISK_COLORS,
            title=
                "Fleet Risk Distribution",
        )

        risk_fig.update_traces(
            textposition="inside",
            textinfo="percent",
        )

        risk_fig = configure_plot(
            risk_fig,
            height=430,
        )

        st.plotly_chart(
            risk_fig,
            use_container_width=True,
        )

    with right:

        st.markdown(
            "#### Top priority engines"
        )

        top_priority = (
            financial_results
            .sort_values(
                [
                    "high_risk_probability",
                    "predicted_rul_capped",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .head(
                10
            )
            .copy()
        )

        top_priority[
            "Action"
        ] = (
            top_priority[
                "economic_action"
            ]
            .apply(
                short_action
            )
        )

        top_priority = (
            top_priority[
                [
                    "engine_id",
                    "cycle",
                    "predicted_rul_capped",
                    "high_risk_probability",
                    "risk_level",
                    "Action",
                ]
            ]
            .rename(
                columns={
                    "engine_id":
                        "Engine",

                    "cycle":
                        "Cycle",

                    "predicted_rul_capped":
                        "Predicted RUL",

                    "high_risk_probability":
                        "HIGH-Risk Probability",

                    "risk_level":
                        "Risk",
                }
            )
        )

        st.dataframe(
            top_priority,
            hide_index=True,
            use_container_width=True,
            height=430,
            column_config={
                "Predicted RUL":
                    st.column_config.NumberColumn(
                        format="%.1f",
                    ),

                "HIGH-Risk Probability":
                    st.column_config.ProgressColumn(
                        format="percent",
                        min_value=0,
                        max_value=1,
                    ),
            },
        )

    with st.expander(
        "How to interpret this dashboard"
    ):

        st.markdown(
            f"""
            - **RUL** means Remaining Useful Life in engine cycles.
            - **HIGH risk** means predicted RUL ≤ **{optimised_threshold} cycles**.
            - **HIGH-risk probability** estimates whether actual RUL is within **{ACTUAL_HIGH_THRESHOLD} cycles**.
            - **Financially selected** means the estimated risk-adjusted exposure exceeds the assumed maintenance cost.
            - Actual future RUL is only available for evaluation and is intentionally excluded from operational decision views.
            - Financial values are illustrative scenario outputs, not demonstrated real-world savings.
            """
        )


# =========================================================
# FLEET RISK
# =========================================================

elif page == "Fleet Risk":

    st.subheader(
        "Fleet Risk"
    )

    st.caption(
        "Operational fleet view using predicted RUL, estimated "
        "short-horizon risk and economic action."
    )

    filter_col1, filter_col2, filter_col3 = st.columns(
        [
            1.35,
            1,
            1,
        ]
    )

    risk_filter = (
        filter_col1.multiselect(
            "Risk level",
            options=[
                "HIGH",
                "MEDIUM",
                "LOW",
            ],
            default=[
                "HIGH",
                "MEDIUM",
                "LOW",
            ],
        )
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

    financially_selected_only = (
        filter_col3.toggle(
            "Financially selected only",
            value=False,
        )
    )

    fleet = (
        financial_results[
            financial_results[
                "risk_level"
            ].isin(
                risk_filter
            )
        ]
        .copy()
    )

    fleet = fleet[
        fleet[
            "high_risk_probability"
        ]
        >= minimum_probability
    ]

    if financially_selected_only:

        fleet = fleet[
            fleet[
                "maintenance_economically_justified"
            ]
        ]

    fleet = (
        fleet
        .sort_values(
            [
                "high_risk_probability",
                "predicted_rul_capped",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    col1, col2, col3 = st.columns(
        3
    )

    col1.metric(
        "Engines Shown",
        len(
            fleet
        ),
    )

    col2.metric(
        "Mean HIGH-Risk Probability",
        (
            f"{fleet['high_risk_probability'].mean():.1%}"
            if not fleet.empty
            else "—"
        ),
    )

    col3.metric(
        "Financially Selected",
        (
            int(
                fleet[
                    "maintenance_economically_justified"
                ].sum()
            )
            if not fleet.empty
            else 0
        ),
    )

    fleet_display = (
        fleet[
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
        ]
        .copy()
    )

    fleet_display[
        "risk_adjusted_failure_exposure"
    ] = (
        fleet_display[
            "risk_adjusted_failure_exposure"
        ]
        .apply(
            money
        )
    )

    fleet_display[
        "risk_adjusted_net_benefit"
    ] = (
        fleet_display[
            "risk_adjusted_net_benefit"
        ]
        .apply(
            money
        )
    )

    fleet_display[
        "economic_action"
    ] = (
        fleet_display[
            "economic_action"
        ]
        .apply(
            short_action
        )
    )

    fleet_display = (
        fleet_display
        .rename(
            columns={
                "engine_id":
                    "Engine",

                "cycle":
                    "Current Cycle",

                "predicted_rul_capped":
                    "Predicted RUL",

                "risk_level":
                    "Risk",

                "high_risk_probability":
                    "HIGH-Risk Probability",

                "risk_adjusted_failure_exposure":
                    "Risk-Adjusted Exposure",

                "risk_adjusted_net_benefit":
                    "Risk-Adjusted Net Benefit",

                "economic_action":
                    "Action",
            }
        )
    )

    st.dataframe(
        fleet_display,
        hide_index=True,
        use_container_width=True,
        height=620,
        column_config={
            "Predicted RUL":
                st.column_config.NumberColumn(
                    format="%.1f",
                ),

            "HIGH-Risk Probability":
                st.column_config.ProgressColumn(
                    format="percent",
                    min_value=0,
                    max_value=1,
                ),
        },
    )

    st.download_button(
        "Download filtered fleet CSV",
        data=csv_bytes(
            fleet
        ),
        file_name=
            "fleet_risk_filtered.csv",
        mime="text/csv",
    )

    st.caption(
        "Actual RUL is intentionally hidden here because it "
        "would not be known in a live deployment."
    )


# =========================================================
# INDIVIDUAL ENGINE
# =========================================================

elif page == "Individual Engine":

    st.subheader(
        "Individual Engine Analysis"
    )

    st.caption(
        "Operational decision view for one engine. "
        "Ground-truth RUL is available separately for evaluation only."
    )

    engine_ids = sorted(
        financial_results[
            "engine_id"
        ].unique()
    )

    selected_engine = (
        st.selectbox(
            "Select engine",
            engine_ids,
        )
    )

    engine = (
        financial_results[
            financial_results[
                "engine_id"
            ]
            == selected_engine
        ]
        .iloc[0]
    )

    risk = (
        engine[
            "risk_level"
        ]
    )

    recommendation = (
        recommendation_for(
            risk
        )
    )

    risk_color = (
        RISK_COLORS[
            risk
        ]
    )

    st.markdown(
        f"""
        <div class="status-card" style="border-left: 5px solid {risk_color};">
            <div class="small-muted">Current operational status</div>
            <div style="font-size:1.35rem;font-weight:750;margin-top:.2rem;">
                Engine {int(engine['engine_id'])} ·
                <span style="color:{risk_color};">{risk}</span>
            </div>
            <div class="small-muted" style="margin-top:.35rem;">
                {recommendation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(
        3
    )

    col1.metric(
        "Current Cycle",
        int(
            engine[
                "cycle"
            ]
        ),
    )

    col2.metric(
        "Predicted RUL",
        f"{engine['predicted_rul_capped']:.1f} cycles",
    )

    col3.metric(
        "Operational Risk",
        risk,
    )

    left, right = st.columns(
        [
            1,
            1,
        ]
    )

    with left:

        st.markdown(
            "#### Engineering risk"
        )

        probability = float(
            engine[
                "high_risk_probability"
            ]
        )

        gauge = go.Figure(
            go.Indicator(
                mode=
                    "gauge+number",
                value=
                    probability
                    * 100,
                number={
                    "suffix":
                        "%",

                    "valueformat":
                        ".1f",
                },
                title={
                    "text":
                        "Probability actual RUL ≤ "
                        f"{ACTUAL_HIGH_THRESHOLD} cycles"
                },
                gauge={
                    "axis": {
                        "range":
                            [
                                0,
                                100,
                            ],

                        "tickcolor":
                            MUTED_TEXT,
                    },

                    "bar": {
                        "color":
                            risk_color,
                    },

                    "bgcolor":
                        "rgba(15,23,42,0.45)",

                    "bordercolor":
                        "rgba(148,163,184,0.18)",

                    "steps": [
                        {
                            "range":
                                [
                                    0,
                                    base_break_even
                                    * 100,
                                ],

                            "color":
                                "rgba(34,197,94,0.08)",
                        },
                        {
                            "range":
                                [
                                    base_break_even
                                    * 100,
                                    100,
                                ],

                            "color":
                                "rgba(239,68,68,0.06)",
                        },
                    ],

                    "threshold": {
                        "line": {
                            "color":
                                "#f8fafc",

                            "width":
                                4,
                        },

                        "thickness":
                            0.8,

                        "value":
                            base_break_even
                            * 100,
                    },
                },
            )
        )

        gauge.update_layout(
            height=285,
            paper_bgcolor=PLOT_BG,
            font={
                "color":
                    TEXT_COLOR,
            },
            margin={
                "l": 30,
                "r": 30,
                "t": 75,
                "b": 15,
            },
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
        )

        st.caption(
            f"White marker = current financial break-even "
            f"probability ({base_break_even:.1%})."
        )

        st.write(
            f"Operational HIGH threshold: "
            f"**{optimised_threshold} cycles**"
        )

        st.write(
            f"Recommendation: "
            f"**{recommendation}**"
        )

    with right:

        st.markdown(
            "#### Financial decision"
        )

        st.metric(
            "Risk-Adjusted Exposure",
            money(
                engine[
                    "risk_adjusted_failure_exposure"
                ]
            ),
        )

        st.metric(
            "Illustrative Net Benefit",
            money(
                engine[
                    "risk_adjusted_net_benefit"
                ]
            ),
        )

        st.metric(
            "Break-Even Probability",
            f"{base_break_even:.1%}",
        )

        st.write(
            "Economic action: "
            f"**{engine['economic_action']}**"
        )

        probability_gap = (
            probability
            - base_break_even
        )

        if probability_gap >= 0:

            st.success(
                "Estimated HIGH-risk probability is "
                f"{probability_gap:.1%} above the current "
                "break-even threshold."
            )

        else:

            st.info(
                "Estimated HIGH-risk probability is "
                f"{abs(probability_gap):.1%} below the current "
                "break-even threshold."
            )

    with st.expander(
        "Evaluation-only ground truth"
    ):

        prediction_error = (
            engine[
                "predicted_rul_capped"
            ]
            - engine[
                "actual_rul_capped"
            ]
        )

        eval_col1, eval_col2 = st.columns(
            2
        )

        eval_col1.metric(
            "Actual RUL",
            f"{engine['actual_rul_capped']:.0f} cycles",
        )

        eval_col2.metric(
            "Prediction Error",
            f"{prediction_error:+.1f} cycles",
        )

        st.caption(
            "These values are available because this is an "
            "evaluation dataset. A live system would not know "
            "future actual RUL."
        )

    st.warning(
        "Financial values use illustrative assumptions "
        "and are not demonstrated real-world savings."
    )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.subheader(
        "Model Performance"
    )

    st.caption(
        "Evaluation-only view of RUL accuracy, threshold behaviour "
        "and probability calibration."
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    col1.metric(
        "Capped Test MAE",
        f"{capped_mae:.2f} cycles",
    )

    col2.metric(
        "HIGH-Risk Recall",
        f"{high_recall:.1%}",
    )

    col3.metric(
        "ROC AUC",
        f"{roc_auc:.3f}",
    )

    col4.metric(
        "Brier Score",
        f"{brier_score:.4f}",
    )

    chart_col1, chart_col2 = st.columns(
        2
    )

    with chart_col1:

        st.markdown(
            "#### RUL prediction performance"
        )

        rul_fig = px.scatter(
            financial_results,
            x="actual_rul_capped",
            y="predicted_rul_capped",
            hover_data={
                "engine_id":
                    True,

                "actual_rul_capped":
                    ":.0f",

                "predicted_rul_capped":
                    ":.1f",

                "high_risk_probability":
                    ":.1%",
            },
            labels={
                "actual_rul_capped":
                    "Actual RUL (cycles)",

                "predicted_rul_capped":
                    "Predicted RUL (cycles)",

                "engine_id":
                    "Engine",
            },
            title=
                "Capped Actual vs Predicted RUL",
        )

        max_rul = max(
            float(
                financial_results[
                    "actual_rul_capped"
                ].max()
            ),
            float(
                financial_results[
                    "predicted_rul_capped"
                ].max()
            ),
        )

        rul_fig.add_trace(
            go.Scatter(
                x=[
                    0,
                    max_rul,
                ],
                y=[
                    0,
                    max_rul,
                ],
                mode=
                    "lines",
                name=
                    "Perfect prediction",
                line={
                    "dash":
                        "dash",

                    "color":
                        MUTED_TEXT,
                },
            )
        )

        rul_fig = configure_plot(
            rul_fig,
            height=430,
        )

        st.plotly_chart(
            rul_fig,
            use_container_width=True,
        )

    with chart_col2:

        st.markdown(
            "#### Probability calibration"
        )

        calibration_plot_data = (
            calibration_results
            .copy()
        )

        calibration_fig = go.Figure()

        calibration_fig.add_trace(
            go.Scatter(
                x=
                    calibration_plot_data[
                        "mean_predicted_probability"
                    ],
                y=
                    calibration_plot_data[
                        "actual_high_rate"
                    ],
                mode=
                    "lines+markers",
                name=
                    "Observed calibration",
                line={
                    "color":
                        "#60a5fa",
                },
                marker={
                    "size":
                        9,
                },
            )
        )

        calibration_fig.add_trace(
            go.Scatter(
                x=[
                    0,
                    1,
                ],
                y=[
                    0,
                    1,
                ],
                mode=
                    "lines",
                name=
                    "Perfect calibration",
                line={
                    "dash":
                        "dash",

                    "color":
                        MUTED_TEXT,
                },
            )
        )

        calibration_fig.update_layout(
            title=
                "HIGH-Risk Probability Calibration",
            xaxis_title=
                "Mean Predicted Probability",
            yaxis_title=
                "Observed HIGH Rate",
        )

        calibration_fig = configure_plot(
            calibration_fig,
            height=430,
        )

        st.plotly_chart(
            calibration_fig,
            use_container_width=True,
        )

    st.markdown(
        "#### Default vs optimised HIGH-risk threshold"
    )

    actual_high = (
        financial_results[
            "actual_rul_capped"
        ]
        <= ACTUAL_HIGH_THRESHOLD
    ).astype(
        int
    )

    default_high = (
        financial_results[
            "predicted_rul_capped"
        ]
        <= ACTUAL_HIGH_THRESHOLD
    ).astype(
        int
    )

    optimised_high = (
        financial_results[
            "predicted_rul_capped"
        ]
        <= optimised_threshold
    ).astype(
        int
    )

    comparison = pd.DataFrame(
        {
            "Threshold": [
                f"Default ({ACTUAL_HIGH_THRESHOLD} cycles)",
                f"Optimised ({optimised_threshold} cycles)",
            ],

            "Recall": [
                recall_score(
                    actual_high,
                    default_high,
                ),
                recall_score(
                    actual_high,
                    optimised_high,
                ),
            ],

            "Precision": [
                precision_score(
                    actual_high,
                    default_high,
                    zero_division=0,
                ),
                precision_score(
                    actual_high,
                    optimised_high,
                    zero_division=0,
                ),
            ],

            "HIGH-Risk Engines Flagged": [
                int(
                    default_high.sum()
                ),
                int(
                    optimised_high.sum()
                ),
            ],
        }
    )

    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Recall":
                st.column_config.NumberColumn(
                    format="percent",
                ),

            "Precision":
                st.column_config.NumberColumn(
                    format="percent",
                ),
        },
    )

    st.markdown(
        "#### Calibration data"
    )

    st.dataframe(
        calibration_results,
        hide_index=True,
        use_container_width=True,
        column_config={
            "mean_predicted_probability":
                st.column_config.NumberColumn(
                    "Mean Predicted Probability",
                    format="percent",
                ),

            "actual_high_rate":
                st.column_config.NumberColumn(
                    "Observed HIGH Rate",
                    format="percent",
                ),
        },
    )

    with st.expander(
        "Metric interpretation"
    ):

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

    st.subheader(
        "Financial Scenario Analysis"
    )

    st.caption(
        "Test how preventative maintenance and failure-cost assumptions "
        "change the illustrative economic decision threshold."
    )

    maintenance_options = sorted(
        sensitivity_results[
            "maintenance_cost"
        ].unique()
    )

    failure_options = sorted(
        sensitivity_results[
            "failure_cost"
        ].unique()
    )

    col1, col2 = st.columns(
        2
    )

    selected_maintenance_cost = (
        col1.selectbox(
            "Preventative maintenance cost",
            maintenance_options,
            index=(
                maintenance_options.index(
                    8000
                )
                if 8000
                in maintenance_options
                else 0
            ),
            format_func=
                money,
        )
    )

    selected_failure_cost = (
        col2.selectbox(
            "Unplanned failure cost",
            failure_options,
            index=(
                failure_options.index(
                    40000
                )
                if 40000
                in failure_options
                else 0
            ),
            format_func=
                money,
        )
    )

    scenario = (
        sensitivity_results[
            (
                sensitivity_results[
                    "maintenance_cost"
                ]
                == selected_maintenance_cost
            )
            &
            (
                sensitivity_results[
                    "failure_cost"
                ]
                == selected_failure_cost
            )
        ]
        .iloc[0]
    )

    st.markdown(
        "#### Selected scenario"
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    col1.metric(
        "Break-Even Probability",
        f"{scenario['break_even_probability']:.1%}",
    )

    col2.metric(
        "Engines Selected",
        int(
            scenario[
                "engines_selected"
            ]
        ),
    )

    col3.metric(
        "Maintenance Outlay",
        money(
            scenario[
                "maintenance_outlay"
            ]
        ),
    )

    col4.metric(
        "Illustrative Net Benefit",
        money(
            scenario[
                "risk_adjusted_net_benefit"
            ]
        ),
    )

    st.info(
        "The break-even probability is the estimated HIGH-risk "
        "probability at which risk-adjusted exposure equals the "
        "assumed preventative maintenance cost."
    )

    chart_col1, chart_col2 = st.columns(
        2
    )

    with chart_col1:

        st.markdown(
            "#### Net benefit vs failure cost"
        )

        line_data = (
            sensitivity_results[
                sensitivity_results[
                    "maintenance_cost"
                ]
                == selected_maintenance_cost
            ]
            .copy()
        )

        line_fig = px.line(
            line_data,
            x=
                "failure_cost",
            y=
                "risk_adjusted_net_benefit",
            markers=
                True,
            labels={
                "failure_cost":
                    "Failure Cost (£)",

                "risk_adjusted_net_benefit":
                    "Illustrative Net Benefit (£)",
            },
        )

        line_fig.update_traces(
            line={
                "color":
                    "#60a5fa",
            },
            marker={
                "size":
                    9,
            },
        )

        selected_line_value = float(
            scenario[
                "risk_adjusted_net_benefit"
            ]
        )

        line_fig.add_trace(
            go.Scatter(
                x=[
                    selected_failure_cost
                ],
                y=[
                    selected_line_value
                ],
                mode=
                    "markers",
                name=
                    "Selected scenario",
                marker={
                    "size":
                        16,

                    "symbol":
                        "diamond",

                    "color":
                        "#f59e0b",

                    "line": {
                        "width":
                            2,

                        "color":
                            "#f8fafc",
                    },
                },
            )
        )

        line_fig = configure_plot(
            line_fig,
            height=420,
        )

        st.plotly_chart(
            line_fig,
            use_container_width=True,
        )

    with chart_col2:

        st.markdown(
            "#### Scenario heatmap"
        )

        heatmap_data = (
            sensitivity_results
            .pivot(
                index=
                    "maintenance_cost",
                columns=
                    "failure_cost",
                values=
                    "risk_adjusted_net_benefit",
            )
            .sort_index()
        )

        heatmap_fig = px.imshow(
            heatmap_data,
            aspect=
                "auto",
            labels={
                "x":
                    "Failure Cost (£)",

                "y":
                    "Maintenance Cost (£)",

                "color":
                    "Illustrative Net Benefit (£)",
            },
            color_continuous_scale=
                "Blues",
        )

        heatmap_fig.add_trace(
            go.Scatter(
                x=[
                    selected_failure_cost
                ],
                y=[
                    selected_maintenance_cost
                ],
                mode=
                    "markers",
                name=
                    "Selected scenario",
                marker={
                    "size":
                        18,

                    "symbol":
                        "x",

                    "color":
                        "#f59e0b",

                    "line": {
                        "width":
                            2,
                    },
                },
            )
        )

        heatmap_fig = configure_plot(
            heatmap_fig,
            height=420,
        )

        st.plotly_chart(
            heatmap_fig,
            use_container_width=True,
        )

    st.markdown(
        "#### All cost scenarios"
    )

    scenarios_display = (
        sensitivity_results
        .copy()
    )

    scenarios_display[
        "maintenance_cost"
    ] = (
        scenarios_display[
            "maintenance_cost"
        ]
        .apply(
            money
        )
    )

    scenarios_display[
        "failure_cost"
    ] = (
        scenarios_display[
            "failure_cost"
        ]
        .apply(
            money
        )
    )

    scenarios_display[
        "break_even_probability"
    ] = (
        scenarios_display[
            "break_even_probability"
        ]
        .apply(
            lambda value:
            f"{value:.1%}"
        )
    )

    scenarios_display[
        "maintenance_outlay"
    ] = (
        scenarios_display[
            "maintenance_outlay"
        ]
        .apply(
            money
        )
    )

    scenarios_display[
        "risk_adjusted_exposure"
    ] = (
        scenarios_display[
            "risk_adjusted_exposure"
        ]
        .apply(
            money
        )
    )

    scenarios_display[
        "risk_adjusted_net_benefit"
    ] = (
        scenarios_display[
            "risk_adjusted_net_benefit"
        ]
        .apply(
            money
        )
    )

    scenarios_display = (
        scenarios_display
        .rename(
            columns={
                "maintenance_cost":
                    "Maintenance Cost",

                "failure_cost":
                    "Failure Cost",

                "break_even_probability":
                    "Break-Even Probability",

                "engines_selected":
                    "Engines Selected",

                "maintenance_outlay":
                    "Maintenance Outlay",

                "risk_adjusted_exposure":
                    "Risk-Adjusted Exposure",

                "risk_adjusted_net_benefit":
                    "Illustrative Net Benefit",
            }
        )
    )

    st.dataframe(
        scenarios_display,
        hide_index=True,
        use_container_width=True,
        height=560,
    )

    download_col1, download_col2 = st.columns(
        2
    )

    download_col1.download_button(
        "Download sensitivity analysis",
        data=csv_bytes(
            sensitivity_results
        ),
        file_name=
            "cost_sensitivity_analysis.csv",
        mime=
            "text/csv",
    )

    download_col2.download_button(
        "Download financial risk results",
        data=csv_bytes(
            financial_results
        ),
        file_name=
            "financial_risk_results.csv",
        mime=
            "text/csv",
    )

    st.warning(
        "Financial outputs are scenario-based and use "
        "illustrative cost assumptions rather than observed "
        "NASA maintenance or failure costs."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        NASA C-MAPSS FD001 · Portfolio project ·
        Financial assumptions are illustrative
    </div>
    """,
    unsafe_allow_html=True,
)

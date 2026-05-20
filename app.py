# ═══════════════════════════════════════════════════════════════════════════════
# IMPORT LIBRARIES
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
import time

from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, LabelEncoder,
)
from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False

matplotlib.use("Agg")  # non-interactive backend for Streamlit

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

# Neo-Brutalist Pastel Palette
MINT = "#A7DBD8"
LIME = "#BAFCA2"
YELLOW = "#FFDB58"
ORANGE = "#FFA07A"
PINK = "#FFC0CB"
PURPLE = "#C4A1FF"

WORKFLOW_STEPS = [
    {"label": "Home",              "icon": "🏠", "accent": YELLOW},
    {"label": "Dataset Overview",    "icon": "📤", "accent": MINT},
    {"label": "EDA",               "icon": "📊", "accent": PINK},
    {"label": "Feature Selection", "icon": "🎯", "accent": PURPLE},
    {"label": "Preprocessing",     "icon": "⚙️", "accent": ORANGE},
    {"label": "Training",          "icon": "🚀", "accent": LIME},
    {"label": "Evaluation",        "icon": "📈", "accent": YELLOW},
    {"label": "Prediction",        "icon": "🔮", "accent": MINT},
    {"label": "About",             "icon": "📚", "accent": PINK},
]

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STEPS
# ═══════════════════════════════════════════════════════════════════════════════

STEP_HOME = 0
STEP_DATASET = 1
STEP_EDA = 2
STEP_FEAT_SEL = 3
STEP_PREPROCESS = 4
STEP_TRAINING = 5
STEP_EVALUATION = 6
STEP_PREDICTION = 7
STEP_ABOUT = 8

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_session_state():
    """Initialize every session key the app will ever need."""
    defaults = {
        # Workflow control
        "current_step": STEP_HOME,
        "max_step": STEP_HOME,
        # Data
        "raw_data": None,
        "engineered_data": None,
        "target_col": "Revenue",
        # Feature selection
        "selected_features": None,
        "_fs_selection": None,
        # Preprocessing pipeline state
        "pp_working_df": None,
        "pp_encoded": False,
        "pp_capped": False,
        "pp_skew_fixed": False,
        "pp_split_done": False,
        "pp_scaled": False,
        "pp_smote_done": False,
        # Preprocessing snapshots (before/after)
        "_snap_enc_before": None,
        "_snap_enc_after": None,
        "_snap_enc_cols": None,
        "_snap_cap_before": None,
        "_snap_cap_after": None,
        "_snap_yj_before": None,
        "_snap_yj_after": None,
        "_snap_scale_before": None,
        "_snap_scale_after": None,
        "_snap_smote_before": None,
        "_snap_smote_after": None,
        "_snap_split_cfg": None,
        "_snap_dup_before": None,
        "_snap_dup_after": None,
        "X_train": None,
        "X_test": None,
        "y_train": None,
        "y_test": None,
        "scaler": None,
        "power_transformer": None,
        "label_encoder_visitor": None,
        "month_dummies_cols": None,
        # Training
        "trained_model": None,
        "model_name": None,
        # Evaluation
        "metrics": {},
        # Prediction history
        "prediction_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def unlock_step(step_num: int):
    """Unlock a workflow step (only advances forward)."""
    if step_num > st.session_state.max_step:
        st.session_state.max_step = step_num


def navigate_to(step_num: int):
    """Navigate to a step and trigger rerun."""
    st.session_state.current_step = step_num
    st.rerun()


def can_access(step_num: int) -> bool:
    """Check whether the user may visit a step."""
    if step_num == STEP_ABOUT:
        return True
    return step_num <= st.session_state.max_step


def reset_downstream(from_step: int):
    """Null all state produced at or after *from_step*."""
    if from_step <= STEP_DATASET:
        st.session_state.raw_data = None
    if from_step <= STEP_FEAT_SEL:
        st.session_state.selected_features = None
        st.session_state.engineered_data = None
    if from_step <= STEP_PREPROCESS:
        for k in ("X_train", "X_test", "y_train", "y_test",
                   "scaler", "power_transformer",
                   "label_encoder_visitor", "month_dummies_cols"):
            st.session_state[k] = None
    if from_step <= STEP_TRAINING:
        st.session_state.trained_model = None
        st.session_state.model_name = None
    if from_step <= STEP_EVALUATION:
        st.session_state.metrics = {}
    if from_step <= STEP_PREDICTION:
        st.session_state.prediction_history = []
    st.session_state.max_step = max(from_step - 1, STEP_HOME)

# ═══════════════════════════════════════════════════════════════════════════════
# CSS  
# ═══════════════════════════════════════════════════════════════════════════════

def inject_global_css():
    """Inject the full neo-brutalist design system with fixed dataframe overflow."""
    st.markdown("""
    <style>
    /* ── Imports ─────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Syne:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

    /* ── Root tokens ─────────────────────────────────────────────── */
    :root {
        --canvas:       #EFE9DC; /* Warm paper-like beige */
        --surface:      #F7F3EB; /* Soft warm off-white for containers */
        --border:       #000000;
        --text-primary: #000000;
        --text-secondary: #000000;
        --mint:   #A7DBD8;
        --lime:   #BAFCA2;
        --yellow: #FFDB58;
        --orange: #FFA07A;
        --pink:   #FFC0CB;
        --purple: #C4A1FF;
    }

    /* ── Global ──────────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--canvas) !important;
        background-image: radial-gradient(var(--border) 2px, transparent 2px);
        background-size: 24px 24px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: var(--text-primary);
    }
    [data-testid="stHeader"] { background: transparent !important; }

    /* ── Typography ──────────────────────────────────────────────── */
    h1, h2, h3, h4, h5, h6 { font-family: 'Syne', sans-serif !important; text-transform: uppercase; color: var(--text-primary); }
    h1 { font-size:4rem; font-weight:800; letter-spacing:-0.04em; margin-bottom: 2rem; line-height: 1.1; }
    h2 { font-size:2.8rem; font-weight:800; letter-spacing:-0.03em; margin-top: 2rem; margin-bottom: 1.5rem; line-height: 1.1; border-bottom: none; }
    h3 { font-size:1.8rem; font-weight:800; margin-bottom: 1rem; }
    
    p, li, label { font-family:'Plus Jakarta Sans', sans-serif; font-weight: 600; font-size: 1.15rem; color: var(--text-primary); line-height: 1.6; }
    code, [data-testid="stMetricValue"] {
        font-family:'IBM Plex Mono', monospace !important;
    }
    
    /* Exempt markdown blocks that contain our custom cards from double-framing */
    .neo-card p, .neo-hero p, .neo-result p, .placeholder-box p {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        margin-bottom: 0 !important;
    }

    /* ── Sidebar ─────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 5px solid var(--border);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
        font-weight: 700;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
        border-width: 4px !important;
    }
    
    /* ── Buttons (General Neo-Brutalism style) ───────────────────── */
    button[kind="primary"], button[kind="secondary"] {
        border: 4px solid var(--border) !important;
        border-radius: 0 !important;
        padding: 1.5rem 2rem !important;
        font-weight: 800 !important;
        font-size: 1.25rem !important;
        font-family: 'Syne', sans-serif !important;
        text-transform: uppercase;
        box-shadow: 6px 6px 0px var(--border) !important;
        transition: all 0.1s ease-in-out;
        cursor: pointer;
        width: 100% !important;
        min-height: 4.5rem !important;
        white-space: normal !important;
        word-wrap: break-word !important;
        color: var(--border) !important;
    }
    button[kind="primary"]:active, button[kind="secondary"]:active {
        box-shadow: 2px 2px 0px var(--border) !important;
        transform: translateY(0);
    }
    button:disabled {
        opacity: 1 !important;
        background-color: var(--surface) !important;
        color: #888888 !important;
        cursor: not-allowed !important;
        box-shadow: none !important;
        transform: none !important;
    }

    /* Primary Buttons (Next, Confirm, Predict) -> Yellow to Lime */
    button[kind="primary"] {
        background-color: var(--yellow) !important;
    }
    button[kind="primary"]:hover:not(:disabled) {
        background-color: var(--lime) !important;
        box-shadow: 10px 10px 0px var(--border) !important;
        transform: translate(-4px, -4px);
    }

    /* Secondary Buttons (Back, Cancel) -> Pink to Orange */
    button[kind="secondary"] {
        background-color: var(--pink) !important;
    }
    button[kind="secondary"]:hover:not(:disabled) {
        background-color: var(--orange) !important;
        box-shadow: 10px 10px 0px var(--border) !important;
        transform: translate(-4px, -4px);
    }

    /* Sidebar Buttons Specific Override -> White to Lime */
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #FFFFFF !important;
        padding: 1rem !important;
        min-height: 3rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1rem !important;
        text-align: left !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover:not(:disabled) {
        background-color: var(--lime) !important;
        box-shadow: 6px 6px 0px var(--border) !important;
        transform: translate(-2px, -2px);
    }

    /* ── Metrics & Visuals (Fixed for DataFrames) ─────────────────── */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        color: var(--text-primary) !important;
    }
                
    [class*="st-key-neo-"] {
        background-color: #F7F3EB !important;
        border: 4px solid #000000 !important;
        box-shadow: 8px 8px 0px #000000 !important;
        border-radius: 0 !important;
        padding: 1.5rem !important;
        margin-bottom: 2rem !important;
    }

    /* Komponen standard dengan padding utuh */
    [data-testid="metric-container"], [data-testid="stFileUploader"], [data-testid="stImage"] {
        background: var(--surface) !important;
        border: 4px solid var(--border) !important;
        padding: 1.25rem !important;
        box-shadow: 6px 6px 0px var(--border) !important;
        margin-bottom: 1.5rem !important;
        border-radius: 0 !important;
    }

    /* FIX UNTUK EXPANDER */
    [data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 4px solid var(--border) !important;
        padding: 0.25rem 0.5rem !important;
        box-shadow: 6px 6px 0px var(--border) !important;
        margin-bottom: 1.5rem !important;
        border-radius: 0 !important;
    }

    /* FIX UNTUK DATAFRAME */
    [data-testid="stDataFrame"] {
        background: var(--surface) !important;
        border: 4px solid var(--border) !important;
        padding: 0px !important;
        box-shadow: 6px 6px 0px var(--border) !important;
        margin-bottom: 1.5rem !important;
        border-radius: 0 !important;
    }
    
    [data-testid="stDataFrame"] > div {
        border-radius: 0 !important;
    }

    /* ── Inputs / Selectbox ──────────────────────────────────────── */
    .stSelectbox [data-baseweb="select"],
    .stMultiSelect [data-baseweb="select"],
    .stNumberInput input, .stTextInput input,
    .stSlider > div, .stCheckbox > div, [data-testid="stRadio"] > div {
        background-color: var(--surface) !important;
        border: 4px solid var(--border) !important;
        border-radius: 0 !important;
        box-shadow: 4px 4px 0px var(--border) !important;
        padding: 0.5rem;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        color: var(--text-primary) !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stCheckbox label, [data-testid="stRadio"] label {
        font-weight: 800 !important;
    }

    /* ── Tabs ─────────────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 5px solid var(--border);
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        border: 4px solid var(--border);
        border-bottom: none;
        background: var(--surface);
        font-weight: 800;
        font-size: 1.2rem;
        padding: 1rem 2rem;
        margin-right: 0.5rem;
        box-shadow: 4px 4px 0px var(--border);
        border-radius: 0;
    }
    .stTabs [aria-selected="true"] {
        background: var(--purple) !important;
        color: var(--border) !important;
        box-shadow: 6px 6px 0px var(--border);
        transform: translateY(-4px);
    }

    /* ── Expanders ────────────────────────────────────────────────── */
    [data-testid="stExpander"] summary {
        font-weight: 800 !important;
        font-size: 1.2rem !important;
        padding: 1rem !important;
    }

    /* ── Alerts ───────────────────────────────────────────────────── */
    .stAlert {
        background-color: var(--mint) !important;
        border: 4px solid var(--border) !important;
        box-shadow: 6px 6px 0px var(--border) !important;
        padding: 1.5rem !important;
        font-weight: 800;
        color: #000000 !important;
    }

    /* ── Progress bar ────────────────────────────────────────────── */
    .stProgress > div > div > div {
        background: var(--text-primary) !important;
        border-radius: 0;
    }
    .stProgress > div > div {
        background: #FFFFFF;
        border: 4px solid var(--border);
        border-radius: 0;
        height: 28px;
    }

    /* ── Section Containers (Container-Container) ────────────────── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: var(--surface) !important;
        border: 4px solid var(--border) !important;
        box-shadow: 8px 8px 0px var(--border) !important;
        padding: 2rem !important;
        margin-bottom: 2rem !important;
        border-radius: 0 !important;
    }
    
    /* ── Helper classes ──────────────────────────────────────────── */
    .neo-card {
        background: #FFFFFF;
        border: 4px solid #000000;
        border-radius: 0;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 10px 10px 0px #000000;
    }
    .neo-hero {
        background: var(--lime);
        border: 4px solid #000000;
        border-radius: 0;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 8px 8px 0px #000000;
    }
    .neo-hero h1 { margin-bottom: 0.25rem; font-size: 3rem; }
    .neo-hero p  { margin: 0; font-size: 1.25rem; font-weight: 800; }

    .neo-result {
        border: 4px solid #000000;
        border-radius: 0;
        padding: 2rem 3rem;
        box-shadow: 10px 10px 0px #000000;
        text-align: left;
    }
    .neo-result h2 { margin-bottom: 0.5rem; font-size: 3rem; }

    .placeholder-box {
        background: #FFFFFF;
        border: 4px solid var(--border);
        border-radius: 0;
        padding: 4rem 3rem;
        text-align: center;
        color: var(--text-primary);
        margin: 3rem 0;
        box-shadow: 10px 10px 0px var(--border);
        font-weight: 800;
    }
    .placeholder-box h3 { color: var(--text-primary); font-size: 2.5rem; }
    </style>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CHART  
# ═══════════════════════════════════════════════════════════════════════════════

_SERIES_COLORS = [MINT, LIME, YELLOW, ORANGE, PINK, PURPLE]


def _style_ax(ax, title="", xlabel="", ylabel=""):
    """Apply neo-brutalist styling to a matplotlib Axes."""
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontweight="bold", fontsize=10)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", color="#E0E0D8", linewidth=0.5)
    ax.set_facecolor("none")


def _detect_columns(df):
    """Return (numerical_cols, categorical_cols) lists.
    Boolean columns are excluded from both lists — they are handled separately.
    """
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return num, cat


def _safe_target(df):
    """Return the target column name from session state, or None."""
    t = st.session_state.get("target_col")
    if t and t in df.columns:
        return t
    return None

# ═══════════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def hero(title: str, subtitle: str, color: str):
    """Render a page hero banner."""
    st.markdown(
        f'<div class="neo-hero" style="background:{color};">'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

def section_title(title: str, color: str = LIME):
    """Render a neo-brutalist 'sticker' title for a section."""
    st.markdown(
        f'<div style="margin-top: 1.5rem; margin-bottom: 1.5rem;">'
        f'<h2 style="display: inline-block; background-color: {color}; '
        f'border: 4px solid #000; padding: 0.25rem 1rem; box-shadow: 6px 6px 0px #000; '
        f'margin: 0; width: fit-content; font-size: 2rem;">{title}</h2>'
        f'</div>',
        unsafe_allow_html=True
    )


def card(content: str, color: str = "#FFFFFF"):
    """Render a styled card."""
    st.markdown(
        f'<div class="neo-card" style="background:{color};">{content}</div>',
        unsafe_allow_html=True,
    )


def placeholder_page(step_idx: int, description: str):
    """Render a polished placeholder for an unfinished page."""
    meta = WORKFLOW_STEPS[step_idx]
    hero(
        f'{meta["icon"]} {meta["label"]}',
        "This section is under construction and will be available in the next implementation phase.",
        meta["accent"],
    )
    st.markdown(
        f'<div class="placeholder-box">'
        f'<h3>{meta["icon"]}  Coming Soon</h3>'
        f'<p>{description}</p></div>',
        unsafe_allow_html=True,
    )
    # Footer navigation
    render_page_nav(step_idx)


def render_page_nav(step_idx: int):
    """Render ← Back / Next → buttons at the bottom of each page."""
    cols = st.columns([1, 1, 1])

    # Back button (Secondary -> Pink/Orange hover)
    if step_idx > STEP_HOME:
        prev_label = WORKFLOW_STEPS[step_idx - 1]["label"]
        if cols[0].button(f"← {prev_label}", key=f"nav_back_{step_idx}", use_container_width=True, type="secondary"):
            navigate_to(step_idx - 1)

    # Next button (Primary -> Yellow/Lime hover)
    if step_idx < STEP_ABOUT:
        nxt = step_idx + 1
        nxt_label = WORKFLOW_STEPS[nxt]["label"]
        disabled = not can_access(nxt)
        if cols[2].button(f"{nxt_label} →", key=f"nav_next_{step_idx}", disabled=disabled, use_container_width=True, type="primary"):
            navigate_to(nxt)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    """Build the sidebar navigation spine."""
    with st.sidebar:
        st.markdown("## ML PIPELINE")
        st.markdown("---")

        # Progress bar (exclude About page from progress)
        pipeline_steps = len(WORKFLOW_STEPS) - 1
        active_step = st.session_state.max_step if st.session_state.current_step == STEP_ABOUT else st.session_state.current_step
        progress = active_step / (pipeline_steps - 1)
        
        st.progress(progress)
        st.caption(f"PHASE {active_step + 1} OF {pipeline_steps}")
        st.markdown("---")

        for i, step in enumerate(WORKFLOW_STEPS):
            if i == STEP_ABOUT:
                st.markdown("---")

            is_current = i == st.session_state.current_step
            accessible = can_access(i)

            # Define visual status
            if i == STEP_ABOUT:
                icon = "📚"
            elif i < st.session_state.max_step:
                icon = "✅"
            elif is_current:
                icon = "📍"
            elif accessible:
                icon = "▶️"
            else:
                icon = "🔒"

            # Add arrows to current step for emphasis
            label_prefix = "→ " if is_current else ""
            label = f"{label_prefix}{icon} {step['label']}"

            if st.button(label, key=f"sb_{i}", disabled=not accessible,
                         use_container_width=True, type="secondary"):
                if accessible and i != st.session_state.current_step:
                    navigate_to(i)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ═══════════════════════════════════════════════════════════════════════════════

def render_home():
    hero("Online Shoppers Prediction Engine",
         "A sequential ML pipeline to predict purchase intent from shopper behaviour.",
         WORKFLOW_STEPS[STEP_HOME]["accent"])

    with st.container(key="neo-home-how", border=True):
        section_title("How It Works", LIME)
    
        c1, c2, c3 = st.columns(3)
        with c1:
            card("<h3> Dataset</h3><p>Explore the bundled dataset and understand its structure.</p>", PURPLE)
        with c2:
            card("<h3> Explore & Build</h3><p>Visualise patterns, engineer features, train models.</p>", ORANGE)
        with c3:
            card("<h3> Predict</h3><p>Run inference on new data using the trained pipeline.</p>", PINK)
    
        st.info(
            "**Dataset:** Online Shoppers Purchasing Intention Dataset  |  "
            "**Accessed at:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)   "
        )
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("  Begin Workflow", use_container_width=True, key="home_start", type="primary"):
            unlock_step(STEP_DATASET)
            navigate_to(STEP_DATASET)
    with c2:
        if st.button("  About This App", use_container_width=True, key="home_about", type="secondary"):
                navigate_to(STEP_ABOUT)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def render_dataset():
    hero("Dataset Overview",
         "Explore the Online Shoppers Purchasing Intention dataset.",
         WORKFLOW_STEPS[STEP_DATASET]["accent"])

    # Auto-load bundled dataset
    if st.session_state.raw_data is None:
        try:
            st.session_state.raw_data = _load_csv("online_shoppers_intention.csv")
        except FileNotFoundError:
            st.error("Sample file `online_shoppers_intention.csv` not found.")
            return

    df = st.session_state.raw_data

    # ── Context & background ─────────────────────────────────────
    with st.container(key="neo-ds-about", border=True):
        section_title("About This Dataset", PURPLE)
        st.markdown(
            "The **Online Shoppers Purchasing Intention** dataset captures browsing "
            "sessions from an e-commerce website.  Each row represents one user session "
            "and records page-visit metrics, temporal attributes, and traffic metadata.  "
            "The goal is to predict whether the session ends in a **purchase** "
            "(the `Revenue` column)."
        )
        c1, c2 = st.columns(2)
    with c1:
        card(
            "<h3> Target Variable</h3>"
            "<p><b>Revenue</b> — binary (True / False).<br>"
            "<b>True</b> = the visitor completed a purchase.<br>"
            "<b>False</b> = the visitor left without buying.</p>",
            LIME,
        )
    with c2:
        card(
            """
            <h3>Source</h3>
            <p>
                Sakar, C.O. et al. (2018) — <i>Online Shoppers Purchasing Intention Dataset</i>, 
                <b>Accessed at:</b> <a href="https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset" target="_blank">UCI ML Repository</a> <br>
                ~12,330 sessions &middot; 18 features &middot; binary classification.
            </p>
            """,
            ORANGE,
        )

    with st.expander("📖 Feature Glossary"):
        glossary = {
            "Administrative / Informational / ProductRelated": "Number of pages visited in each category.",
            "Administrative_Duration / Informational_Duration / ProductRelated_Duration": "Total seconds spent on pages of that category.",
            "BounceRates": "Average bounce rate of pages visited (% who left after one page).",
            "ExitRates": "Average exit rate of pages visited (% of last views that were exits).",
            "PageValues": "Average value of a page visited before completing an e-commerce transaction.",
            "SpecialDay": "Closeness of the session date to a special day (e.g. Valentine's, Mother's Day). 0 = not close.",
            "Month": "Month of the session.",
            "OperatingSystems / Browser / Region / TrafficType": "Categorical IDs for environment and traffic source.",
            "VisitorType": "Returning visitor, new visitor, or other.",
            "Weekend": "Whether the session occurred on a weekend.",
            "Revenue": "Target — whether the session ended with a purchase.",
        }
        for feat, desc in glossary.items():
            st.markdown(f"**{feat}** — {desc}")

    # ── Overview metrics ─────────────────────────────────────────
    with st.container(key="neo-ds-metrics", border=True):
        section_title("Dataset Metrics", YELLOW)
        num_cols, cat_cols = _detect_columns(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", str(len(df.columns)))
        c3.metric("Missing Values", str(df.isnull().sum().sum()))
        c4.metric("Duplicates", str(df.duplicated().sum()))

    # ── Preview ──────────────────────────────────────────────────
    with st.container(key="neo-ds-preview", border=True):
        section_title("Preview (first 5 rows)", PINK)
        st.dataframe(df.head(5), use_container_width=True)

    # ── Scrollable dtypes ────────────────────────────────────────
    with st.container(key="neo-ds-dtypes", border=True):
        section_title("Column Data Types", PURPLE)
        st.dataframe(
            df.dtypes.rename("dtype").to_frame(),
        )

    # ── Confirm & navigate ───────────────────────────────────────
    unlock_step(STEP_EDA)
    render_page_nav(STEP_DATASET)

# ═══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER PAGES
# ═══════════════════════════════════════════════════════════════════════════════

def render_eda():
    hero("Exploratory Data Analysis",
         "Understand the dataset through interactive visualisations.",
         WORKFLOW_STEPS[STEP_EDA]["accent"])

    df = st.session_state.raw_data
    if df is None:
        st.warning("No dataset loaded. Please go back to **Dataset Upload**.")
        render_page_nav(STEP_EDA)
        return

    target = _safe_target(df)
    num_cols, cat_cols = _detect_columns(df)

    # ── 1. Dataset overview (coloured cards) ─────────────────────
    with st.container(key="neo-eda-overview", border=True):
        section_title("Dataset Overview", YELLOW)
        
        c1, c2, c3 = st.columns(3)
        with c1:
            card(f"<h3> Shape</h3><p><b>{len(df):,}</b> rows × <b>{len(df.columns)}</b> columns</p>", LIME)
        with c2:
            card(f"<h3> Missing Values</h3><p><b>{df.isnull().sum().sum()}</b> total missing cells</p>", YELLOW)
        with c3:
            card(f"<h3> Duplicates</h3><p><b>{df.duplicated().sum()}</b> duplicate rows</p>", PINK)
    
        c4, c5 = st.columns(2)
        with c4:
            card(f"<h3> Numerical</h3><p><b>{len(num_cols)}</b> numeric features</p>", MINT)
        with c5:
            card(f"<h3> Categorical</h3><p><b>{len(cat_cols)}</b> categorical features</p>", PURPLE)
    
        with st.expander(" Statistical Summary"):
            st.dataframe(df.describe().T, use_container_width=True)

    # ── 2. Duplicate handling (always before/after) ──────────────
    with st.container(key="neo-eda-duplicates", border=True):
        section_title("Duplicate Handling", PURPLE)
    
        dup_count = df.duplicated().sum()
    
        # Show before/after snapshot if duplicates were already handled
        if st.session_state._snap_dup_before is not None:
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Rows Before", f"{st.session_state._snap_dup_before:,}")
            dc2.metric("Rows After", f"{st.session_state._snap_dup_after:,}")
            dc3.metric("Duplicates Removed", str(st.session_state._snap_dup_before - st.session_state._snap_dup_after))
            st.success(" Duplicates already handled.")
        elif dup_count > 0:
            st.warning(f"**{dup_count}** duplicate rows detected out of **{len(df):,}** total rows.")
            if st.button("  Remove Duplicates", key="eda_rm_dup", type="primary"):
                with st.spinner("Removing duplicates…"):
                    rows_before = len(df)
                    cleaned = df.drop_duplicates().reset_index(drop=True)
                    st.session_state.raw_data = cleaned
                    st.session_state._snap_dup_before = rows_before
                    st.session_state._snap_dup_after = len(cleaned)
                st.rerun()
        else:
            dc1, dc2, dc3 = st.columns(3)
            dc1.metric("Rows Before", f"{len(df):,}")
            dc2.metric("Rows After", f"{len(df):,}")
            dc3.metric("Duplicates Removed", "0")
            st.success("No duplicate rows found — dataset is clean.")

    # ── 3. Target distribution (standalone, with explanation) ────
    with st.container(key="neo-eda-target", border=True):
        section_title("Target Distribution", ORANGE)
    
        if not target:
            st.info("No target column set.")
        else:
            counts = df[target].value_counts()
            total = counts.sum()
            minority_pct = (counts.min() / total * 100)
    
            st.markdown(
                f"The target column **`{target}`** indicates whether a shopping session "
                f"ended in a purchase.\n\n"
                f"- **`True`** (Revenue = 1) — the visitor **completed a purchase**.\n"
                f"- **`False`** (Revenue = 0) — the visitor **left without buying**.\n\n"
                f"The minority class represents **{minority_pct:.1f}%** of all sessions.  "
                f"{'This is a **significant class imbalance** — the model may need oversampling (e.g. SMOTE) during preprocessing to learn the minority class effectively.' if minority_pct < 30 else 'The classes are reasonably balanced.'}"
            )
    
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(5, 2.5))
                fig.patch.set_alpha(0.0)
                ax.bar(counts.index.astype(str), counts.values,
                       color=[_SERIES_COLORS[0], _SERIES_COLORS[1]],
                       edgecolor="black", linewidth=2)
                _style_ax(ax, f"{target} Counts", target, "Count")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            with c2:
                fig, ax = plt.subplots(figsize=(2.5, 2.5))
                fig.patch.set_alpha(0.0)
                ax.pie(counts.values, labels=counts.index.astype(str),
                       autopct="%1.1f%%", startangle=90,
                       colors=[_SERIES_COLORS[0], _SERIES_COLORS[1]],
                       wedgeprops={"edgecolor": "black", "linewidth": 2})
                _style_ax(ax, f"{target} Proportion")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ── Tabs for analysis sections ───────────────────────────────
    with st.container(key="neo-eda-analysis", border=True):
        section_title("Detailed Analysis", PURPLE)
        tabs = st.tabs(["Univariate", "Outliers", "Bivariate", "Correlation"])
    
        # ── 4. Univariate ────────────────────────────────────────────
        with tabs[0]:
            section_title("Univariate Analysis", LIME)
            if not num_cols:
                st.info("No numerical columns detected.")
            else:
                sel = st.selectbox("Select feature", num_cols, key="eda_uni_sel")
                c1, c2 = st.columns(2)
                with c1:
                    fig, ax = plt.subplots(figsize=(5, 2.5))
                    fig.patch.set_alpha(0.0)
                    ax.hist(df[sel].dropna(), bins=40, color=_SERIES_COLORS[2],
                            edgecolor="black", linewidth=1)
                    _style_ax(ax, f"Distribution of {sel}", sel, "Frequency")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                with c2:
                    fig, ax = plt.subplots(figsize=(5, 2.5))
                    fig.patch.set_alpha(0.0)
                    ax.barh(["Skewness", "Kurtosis"],
                            [df[sel].skew(), df[sel].kurtosis()],
                            color=[_SERIES_COLORS[3], _SERIES_COLORS[4]],
                            edgecolor="black", linewidth=1.5)
                    _style_ax(ax, f"Shape Metrics — {sel}")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
    
        # ── 5. Outliers ──────────────────────────────────────────────
        with tabs[1]:
            section_title("Outlier Analysis", PINK)
            if not num_cols:
                st.info("No numerical columns detected.")
            else:
                sel_o = st.selectbox("Select feature", num_cols, key="eda_out_sel")
                fig, ax = plt.subplots(figsize=(5, 2))
                fig.patch.set_alpha(0.0)
                bp = ax.boxplot(df[sel_o].dropna(), vert=False, patch_artist=True,
                                boxprops=dict(facecolor=_SERIES_COLORS[5],
                                              edgecolor="black", linewidth=2),
                                medianprops=dict(color="black", linewidth=2),
                                whiskerprops=dict(linewidth=1.5),
                                flierprops=dict(marker="o", markerfacecolor="#FF6B6B",
                                                markersize=5, markeredgecolor="black"))
                _style_ax(ax, f"Boxplot — {sel_o}", sel_o)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption("Red dots indicate potential outliers beyond 1.5 × IQR.")
    
        # ── 6. Bivariate ─────────────────────────────────────────────
        with tabs[2]:
            section_title("Bivariate Analysis", YELLOW)
            biv_type = st.radio("Comparison type",
                                ["Numerical vs Numerical", "Categorical vs Target"],
                                horizontal=True, key="eda_biv_type")
            if biv_type == "Numerical vs Numerical":
                if len(num_cols) < 2:
                    st.info("Need at least two numerical columns.")
                else:
                    bc1, bc2 = st.columns(2)
                    fx = bc1.selectbox("X-axis", num_cols, key="biv_x")
                    fy = bc2.selectbox("Y-axis", num_cols, index=min(1, len(num_cols)-1), key="biv_y")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    fig.patch.set_alpha(0.0)
                    ax.scatter(df[fx], df[fy], alpha=0.45, edgecolors="black",
                               linewidth=0.5, color=_SERIES_COLORS[0], s=30)
                    _style_ax(ax, f"{fx} vs {fy}", fx, fy)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                if not cat_cols or not target:
                    st.info("Need categorical columns and a target variable.")
                else:
                    sel_cat = st.selectbox("Categorical feature", cat_cols, key="biv_cat")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    fig.patch.set_alpha(0.0)
                    ct = df.groupby([sel_cat, target]).size().unstack(fill_value=0)
                    ct.plot.bar(ax=ax, edgecolor="black", linewidth=1,
                                color=[_SERIES_COLORS[1], _SERIES_COLORS[3]])
                    _style_ax(ax, f"{sel_cat} by {target}", sel_cat, "Count")
                    ax.legend(title=target)
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
    
        # ── 7. Correlation ───────────────────────────────────────────
        with tabs[3]:
            section_title("Correlation Matrix", ORANGE)
            if len(num_cols) < 2:
                st.info("Need at least two numerical columns.")
            else:
                corr = df[num_cols].corr()
                fig, ax = plt.subplots(figsize=(8, 8))
                fig.patch.set_alpha(0.0)
                sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                            linewidths=0.5, linecolor="black", square=True,
                            cbar_kws={"shrink": 0.8})
                _style_ax(ax, "Correlation Heatmap")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ── Footer ───────────────────────────────────────────────────
    unlock_step(STEP_FEAT_SEL)

    render_page_nav(STEP_EDA)


def render_feature_selection():
    hero("Feature Selection",
         "Choose which features to include in the model.",
         WORKFLOW_STEPS[STEP_FEAT_SEL]["accent"])

    df = st.session_state.raw_data
    if df is None:
        st.warning("No dataset loaded.")
        render_page_nav(STEP_FEAT_SEL)
        return

    target = _safe_target(df)
    if not target:
        st.error("No target column configured in session state.")
        render_page_nav(STEP_FEAT_SEL)
        return

    all_features = [c for c in df.columns if c != target]
    num_cols, cat_cols = _detect_columns(df.drop(columns=[target]))

    # Initialise widget key on first visit
    if "fs_multiselect" not in st.session_state:
        if st.session_state._fs_selection is None:
            if st.session_state.selected_features is not None:
                init = [f for f in st.session_state.selected_features if f in all_features]
            else:
                init = list(all_features)
        else:
            init = st.session_state._fs_selection
        st.session_state.fs_multiselect = init

    # ── Quick-select toolbar & Feature Selection ─────────────────
    with st.container(key="neo-fs-select", border=True):
        section_title("1 · Select Features", YELLOW)
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            if st.button("  Select All", key="fs_all", use_container_width=True, type="primary"):
                st.session_state.fs_multiselect = list(all_features)
                st.rerun()
        with qc2:
            if st.button("  Numerical Only", key="fs_num", use_container_width=True, type="primary"):
                st.session_state.fs_multiselect = list(num_cols)
                st.rerun()
        with qc3:
            if st.button("  Deselect All", key="fs_none", use_container_width=True, type="secondary"):
                st.session_state.fs_multiselect = []
                st.rerun()
    
        # ── Single multiselect (sole source of truth) ────────────────
        selected = st.multiselect(
            "Selected Features",
            options=all_features,
            key="fs_multiselect",
        )
        st.caption("Add or remove features directly from the selection field.")
        st.session_state._fs_selection = selected
    
        # ── Compact summary metrics ──────────────────────────────────
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected", str(len(selected)))
        c2.metric("Available", str(len(all_features)))
        c3.metric("Excluded", str(len(all_features) - len(selected)))
    
        if not selected:
            st.warning("Select at least one feature to proceed.")
    
    # ── Confirm & navigate ───────────────────────────────────────
    with st.container(key="neo-fs-confirm", border=True):
        section_title("2 · Confirm", MINT)
        if st.button("  Confirm Features", use_container_width=True,
            key="fs_confirm", disabled=len(selected) == 0, type="primary"):
                st.session_state.selected_features = selected
                st.session_state.engineered_data = df[selected + [target]].copy()
                for k in ("pp_working_df", "pp_encoded", "pp_capped",
                          "pp_skew_fixed", "pp_split_done", "pp_scaled", "pp_smote_done"):
                    st.session_state[k] = None if "df" in k else False
                st.session_state.pp_working_df = None
                for k in ("_snap_enc_before", "_snap_enc_after", "_snap_enc_cols",
                          "_snap_cap_before", "_snap_cap_after",
                          "_snap_yj_before", "_snap_yj_after",
                          "_snap_scale_before", "_snap_scale_after",
                          "_snap_smote_before", "_snap_smote_after",
                          "_snap_split_cfg"):
                    st.session_state[k] = None
                unlock_step(STEP_PREPROCESS)
                st.success(" Features confirmed! You may now proceed to Preprocessing.")

    render_page_nav(STEP_FEAT_SEL)

def render_preprocessing():
    hero("Preprocessing Pipeline",
         "Execute each transformation step manually. You control the pipeline.",
         WORKFLOW_STEPS[STEP_PREPROCESS]["accent"])

    df = st.session_state.engineered_data
    if df is None:
        df = st.session_state.raw_data
    if df is None:
        st.warning("No dataset available. Complete earlier steps first.")
        render_page_nav(STEP_PREPROCESS)
        return

    target = _safe_target(df)
    if not target:
        st.error("No target column detected. Go back to Feature Selection.")
        render_page_nav(STEP_PREPROCESS)
        return

    if st.session_state.pp_working_df is None:
        st.session_state.pp_working_df = df.copy()

    working = st.session_state.pp_working_df
    num_cols, cat_cols = _detect_columns(working)
    feat_cols = [c for c in num_cols if c != target]

    # ═════════════════════════════════════════════════════════════
    # 1 · CATEGORICAL ENCODING
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-pp-encoding", border=True):
        section_title("1 · Categorical Encoding", YELLOW)
    
        # Detect ONLY object/category columns (exclude bool and target)
        enc_candidates = [
            c for c in working.select_dtypes(include=["object", "category"]).columns
            if c != target
        ]
    
        if st.session_state.pp_encoded:
            st.success(" Encoding already applied.")
            if st.session_state._snap_enc_before is not None:
                with st.expander("View Before / After Comparison"):
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.markdown("<h3>Before</h3>", unsafe_allow_html=True)
                        st.dataframe(st.session_state._snap_enc_before, use_container_width=True)
                    with bc2:
                        st.markdown("<h3>After</h3>", unsafe_allow_html=True)
                        st.dataframe(st.session_state._snap_enc_after, use_container_width=True)
                    st.caption(f"Encoded columns: {', '.join(st.session_state._snap_enc_cols or [])}")
        elif not enc_candidates:
            st.success("No categorical columns to encode (boolean columns are handled separately).")
        else:
            enc_method = st.selectbox("Encoding strategy",
                                      ["Label Encoding", "Frequency Encoding"],
                                      key="pp_enc")
            enc_cols = st.multiselect("Columns to encode", enc_candidates,
                                      default=enc_candidates, key="pp_enc_cols")
    
            if enc_cols:
                with st.expander("Preview (before)"):
                    st.dataframe(working[enc_cols].head(8), use_container_width=True)
    
            if st.button("  Apply Encoding", key="pp_run_enc", use_container_width=True, type="primary"):
                if not enc_cols:
                    st.warning("Select at least one column.")
                else:
                    with st.spinner("Encoding…"):
                        snap_before = working[enc_cols].head(8).copy()
                        w = working.copy()
                        if enc_method == "Label Encoding":
                            encoders = {}
                            for c in enc_cols:
                                le = LabelEncoder()
                                w[c] = le.fit_transform(w[c].astype(str))
                                encoders[c] = le
                            st.session_state.label_encoder_visitor = encoders
                        else:
                            for c in enc_cols:
                                freq = w[c].value_counts(normalize=True)
                                w[c] = w[c].map(freq).astype(float)
                        st.session_state._snap_enc_before = snap_before
                        st.session_state._snap_enc_after = w[enc_cols].head(8).copy()
                        st.session_state._snap_enc_cols = list(enc_cols)
                        st.session_state.pp_working_df = w
                        st.session_state.pp_encoded = True
                    st.rerun()
    
        # Auto-convert booleans
        bool_cols = st.session_state.pp_working_df.select_dtypes(include=["bool"]).columns.tolist()
        if bool_cols:
            for c in bool_cols:
                st.session_state.pp_working_df[c] = st.session_state.pp_working_df[c].astype(int)

    working = st.session_state.pp_working_df
    num_cols, _ = _detect_columns(working)
    feat_cols = [c for c in num_cols if c != target]

    # ═════════════════════════════════════════════════════════════
    # 2 · OUTLIER CAPPING
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-pp-outlier", border=True):
        section_title("2 · Outlier Capping", LIME)
        enable_cap = st.checkbox("Enable outlier capping", value=True, key="pp_cap_on")
    
        if st.session_state.pp_capped:
            st.success(" Outlier capping already applied.")
            if st.session_state._snap_cap_before is not None:
                with st.expander("Before / After Statistics"):
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.markdown("<h3>Before</h3>", unsafe_allow_html=True)
                        st.dataframe(st.session_state._snap_cap_before, use_container_width=True)
                    with bc2:
                        st.markdown("<h3>After</h3>", unsafe_allow_html=True)
                        st.dataframe(st.session_state._snap_cap_after, use_container_width=True)
        elif enable_cap and feat_cols:
            cap_pct = st.slider("Percentile threshold", 90, 100, 99, key="pp_cap_pct")
            cap_cols = st.multiselect("Columns to cap", feat_cols,
                                      default=feat_cols, key="pp_cap_cols")
            if st.button("  Cap Outliers", key="pp_run_cap", use_container_width=True, type="primary"):
                if not cap_cols:
                    st.warning("Select at least one column.")
                else:
                    with st.spinner("Capping outliers…"):
                        w = st.session_state.pp_working_df.copy()
                        snap_b = w[cap_cols].describe().T[["mean", "std", "max"]].round(2)
                        for c in cap_cols:
                            upper = w[c].quantile(cap_pct / 100)
                            w[c] = w[c].clip(upper=upper)
                        snap_a = w[cap_cols].describe().T[["mean", "std", "max"]].round(2)
                        st.session_state._snap_cap_before = snap_b
                        st.session_state._snap_cap_after = snap_a
                        st.session_state.pp_working_df = w
                        st.session_state.pp_capped = True
                    st.rerun()
        elif not feat_cols:
            st.info("No numeric feature columns available.")

    # ═════════════════════════════════════════════════════════════
    # 3 · SKEWNESS CORRECTION
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-pp-skewness", border=True):
        section_title("3 · Skewness Correction (Yeo-Johnson)", MINT)
        enable_yj = st.checkbox("Enable Yeo-Johnson transform", value=True, key="pp_yj_on")
    
        working = st.session_state.pp_working_df
        num_cols, _ = _detect_columns(working)
        feat_cols = [c for c in num_cols if c != target]
    
        if st.session_state.pp_skew_fixed:
            st.success(" Yeo-Johnson already applied.")
            if st.session_state._snap_yj_before is not None:
                with st.expander("View Before / After Skewness"):
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.markdown("<h3>Before</h3>", unsafe_allow_html=True)
                        st.dataframe(st.session_state._snap_yj_before, use_container_width=True)
                    with bc2:
                        st.markdown("<h3>After</h3>", unsafe_allow_html=True)
                        st.dataframe(st.session_state._snap_yj_after, use_container_width=True)
        elif enable_yj and feat_cols:
            yj_cols = st.multiselect("Columns for Yeo-Johnson", feat_cols,
                                      default=feat_cols[:min(6, len(feat_cols))],
                                      key="pp_yj_cols")
            if yj_cols:
                with st.expander("Skewness before transform"):
                    sk_before = working[yj_cols].skew().sort_values(ascending=False)
                    st.dataframe(sk_before.rename("skewness").to_frame(), use_container_width=True)
    
            if st.button("  Run Yeo-Johnson", key="pp_run_yj", use_container_width=True, type="primary"):
                if not yj_cols:
                    st.warning("Select at least one column.")
                else:
                    with st.spinner("Applying Yeo-Johnson…"):
                        w = st.session_state.pp_working_df.copy()
                        snap_b = w[yj_cols].skew().sort_values(ascending=False)
                        pt = PowerTransformer(method="yeo-johnson")
                        w[yj_cols] = pt.fit_transform(w[yj_cols])
                        snap_a = w[yj_cols].skew().sort_values(ascending=False)
                        st.session_state._snap_yj_before = snap_b.rename("skewness").to_frame()
                        st.session_state._snap_yj_after = snap_a.rename("skewness").to_frame()
                        st.session_state.pp_working_df = w
                        st.session_state.power_transformer = pt
                        st.session_state.pp_skew_fixed = True
                    st.rerun()

    # ═════════════════════════════════════════════════════════════
    # 4 · TRAIN / TEST SPLIT
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-pp-split", border=True):
        section_title("4 · Train / Test Split", PINK)
    
        working = st.session_state.pp_working_df
    
        if st.session_state.pp_split_done and st.session_state.X_train is not None:
            st.success(" Split already completed.")
            cfg = st.session_state._snap_split_cfg or {}
            c1, c2, c3 = st.columns(3)
            c1.metric("Train", f"{np.array(st.session_state.X_train).shape[0]:,} rows ({cfg.get('train_pct', '')})")
            c2.metric("Test", f"{np.array(st.session_state.X_test).shape[0]:,} rows ({cfg.get('test_pct', '')})")
            c3.metric("Random State", str(cfg.get("rs", "42")))
        else:
            sc1, sc2 = st.columns(2)
            test_size = sc1.slider("Test size (%)", 10, 50, 20, step=5, key="pp_test") / 100
            rand_state = sc2.number_input("Random state", value=42, step=1, key="pp_rs")
    
            if st.button("  Split Data", key="pp_run_split", use_container_width=True, type="primary"):
                with st.spinner("Splitting…"):
                    X = working.drop(columns=[target])
                    y = working[target]
                    stratify_y = y if y.nunique() <= 20 else None
                    X_tr, X_te, y_tr, y_te = train_test_split(
                        X, y, test_size=test_size,
                        random_state=int(rand_state), stratify=stratify_y,
                    )
                    st.session_state.X_train = X_tr
                    st.session_state.X_test = X_te
                    st.session_state.y_train = y_tr
                    st.session_state.y_test = y_te
                    st.session_state.selected_features = X_tr.columns.tolist()
                    st.session_state.pp_split_done = True
                    st.session_state._snap_split_cfg = {
                        "train_pct": f"{100 - int(test_size*100)}%",
                        "test_pct": f"{int(test_size*100)}%",
                        "rs": int(rand_state),
                    }
                st.rerun()

    # ═════════════════════════════════════════════════════════════
    # 5 · FEATURE SCALING
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-pp-scaling", border=True):
        section_title("5 · Feature Scaling", ORANGE)
    
        if not st.session_state.pp_split_done:
            st.info(" Complete the Train/Test Split first.")
        elif st.session_state.pp_scaled:
            st.success(" Scaling already applied.")
            if st.session_state._snap_scale_before is not None:
                with st.expander("Before / After Scaling Statistics"):
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.markdown("<h3>Before</h3>", unsafe_allow_html=True)
                        st.dataframe(st.session_state._snap_scale_before, use_container_width=True)
                    with bc2:
                        st.markdown("<h3>After</h3>", unsafe_allow_html=True)
                        st.dataframe(st.session_state._snap_scale_after, use_container_width=True)
        else:
            scaler_name = st.selectbox("Scaler",
                                        ["StandardScaler", "MinMaxScaler", "RobustScaler"],
                                        key="pp_scaler")
            scaler_map = {"StandardScaler": StandardScaler,
                          "MinMaxScaler": MinMaxScaler,
                          "RobustScaler": RobustScaler}
    
            if st.button("  Run Scaling", key="pp_run_scale", use_container_width=True, type="primary"):
                with st.spinner(f"Applying {scaler_name}…"):
                    X_tr = st.session_state.X_train
                    feat_names = st.session_state.selected_features or [f"f{i}" for i in range(X_tr.shape[1] if hasattr(X_tr, 'shape') else len(X_tr))]
                    snap_b = pd.DataFrame(X_tr, columns=feat_names[:np.array(X_tr).shape[1]]).describe().T[["mean", "std"]].round(4)
                    scaler = scaler_map[scaler_name]()
                    X_tr_scaled = scaler.fit_transform(X_tr)
                    X_te_scaled = scaler.transform(st.session_state.X_test)
                    snap_a = pd.DataFrame(X_tr_scaled, columns=feat_names[:X_tr_scaled.shape[1]]).describe().T[["mean", "std"]].round(4)
                    st.session_state._snap_scale_before = snap_b
                    st.session_state._snap_scale_after = snap_a
                    st.session_state.X_train = X_tr_scaled
                    st.session_state.X_test = X_te_scaled
                    st.session_state.scaler = scaler
                    st.session_state.pp_scaled = True
                st.rerun()

    # ═════════════════════════════════════════════════════════════
    # 6 · SMOTE
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-pp-smote", border=True):
        section_title("6 · Class Imbalance (SMOTE)", PURPLE)
    
        if not st.session_state.pp_scaled:
            st.info(" Complete Scaling first.")
        elif st.session_state.pp_smote_done:
            st.success(" SMOTE already applied.")
            if st.session_state._snap_smote_before is not None:
                with st.expander("View Before / After SMOTE"):
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.markdown(f"<h3>Before SMOTE</h3><p>{st.session_state._snap_smote_before}</p>", unsafe_allow_html=True)
                    with bc2:
                        st.markdown(f"<h3>After SMOTE</h3><p>{st.session_state._snap_smote_after}</p>", unsafe_allow_html=True)
        else:
            use_smote = st.checkbox("Enable SMOTE oversampling", value=False, key="pp_smote_chk")
            if use_smote:
                if not _HAS_SMOTE:
                    st.error("`imbalanced-learn` not installed. SMOTE unavailable.")
                else:
                    smote_rs = st.number_input("SMOTE random state", value=42, step=1, key="pp_smote_rs")
    
                    before_dist = pd.Series(np.array(st.session_state.y_train)).value_counts()
                    st.markdown(f"**Current Class Distribution:** {before_dist.to_dict()}")
    
                    if st.button("  Apply SMOTE", key="pp_run_smote", use_container_width=True, type="primary"):
                        with st.spinner("Applying SMOTE…"):
                            sm = SMOTE(random_state=int(smote_rs))
                            X_res, y_res = sm.fit_resample(
                                st.session_state.X_train, st.session_state.y_train
                            )
                            after_dist = pd.Series(np.array(y_res)).value_counts()
                            st.session_state._snap_smote_before = before_dist.to_dict()
                            st.session_state._snap_smote_after = after_dist.to_dict()
                            st.session_state.X_train = X_res
                            st.session_state.y_train = y_res
                            st.session_state.pp_smote_done = True
                        st.rerun()

    # ═════════════════════════════════════════════════════════════
    # 7 · SUMMARY & CONTINUE
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-pp-summary", border=True):
        section_title("7 · Pipeline Summary", LIME)
    
        ready = st.session_state.pp_split_done and st.session_state.pp_scaled
        if ready:
            c1, c2, c3 = st.columns(3)
            c1.metric("X_train", f"{np.array(st.session_state.X_train).shape[0]:,} × {np.array(st.session_state.X_train).shape[1]}")
            c2.metric("X_test", f"{np.array(st.session_state.X_test).shape[0]:,} × {np.array(st.session_state.X_test).shape[1]}")
            c3.metric("SMOTE", "Applied" if st.session_state.pp_smote_done else "Skipped")
    
            steps_done = []
            if st.session_state.pp_encoded:  steps_done.append(" Encoding")
            if st.session_state.pp_capped:   steps_done.append(" Outlier Capping")
            if st.session_state.pp_skew_fixed: steps_done.append(" Yeo-Johnson")
            steps_done.append(" Train/Test Split")
            steps_done.append(" Scaling")
            if st.session_state.pp_smote_done: steps_done.append(" SMOTE")
            with st.expander("Steps executed", expanded=True):
                for s in steps_done:
                    st.markdown(f"- {s}")
    
            st.session_state.engineered_data = st.session_state.pp_working_df
            unlock_step(STEP_TRAINING)
        else:
            st.warning("Complete **Train/Test Split** and **Scaling** before continuing.")

    render_page_nav(STEP_PREPROCESS)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def render_training():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    try:
        from xgboost import XGBClassifier
        _HAS_XGB = True
    except ImportError:
        _HAS_XGB = False

    hero("Model Training",
         "Configure, train, and inspect your machine learning model.",
         WORKFLOW_STEPS[STEP_TRAINING]["accent"])

    if st.session_state.X_train is None or st.session_state.y_train is None:
        st.warning(" Complete **Preprocessing** before training a model.")
        render_page_nav(STEP_TRAINING)
        return

    X_train = np.array(st.session_state.X_train)
    y_train = np.array(st.session_state.y_train).ravel()
    X_test  = np.array(st.session_state.X_test)
    y_test  = np.array(st.session_state.y_test).ravel()

    feat_names = st.session_state.selected_features or [f"f{i}" for i in range(X_train.shape[1])]

    # ═════════════════════════════════════════════════════════════
    # 1 · DATASET SUMMARY
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-tr-dataset", border=True):
        section_title("1 · Dataset Summary", YELLOW)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("X_train", f"{X_train.shape[0]:,} × {X_train.shape[1]}")
        c2.metric("X_test", f"{X_test.shape[0]:,} × {X_test.shape[1]}")
        c3.metric("Features", str(len(feat_names)))
        c4.metric("Target", st.session_state.target_col or "Revenue")
    
        with st.expander("Selected Features"):
            for f in feat_names:
                st.markdown(f"- `{f}`")

    # ═════════════════════════════════════════════════════════════
    # 2 · PIPELINE SUMMARY
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-tr-pipeline", border=True):
        section_title("2 · Pipeline Summary", MINT)
    
        steps_done = []
        if st.session_state.pp_encoded:    steps_done.append(" Categorical Encoding")
        if st.session_state.pp_capped:     steps_done.append(" Outlier Capping")
        if st.session_state.pp_skew_fixed: steps_done.append(" Yeo-Johnson Skewness Correction")
        if st.session_state.pp_split_done: steps_done.append(" Train / Test Split")
        if st.session_state.pp_scaled:     steps_done.append(" Feature Scaling")
        if st.session_state.pp_smote_done: steps_done.append(" SMOTE Oversampling")
    
        with st.expander("Executed Preprocessing Steps", expanded=True):
            if steps_done:
                for s in steps_done:
                    st.markdown(f"- {s}")
            else:
                st.info("No preprocessing steps recorded.")

    # ═════════════════════════════════════════════════════════════
    # 3 · MODEL SELECTION & CONFIGURATION
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-tr-model", border=True):
        section_title("3 · Model Selection", LIME)
    
        model_options = ["Logistic Regression", "Random Forest"]
        if _HAS_XGB:
            model_options.append("XGBoost")
    
        if st.session_state.trained_model is not None:
            st.success(f" Model already trained: **{st.session_state.model_name}**")
    
            model = st.session_state.trained_model
            if hasattr(model, "feature_importances_"):
                with st.expander("View Feature Importance"):
                    importances = model.feature_importances_
                    fi_df = pd.DataFrame({
                        "Feature": feat_names[:len(importances)],
                        "Importance": importances,
                    }).sort_values("Importance", ascending=False)
    
                    fig, ax = plt.subplots(figsize=(10, max(4, len(fi_df) * 0.35)))
                    fig.patch.set_alpha(0.0)
                    colors = [_SERIES_COLORS[i % len(_SERIES_COLORS)] for i in range(len(fi_df))]
                    ax.barh(fi_df["Feature"].values[::-1], fi_df["Importance"].values[::-1],
                            color=colors[:len(fi_df)][::-1], edgecolor="#000", linewidth=1.5)
                    _style_ax(ax, title="Feature Importance", xlabel="Importance")
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
    
            elif hasattr(model, "coef_"):
                with st.expander("View Feature Coefficients"):
                    coefs = model.coef_.ravel()
                    co_df = pd.DataFrame({
                        "Feature": feat_names[:len(coefs)],
                        "Coefficient": coefs,
                    }).sort_values("Coefficient", key=abs, ascending=False)
                    st.dataframe(co_df, use_container_width=True)
    
        else:
            model_choice = st.selectbox("Choose a model", model_options, key="tr_model_sel")
    
            st.markdown("---")
            section_title("Hyperparameters", PINK)
    
            if model_choice == "Logistic Regression":
                lr_max_iter = st.slider("Max iterations", 100, 5000, 1000, step=100, key="tr_lr_iter")
                lr_c = st.number_input("Regularization C", value=1.0, min_value=0.001, step=0.1, key="tr_lr_c")
    
            elif model_choice == "Random Forest":
                rf_n = st.slider("Number of estimators", 10, 500, 100, step=10, key="tr_rf_n")
                rf_depth = st.selectbox("Max depth", [None, 5, 10, 15, 20, 30], key="tr_rf_depth")
                rf_rs = st.number_input("Random state", value=42, step=1, key="tr_rf_rs")
    
            elif model_choice == "XGBoost":
                xgb_n = st.slider("Number of estimators", 50, 500, 100, step=50, key="tr_xgb_n")
                xgb_lr = st.number_input("Learning rate", value=0.3, min_value=0.01, max_value=1.0,
                                          step=0.05, key="tr_xgb_lr")
                xgb_depth = st.slider("Max depth", 3, 15, 6, key="tr_xgb_depth")
    
            st.markdown("---")
            if st.button("  Train Model", key="tr_run", use_container_width=True, type="primary"):
                with st.spinner(f"Training {model_choice}…"):
                    if model_choice == "Logistic Regression":
                        model = LogisticRegression(max_iter=lr_max_iter, C=lr_c)
                    elif model_choice == "Random Forest":
                        model = RandomForestClassifier(
                            n_estimators=rf_n,
                            max_depth=rf_depth,
                            random_state=int(rf_rs),
                        )
                    elif model_choice == "XGBoost":
                        model = XGBClassifier(
                            n_estimators=xgb_n,
                            learning_rate=xgb_lr,
                            max_depth=xgb_depth,
                            eval_metric="logloss",
                            use_label_encoder=False,
                        )
    
                    model.fit(X_train, y_train)
    
                    st.session_state.trained_model = model
                    st.session_state.model_name = model_choice
                    st.session_state.metrics = {}
                    st.session_state.prediction_history = []
    
                    unlock_step(STEP_EVALUATION)
                st.rerun()

    render_page_nav(STEP_TRAINING)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def render_evaluation():
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, roc_curve, confusion_matrix, classification_report,
    )

    hero("Model Evaluation",
         "Understand your model's performance, strengths, and weaknesses.",
         WORKFLOW_STEPS[STEP_EVALUATION]["accent"])

    model = st.session_state.trained_model
    if model is None:
        st.warning(" Train a model first on the **Training** page.")
        render_page_nav(STEP_EVALUATION)
        return

    X_test = np.array(st.session_state.X_test)
    y_test = np.array(st.session_state.y_test).ravel()

    model_name = st.session_state.model_name or "Model"

    # ═════════════════════════════════════════════════════════════
    # 1 · RUN EVALUATION
    # ═════════════════════════════════════════════════════════════
    if st.session_state.metrics:
        st.success(" Evaluation already completed.")
        metrics = st.session_state.metrics
    else:
        if st.button("  Run Evaluation", key="ev_run", use_container_width=True, type="primary"):
            with st.spinner("Evaluating model on test set…"):
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

                st.session_state.metrics = {
                    "accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "auc": auc,
                    "y_pred": y_pred.tolist(),
                    "y_prob": y_prob.tolist() if y_prob is not None else None,
                    "y_test": y_test.tolist(),
                    "report": classification_report(y_test, y_pred, output_dict=True),
                }
                unlock_step(STEP_PREDICTION)
            st.rerun()
        else:
            render_page_nav(STEP_EVALUATION)
            return

        metrics = st.session_state.metrics

    # ═════════════════════════════════════════════════════════════
    # 2 · METRIC OVERVIEW
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-ev-metrics", border=True):
        section_title("1 · Metrics Overview", MINT)
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        mc2.metric("Precision", f"{metrics['precision']:.4f}")
        mc3.metric("Recall", f"{metrics['recall']:.4f}")
        mc4.metric("F1-Score", f"{metrics['f1']:.4f}")
        if metrics["auc"] is not None:
            mc5.metric("AUC-ROC", f"{metrics['auc']:.4f}")
        else:
            mc5.metric("AUC-ROC", "N/A")
    
        f1_val = metrics["f1"]
        if f1_val >= 0.8:
            quality = "strong"
            emoji = ""
        elif f1_val >= 0.6:
            quality = "moderate"
            emoji = ""
        else:
            quality = "weak"
            emoji = ""
    
        st.markdown(
            f"{emoji} **Interpretation:** The model shows **{quality}** overall performance "
            f"with an F1-score of **{f1_val:.4f}**. "
            f"{'The precision/recall balance looks healthy.' if abs(metrics['precision'] - metrics['recall']) < 0.15 else 'There is a noticeable gap between precision and recall — this may indicate the model favors one metric over the other.'}"
        )

    # ═════════════════════════════════════════════════════════════
    # 3 · CONFUSION MATRIX & ROC CURVE
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-ev-diagnostics", border=True):
        section_title("2 · Prediction Diagnostics", ORANGE)
    
        y_pred = np.array(metrics["y_pred"])
        y_test_arr = np.array(metrics["y_test"])
        y_prob = np.array(metrics["y_prob"]) if metrics["y_prob"] is not None else None
    
        col_cm, col_roc = st.columns(2)
    
        with col_cm:
            section_title("Confusion Matrix", YELLOW)
            cm = confusion_matrix(y_test_arr, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_alpha(0.0)
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["No Purchase", "Purchase"],
                        yticklabels=["No Purchase", "Purchase"],
                        linewidths=2, linecolor="#000")
            _style_ax(ax, title=f"{model_name}: Confusion Matrix",
                      xlabel="Predicted", ylabel="Actual")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
    
        with col_roc:
            if y_prob is not None:
                section_title("ROC Curve", LIME)
                fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
                fig, ax = plt.subplots(figsize=(5, 4))
                fig.patch.set_alpha(0.0)
                ax.plot(fpr, tpr, color="#7B2FFF", linewidth=2,
                        label=f"AUC = {metrics['auc']:.4f}")
                ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.5)")
                ax.fill_between(fpr, tpr, alpha=0.1, color="#7B2FFF")
                _style_ax(ax, title=f"{model_name}: ROC Curve",
                          xlabel="False Positive Rate", ylabel="True Positive Rate")
                ax.legend(loc="lower right")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("ROC curve not available for this model.")

    # ═════════════════════════════════════════════════════════════
    # 4 · CLASSIFICATION REPORT
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-ev-report", border=True):
        section_title("3 · Classification Report", PURPLE)
    
        with st.expander("View Full Classification Report"):
            report_dict = metrics["report"]
            report_df = pd.DataFrame(report_dict).T
            st.dataframe(report_df.round(4), use_container_width=True)
    
    # ═════════════════════════════════════════════════════════════
    # 5 · ERROR ANALYSIS & INTERPRETATION
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-ev-interpret", border=True):
        section_title("4 · Interpretation Summary", PINK)
    
        tn, fp, fn, tp = confusion_matrix(y_test_arr, y_pred).ravel()
        total = len(y_test_arr)
    
        st.markdown(f"""
        **{model_name}** was evaluated on **{total:,}** test samples.
    
        | Metric | Value | Meaning |
        |--------|-------|---------|
        | True Positives (TP) | {tp} | Correctly predicted purchases |
        | True Negatives (TN) | {tn} | Correctly predicted non-purchases |
        | False Positives (FP) | {fp} | Incorrectly predicted as purchase |
        | False Negatives (FN) | {fn} | Missed actual purchases |
    
        {" **Overfitting Risk:** If training accuracy is significantly higher than test accuracy, the model may be overfitting. Consider regularization or simpler models." if metrics["accuracy"] > 0.95 else ""}
    
        {" **Class Imbalance Note:** The recall for the minority class may be lower due to natural class imbalance. SMOTE was applied during preprocessing to mitigate this." if metrics["recall"] < 0.5 else ""}
    
        {" The model demonstrates a solid balance between precision and recall." if abs(metrics["precision"] - metrics["recall"]) < 0.15 and f1_val >= 0.6 else ""}
        """)

    render_page_nav(STEP_EVALUATION)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def render_prediction():
    hero("Real-Time Prediction",
         "Input new data and get instant purchase predictions.",
         WORKFLOW_STEPS[STEP_PREDICTION]["accent"])

    model = st.session_state.trained_model
    if model is None or not st.session_state.metrics:
        st.warning(" Complete **Training** and **Evaluation** before making predictions.")
        render_page_nav(STEP_PREDICTION)
        return

    model_name = st.session_state.model_name or "Model"
    feat_names = st.session_state.selected_features or []

    if not feat_names:
        st.error("No features configured. Go back to Feature Selection.")
        render_page_nav(STEP_PREDICTION)
        return

    # ═════════════════════════════════════════════════════════════
    # 1 · MODEL READINESS
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-pred-ready", border=True):
        section_title("1 · Model Readiness", MINT)
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Model", model_name)
        rc2.metric("Features", str(len(feat_names)))
        f1_val = st.session_state.metrics.get("f1", 0)
        rc3.metric("F1-Score", f"{f1_val:.4f}")
    
        with st.expander("Preprocessing Artifacts"):
            artifacts = {
                "Scaler": "Loaded" if st.session_state.scaler is not None else "Not fitted",
                "Power Transformer": "Loaded" if st.session_state.power_transformer is not None else "Not fitted",
                "Label Encoders": "Loaded" if st.session_state.label_encoder_visitor is not None else "Not fitted",
            }
            for k, v in artifacts.items():
                st.markdown(f"- **{k}**: {v}")
    
    # ═════════════════════════════════════════════════════════════
    # 2 · DYNAMIC INPUT FORM
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-pred-input", border=True):
        section_title("2 · Input Features", YELLOW)
        st.caption("Fill in the feature values below. All fields are dynamically generated from your selected features.")
    
        input_values = {}
    
        X_train = st.session_state.X_train
        if X_train is not None:
            if isinstance(X_train, pd.DataFrame):
                ref_df = X_train
            else:
                ref_df = pd.DataFrame(np.array(X_train), columns=feat_names[:np.array(X_train).shape[1]])
        else:
            ref_df = None
    
        cols = st.columns(2)
        for i, feat in enumerate(feat_names):
            col = cols[i % 2]
            with col:
                if ref_df is not None and feat in ref_df.columns:
                    default_val = float(ref_df[feat].median())
                    min_val = float(ref_df[feat].min())
                    max_val = float(ref_df[feat].max())
                    input_values[feat] = st.number_input(
                        f" {feat}",
                        value=round(default_val, 4),
                        step=round((max_val - min_val) / 100, 4) if max_val != min_val else 0.01,
                        key=f"pred_{feat}",
                    )
                else:
                    input_values[feat] = st.number_input(
                        f" {feat}", value=0.0, step=0.01, key=f"pred_{feat}",
                    )

    # ═════════════════════════════════════════════════════════════
    # 3 · PREDICT
    # ═════════════════════════════════════════════════════════════
    with st.container(key="neo-pred-run", border=True):
        section_title("3 · Run Prediction", LIME)
    
        if st.button("  Predict", key="pred_run", use_container_width=True, type="primary"):
            with st.spinner("Running inference…"):
                input_arr = np.array([[input_values.get(f, 0.0) for f in feat_names]])
    
                prediction = model.predict(input_arr)[0]
                probabilities = model.predict_proba(input_arr)[0] if hasattr(model, "predict_proba") else None
    
                result = {
                    "input": {k: round(v, 4) for k, v in input_values.items()},
                    "prediction": int(prediction),
                    "probabilities": probabilities.tolist() if probabilities is not None else None,
                }
                st.session_state.prediction_history.insert(0, result)
    
            st.markdown("---")
            section_title("4 · Result", PINK)
    
            if prediction == 1:
                result_color = LIME
                result_icon = ""
                result_text = "Purchase Predicted"
                result_desc = "The model predicts this visitor <strong>will complete a purchase</strong>."
            else:
                result_color = ORANGE
                result_icon = ""
                result_text = "No Purchase Predicted"
                result_desc = "The model predicts this visitor <strong>will NOT complete a purchase</strong>."
    
            st.markdown(
                f'<div class="neo-result" style="background:{result_color};">'
                f'<h2>{result_icon} {result_text}</h2>'
                f'<p>{result_desc}</p></div>',
                unsafe_allow_html=True,
            )
    
            if probabilities is not None:
                st.markdown("### Class Probabilities")
                pc1, pc2 = st.columns(2)
                pc1.metric("No Purchase (0)", f"{probabilities[0]:.4f}")
                pc2.metric("Purchase (1)", f"{probabilities[1]:.4f}")
    
                # Probability bar
                fig, ax = plt.subplots(figsize=(8, 1.5))
                fig.patch.set_alpha(0.0)
                ax.barh(["Prediction"], [probabilities[1]], color=LIME,
                        edgecolor="#000", linewidth=2, label="Purchase")
                ax.barh(["Prediction"], [probabilities[0]], left=[probabilities[1]],
                        color=ORANGE, edgecolor="#000", linewidth=2, label="No Purchase")
                ax.set_xlim(0, 1)
                ax.legend(loc="upper right", fontsize=9)
                _style_ax(ax, title="Confidence Distribution")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # ═════════════════════════════════════════════════════════════
    # 5 · PREDICTION HISTORY
    # ═════════════════════════════════════════════════════════════
    if st.session_state.prediction_history:
        with st.container(key="neo-pred-history", border=True):
            section_title("5 · Prediction History", PURPLE)
            with st.expander(f"View History ({len(st.session_state.prediction_history)} predictions)"):
                for idx, rec in enumerate(st.session_state.prediction_history):
                    label = " Purchase" if rec["prediction"] == 1 else " No Purchase"
                    prob_str = ""
                    if rec["probabilities"]:
                        prob_str = f" (confidence: {max(rec['probabilities']):.2%})"
                    st.markdown(f"**#{idx + 1}** — {label}{prob_str}")

    render_page_nav(STEP_PREDICTION)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT  (always accessible)
# ═══════════════════════════════════════════════════════════════════════════════

def render_about():
    hero("About This Application",
         "Technical details, preprocessing pipeline, and technology stack.",
         WORKFLOW_STEPS[STEP_ABOUT]["accent"])

    # ── Project Description ──────────────────────────────────────
    with st.container(key="neo-about-desc", border=True):
        section_title("Project Description", YELLOW)
        st.markdown("""
        This application was developed as part of a Machine Learning course assignment
        at Bina Nusantara University. The project focuses on implementing a complete
        machine learning pipeline, starting from Exploratory Data Analysis (EDA) to
        model deployment.
        """)

    # ── Main Objectives ──────────────────────────────────────────
    with st.container(key="neo-about-objectives", border=True):
        section_title("Main Objectives", LIME)
        st.markdown("""
        - Understand and implement the end-to-end machine learning pipeline from EDA to deployment.
        - Evaluate and compare multiple classification algorithms for predictive accuracy.
        - Develop an interactive dashboard to visualize data patterns and provide real-time predictions.
        """)

    # ── Dataset Info + Team Members (side by side) ───────────────
    col1, col2 = st.columns(2)
    with col1:
        with st.container(key="neo-about-dataset", border=True):
            section_title("Dataset Info", MINT)
            st.markdown("""
            **Name:** Online Shoppers Purchasing Intention Dataset  
            **Source:** UCI Machine Learning Repository  
            **Scope:** E-commerce browsing session data  
            **Link:** [UCI ML Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
            """)

    with col2:
        with st.container(key="neo-about-team", border=True):
            section_title("Team Members", PINK)
            st.caption("Group 1 - LC01")
            
            team = {
                "2802397306": "Edwin Antonie",
                "2802391006": "Maximilianus Ronald",
                "2802401846": "Wesley Sumedha Deano"
            }
            
            for nim, name in team.items():
                st.markdown(f"**{nim}** &nbsp; | &nbsp; {name}")

    # ── Technologies Used ────────────────────────────────────────
    with st.container(key="neo-about-tech", border=True):
        section_title("Technologies Used", PURPLE)
        st.markdown("""
        **Streamlit** (User Interface),
        **Scikit-learn** (Machine Learning),
        **Pandas** (Data Processing),
        **NumPy** (Numerical Computation),
        **Matplotlib & Seaborn** (Data Visualization)
        """)

    render_page_nav(STEP_ABOUT)

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

PAGE_ROUTER = {
    STEP_HOME:       render_home,
    STEP_DATASET:    render_dataset,
    STEP_EDA:        render_eda,
    STEP_FEAT_SEL:   render_feature_selection,
    STEP_PREPROCESS: render_preprocessing,
    STEP_TRAINING:   render_training,
    STEP_EVALUATION: render_evaluation,
    STEP_PREDICTION: render_prediction,
    STEP_ABOUT:      render_about,
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="ML Prediction Engine",
        page_icon="🌐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    initialize_session_state()
    inject_global_css()
    render_sidebar()

    # Route to current page
    step = st.session_state.current_step
    renderer = PAGE_ROUTER.get(step, render_home)
    renderer()


if __name__ == "__main__":
    main()
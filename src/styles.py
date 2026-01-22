"""Styling module for the Portfolio Optimizer app.

This module contains all CSS styles for light and dark themes,
keeping the main app.py clean and focused on logic.
"""
import streamlit as st

# Base CSS applied regardless of theme
BASE_CSS = """
/* Style info buttons (only single-character buttons) */
button[kind="secondary"] p:only-child {
    display: flex;
    align-items: center;
    justify-content: center;
}
button[kind="secondary"]:has(p:only-child:not(:empty)) {
    container-type: inline-size;
}
div[data-testid="stColumn"]:last-child button[kind="secondary"] {
    background-color: #6366f1 !important;
    color: white !important;
    border: none !important;
    border-radius: 50% !important;
    width: 32px !important;
    height: 32px !important;
    padding: 0 !important;
    min-height: 0 !important;
}
div[data-testid="stColumn"]:last-child button[kind="secondary"]:hover {
    background-color: #4f46e5 !important;
}
div[data-testid="stColumn"]:last-child button[kind="secondary"] p {
    font-size: 16px !important;
    color: white !important;
}
/* Align title info button with title text */
div[data-testid="stVerticalBlock"] > div:first-child div[data-testid="stColumn"]:last-child {
    display: flex !important;
    align-items: flex-end !important;
    justify-content: flex-end !important;
    padding-bottom: 0.75rem !important;
}

/* Make multiselect X button larger for mobile tappability */
[data-baseweb="tag"] svg {
    width: 12px !important;
    height: 12px !important;
    stroke-width: 1px !important;
}
[data-baseweb="tag"] [aria-label="close"],
[data-baseweb="tag"] span:last-child {
    padding: 5px !important;
    min-width: 22px !important;
    min-height: 22px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
"""

# Dark theme CSS
DARK_THEME_CSS = """
/* Dark theme */
.stApp, [data-testid="stSidebar"], [data-testid="stHeader"] {
    background-color: #1a1a2e !important;
    color: #eaeaea !important;
}
.stApp * {
    color: #eaeaea !important;
}
[data-testid="stSidebar"] {
    background-color: #16213e !important;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #1a1a2e !important;
}
.stTabs [data-baseweb="tab"] {
    color: #eaeaea !important;
}
[data-testid="stMetric"] {
    background-color: #16213e !important;
    padding: 10px;
    border-radius: 8px;
}
.stDataFrame, [data-testid="stDataFrame"] {
    background-color: #16213e !important;
}
.stMarkdown a {
    color: #818cf8 !important;
}

/* Input fields */
input, [data-baseweb="input"], [data-baseweb="select"] > div,
.stTextInput > div > div, .stNumberInput > div > div,
.stDateInput > div > div, .stSelectbox > div > div {
    background-color: #16213e !important;
    color: #eaeaea !important;
    border-color: #334155 !important;
    caret-color: #eaeaea !important;
}
.stMultiSelect > div > div {
    background-color: #16213e !important;
}
[data-baseweb="popover"] {
    background-color: #16213e !important;
}
[data-baseweb="menu"] {
    background-color: #16213e !important;
}
[data-baseweb="menu"] li {
    background-color: #16213e !important;
}
[data-baseweb="menu"] li:hover {
    background-color: #1e3a5f !important;
}

/* Placeholder text */
input::placeholder, textarea::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

/* Number input buttons */
.stNumberInput button {
    background-color: #16213e !important;
    color: #eaeaea !important;
    border-color: #334155 !important;
}

/* Data table */
.stDataFrame, [data-testid="stDataFrame"],
.stDataFrame > div, [data-testid="stDataFrame"] > div,
.stDataFrame iframe, [data-testid="stDataFrame"] iframe {
    background-color: #16213e !important;
}
[data-testid="stDataFrameResizable"] {
    background-color: #16213e !important;
}

/* Download button */
.stDownloadButton button {
    background-color: #16213e !important;
    color: #eaeaea !important;
    border: 1px solid #334155 !important;
}
.stDownloadButton button:hover {
    background-color: #1e3a5f !important;
    border-color: #6366f1 !important;
}

/* Dialog/Modal styling */
[data-testid="stModal"] > div:first-child {
    background-color: rgba(0, 0, 0, 0.6) !important;
}
[data-testid="stModal"] > div > div,
[data-testid="stModal"] [data-testid="stVerticalBlock"],
[data-testid="stModal"] section,
div[data-modal-container="true"] > div > div,
[role="dialog"],
[role="dialog"] > div {
    background-color: #242945 !important;
    color: #eaeaea !important;
    border: 1px solid #3d4566 !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
}
[data-testid="stModal"] *,
[role="dialog"] * {
    color: #eaeaea !important;
}
[data-testid="stModal"] [data-testid="stMarkdownContainer"],
[role="dialog"] [data-testid="stMarkdownContainer"] {
    color: #eaeaea !important;
}
[data-testid="stModal"] h1, [data-testid="stModal"] h2,
[data-testid="stModal"] h3, [data-testid="stModal"] h4,
[role="dialog"] h1, [role="dialog"] h2,
[role="dialog"] h3, [role="dialog"] h4 {
    color: #eaeaea !important;
}
[data-testid="stModal"] hr, [role="dialog"] hr {
    border-color: #334155 !important;
}
[data-testid="stModal"] a, [role="dialog"] a {
    color: #818cf8 !important;
}

/* Modal close button */
[data-testid="stModal"] button[aria-label="Close"],
[role="dialog"] button[aria-label="Close"] {
    color: #eaeaea !important;
}

/* Info buttons in dark mode - brighter colors */
div[data-testid="stColumn"]:last-child button[kind="secondary"] {
    background-color: #a5b4fc !important;
    border: 2px solid #c7d2fe !important;
    box-shadow: 0 0 12px rgba(165, 180, 252, 0.7) !important;
}
div[data-testid="stColumn"]:last-child button[kind="secondary"]:hover {
    background-color: #c7d2fe !important;
    border-color: #e0e7ff !important;
    box-shadow: 0 0 16px rgba(199, 210, 254, 0.9) !important;
}
div[data-testid="stColumn"]:last-child button[kind="secondary"] p {
    color: #1e1b4b !important;
}

/* Tooltip/help popover in dark mode */
[data-testid="stTooltipContent"],
div[data-baseweb="tooltip"] > div,
[role="tooltip"] {
    background-color: #1a1a2e !important;
    color: #eaeaea !important;
    border: 1px solid #334155 !important;
}

/* Widget help icons (circle with ?) in dark mode */
[data-testid="stTooltipHoverTarget"] svg {
    color: #e0e7ff !important;
    stroke: #e0e7ff !important;
}
"""


def apply_base_styles():
    """Apply base styles that work for both light and dark themes."""
    st.markdown(f"<style>{BASE_CSS}</style>", unsafe_allow_html=True)


def apply_theme(dark_mode: bool):
    """Apply theme-specific styles.

    Args:
        dark_mode: If True, apply dark theme. Otherwise, use default light theme.
    """
    if dark_mode:
        st.markdown(f"<style>{DARK_THEME_CSS}</style>", unsafe_allow_html=True)

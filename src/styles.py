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
    background-color: transparent !important;
    color: #94a3b8 !important;
    border: none !important;
    padding: 0 !important;
    min-height: 0 !important;
}
div[data-testid="stColumn"]:last-child button[kind="secondary"]:hover {
    background-color: transparent !important;
    color: #6366f1 !important;
}
div[data-testid="stColumn"]:last-child button[kind="secondary"] p {
    font-size: 18px !important;
    color: inherit !important;
}
/* Align all info buttons with their headers - left aligned, inline */
div[data-testid="stColumn"]:last-child:has(button[kind="secondary"]) {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
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

/* Documentation link */
.doc-link {
    color: #6366f1 !important;
    font-weight: 600;
    text-decoration: none !important;
}
.doc-link:hover {
    text-decoration: underline !important;
}

/* Sidebar toggle button - prominent when collapsed */
[data-testid="collapsedControl"] button {
    background-color: #6366f1 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 8px !important;
}
[data-testid="collapsedControl"] button:hover {
    background-color: #4f46e5 !important;
}
[data-testid="collapsedControl"] svg {
    stroke: white !important;
    width: 20px !important;
    height: 20px !important;
}
/* Sidebar collapse button - always visible when expanded */
[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] {
    opacity: 1 !important;
    visibility: visible !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    opacity: 1 !important;
    visibility: visible !important;
}

/* Prominent floating tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    padding: 0;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: 600;
    font-size: 1rem;
    color: #64748b;
    letter-spacing: 0.01em;
    transition: all 0.2s ease;
    border: 2px solid #e2e8f0;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #475569;
    border-color: #cbd5e1;
    background-color: #f8fafc;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    font-weight: 600;
    border-color: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
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
/* Dark mode tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    border-color: #334155 !important;
    color: #94a3b8 !important;
}
.stTabs [data-baseweb="tab"]:hover {
    border-color: #475569 !important;
    background-color: rgba(255, 255, 255, 0.03) !important;
    color: #e2e8f0 !important;
}
.stTabs [aria-selected="true"] {
    box-shadow: none !important;
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
.doc-link {
    color: #a5b4fc !important;
    text-decoration: none !important;
}
.doc-link:hover {
    text-decoration: underline !important;
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

/* Sidebar toggle in dark mode - prominent when collapsed */
[data-testid="collapsedControl"] button {
    background-color: #818cf8 !important;
}
[data-testid="collapsedControl"] button:hover {
    background-color: #a5b4fc !important;
}
/* Sidebar collapse button always visible in dark mode */
[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"],
[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    opacity: 1 !important;
    visibility: visible !important;
}

/* Info buttons in dark mode */
div[data-testid="stColumn"]:last-child button[kind="secondary"] {
    color: #64748b !important;
}
div[data-testid="stColumn"]:last-child button[kind="secondary"]:hover {
    color: #818cf8 !important;
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

"""
loan_design_system.py — Loan Analytics design system (Step 3)

A self-contained, theme-aware sidebar + UI helper kit for Streamlit.
 * One source of truth for light/dark theme: st.session_state["ld_theme"]
 * No gradients, no blue accents, no prediction logic, no model loading.
 * Never writes outside its own session-state keys (prefix "ld_").
"""
from __future__ import annotations

import html
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import streamlit as st

__version__ = "1.0.0"

BRAND_NAME = "Loan Analytics"
BRAND_SUBTITLE = "Credit Risk Intelligence"
THEME_KEY = "ld_theme"
NAV_KEY = "ld_active_page"
DEFAULT_THEME = "light"

# (page_key, label, glyph)
NAV_ITEMS: Tuple[Tuple[str, str, str], ...] = (
    ("overview", "Overview", "◈"),
    ("prediction", "Loan Prediction", "◇"),
    ("insights", "Model Insights", "❖"),
    ("monitoring", "Prediction Monitoring", "◉"),
)

PALETTES: Dict[str, Dict[str, str]] = {
    "light": {
        "bg": "#F7F3EE", "surface": "#FFFFFF", "surface_alt": "#F0E9E0",
        "sidebar": "#F2ECE4", "border": "#E2D8CB",
        "text": "#292321", "muted": "#6F645D",
        "primary": "#A8442E", "primary_hover": "#8E3724",
        "primary_soft": "#F4E3DC", "primary_text": "#FFFFFF",
        "accent": "#6E9586", "accent_text": "#3E6354", "accent_soft": "#E3EDE8",
        "warning": "#9A6B1F", "warning_soft": "#F5EAD3",
        "danger": "#8F2F22", "danger_soft": "#F3DAD4",
    },
    "dark": {
        "bg": "#19161F", "surface": "#25212C", "surface_alt": "#2E2936",
        "sidebar": "#1E1A25", "border": "#3B3544",
        "text": "#F7F0E8", "muted": "#B0A6AD",
        "primary": "#C9735B", "primary_hover": "#D88A73",
        "primary_soft": "#3C2A28", "primary_text": "#19161F",
        "accent": "#86AB9B", "accent_text": "#A4C6B7", "accent_soft": "#24342F",
        "warning": "#D0A257", "warning_soft": "#3A3022",
        "danger": "#D9806C", "danger_soft": "#3F2724",
    },
}

_FONT_IMPORT = (
    "@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600"
    "&family=Inter:wght@400;500;600&display=swap');"
)

_BASE_CSS = """
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--ld-bg) !important; color: var(--ld-text); font-family: var(--ld-font-body); }
[data-testid="stHeader"] { background: var(--ld-bg) !important; }
.block-container { padding-top: 2.4rem; max-width: 1180px; }
h1, h2, h3, h4, h5 { font-family: var(--ld-font-head) !important; color: var(--ld-text) !important; letter-spacing: -0.01em; font-weight: 600 !important; }
p, li, label, [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] { color: var(--ld-text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--ld-muted) !important; }
hr { border-color: var(--ld-border) !important; }
a { color: var(--ld-primary); }

/* ---------- Sidebar shell ---------- */
section[data-testid="stSidebar"] { background-color: var(--ld-sidebar) !important; background-image: var(--ld-pattern); background-size: 48px 48px; border-right: 1px solid var(--ld-border); }
section[data-testid="stSidebar"] > div { background: transparent !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1rem; }
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: 0; }

/* ---------- Buttons (global) ---------- */
.stButton, .stDownloadButton { width: 100%; }
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button { border-radius: 10px; font-weight: 600; border: 1px solid var(--ld-border); background: var(--ld-surface); color: var(--ld-text); box-shadow: none; transition: background .15s ease, border-color .15s ease, color .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover { border-color: var(--ld-primary); color: var(--ld-primary); background: var(--ld-surface); }
.stButton > button:focus-visible, .stFormSubmitButton > button:focus-visible { outline: 2px solid var(--ld-accent); outline-offset: 2px; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"], .stFormSubmitButton > button[kind="primaryFormSubmit"], .stFormSubmitButton > button[data-testid="stBaseButton-primaryFormSubmit"] { background: var(--ld-primary); border-color: var(--ld-primary); color: var(--ld-primary-text); }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover, .stFormSubmitButton > button[kind="primaryFormSubmit"]:hover, .stFormSubmitButton > button[data-testid="stBaseButton-primaryFormSubmit"]:hover { background: var(--ld-primary-hover); border-color: var(--ld-primary-hover); color: var(--ld-primary-text); }
.stButton button [data-testid="stMarkdownContainer"] *, .stFormSubmitButton button [data-testid="stMarkdownContainer"] *, .stDownloadButton button [data-testid="stMarkdownContainer"] * { color: inherit !important; }

/* ---------- Sidebar navigation buttons ---------- */
section[data-testid="stSidebar"] .stButton > button { width: 100%; justify-content: flex-start; text-align: left; background: transparent; border: 1px solid transparent; border-left: 3px solid transparent; border-radius: 10px; padding: 0.55rem 0.85rem; font-weight: 500; color: var(--ld-text); }
section[data-testid="stSidebar"] .stButton > button p { text-align: left; width: 100%; }
section[data-testid="stSidebar"] .stButton > button:hover { background: var(--ld-surface-alt); border-color: transparent; border-left-color: var(--ld-accent); color: var(--ld-text); }
section[data-testid="stSidebar"] .stButton > button[kind="primary"], section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] { background: var(--ld-primary); border-color: var(--ld-primary); border-left-color: var(--ld-accent); color: var(--ld-primary-text); font-weight: 600; }
section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover, section[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"]:hover { background: var(--ld-primary-hover); border-color: var(--ld-primary-hover); border-left-color: var(--ld-accent); color: var(--ld-primary-text); }

/* ---------- Inputs ---------- */
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="textarea"], [data-baseweb="select"] > div { background-color: var(--ld-surface) !important; border-color: var(--ld-border) !important; border-radius: 10px !important; }
input, textarea { color: var(--ld-text) !important; -webkit-text-fill-color: var(--ld-text) !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"] { background: var(--ld-surface) !important; color: var(--ld-text) !important; }
[data-baseweb="popover"] li { color: var(--ld-text) !important; }
[data-baseweb="slider"] div[role="slider"] { background-color: var(--ld-primary) !important; }

/* ---------- Tabs / expanders / metrics / tables ---------- */
button[data-baseweb="tab"] { color: var(--ld-muted); }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--ld-primary); }
[data-baseweb="tab-highlight"] { background-color: var(--ld-primary) !important; }
[data-baseweb="tab-border"] { background-color: var(--ld-border) !important; }
[data-testid="stExpander"] { border: 1px solid var(--ld-border); border-radius: 12px; background: var(--ld-surface); }
[data-testid="stMetric"] { background: var(--ld-surface); border: 1px solid var(--ld-border); border-radius: 14px; padding: 14px 16px; }
[data-testid="stMetricLabel"] p { color: var(--ld-muted) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--ld-border); border-radius: 12px; overflow: hidden; }
[data-testid="stAlert"] { border-radius: 12px; }
[data-testid="stElementContainer"]:has(.ld-surface-mark) { display: none; }
[data-testid="stVerticalBlockBorderWrapper"]:has(.ld-surface-mark) { background: var(--ld-surface); border: 1px solid var(--ld-border) !important; border-radius: 16px; padding: 6px 8px; }

/* ---------- Brand + decorative ---------- */
.ld-brand { display: flex; align-items: center; gap: 12px; padding: 4px 4px 2px; }
.ld-brand-mark { flex: 0 0 auto; line-height: 0; }
.ld-brand-name { font-family: var(--ld-font-head); font-size: 1.22rem; font-weight: 600; color: var(--ld-text); line-height: 1.1; }
.ld-brand-sub { font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ld-muted); margin-top: 4px; }
.ld-side-label { font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase; color: var(--ld-muted); font-weight: 600; margin: 4px 4px 6px; }
.ld-divider { position: relative; height: 1px; background: var(--ld-border); margin: 16px 0; }
.ld-divider::after { content: ""; position: absolute; left: 50%; top: -4px; width: 7px; height: 7px; transform: translateX(-50%) rotate(45deg); background: var(--ld-bg); border: 1px solid var(--ld-primary); }
section[data-testid="stSidebar"] .ld-divider::after { background: var(--ld-sidebar); }

/* ---------- Status pills ---------- */
.ld-pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px; font-size: 0.7rem; font-weight: 600; border: 1px solid transparent; white-space: nowrap; }
.ld-pill::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.ld-tone-success { color: var(--ld-accent-text); background: var(--ld-accent-soft); border-color: var(--ld-accent); }
.ld-tone-warning { color: var(--ld-warning); background: var(--ld-warning-soft); border-color: var(--ld-warning); }
.ld-tone-danger { color: var(--ld-danger); background: var(--ld-danger-soft); border-color: var(--ld-danger); }
.ld-tone-neutral { color: var(--ld-muted); background: var(--ld-surface-alt); border-color: var(--ld-border); }

/* ---------- Model status card ---------- */
.ld-status-card { background: var(--ld-surface); border: 1px solid var(--ld-border); border-radius: 14px; padding: 12px 14px; }
.ld-status-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.ld-status-title { font-weight: 600; font-size: 0.82rem; color: var(--ld-text); }
.ld-status-row { display: flex; justify-content: space-between; gap: 10px; font-size: 0.76rem; padding: 3px 0; color: var(--ld-muted); }
.ld-status-row b { color: var(--ld-text); font-weight: 500; text-align: right; max-width: 62%; overflow-wrap: anywhere; }

/* ---------- Page components ---------- */
.ld-page-header { margin-bottom: 1.2rem; }
.ld-eyebrow { font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase; color: var(--ld-primary); font-weight: 600; margin-bottom: 4px; }
.ld-page-title { font-family: var(--ld-font-head); font-size: 2.1rem; line-height: 1.15; margin: 0; padding: 0; }
.ld-page-sub { color: var(--ld-muted); font-size: 0.98rem; margin-top: 6px; }
.ld-card { background: var(--ld-surface); border: 1px solid var(--ld-border); border-radius: 16px; padding: 18px 20px; height: 100%; }
.ld-card-title { font-family: var(--ld-font-head); font-weight: 600; font-size: 1.04rem; color: var(--ld-text); }
.ld-card-body { color: var(--ld-muted); font-size: 0.9rem; margin-top: 6px; line-height: 1.5; }
.ld-card-footer { color: var(--ld-muted); font-size: 0.75rem; margin-top: 12px; padding-top: 10px; border-top: 1px dashed var(--ld-border); }
.ld-metric-label { font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ld-muted); font-weight: 600; }
.ld-metric-value { font-family: var(--ld-font-head); font-size: 2rem; font-weight: 600; color: var(--ld-text); line-height: 1.15; margin-top: 4px; }
.ld-metric-delta { font-size: 0.8rem; margin-top: 4px; font-weight: 500; }
.ld-callout { border-left: 3px solid var(--ld-border); background: var(--ld-surface); border-radius: 4px 12px 12px 4px; padding: 12px 16px; margin: 8px 0; color: var(--ld-text); font-size: 0.92rem; }
.ld-callout-title { font-weight: 600; margin-bottom: 2px; }
.ld-callout.ld-tone-success { border-left-color: var(--ld-accent); color: var(--ld-text); background: var(--ld-accent-soft); }
.ld-callout.ld-tone-warning { border-left-color: var(--ld-warning); color: var(--ld-text); background: var(--ld-warning-soft); }
.ld-callout.ld-tone-danger { border-left-color: var(--ld-danger); color: var(--ld-text); background: var(--ld-danger-soft); }
.ld-callout.ld-tone-neutral { border-left-color: var(--ld-primary); color: var(--ld-text); background: var(--ld-surface); }
.ld-delta-success { color: var(--ld-accent-text); } .ld-delta-warning { color: var(--ld-warning); } .ld-delta-danger { color: var(--ld-danger); } .ld-delta-neutral { color: var(--ld-muted); }
"""

TONES = ("success", "warning", "danger", "neutral")


# ----------------------------------------------------------------------------
# Low-level helpers
# ----------------------------------------------------------------------------
def _esc(value) -> str:
    return html.escape("" if value is None else str(value))


def _one_line(markup: str) -> str:
    """Streamlit's markdown breaks HTML blocks at blank lines, so collapse everything."""
    return " ".join(line.strip() for line in markup.splitlines() if line.strip())


def _html(markup: str) -> None:
    st.markdown(_one_line(markup), unsafe_allow_html=True)


def _tone(tone: Optional[str]) -> str:
    return tone if tone in TONES else "neutral"


def _star_points(cx: float, cy: float, r_out: float, r_in: float, n: int = 8) -> str:
    pts = []
    for i in range(n * 2):
        r = r_out if i % 2 == 0 else r_in
        ang = math.pi * i / n - math.pi / 2
        pts.append(f"{cx + r * math.cos(ang):.2f},{cy + r * math.sin(ang):.2f}")
    return " ".join(pts)


def _pattern_uri(color: str) -> str:
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='48' height='48' viewBox='0 0 48 48'>"
        f"<g fill='none' stroke='{color}' stroke-opacity='0.14' stroke-width='1'>"
        "<path d='M24 3 L45 24 L24 45 L3 24 Z'/><path d='M24 13 L35 24 L24 35 L13 24 Z'/></g>"
        f"<circle cx='24' cy='24' r='1.6' fill='{color}' fill-opacity='0.2'/></svg>"
    )
    return "data:image/svg+xml," + quote(svg, safe="")


# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
def get_theme(theme_key: str = THEME_KEY) -> str:
    """Return 'light' or 'dark'. Tolerates bool or 'Dark'/'Light' style values."""
    value = st.session_state.get(theme_key, DEFAULT_THEME)
    if isinstance(value, bool):
        return "dark" if value else "light"
    if isinstance(value, str):
        return "dark" if "dark" in value.lower() else "light"
    return DEFAULT_THEME


def set_theme(theme: str, theme_key: str = THEME_KEY) -> None:
    st.session_state[theme_key] = "dark" if str(theme).lower() == "dark" else "light"


def toggle_theme(theme_key: str = THEME_KEY) -> str:
    set_theme("light" if get_theme(theme_key) == "dark" else "dark", theme_key)
    return get_theme(theme_key)


def get_palette(theme: Optional[str] = None) -> Dict[str, str]:
    return dict(PALETTES[theme or get_theme()])


def build_css(theme: str) -> str:
    p = PALETTES[theme]
    variables = " ".join(f"--ld-{k.replace('_', '-')}: {v};" for k, v in p.items())
    root = (
        ":root, .stApp { " + variables +
        ' --ld-font-head: "Fraunces", Georgia, "Times New Roman", serif;'
        ' --ld-font-body: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;'
        f' --ld-pattern: url("{_pattern_uri(p["primary"])}");'
        f" color-scheme: {theme}; }}"
    )
    return _one_line(_FONT_IMPORT + "\n" + root + "\n" + _BASE_CSS)


def inject_theme_css(theme: Optional[str] = None) -> str:
    """Inject the CSS for the active theme. Safe to call on every rerun."""
    theme = theme or get_theme()
    st.markdown(f"<style>{build_css(theme)}</style>", unsafe_allow_html=True)
    return theme


def export_assets(out_dir) -> List[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for theme in PALETTES:
        (out / f"theme_{theme}.css").write_text(build_css(theme), encoding="utf-8")
    tokens = {"brand": {"name": BRAND_NAME, "subtitle": BRAND_SUBTITLE},
              "version": __version__, "palettes": PALETTES}
    (out / "design_tokens.json").write_text(json.dumps(tokens, indent=2), encoding="utf-8")
    return sorted(p.name for p in out.iterdir())


def _lum(hex_color: str) -> float:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast_ratio(fg: str, bg: str) -> float:
    a, b = _lum(fg), _lum(bg)
    hi, lo = max(a, b), min(a, b)
    return (hi + 0.05) / (lo + 0.05)


def chart_colors(theme: Optional[str] = None) -> List[str]:
    theme = theme or get_theme()
    if theme == "dark":
        return ["#C9735B", "#86AB9B", "#D0A257", "#A98BAE", "#C9A27A", "#6F9486"]
    return ["#A8442E", "#6E9586", "#B9822F", "#8A6F8F", "#C9A27A", "#5E7F73"]


def apply_plotly_theme(fig, theme: Optional[str] = None):
    """Theme a Plotly figure (works on any object with update_layout/update_xaxes)."""
    theme = theme or get_theme()
    p = PALETTES[theme]
    fig.update_layout(
        paper_bgcolor=p["surface"], plot_bgcolor=p["surface"],
        font=dict(color=p["text"], family="Inter, system-ui, sans-serif"),
        colorway=chart_colors(theme), margin=dict(l=24, r=24, t=48, b=24),
    )
    fig.update_xaxes(gridcolor=p["border"], zerolinecolor=p["border"], linecolor=p["border"])
    fig.update_yaxes(gridcolor=p["border"], zerolinecolor=p["border"], linecolor=p["border"])
    return fig


def apply_matplotlib_theme(theme: Optional[str] = None) -> None:
    """Set matplotlib rcParams to match the active theme (call before plotting)."""
    import matplotlib as mpl
    from cycler import cycler
    theme = theme or get_theme()
    p = PALETTES[theme]
    mpl.rcParams.update({
        "figure.facecolor": p["surface"], "axes.facecolor": p["surface"],
        "savefig.facecolor": p["surface"], "axes.edgecolor": p["border"],
        "axes.labelcolor": p["text"], "text.color": p["text"],
        "xtick.color": p["muted"], "ytick.color": p["muted"],
        "grid.color": p["border"], "axes.prop_cycle": cycler(color=chart_colors(theme)),
    })


# ----------------------------------------------------------------------------
# HTML builders (pure functions — easy to test)
# ----------------------------------------------------------------------------
def brand_html(theme: Optional[str] = None) -> str:
    p = PALETTES[theme or get_theme()]
    star = _star_points(21, 21, 14, 7.2)
    mark = (
        "<svg width='42' height='42' viewBox='0 0 42 42' xmlns='http://www.w3.org/2000/svg'>"
        f"<rect width='42' height='42' rx='12' fill='{p['primary']}'/>"
        f"<polygon points='{star}' fill='none' stroke='{p['primary_text']}' stroke-width='1.4' stroke-linejoin='round'/>"
        f"<circle cx='21' cy='21' r='3' fill='{p['primary_text']}'/></svg>"
    )
    return (
        f"<div class='ld-brand'><div class='ld-brand-mark'>{mark}</div><div>"
        f"<div class='ld-brand-name'>{_esc(BRAND_NAME)}</div>"
        f"<div class='ld-brand-sub'>{_esc(BRAND_SUBTITLE)}</div></div></div>"
    )


def status_pill_html(text: str, tone: str = "neutral") -> str:
    return f"<span class='ld-pill ld-tone-{_tone(tone)}'>{_esc(text)}</span>"


def card_html(title: Optional[str] = None, body: str = "", footer: Optional[str] = None,
              body_is_html: bool = False) -> str:
    parts = ["<div class='ld-card'>"]
    if title:
        parts.append(f"<div class='ld-card-title'>{_esc(title)}</div>")
    if body:
        parts.append(f"<div class='ld-card-body'>{body if body_is_html else _esc(body)}</div>")
    if footer:
        parts.append(f"<div class='ld-card-footer'>{_esc(footer)}</div>")
    parts.append("</div>")
    return "".join(parts)


def metric_card_html(label: str, value, delta: Optional[str] = None, tone: str = "neutral") -> str:
    d = f"<div class='ld-metric-delta ld-delta-{_tone(tone)}'>{_esc(delta)}</div>" if delta else ""
    return (f"<div class='ld-card'><div class='ld-metric-label'>{_esc(label)}</div>"
            f"<div class='ld-metric-value'>{_esc(value)}</div>{d}</div>")


def callout_html(text: str, tone: str = "neutral", title: Optional[str] = None) -> str:
    t = f"<div class='ld-callout-title'>{_esc(title)}</div>" if title else ""
    return f"<div class='ld-callout ld-tone-{_tone(tone)}'>{t}{_esc(text)}</div>"


def page_header_html(title: str, subtitle: Optional[str] = None, eyebrow: Optional[str] = None) -> str:
    e = f"<div class='ld-eyebrow'>{_esc(eyebrow)}</div>" if eyebrow else ""
    s = f"<div class='ld-page-sub'>{_esc(subtitle)}</div>" if subtitle else ""
    return f"<div class='ld-page-header'>{e}<h1 class='ld-page-title'>{_esc(title)}</h1>{s}</div>"


def model_status_html(status: Dict) -> str:
    state = status.get("state", "missing")
    tone = {"ready": "success", "degraded": "warning"}.get(state, "danger")
    label = {"ready": "Ready", "degraded": "Degraded"}.get(state, "Not found")
    rows = "".join(
        f"<div class='ld-status-row'><span>{_esc(k)}</span><b>{_esc(v)}</b></div>"
        for k, v in status.get("rows", [])
    )
    return (
        "<div class='ld-status-card'><div class='ld-status-head'>"
        f"<span class='ld-status-title'>Model status</span>{status_pill_html(label, tone)}</div>{rows}</div>"
    )


# ----------------------------------------------------------------------------
# Render helpers (call Streamlit)
# ----------------------------------------------------------------------------
def page_header(title: str, subtitle: Optional[str] = None, eyebrow: Optional[str] = None) -> None:
    _html(page_header_html(title, subtitle, eyebrow))
    _html("<div class='ld-divider'></div>")


def card(title: Optional[str] = None, body: str = "", footer: Optional[str] = None,
         body_is_html: bool = False) -> None:
    _html(card_html(title, body, footer, body_is_html))


def metric_card(label: str, value, delta: Optional[str] = None, tone: str = "neutral") -> None:
    _html(metric_card_html(label, value, delta, tone))


def callout(text: str, tone: str = "neutral", title: Optional[str] = None) -> None:
    _html(callout_html(text, tone, title))


def status_pill(text: str, tone: str = "neutral") -> None:
    _html(status_pill_html(text, tone))


def section_divider() -> None:
    _html("<div class='ld-divider'></div>")


def surface():
    """Bordered container for widgets: `with surface(): st.slider(...)`."""
    box = st.container(border=True)
    with box:
        st.markdown("<span class='ld-surface-mark'></span>", unsafe_allow_html=True)
    return box


# ----------------------------------------------------------------------------
# Model status discovery (file inspection only — never loads a model)
# ----------------------------------------------------------------------------
_MODEL_EXT = {".pkl", ".joblib", ".sav", ".h5", ".keras", ".onnx", ".pt", ".pth"}
_SKIP_DIRS = {"parallel_work", "venv", ".venv", "env", "node_modules", "__pycache__", "sample_data"}


def _find_repo_root() -> Path:
    starts = [Path.cwd()]
    try:
        starts.append(Path(__file__).resolve().parent)
    except NameError:
        pass
    for start in starts:
        for cand in [start, *start.parents]:
            if (cand / ".git").exists() or (cand / "app.py").exists():
                return cand
    return Path.cwd()


def _scan_models(root: str) -> Dict:
    found: List[Tuple[Path, float]] = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in _SKIP_DIRS and not d.startswith(".")]
        for f in fn:
            if Path(f).suffix.lower() in _MODEL_EXT:
                p = Path(dp) / f
                try:
                    found.append((p, p.stat().st_mtime))
                except OSError:
                    pass
    if not found:
        return {"state": "missing",
                "rows": [("Artifact", "No model file found"), ("Checked", "pkl · joblib · h5 · onnx · pt")]}
    named = [x for x in found if "model" in x[0].name.lower()]
    best = max(named or found, key=lambda x: x[1])
    p, mtime = best
    return {"state": "ready", "rows": [
        ("Model file", p.name),
        ("Artifacts", f"{len(found)} found"),
        ("Updated", datetime.fromtimestamp(mtime).strftime("%d %b %Y, %H:%M")),
        ("Size", f"{p.stat().st_size / 1024:,.0f} KB"),
    ]}


try:
    _scan_models_cached = st.cache_data(ttl=30, show_spinner=False)(_scan_models)
except Exception:  # pragma: no cover
    _scan_models_cached = _scan_models


def discover_model_status(repo_root=None) -> Dict:
    root = str(repo_root) if repo_root else str(_find_repo_root())
    try:
        return _scan_models_cached(root)
    except Exception:
        return _scan_models(root)


def render_model_status(status: Optional[Dict] = None) -> None:
    _html(model_status_html(status or discover_model_status()))


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
def _go(nav_key: str, page_key: str, callback: Optional[Callable[[str], None]]) -> None:
    st.session_state[nav_key] = page_key
    if callback:
        callback(page_key)


def _on_theme_toggle(theme_key: str, toggle_key: str) -> None:
    st.session_state[theme_key] = "dark" if st.session_state.get(toggle_key) else "light"


def render_sidebar(
    pages: Optional[Sequence[Tuple[str, str, str]]] = None,
    *,
    nav_key: str = NAV_KEY,
    theme_key: str = THEME_KEY,
    model_status: Optional[Dict] = None,
    show_model_status: bool = True,
    footer: Optional[str] = None,
    on_navigate: Optional[Callable[[str], None]] = None,
) -> str:
    """Render brand, navigation, model status and theme toggle. Returns active page key."""
    pages = tuple(pages or NAV_ITEMS)
    keys = [p[0] for p in pages]
    if st.session_state.get(nav_key) not in keys:
        st.session_state[nav_key] = keys[0]
    theme = get_theme(theme_key)
    inject_theme_css(theme)
    active = st.session_state[nav_key]

    with st.sidebar:
        _html(brand_html(theme))
        _html("<div class='ld-divider'></div>")
        _html("<div class='ld-side-label'>Navigate</div>")
        for key, label, glyph in pages:
            st.button(
                f"{glyph}   {label}",
                key=f"{nav_key}__{key}",
                type="primary" if key == active else "secondary",
                on_click=_go,
                args=(nav_key, key, on_navigate),
            )
        if show_model_status:
            _html("<div class='ld-divider'></div>")
            render_model_status(model_status)
        _html("<div class='ld-divider'></div>")
        _html("<div class='ld-side-label'>Appearance</div>")
        toggle_key = f"{theme_key}__toggle"
        st.session_state[toggle_key] = theme == "dark"
        control = st.toggle if hasattr(st, "toggle") else st.checkbox
        control("Dark mode", key=toggle_key, on_change=_on_theme_toggle, args=(theme_key, toggle_key))
        if footer:
            _html(f"<div class='ld-side-label' style='margin-top:14px'>{_esc(footer)}</div>")
    return st.session_state[nav_key]

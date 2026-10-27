"""Design system for the Loan Analytics Streamlit app.

Palettes, one CSS injector, small HTML component helpers and theme-aware
Plotly helpers. Nothing here touches the model, predictions, SHAP,
monitoring or prediction history.
"""
from __future__ import annotations

import html
from urllib.parse import quote

import streamlit as st

LIGHT = {
    "bg": "#F7F3EE", "sidebar": "#EFE9E1", "card": "#FFFFFF",
    "text": "#292321", "muted": "#6B605A", "border": "#E2D9CF",
    "accent": "#A8442E", "accent_hover": "#8F3925", "on_accent": "#FFFFFF",
    "accent_soft": "#F4E3DC", "accent_text": "#A8442E",
    "sage": "#6E9586", "sage_text": "#3F6455", "sage_soft": "#E6EFEB",
    "ochre": "#B8862F", "ochre_text": "#7A5614", "ochre_soft": "#F6EBD3",
    "rose": "#A9667A", "neutral_soft": "#EEE8E0", "plot_grid": "#E8E0D6",
}
DARK = {
    "bg": "#19161F", "sidebar": "#141118", "card": "#25212C",
    "text": "#F7F0E8", "muted": "#B8ADA9", "border": "#3A3441",
    "accent": "#D98B76", "accent_hover": "#E49E8B", "on_accent": "#19161F",
    "accent_soft": "#3A2A2C", "accent_text": "#E8A593",
    "sage": "#8DB3A3", "sage_text": "#A9CBBB", "sage_soft": "#243029",
    "ochre": "#C9A25A", "ochre_text": "#DDB878", "ochre_soft": "#35301F",
    "rose": "#C48B9F", "neutral_soft": "#2F2A36", "plot_grid": "#342F3B",
}
PALETTES = {"light": LIGHT, "dark": DARK}

DISPLAY_FONT = '"Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif'
BODY_FONT = '"Source Sans Pro", "Segoe UI", system-ui, -apple-system, sans-serif'
SPACING = {"xs": 4, "sm": 8, "md": 16, "lg": 24, "xl": 40}
BADGE_KINDS = ("success", "warning", "danger", "neutral")

_VAR_NAMES = (
    "bg", "sidebar", "card", "text", "muted", "border", "accent",
    "accent_hover", "on_accent", "accent_soft", "accent_text", "sage",
    "sage_text", "sage_soft", "ochre", "ochre_text", "ochre_soft", "rose",
    "neutral_soft",
)


# ----------------------------------------------------------------- theme --
def current_theme():
    """Return 'light' or 'dark' for the active Streamlit theme, else None."""
    try:
        value = st.context.theme.type
        if value in PALETTES:
            return value
    except Exception:
        pass
    try:
        base = st.get_option("theme.base")
        if base in PALETTES:
            return base
    except Exception:
        pass
    return None


def get_palette(theme=None):
    return PALETTES.get(theme or current_theme(), LIGHT)


def _motif_uri(color):
    """Flat eight-pointed-star lattice (jaali-style) as an SVG data URI."""
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='56' height='56' "
        "viewBox='0 0 56 56'><g fill='none' stroke='{c}' stroke-width='1'>"
        "<rect x='16' y='16' width='24' height='24'/>"
        "<rect x='16' y='16' width='24' height='24' transform='rotate(45 28 28)'/>"
        "<circle cx='28' cy='28' r='3.5'/>"
        "<path d='M0 28 L28 0 L56 28 L28 56 Z'/></g></svg>"
    ).format(c=color)
    return 'url("data:image/svg+xml,' + quote(svg, safe="") + '")'


def _vars(p):
    out = "".join("--ll-%s:%s;" % (n.replace("_", "-"), p[n]) for n in _VAR_NAMES)
    out += "--ll-motif:%s;" % _motif_uri(p["accent"])
    out += "--ll-display:%s;" % DISPLAY_FONT
    out += "--ll-radius:14px;"
    out += "".join("--ll-sp-%s:%spx;" % (k, v) for k, v in SPACING.items())
    return out


_BASE_RULES = [
    # page surfaces
    ".stApp{background:var(--ll-bg);color:var(--ll-text);}",
    "[data-testid='stHeader']{background:transparent;}",
    "[data-testid='stSidebar']{background:var(--ll-sidebar);border-right:1px solid var(--ll-border);}",
    "h1,h2,h3,h4{font-family:var(--ll-display);color:var(--ll-text);font-weight:600;letter-spacing:-0.01em;}",
    ".stMarkdown,.stMarkdown p,[data-testid='stWidgetLabel'] p{color:var(--ll-text);}",
    "[data-testid='stCaptionContainer'],[data-testid='stCaptionContainer'] p{color:var(--ll-muted);}",
    "hr{border-color:var(--ll-border);}",
    # native metric -> card
    "[data-testid='stMetric']{background:var(--ll-card);border:1px solid var(--ll-border);border-radius:var(--ll-radius);padding:18px 20px;}",
    "[data-testid='stMetricLabel'] p{color:var(--ll-muted);font-size:.78rem;letter-spacing:.06em;text-transform:uppercase;}",
    "[data-testid='stMetricValue']{color:var(--ll-text);font-family:var(--ll-display);}",
    # forms, tables, bordered containers
    "[data-testid='stForm']{background:var(--ll-card);border:1px solid var(--ll-border);border-radius:var(--ll-radius);}",
    "[data-testid='stDataFrame'],[data-testid='stTable']{border:1px solid var(--ll-border);border-radius:12px;overflow:hidden;}",
    "[data-testid='stVerticalBlockBorderWrapper']{border-color:var(--ll-border);border-radius:var(--ll-radius);}",
    # alerts: one warm neutral treatment (no default blue)
    "[data-testid='stAlert']{border-radius:12px;}",
    "[data-testid='stAlert'] [data-baseweb='notification']{background:var(--ll-neutral-soft);border:1px solid var(--ll-border);border-left:3px solid var(--ll-accent);color:var(--ll-text);border-radius:12px;}",
    "[data-testid='stAlert'] [data-baseweb='notification'] p{color:var(--ll-text);}",
    # buttons
    ".stButton>button,.stDownloadButton>button{background:var(--ll-card);color:var(--ll-accent-text);border:1px solid var(--ll-accent);border-radius:10px;font-weight:600;box-shadow:none;transition:background-color .15s ease,color .15s ease;}",
    ".stButton>button p,.stDownloadButton>button p{color:inherit;}",
    ".stButton>button:hover,.stDownloadButton>button:hover{background:var(--ll-accent-soft);color:var(--ll-accent-text);border-color:var(--ll-accent);}",
    ".stFormSubmitButton>button{background:var(--ll-accent);color:var(--ll-on-accent);border:1px solid var(--ll-accent);border-radius:10px;font-weight:600;box-shadow:none;transition:background-color .15s ease;}",
    ".stFormSubmitButton>button p{color:inherit;}",
    ".stFormSubmitButton>button:hover{background:var(--ll-accent-hover);border-color:var(--ll-accent-hover);color:var(--ll-on-accent);}",
    ".stButton>button:focus-visible,.stFormSubmitButton>button:focus-visible{outline:2px solid var(--ll-sage);outline-offset:2px;}",
    # components: card
    ".ll-card{background:var(--ll-card);border:1px solid var(--ll-border);border-radius:var(--ll-radius);padding:20px 22px;margin-bottom:var(--ll-sp-md);}",
    ".ll-card--accent{border-left:3px solid var(--ll-accent);}",
    ".ll-card--sage{border-left:3px solid var(--ll-sage);}",
    ".ll-card__label{font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:var(--ll-muted);font-weight:600;}",
    ".ll-card__value{font-family:var(--ll-display);font-size:1.9rem;line-height:1.2;color:var(--ll-text);margin-top:6px;}",
    ".ll-card__body{color:var(--ll-text);font-size:.95rem;line-height:1.6;margin-top:8px;}",
    ".ll-card__caption{color:var(--ll-muted);font-size:.85rem;margin-top:6px;}",
    ".ll-card__badge{margin-top:10px;}",
    # components: section header
    ".ll-section{margin:var(--ll-sp-lg) 0 var(--ll-sp-md);}",
    ".ll-eyebrow{font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;color:var(--ll-accent-text);}",
    ".ll-section__title{font-family:var(--ll-display);font-size:1.5rem;font-weight:600;color:var(--ll-text);margin:4px 0 0;}",
    ".ll-section__desc{color:var(--ll-muted);font-size:.95rem;margin-top:6px;max-width:60ch;}",
    ".ll-section__rule{display:flex;align-items:center;gap:8px;margin-top:12px;}",
    ".ll-section__rule::before{content:'';width:36px;height:2px;background:var(--ll-accent);}",
    ".ll-section__rule::after{content:'';flex:1;height:1px;background:var(--ll-border);}",
    ".ll-diamond{display:inline-block;width:7px;height:7px;background:var(--ll-accent);transform:rotate(45deg);}",
    # components: badges
    ".ll-badge{display:inline-flex;align-items:center;gap:6px;padding:3px 10px;border-radius:999px;font-size:.75rem;font-weight:600;line-height:1.4;border:1px solid transparent;}",
    ".ll-badge::before{content:'';width:6px;height:6px;border-radius:50%;background:currentColor;}",
    ".ll-badge--success{background:var(--ll-sage-soft);color:var(--ll-sage-text);border-color:var(--ll-sage);}",
    ".ll-badge--warning{background:var(--ll-ochre-soft);color:var(--ll-ochre-text);border-color:var(--ll-ochre);}",
    ".ll-badge--danger{background:var(--ll-accent-soft);color:var(--ll-accent-text);border-color:var(--ll-accent);}",
    ".ll-badge--neutral{background:var(--ll-neutral-soft);color:var(--ll-text);border-color:var(--ll-border);}",
    # components: hero banner (flat, with a subtle geometric lattice)
    ".ll-hero{position:relative;overflow:hidden;background:var(--ll-card);border:1px solid var(--ll-border);border-radius:20px;padding:28px 32px;margin:8px 0 24px;}",
    ".ll-hero::after{content:'';position:absolute;top:0;right:0;bottom:0;width:34%;background-image:var(--ll-motif);background-size:56px 56px;background-position:center;opacity:.35;pointer-events:none;}",
    ".ll-hero>*{position:relative;z-index:1;max-width:62%;}",
    ".ll-hero__title{font-family:var(--ll-display);font-size:clamp(1.7rem,3.2vw,2.4rem);line-height:1.15;font-weight:600;color:var(--ll-text);margin:10px 0 8px;}",
    ".ll-hero__sub{color:var(--ll-muted);font-size:.98rem;line-height:1.6;}",
    "@media (max-width:640px){.ll-hero::after{display:none;}.ll-hero>*{max-width:100%;}}",
    "@media (prefers-reduced-motion:reduce){*,*::before,*::after{transition:none!important;animation:none!important;}}",
]


def build_css(theme=None):
    """Return one minified <style> block. theme=None emits both palettes."""
    if theme in PALETTES:
        root = ":root{%s}" % _vars(PALETTES[theme])
    else:
        root = ":root{%s}@media (prefers-color-scheme: dark){:root{%s}}" % (
            _vars(LIGHT), _vars(DARK))
    return "<style>" + root + "".join(_BASE_RULES) + "</style>"


def apply_theme():
    """Inject the design system CSS. Call once, right after set_page_config."""
    theme = current_theme()
    st.markdown(build_css(theme), unsafe_allow_html=True)
    st.markdown("""<style>
/* Loan prediction: accessible form input contrast */
.stApp input,
.stApp textarea,
.stApp [data-baseweb="input"] input,
.stApp [data-baseweb="textarea"] textarea {
    color: #F8FAFC !important;
    -webkit-text-fill-color: #F8FAFC !important;
    caret-color: #FFFFFF !important;
    opacity: 1 !important;
}
.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #D1CBD6 !important;
    -webkit-text-fill-color: #D1CBD6 !important;
    opacity: 1 !important;
}
.stApp [data-baseweb="select"] *,
.stApp [role="combobox"] {
    color: #F8FAFC !important;
    -webkit-text-fill-color: #F8FAFC !important;
}
.stApp [data-baseweb="menu"],
.stApp [role="listbox"],
.stApp [role="option"] {
    color: #F8FAFC !important;
}
</style>""", unsafe_allow_html=True)
    return theme or "auto"


# ------------------------------------------------------------ components --
def _esc(value):
    return html.escape(str(value), quote=True)


def _render(markup):
    st.markdown(markup, unsafe_allow_html=True)


def spacer(size="md"):
    """Vertical space. size: xs, sm, md, lg, xl (or a pixel int)."""
    px = size if isinstance(size, int) else SPACING.get(size, SPACING["md"])
    _render("<div style='height:%dpx' aria-hidden='true'></div>" % px)


def badge_html(text, kind="neutral"):
    kind = kind if kind in BADGE_KINDS else "neutral"
    return "<span class='ll-badge ll-badge--%s'>%s</span>" % (kind, _esc(text))


def status_badge(text, kind="neutral"):
    _render(badge_html(text, kind))


def card_html(body=None, title=None, accent=None, allow_html=False):
    mod = " ll-card--%s" % accent if accent in ("accent", "sage") else ""
    parts = ["<div class='ll-card%s'>" % mod]
    if title:
        parts.append("<div class='ll-card__label'>%s</div>" % _esc(title))
    if body:
        text = body if allow_html else _esc(body)
        parts.append("<div class='ll-card__body'>%s</div>" % text)
    parts.append("</div>")
    return "".join(parts)


def card(body=None, title=None, accent=None, allow_html=False):
    _render(card_html(body, title, accent, allow_html))


def metric_card_html(label, value, caption=None, badge_text=None,
                     badge_kind="neutral", accent=None):
    mod = " ll-card--%s" % accent if accent in ("accent", "sage") else ""
    parts = [
        "<div class='ll-card%s'>" % mod,
        "<div class='ll-card__label'>%s</div>" % _esc(label),
        "<div class='ll-card__value'>%s</div>" % _esc(value),
    ]
    if caption:
        parts.append("<div class='ll-card__caption'>%s</div>" % _esc(caption))
    if badge_text:
        parts.append("<div class='ll-card__badge'>%s</div>" % badge_html(badge_text, badge_kind))
    parts.append("</div>")
    return "".join(parts)


def metric_card(label, value, caption=None, badge_text=None,
                badge_kind="neutral", accent=None):
    _render(metric_card_html(label, value, caption, badge_text, badge_kind, accent))


def section_header_html(title, eyebrow=None, description=None):
    parts = ["<div class='ll-section'>"]
    if eyebrow:
        parts.append("<div class='ll-eyebrow'>%s</div>" % _esc(eyebrow))
    parts.append("<div class='ll-section__title' role='heading' aria-level='2'>%s</div>" % _esc(title))
    if description:
        parts.append("<div class='ll-section__desc'>%s</div>" % _esc(description))
    parts.append("<div class='ll-section__rule'><span class='ll-diamond'></span></div></div>")
    return "".join(parts)


def section_header(title, eyebrow=None, description=None):
    _render(section_header_html(title, eyebrow, description))


def hero_banner_html(eyebrow, title, subtitle=None):
    parts = [
        "<div class='ll-hero'>",
        "<div class='ll-eyebrow'>%s</div>" % _esc(eyebrow),
        "<div class='ll-hero__title' role='heading' aria-level='2'>%s</div>" % _esc(title),
    ]
    if subtitle:
        parts.append("<div class='ll-hero__sub'>%s</div>" % _esc(subtitle))
    parts.append("</div>")
    return "".join(parts)


def hero_banner(eyebrow, title, subtitle=None):
    _render(hero_banner_html(eyebrow, title, subtitle))


def card_container():
    """A native bordered container (styled as a card). Use with `with`."""
    return st.container(border=True)


# --------------------------------------------------------------- plotly --
def plotly_layout(theme=None):
    """Layout dict for Plotly figures, matching the active palette."""
    p = get_palette(theme)
    axis = dict(
        gridcolor=p["plot_grid"], linecolor=p["border"], zeroline=False,
        ticks="", tickfont=dict(color=p["muted"]),
        title=dict(font=dict(color=p["muted"])),
    )
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=BODY_FONT, color=p["text"], size=13),
        colorway=[p["accent"], p["sage"], p["ochre"], p["rose"], p["muted"]],
        margin=dict(l=8, r=8, t=44, b=8),
        title=dict(font=dict(family=DISPLAY_FONT, size=16, color=p["text"]),
                   x=0, xanchor="left"),
        xaxis=dict(axis), yaxis=dict(axis),
        legend=dict(font=dict(color=p["muted"]), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=p["card"], bordercolor=p["border"],
                        font=dict(color=p["text"])),
        bargap=0.35,
    )


def style_figure(fig, height=None, theme=None):
    """Apply the design-system look to any Plotly figure and return it."""
    fig.update_layout(**plotly_layout(theme))
    if height:
        fig.update_layout(height=height)
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    return fig


def bar_figure(labels, values, title=None, horizontal=False, height=300, theme=None):
    """A ready-styled bar chart from two sequences."""
    import plotly.graph_objects as go

    labels, values = list(labels), list(values)
    if horizontal:
        fig = go.Figure(go.Bar(x=values, y=labels, orientation="h"))
    else:
        fig = go.Figure(go.Bar(x=labels, y=values))
    if title:
        fig.update_layout(title_text=title)
    return style_figure(fig, height=height, theme=theme)


def show_plotly(fig, key=None):
    """Render a Plotly figure without Streamlit overriding its theme."""
    kwargs = {"theme": None, "config": {"displayModeBar": False}}
    if key:
        kwargs["key"] = key
    try:
        st.plotly_chart(fig, width="stretch", **kwargs)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, **kwargs)


# ------------------------------------------------------------- utilities --
def contrast_ratio(hex_a, hex_b):
    """WCAG contrast ratio between two #RRGGBB colours."""
    def lum(h):
        h = h.lstrip("#")
        rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
        lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
        return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    la, lb = lum(hex_a), lum(hex_b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)

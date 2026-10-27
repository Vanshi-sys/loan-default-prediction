"""Credit Risk Overview page (Step 4).

Self-contained Streamlit page. Reads model_metadata.json only; it contains no
model, preprocessing, or prediction logic and no hardcoded metric values.

Public entry point:  render_overview(metadata_path=None, metadata=None) -> None
The host app must call st.set_page_config() itself (this module never does).
"""
from __future__ import annotations

import html
import inspect
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import plotly.graph_objects as go
import streamlit as st

__all__ = [
    "OverviewMetrics",
    "find_metadata_path",
    "load_metadata",
    "extract_overview_metrics",
    "format_metric",
    "header_html",
    "kpi_cards_html",
    "pipeline_html",
    "explanations_html",
    "scope_note_html",
    "build_metrics_figure",
    "build_feature_expansion_figure",
    "render_overview",
]

METADATA_FILENAME = "model_metadata.json"
TERRACOTTA = "#C4623F"
SAGE = "#7F9C85"
EM_DASH = "—"

_CSS = """
<style>
:root{--ov-terra:#C4623F;--ov-sage:#7F9C85;}
.ov-header{margin:0 0 .6rem 0}
.ov-title{font-size:1.5rem;font-weight:700;line-height:1.2;margin:0;letter-spacing:-.01em}
.ov-sub{font-size:.9rem;opacity:.75;margin:.25rem 0 0 0;max-width:60rem;line-height:1.45}
.ov-grid{display:grid;gap:.7rem;margin:.4rem 0 .8rem 0}
.ov-kpis{grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
.ov-two{grid-template-columns:repeat(auto-fit,minmax(290px,1fr))}
.ov-card{border:1px solid rgba(128,128,128,.28);background:rgba(128,128,128,.07);border-radius:12px;padding:.75rem 1rem}
.ov-kpi{border-top:4px solid var(--ov-accent,#C4623F)}
.ov-kpi-label{font-size:.74rem;text-transform:uppercase;letter-spacing:.05em;opacity:.7;margin:0}
.ov-kpi-value{font-size:1.7rem;font-weight:700;line-height:1.15;margin:.2rem 0 .1rem 0}
.ov-kpi-note{font-size:.78rem;opacity:.65;margin:0;line-height:1.35}
.ov-section{font-size:1.02rem;font-weight:650;margin:.5rem 0 .1rem 0}
.ov-card h4{font-size:.95rem;font-weight:650;margin:0 0 .35rem 0}
.ov-card p{font-size:.86rem;line-height:1.5;margin:.25rem 0}
.ov-card .ov-hint{opacity:.7;font-size:.8rem}
.ov-flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:.5rem;margin:.4rem 0 .6rem 0}
.ov-step{flex:1 1 200px;border:1px solid rgba(128,128,128,.28);background:rgba(128,128,128,.07);border-radius:12px;padding:.7rem .9rem;border-left:4px solid var(--ov-accent,#7F9C85)}
.ov-step-label{font-size:.74rem;text-transform:uppercase;letter-spacing:.05em;opacity:.7;margin:0}
.ov-step-value{font-size:1.45rem;font-weight:700;line-height:1.2;margin:.1rem 0}
.ov-step-note{font-size:.8rem;opacity:.7;margin:0;line-height:1.4}
.ov-arrow{display:flex;align-items:center;justify-content:center;font-size:1.5rem;color:var(--ov-terra);flex:0 0 auto;padding:0 .1rem}
.ov-arrow::before{content:"→"}
@media (max-width:640px){.ov-arrow{flex:1 1 100%}.ov-arrow::before{content:"↓"}}
.ov-scope{border-left:4px solid var(--ov-terra);background:rgba(196,98,63,.09);border-radius:8px;padding:.6rem .9rem;font-size:.84rem;line-height:1.5;margin:.6rem 0 .2rem 0}
</style>
"""

# --------------------------------------------------------------------------
# Metadata discovery
# --------------------------------------------------------------------------
def find_metadata_path(explicit: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Return the path to model_metadata.json, or None if it cannot be found."""
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    here = Path(__file__).resolve()
    for root in list(here.parents[:4]) + [Path.cwd()]:
        candidate = root / METADATA_FILENAME
        if candidate.is_file():
            return candidate
    return None


def load_metadata(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load metadata JSON. Raises FileNotFoundError or ValueError on failure."""
    p = find_metadata_path(path)
    if p is None:
        raise FileNotFoundError(f"{METADATA_FILENAME} was not found.")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{p.name} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{p.name} must contain a JSON object at the top level.")
    return data


# --------------------------------------------------------------------------
# Metric extraction (key-name discovery; no hardcoded values)
# --------------------------------------------------------------------------
def _norm(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).lower()).strip("_")


def _flatten(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = _norm(k)
            out.extend(_flatten(v, f"{prefix}_{key}" if prefix else key))
    elif isinstance(obj, list):
        if prefix:
            out.append((prefix, obj))
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                out.extend(_flatten(v, f"{prefix}_{i}" if prefix else str(i)))
    elif prefix:
        out.append((prefix, obj))
    return out


def _to_number(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    elif isinstance(v, str):
        try:
            x = float(v.strip())
        except ValueError:
            return None
    else:
        return None
    return x if math.isfinite(x) else None


def _to_count(v: Any) -> Optional[int]:
    if isinstance(v, (list, tuple)):
        return len(v) if len(v) > 0 else None
    x = _to_number(v)
    if x is not None and x > 0 and float(x).is_integer():
        return int(x)
    return None


_EXCLUDED = ("train", "training", "test", "testing", "cv", "cross", "holdout", "oof")
_VAL_TOKENS = ("val", "valid", "validation")


def _excluded(path: str) -> bool:
    toks = path.split("_")
    compact = path.replace("_", "")
    return any(t in toks for t in _EXCLUDED) or any(compact.startswith(t) for t in _EXCLUDED)


def _is_val(path: str) -> bool:
    toks = path.split("_")
    return any(t in toks for t in _VAL_TOKENS) or "val" in path.replace("_", "")


def _is_roc(path: str) -> bool:
    c = path.replace("_", "")
    return "rocauc" in c or "auroc" in c


def _is_pr(path: str) -> bool:
    c = path.replace("_", "")
    if _is_roc(path):
        return False
    return any(s in c for s in ("prauc", "auprc", "averageprecision", "avgprecision", "precisionrecallauc"))


def _pick_score(flat, predicate, low=0.0, high=1.0, inclusive_low=True):
    best = None
    for path, val in flat:
        if not predicate(path) or _excluded(path):
            continue
        x = _to_number(val)
        if x is None or x > high or x < low or (not inclusive_low and x == low):
            continue
        rank = (1 if _is_val(path) else 0, -len(path))
        if best is None or rank > best[0]:
            best = (rank, path, x)
    return (best[2], best[1]) if best else (None, None)


_INPUT_ALIASES = (
    "n_features_in", "num_input_features", "n_input_features", "input_feature_count",
    "input_features", "raw_features", "raw_input_features", "n_raw_features",
    "original_features", "feature_names_in", "input_columns",
)
_INPUT_GENERIC = ("n_features", "num_features", "feature_columns", "feature_names", "features")
_INPUT_BANNED = ("transformed", "encoded", "processed", "preprocessed", "out", "output", "expanded", "after")
_TRANSFORMED_ALIASES = (
    "n_transformed_features", "num_transformed_features", "transformed_feature_count",
    "transformed_features", "transformed_feature_names", "n_features_transformed",
    "feature_names_out", "n_features_out", "n_encoded_features", "encoded_features",
    "n_processed_features", "processed_features", "preprocessed_features",
    "expanded_features", "output_features",
)
_RATE_ALIASES = (
    "positive_class_rate", "positive_rate", "positive_share", "prevalence",
    "base_rate", "default_rate", "class_1_rate",
)


def _pick_count(flat, aliases, generic=(), banned=()):
    def ok(path):
        return not any(b in path.split("_") for b in banned)

    for alias in aliases:
        best = None
        for path, val in flat:
            if alias not in path or not ok(path):
                continue
            n = _to_count(val)
            if n is None:
                continue
            if best is None or len(path) < len(best[0]):
                best = (path, n)
        if best:
            return best[1], best[0]
    for alias in generic:
        for path, val in flat:
            if (path == alias or path.endswith("_" + alias)) and ok(path):
                n = _to_count(val)
                if n is not None:
                    return n, path
    return None, None


@dataclass
class OverviewMetrics:
    roc_auc: Optional[float] = None
    pr_auc: Optional[float] = None
    n_input: Optional[int] = None
    n_transformed: Optional[int] = None
    positive_rate: Optional[float] = None
    sources: Dict[str, str] = field(default_factory=dict)

    @property
    def expansion_factor(self) -> Optional[float]:
        if self.n_input and self.n_transformed:
            return self.n_transformed / self.n_input
        return None


def extract_overview_metrics(metadata: Any) -> OverviewMetrics:
    """Pull the four overview values from metadata; anything missing stays None."""
    flat = _flatten(metadata) if isinstance(metadata, (dict, list)) else []
    m = OverviewMetrics()

    m.roc_auc, p = _pick_score(flat, _is_roc)
    if p:
        m.sources["roc_auc"] = p
    m.pr_auc, p = _pick_score(flat, _is_pr)
    if p:
        m.sources["pr_auc"] = p

    m.n_input, p = _pick_count(flat, _INPUT_ALIASES, _INPUT_GENERIC, _INPUT_BANNED)
    if p:
        m.sources["n_input"] = p
    m.n_transformed, p = _pick_count(flat, _TRANSFORMED_ALIASES)
    if p:
        m.sources["n_transformed"] = p

    for alias in _RATE_ALIASES:
        hit = [
            (path, _to_number(v)) for path, v in flat
            if alias in path and not _excluded(path) and _to_number(v) is not None
            and 0.0 < _to_number(v) < 1.0
        ]
        if hit:
            hit.sort(key=lambda t: len(t[0]))
            m.positive_rate, m.sources["positive_rate"] = hit[0][1], hit[0][0]
            break
    return m


def format_metric(value: Optional[float], kind: str = "score") -> str:
    """Format a score (4 decimals) or a count (thousands separator); None -> em dash."""
    if value is None:
        return EM_DASH
    if kind == "count":
        return f"{int(round(value)):,}"
    return f"{value:.4f}"


# --------------------------------------------------------------------------
# HTML builders (no blank lines / indentation so Markdown keeps them as HTML)
# --------------------------------------------------------------------------
_esc = html.escape


def header_html() -> str:
    return (
        '<div class="ov-header">'
        '<p class="ov-title">Credit Risk Overview</p>'
        '<p class="ov-sub">How the saved loan-default model performed on its validation data, '
        'and how raw input fields expand into the features the model actually uses. '
        'These are model-level results, not a prediction for any individual borrower.</p>'
        "</div>"
    )


def kpi_cards_html(m: OverviewMetrics) -> str:
    specs = [
        ("Validation ROC-AUC", m.roc_auc, "score", "Ranking quality on the validation split", TERRACOTTA, "roc_auc"),
        ("Validation PR-AUC", m.pr_auc, "score", "Precision–recall summary on the validation split", SAGE, "pr_auc"),
        ("Input features", m.n_input, "count", "Fields accepted by the saved pipeline", TERRACOTTA, "n_input"),
        ("Transformed features", m.n_transformed, "count", "Columns after preprocessing", SAGE, "n_transformed"),
    ]
    cards = []
    for label, value, kind, note, accent, key in specs:
        src = m.sources.get(key)
        tip = f"Read from {METADATA_FILENAME}: {src}" if src else f"Not found in {METADATA_FILENAME}"
        shown_note = note if value is not None else f"Not found in {METADATA_FILENAME}"
        cards.append(
            f'<div class="ov-card ov-kpi" style="--ov-accent:{accent}" title="{_esc(tip)}">'
            f'<p class="ov-kpi-label">{_esc(label)}</p>'
            f'<p class="ov-kpi-value">{_esc(format_metric(value, kind))}</p>'
            f'<p class="ov-kpi-note">{_esc(shown_note)}</p></div>'
        )
    return '<div class="ov-grid ov-kpis">' + "".join(cards) + "</div>"


def pipeline_html(m: OverviewMetrics) -> str:
    n_in = format_metric(m.n_input, "count")
    n_tr = format_metric(m.n_transformed, "count")
    factor = m.expansion_factor
    factor_txt = f" · {factor:.1f}× more columns" if factor else ""
    return (
        '<div class="ov-flow">'
        '<div class="ov-step" style="--ov-accent:#7F9C85">'
        '<p class="ov-step-label">Input fields</p>'
        f'<p class="ov-step-value">{_esc(n_in)}</p>'
        '<p class="ov-step-note">Raw fields passed to the saved pipeline</p></div>'
        '<div class="ov-arrow"></div>'
        '<div class="ov-step" style="--ov-accent:#C4623F">'
        '<p class="ov-step-label">Preprocessing</p>'
        '<p class="ov-step-value">Saved pipeline</p>'
        '<p class="ov-step-note">Runs inside the stored model artifact, so no preprocessing logic is duplicated here</p></div>'
        '<div class="ov-arrow"></div>'
        '<div class="ov-step" style="--ov-accent:#7F9C85">'
        '<p class="ov-step-label">Transformed features</p>'
        f'<p class="ov-step-value">{_esc(n_tr)}</p>'
        f'<p class="ov-step-note">Model-ready columns{_esc(factor_txt)}</p></div>'
        "</div>"
    )


def explanations_html(m: OverviewMetrics) -> str:
    roc_extra = ""
    if m.roc_auc is not None:
        roc_extra = (
            f'<p>With a validation ROC-AUC of {m.roc_auc:.4f}, about {round(m.roc_auc * 100)} of every '
            "100 such pairs are ordered correctly.</p>"
        )
    if m.positive_rate is not None:
        pr_extra = (
            f'<p>The metadata lists a positive-class share of {m.positive_rate:.4f}, '
            "which is the approximate PR-AUC of random scoring.</p>"
        )
    else:
        pr_extra = (
            '<p class="ov-hint">The positive-class share of the validation data is not recorded in the '
            "metadata, so no random-scoring baseline is drawn.</p>"
        )
    return (
        '<div class="ov-grid ov-two">'
        '<div class="ov-card" style="border-top:4px solid #C4623F">'
        "<h4>ROC-AUC: how well does the model rank cases?</h4>"
        "<p>Pick one case from each outcome class at random. ROC-AUC is the chance that the model "
        "gives the higher score to the positive-class case (encoded target 1). "
        "0.5 is no better than random ordering; 1.0 is a perfect ordering.</p>"
        f"{roc_extra}"
        '<p class="ov-hint">It measures ordering only. It says nothing about how large any single score or probability should be.</p>'
        "</div>"
        '<div class="ov-card" style="border-top:4px solid #7F9C85">'
        "<h4>PR-AUC: how good is the precision–recall trade-off?</h4>"
        "<p>Precision asks: of the cases the model flags, how many are truly positive? "
        "Recall asks: of all truly positive cases, how many did the model find? "
        "PR-AUC summarizes that trade-off across all thresholds.</p>"
        "<p>It is most informative when the positive class is uncommon, because random scoring "
        "earns roughly the positive-class share, not 0.5. Judge PR-AUC against that share.</p>"
        f"{pr_extra}"
        "</div></div>"
    )


def scope_note_html() -> str:
    return (
        '<div class="ov-scope"><b>Scope of these numbers.</b> They describe the saved model on its '
        "validation split. They are not the probability for any specific borrower, and they do not "
        "establish probability calibration, fairness, or suitability for lending decisions. "
        "This page reads the saved metadata only and shows no live production traffic.</div>"
    )


# --------------------------------------------------------------------------
# Plotly figures
# --------------------------------------------------------------------------
def _theme_font_color() -> Optional[str]:
    """Explicit text colour only when the active theme can be detected."""
    try:
        t = st.context.theme.type  # Streamlit >= 1.46
        if t in ("dark", "light"):
            return "#E8E6E3" if t == "dark" else "#262730"
    except Exception:
        pass
    try:
        base = st.get_option("theme.base")
        if base in ("dark", "light"):
            return "#E8E6E3" if base == "dark" else "#262730"
    except Exception:
        pass
    return None


def _base_layout(fig, title, height):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#292321")),
        height=max(height, 250),
        font=dict(family="Arial", size=13, color="#292321"),
        title_font=dict(family="Arial", size=14, color="#292321"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        uniformtext_minsize=12,
        uniformtext_mode="show",
        margin=dict(l=65, r=35, t=75, b=75),
    )
    fig.update_xaxes(
        showline=True, linecolor="#64748B", linewidth=1.5,
        ticks="outside", tickcolor="#64748B",
        tickfont=dict(size=12, color="#334155"),
        showgrid=True, gridcolor="#D9DDE1", gridwidth=1,
        zeroline=False, automargin=True, fixedrange=True,
    )
    fig.update_yaxes(
        showline=True, linecolor="#64748B", linewidth=1.5,
        ticks="outside", tickcolor="#64748B",
        tickfont=dict(size=13, color="#334155"),
        showgrid=True, gridcolor="#D9DDE1", gridwidth=1,
        zeroline=False, automargin=True, fixedrange=True,
    )
    return fig

def build_metrics_figure(m: OverviewMetrics) -> Optional[go.Figure]:
    """Horizontal bars for validation ROC-AUC / PR-AUC with direct labels."""
    rows = []
    if m.roc_auc is not None:
        rows.append(dict(
            label="ROC-AUC", value=m.roc_auc, color=TERRACOTTA, ref=0.5,
            ref_text="Random ordering = 0.5",
            desc="Chance a random positive-class case is scored above a random negative-class case",
        ))
    if m.pr_auc is not None:
        rows.append(dict(
            label="PR-AUC", value=m.pr_auc, color=SAGE, ref=m.positive_rate,
            ref_text="Positive-class share (from metadata)",
            desc="Precision–recall trade-off across thresholds; compare with the positive-class share, not 0.5",
        ))
    if not rows:
        return None
    rows = rows[::-1]  # horizontal bars plot bottom-up; keep ROC-AUC on top

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=[r["label"] for r in rows],
        x=[r["value"] for r in rows],
        orientation="h",
        marker=dict(color=[r["color"] for r in rows]),
        text=[f"{r['value']:.4f}" for r in rows],
        textposition="outside",
        textfont=dict(size=13),
        cliponaxis=False,
        customdata=[[r["desc"]] for r in rows],
        name="Validation score",
        showlegend=False,
        hovertemplate="<b>%{y}</b><br>Validation score: %{x:.4f}<br>%{customdata[0]}<extra></extra>",
    ))
    refs = [r for r in rows if r["ref"] is not None]
    if refs:
        fig.add_trace(go.Scatter(
            x=[r["ref"] for r in refs],
            y=[r["label"] for r in refs],
            mode="markers",
            marker=dict(symbol="line-ns-open", size=34, line=dict(width=3, color="rgba(128,128,128,0.95)")),
            text=[r["ref_text"] for r in refs],
            textposition="middle right",
            textfont=dict(color="#292321", size=12, family="Arial"),
            name="No-skill reference",
            hovertemplate="<b>%{y}</b><br>%{text}: %{x:.4f}<extra></extra>",
        ))
    _base_layout(fig, "Validation metrics (higher is better)", 120 + 62 * len(rows))
    fig.update_xaxes(
        range=[0, 1.12], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], tickformat=".1f",
        title=dict(text="Score (0 to 1)", font=dict(size=12, color="#292321"), standoff=12),
    )
    fig.update_layout(showlegend=bool(refs))
    return fig


def build_feature_expansion_figure(m: OverviewMetrics) -> Optional[go.Figure]:
    """Horizontal bars comparing input fields with transformed features."""
    rows = []
    if m.n_input is not None:
        rows.append(("Input fields", m.n_input, SAGE))
    if m.n_transformed is not None:
        rows.append(("Transformed features", m.n_transformed, TERRACOTTA))
    if not rows:
        return None
    rows = rows[::-1]
    factor = m.expansion_factor
    note = f"{factor:.1f}× the input fields" if (factor and m.n_transformed) else ""
    fig = go.Figure(go.Bar(
        y=[r[0] for r in rows],
        x=[r[1] for r in rows],
        orientation="h",
        marker=dict(color=[r[2] for r in rows]),
        text=[f"{r[1]:,}" for r in rows],
        textposition="outside",
        textfont=dict(size=13),
        cliponaxis=False,
        customdata=[[note if r[0] == "Transformed features" else "Fields supplied to the pipeline"] for r in rows],
        hovertemplate="<b>%{y}</b><br>Count: %{x:,}<br>%{customdata[0]}<extra></extra>",
    ))
    _base_layout(fig, "Feature count before and after preprocessing", 120 + 62 * len(rows))
    fig.update_xaxes(
        range=[0, max(r[1] for r in rows) * 1.2], tickformat=",d",
        title=dict(text="Number of columns", font=dict(size=12, color="#292321"), standoff=12),
    )
    fig.update_layout(showlegend=False)
    return fig


def _show_chart(fig: go.Figure, key: str) -> None:
    params = inspect.signature(st.plotly_chart).parameters
    kwargs = dict(key=key, theme="streamlit", config={"displayModeBar": False, "responsive": True})
    if "width" in params:
        kwargs["width"] = "stretch"
    else:
        kwargs["use_container_width"] = True
    st.plotly_chart(fig, **kwargs)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
def render_overview(
    metadata_path: Optional[Union[str, Path]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Render the Credit Risk Overview page.

    metadata_path: optional explicit path to model_metadata.json.
    metadata: optional already-loaded dict (skips file loading).
    """
    source_name = None
    if metadata is None:
        try:
            path = find_metadata_path(metadata_path)
            if path is None:
                raise FileNotFoundError(f"{METADATA_FILENAME} was not found.")
            metadata = load_metadata(path)
            source_name = path.name
        except (OSError, ValueError) as exc:
            st.error(f"Could not load model metadata: {exc}")
            metadata = {}
    else:
        source_name = "provided metadata"

    m = extract_overview_metrics(metadata)

    st.markdown(_CSS + header_html(), unsafe_allow_html=True)
    st.markdown(kpi_cards_html(m), unsafe_allow_html=True)

    left, right = st.columns(2, gap="medium")
    with left:
        fig = build_metrics_figure(m)
        if fig is not None:
            _show_chart(fig, "ov_metrics_chart")
        else:
            st.info("Validation ROC-AUC / PR-AUC were not found in the metadata.")
    with right:
        fig = build_feature_expansion_figure(m)
        if fig is not None:
            _show_chart(fig, "ov_features_chart")
        else:
            st.info("Feature counts were not found in the metadata.")

    st.markdown('<p class="ov-section">What the metrics mean</p>' + explanations_html(m), unsafe_allow_html=True)
    st.markdown('<p class="ov-section">Feature pipeline</p>' + pipeline_html(m), unsafe_allow_html=True)
    st.markdown(scope_note_html(), unsafe_allow_html=True)
    if source_name:
        st.caption(f"Source: {source_name}. Values missing from the file are shown as {EM_DASH}.")

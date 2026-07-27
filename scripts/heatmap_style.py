"""Styled heatmap renderer (no model dependency).

Pure matplotlib + numpy. Takes rows of {label, tokens, acts, group} and renders
a clean, editorial heatmap: single-hue sequential ramp (magnitude -> one blue,
light->dark, per the data-viz method), tile gaps, language grouping, luminance-
picked cell text, slim colorbar. Shared by 04 (compute) and 04b (restyle).
"""
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

# --- Design tokens (from the validated reference palette) --------------------
LIGHT = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "muted": "#52514e",
    # blue sequential 100->700, prefixed with the surface so "near zero" recedes
    "ramp": ["#fcfcfb", "#eaf2fd", "#cde2fb", "#9ec5f4",
             "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
}
DARK = {
    "surface": "#1a1a19",
    "ink": "#ffffff",
    "muted": "#c3c2b7",
    "ramp": ["#1a1a19", "#123152", "#184f95", "#256abf",
             "#3987e5", "#5598e7", "#86b6ef", "#b7d3f6", "#cde2fb"],
}

GROUP_ORDER = ["English", "Deutsch", "Русский", "日本語"]

# Font stack with CJK fallback (matplotlib >=3.6 does per-glyph fallback).
_FONT_STACK = ["Segoe UI", "Noto Sans", "Noto Sans CJK JP", "Yu Gothic",
               "Meiryo", "DejaVu Sans"]
# On Colab (apt install fonts-noto-cjk) the font must be registered explicitly.
_CJK_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def _register_fonts():
    from matplotlib import font_manager
    for p in _CJK_FONT_PATHS:
        try:
            font_manager.fontManager.addfont(p)
        except Exception:
            pass


def detect_group(text: str) -> str:
    for ch in text:
        o = ord(ch)
        if 0x3040 <= o <= 0x30FF or 0x4E00 <= o <= 0x9FFF:
            return "日本語"
    for ch in text:
        if 0x0400 <= ord(ch) <= 0x04FF:
            return "Русский"
    low = text.lower()
    if "hauptstadt" in low or "frankreich" in low or "deutsch" in low:
        return "Deutsch"
    return "English"


def _cmap(mode):
    tok = LIGHT if mode == "light" else DARK
    return LinearSegmentedColormap.from_list(f"france_{mode}", tok["ramp"]), tok


def _text_color(rgba, tok):
    r, g, b = rgba[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # Contrast against the cell itself, not the theme: dark ink on light/bright
    # cells, white on dark cells. Fixes white-on-bright-blue blending in dark mode.
    return "#111111" if lum > 0.6 else "#ffffff"


def render_heatmap(rows, out_path, mode="light",
                   title="One direction, many surfaces",
                   subtitle="Feature 20583 (layer 12) activation across prompts · Qwen3.5-2B-Base"):
    _register_fonts()
    matplotlib.rcParams["font.family"] = _FONT_STACK
    import matplotlib.pyplot as plt

    cmap, tok = _cmap(mode)
    surface = tok["surface"]

    # order by language group, then insert a spacer row between groups
    ordered = sorted(rows, key=lambda r: GROUP_ORDER.index(r["group"]))
    disp, groups_at = [], []
    prev = None
    for r in ordered:
        if prev is not None and r["group"] != prev:
            disp.append(None)  # spacer
        disp.append(r)
        groups_at.append((len(disp) - 1, r["group"]))
        prev = r["group"]

    n = len(disp)
    max_len = max(len(r["tokens"]) for r in rows)
    M = np.full((n, max_len), np.nan)
    for i, r in enumerate(disp):
        if r is None:
            continue
        a = r["acts"]
        M[i, : len(a)] = a
    vmax = float(np.nanmax(M)) or 1.0
    Mm = np.ma.masked_invalid(M)
    cmap.set_bad(surface)

    fig, ax = plt.subplots(figsize=(max(11, max_len * 0.82), n * 0.46 + 1.7))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    mesh = ax.pcolormesh(np.arange(max_len + 1), np.arange(n + 1), Mm,
                         cmap=cmap, vmin=0, vmax=vmax,
                         edgecolors=surface, linewidth=2.0)
    ax.set_ylim(n, 0)  # first row on top
    ax.set_xlim(0, max_len)

    # per-cell token text, colour picked for contrast
    for i, r in enumerate(disp):
        if r is None:
            continue
        for j, t in enumerate(r["tokens"]):
            t = t.strip()
            if not t:
                continue
            v = M[i, j]
            rgba = cmap(0.0 if np.isnan(v) else v / vmax)
            ax.text(j + 0.5, i + 0.5, t[:9], ha="center", va="center",
                    fontsize=7, color=_text_color(rgba, tok))

    # the tokens in each row already spell the sentence, so no left labels:
    # the left margin is reserved for group names only.
    ax.set_yticks([])
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    # group labels + separators in the far-left margin
    spans = {}
    for idx, g in groups_at:
        spans.setdefault(g, [idx, idx])
        spans[g][0] = min(spans[g][0], idx)
        spans[g][1] = max(spans[g][1], idx)
    for g, (lo, hi) in spans.items():
        ax.text(-0.5, (lo + hi) / 2 + 0.5, g.upper(),
                ha="center", va="center", rotation=90, fontsize=10,
                color=tok["muted"], weight="bold")

    # title / subtitle, left-aligned above the plot
    ax.set_title("")
    fig.text(0.012, 0.985, title, ha="left", va="top",
             fontsize=16, color=tok["ink"], weight="bold")
    fig.text(0.012, 0.945, subtitle, ha="left", va="top",
             fontsize=10, color=tok["muted"])

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.018, pad=0.015)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8, colors=tok["muted"])
    cbar.set_label("feature activation", fontsize=9, color=tok["muted"])

    fig.subplots_adjust(top=0.90, left=0.05, right=0.94, bottom=0.03)
    fig.savefig(out_path, dpi=200, facecolor=surface, bbox_inches="tight")
    plt.close(fig)
    return out_path

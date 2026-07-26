"""Hero figure: the France feature across languages, scripts, and cultural markers.

For each prompt, records the feature's activation on every token at the target
layer and draws a heatmap. The feature fires on France / Frankreich / Франции /
フランス and on "the Louvre" / "the Eiffel Tower", but stays dark for Germany,
Russia, Japan, math, and weather.

Font note: the default DejaVu Sans has no CJK glyphs, so フランス renders as
boxes. ``_set_cjk_font`` installs a CJK-capable font best-effort. On Colab, run
``!apt-get -qq install fonts-noto-cjk`` first.
"""
import _bootstrap  # noqa: F401
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from data.prompts import HEAT_PROMPTS
from france_feature import BEST_FEATURE, TARGET_LAYER, capture, load_model, load_saes

FIG = _bootstrap.FIGURES / "heatmap_france.png"

# Common install paths for a CJK-capable font (Colab: fonts-noto-cjk).
_CJK_FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]
_CJK_FONT_NAMES = ["Noto Sans CJK JP", "Noto Sans CJK SC", "Noto Sans JP",
                   "Arial Unicode MS", "MS Gothic", "Yu Gothic"]


def _set_cjk_font():
    """Best-effort: make matplotlib render CJK glyphs so フランス is not boxes."""
    from matplotlib import font_manager

    for path in _CJK_FONT_PATHS:
        try:
            font_manager.fontManager.addfont(path)
        except Exception:
            pass
    for name in _CJK_FONT_NAMES:
        try:
            font_manager.findfont(name, fallback_to_default=False)
            matplotlib.rcParams["font.family"] = name
            print(f"heatmap font: {name}")
            return
        except Exception:
            continue
    print("WARNING: no CJK font found — フランス will render as boxes. "
          "On Colab run: !apt-get -qq install fonts-noto-cjk")


def main():
    _set_cjk_font()
    model, tokenizer, device = load_model()
    saes = load_saes(model, [TARGET_LAYER], device)
    sae = saes[TARGET_LAYER]

    records = []
    for prompt in HEAT_PROMPTS:
        h, ids = capture(model, tokenizer, prompt, TARGET_LAYER, device)
        z = sae.encode(h.to(sae.W_enc.dtype))
        acts = z[0, :, BEST_FEATURE].float().cpu().numpy()
        tokens = [tokenizer.decode([t]) for t in ids["input_ids"][0].tolist()]
        records.append((prompt, tokens, acts))

    max_len = max(len(t) for _, t, _ in records)
    heat = np.zeros((len(records), max_len))
    for i, (_, _, acts) in enumerate(records):
        heat[i, : len(acts)] = acts

    fig, ax = plt.subplots(figsize=(max(10, max_len * 0.9), len(records) * 0.7))
    im = ax.imshow(heat, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_yticks(range(len(records)))
    ax.set_yticklabels([p[:40] for p, _, _ in records], fontsize=9)
    for i, (_, tokens, _) in enumerate(records):
        for j, tok in enumerate(tokens):
            val = heat[i, j]
            ax.text(j, i, tok.strip()[:8], ha="center", va="center", fontsize=7,
                    color="white" if val > heat.max() * 0.5 else "black")
    ax.set_xlabel("Token position")
    ax.set_title(f"Activation of feature {BEST_FEATURE} (layer {TARGET_LAYER}) across prompts")
    plt.colorbar(im, ax=ax, label="z value")
    plt.tight_layout()
    fig.savefig(FIG, dpi=150, bbox_inches="tight")
    print(f"saved {FIG}")


if __name__ == "__main__":
    main()

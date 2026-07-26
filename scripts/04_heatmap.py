"""Hero figure: the France feature across languages, scripts, and cultural markers.

For each prompt, records the feature's activation on every token at the target
layer, dumps the raw matrix to results/heatmap_data.json, and renders the styled
heatmap (see scripts/heatmap_style.py). The feature fires on France / Frankreich /
Франции / フランス and on "the Louvre" / "the Eiffel Tower", but stays dark for
Germany, Russia, Japan, math, and weather.

Compute and style are separated on purpose: this script needs the model, but once
heatmap_data.json exists, scripts/04b_heatmap_style.py re-renders the figure in
seconds without loading anything. On Colab, run `!apt-get -qq install fonts-noto-cjk`
first so フランス renders instead of boxes.
"""
import json

import _bootstrap  # noqa: F401
from heatmap_style import detect_group, render_heatmap

from data.prompts import HEAT_PROMPTS
from france_feature import BEST_FEATURE, TARGET_LAYER, capture, load_model, load_saes

DATA = _bootstrap.RESULTS / "heatmap_data.json"
FIG_LIGHT = _bootstrap.FIGURES / "heatmap_france.png"
FIG_DARK = _bootstrap.FIGURES / "heatmap_france_dark.png"


def main():
    model, tokenizer, device = load_model()
    saes = load_saes(model, [TARGET_LAYER], device)
    sae = saes[TARGET_LAYER]

    rows = []
    for prompt in HEAT_PROMPTS:
        h, ids = capture(model, tokenizer, prompt, TARGET_LAYER, device)
        z = sae.encode(h.to(sae.W_enc.dtype))
        acts = z[0, :, BEST_FEATURE].float().cpu().numpy().tolist()
        tokens = [tokenizer.decode([t]) for t in ids["input_ids"][0].tolist()]
        rows.append({"label": prompt, "tokens": tokens, "acts": acts,
                     "group": detect_group(prompt)})

    DATA.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"saved {DATA}")

    render_heatmap(rows, FIG_LIGHT, mode="light")
    render_heatmap(rows, FIG_DARK, mode="dark")
    print(f"saved {FIG_LIGHT}\nsaved {FIG_DARK}")


if __name__ == "__main__":
    main()

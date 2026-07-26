"""Re-render the heatmap from cached data, without loading the model.

Run scripts/04_heatmap.py once (it needs the GPU) to produce
results/heatmap_data.json; after that this script restyles the figure in seconds,
so palette and layout can be iterated freely. Also useful for anyone who wants the
figure but doesn't want to run the model: the data file is committed.
"""
import json

import _bootstrap  # noqa: F401
from heatmap_style import render_heatmap

DATA = _bootstrap.RESULTS / "heatmap_data.json"
FIG_LIGHT = _bootstrap.FIGURES / "heatmap_france.png"
FIG_DARK = _bootstrap.FIGURES / "heatmap_france_dark.png"


def main():
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found — run scripts/04_heatmap.py first.")
    rows = json.loads(DATA.read_text(encoding="utf-8"))["rows"]
    render_heatmap(rows, FIG_LIGHT, mode="light")
    render_heatmap(rows, FIG_DARK, mode="dark")
    print(f"saved {FIG_LIGHT}\nsaved {FIG_DARK}")


if __name__ == "__main__":
    main()

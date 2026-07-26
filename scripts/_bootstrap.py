"""Make ``france_feature`` and ``data`` importable when a script is run directly.

Lets ``python scripts/04_heatmap.py`` work in Colab without ``pip install -e .``.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "."):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
FIGURES.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

"""One shared plotting style so every figure in the project looks like one system.

Consistent colours, fonts, and sizes are a small thing that makes an analysis read as
professional rather than ad-hoc. We centralise them here and call ``set_style()`` once
at the top of any script or notebook that draws.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

# A small, colour-blind-friendly palette. TREATED/CONTROL are used consistently so a
# reader learns "blue = emailed, grey = not" once and it holds across every chart.
COLORS = {
    "treated": "#2166AC",   # blue  — received email
    "control": "#B2182B",   # red   — no email
    "neutral": "#4D4D4D",   # dark grey
    "accent": "#1B7837",    # green — highlights / positive lift
}
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#762A83", "#E08214", "#4D4D4D"]


def set_style() -> None:
    """Apply the project's matplotlib/seaborn style. Call once before plotting."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )

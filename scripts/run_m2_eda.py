"""M2 runner: generate every EDA figure to artifacts/figures/ (no notebook needed).

This makes the visuals reproducible from the command line. The notebook
(notebooks/01_eda.ipynb) uses the same functions but adds the written narrative.

Run:
    python scripts/run_m2_eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: save files without opening a window

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdip.config import load_config  # noqa: E402
from mdip.data.load import load_hillstrom  # noqa: E402
from mdip.eda import plots  # noqa: E402
from mdip.logging_setup import get_logger  # noqa: E402
from mdip.viz.style import set_style  # noqa: E402


def main() -> None:
    cfg = load_config()
    log = get_logger("run_m2_eda", cfg.logging.level)
    set_style()

    df = load_hillstrom(cfg)
    fig_dir = cfg.path("figures_dir")
    fig_dir.mkdir(parents=True, exist_ok=True)

    figures = {
        "eda_01_arm_sizes": plots.plot_arm_sizes(df),
        "eda_02_history_distribution": plots.plot_history_distribution(df),
        "eda_03_response_by_arm": plots.plot_response_by_arm(df),
        "eda_04_lift_by_channel": plots.plot_lift_by_attribute(df, by="channel"),
        "eda_05_lift_by_history_segment": plots.plot_lift_by_attribute(df, by="history_segment"),
        "eda_06_correlation": plots.plot_correlation_heatmap(df),
    }

    for name, fig in figures.items():
        out = fig_dir / f"{name}.png"
        fig.savefig(out)
        log.info("saved %s", out)

    log.info("M2 complete: %d figures written to %s", len(figures), fig_dir)


if __name__ == "__main__":
    main()

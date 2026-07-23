"""M3 runner: the experiment readout — does emailing cause more visits/conversions?

Runs the SRM check, proportion z-tests, bootstrap CI for spend, BH correction, and a
power/MDE analysis; prints them and saves forest-plot figures.

Run:
    python scripts/run_m3_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from tabulate import tabulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdip.config import load_config  # noqa: E402
from mdip.data.load import load_hillstrom  # noqa: E402
from mdip.logging_setup import get_logger  # noqa: E402
from mdip.stats import plots  # noqa: E402
from mdip.stats.experiment import run_readout  # noqa: E402
from mdip.stats.power import detectable_effect, required_sample_size  # noqa: E402
from mdip.viz.style import set_style  # noqa: E402


def main() -> None:
    cfg = load_config()
    log = get_logger("run_m3_stats", cfg.logging.level)
    set_style()

    df = load_hillstrom(cfg)
    results = run_readout(df, seed=cfg.seed, alpha=cfg.stats.alpha)

    # 1. SRM check.
    srm = results["srm"]
    print("\n" + "=" * 78 + "\nSAMPLE RATIO MISMATCH CHECK\n" + "=" * 78)
    print(f"observed arms : {srm['observed']}")
    print(f"expected arms : {srm['expected']}")
    print(f"chi2 = {srm['chi2_statistic']:.3f}, p = {srm['p_value']:.3f}")
    print(f"verdict: {srm['verdict']}")

    # 2. Test table.
    tests = results["tests"]
    show = tests.copy()
    for col in ["treated_rate", "control_rate", "lift", "ci_low", "ci_high"]:
        show[col] = show[col].round(5)
    print("\n" + "=" * 78)
    print("EXPERIMENT READOUT (with Benjamini-Hochberg correction)")
    print("=" * 78)
    cols = ["arm", "outcome", "lift", "ci_low", "ci_high", "p_value", "p_adjusted"]
    cols.append("significant_bh")
    print(tabulate(show[cols], headers="keys", tablefmt="github", showindex=False))

    # 3. Power / MDE analysis (on the primary outcome, visit).
    control_visit_rate = df.loc[df["treatment"] == 0, "visit"].mean()
    n_per_arm = int(df["segment"].value_counts().min())
    mde = detectable_effect(control_visit_rate, n_per_arm)
    n_needed = required_sample_size(control_visit_rate, mde_absolute=cfg.stats.mde_lift)
    print("\n" + "=" * 78 + "\nPOWER / MINIMUM DETECTABLE EFFECT (visit)\n" + "=" * 78)
    print(f"control visit rate        : {control_visit_rate:.4f}")
    print(f"customers per arm         : {n_per_arm:,}")
    print(f"MDE at 80% power          : {mde * 100:.3f} pp (smallest lift we could detect)")
    print(f"n/arm to detect +{cfg.stats.mde_lift * 100:.0f}pp : {n_needed:,}")

    # 4. Figures.
    fig_dir = cfg.path("figures_dir")
    fig_dir.mkdir(parents=True, exist_ok=True)
    for outcome in ["visit", "conversion", "spend"]:
        fig = plots.plot_forest(tests, outcome=outcome)
        out = fig_dir / f"stats_forest_{outcome}.png"
        fig.savefig(out)
        log.info("saved %s", out)

    print("\n" + "=" * 78)
    log.info("M3 complete.")


if __name__ == "__main__":
    main()

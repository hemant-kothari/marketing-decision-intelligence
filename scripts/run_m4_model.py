"""M4 runner: train the predictive response model and report its evaluation.

Trains logistic regression and LightGBM to predict P(visit | attributes) among emailed
customers, compares them on held-out PR-AUC / ROC-AUC / calibration, and prints odds
ratios. Saves a PR-curve + calibration figure.

Run:
    python scripts/run_m4_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402
from tabulate import tabulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdip.config import load_config  # noqa: E402
from mdip.data.load import load_hillstrom  # noqa: E402
from mdip.logging_setup import get_logger  # noqa: E402
from mdip.response_model import calibration_points, train_and_evaluate_both  # noqa: E402
from mdip.viz.style import COLORS, set_style  # noqa: E402


def main() -> None:
    cfg = load_config()
    log = get_logger("run_m4_model", cfg.logging.level)
    set_style()

    df = load_hillstrom(cfg)
    res = train_and_evaluate_both(df, seed=cfg.seed)

    # Evaluation table.
    print("\n" + "=" * 78 + "\nRESPONSE MODEL EVALUATION (held-out 25%)\n" + "=" * 78)
    rows = []
    for e in [res["eval_logistic"], res["eval_lightgbm"]]:
        rows.append(
            {
                "model": e.name,
                "PR-AUC": round(e.pr_auc, 4),
                "ROC-AUC": round(e.roc_auc, 4),
                "Brier": round(e.brier, 4),
            }
        )
    print(tabulate(rows, headers="keys", tablefmt="github"))
    print(f"(baseline: visitors are {res['eval_logistic'].positive_rate:.1%} of the test set)")

    # Odds ratios (interpretable output of the logistic model).
    print("\n" + "=" * 78 + "\nLOGISTIC ODDS RATIOS (which attributes drive visits)\n" + "=" * 78)
    print(tabulate(res["odds_ratios"].round(3), headers="keys", tablefmt="github", showindex=False))

    # Figure: PR curves + calibration.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for model, name, color in [
        (res["logistic"], "Logistic", COLORS["control"]),
        (res["lightgbm"], "LightGBM", COLORS["treated"]),
    ]:
        proba = model.predict_proba(res["X_test"])[:, 1]
        prec, rec, _ = precision_recall_curve(res["y_test"], proba)
        axes[0].plot(rec, prec, label=name, color=color)
        mean_pred, frac_pos = calibration_points(model, res["X_test"], res["y_test"])
        axes[1].plot(mean_pred, frac_pos, "o-", label=name, color=color)

    axes[0].axhline(res["eval_logistic"].positive_rate, ls="--", color=COLORS["neutral"],
                    label="no-skill baseline")
    axes[0].set_xlabel("recall")
    axes[0].set_ylabel("precision")
    axes[0].set_title("Precision-Recall curves")
    axes[0].legend()

    axes[1].plot([0, 0.5], [0, 0.5], ls="--", color=COLORS["neutral"], label="perfect calibration")
    axes[1].set_xlabel("mean predicted probability")
    axes[1].set_ylabel("observed visit rate")
    axes[1].set_title("Calibration")
    axes[1].legend()
    fig.tight_layout()

    fig_dir = cfg.path("figures_dir")
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / "model_pr_and_calibration.png"
    fig.savefig(out)
    log.info("saved %s", out)
    print("\n" + "=" * 78)
    log.info("M4 complete.")


if __name__ == "__main__":
    main()

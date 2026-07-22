# Design — Marketing Decision Intelligence Platform (Version 1.0)

**A focused Data Science portfolio project.** This is the technical design; see
[ROADMAP.md](ROADMAP.md) for the milestone plan (concept → business → why-here → assumptions →
interview questions → code, for each milestone).

Version 1.0 is **complete and self-contained**. It is intentionally small and deep, not a
framework awaiting more algorithms.

---

## 1. What this project is

From an email-marketing campaign log, it answers:

1. **Does emailing cause more visits/purchases?** — an experiment readout with a confidence
   interval (M3), confirmed by a causal analysis (M5).
2. **Who is persuadable vs a sure thing vs a lost cause vs a sleeping dog?** — subgroup effects
   plus one uplift model (M6).
3. **Who should we email?** — a concrete targeting recommendation (M6).

The spine: **not "who will buy?" but "who will buy *because* we emailed them?"** — the
distinction that marks a Data Scientist.

## 2. Design priorities (tie-breaker for every decision)

`(1) statistical reasoning → (2) business decision → (3) visualization → (4) interview story →
(5) results-focused README → (6) software structure.`

Consequences, stated so they aren't re-argued:
- **Flat package by analysis stage**, not layered architecture: `data/ eda/ stats/ causal/
  uplift/ viz/`. No Protocols, adapters, or hexagonal layers.
- **One YAML + typed dataclass** for config. No Hydra.
- **Results saved to an `artifacts/` folder.** No MLflow.
- **Hybrid notebooks:** reusable logic in tested `src/` modules; thin notebooks import them and
  carry the EDA / statistics narrative and charts.
- **Type hints, logging, one global seed, and unit tests** stay — cheap, and they back the
  "reproducible and documented" claim.
- **Depth over breadth:** one uplift model (a hand-written two-model T-learner), one matching
  method (PSM) plus IPW as a cross-check. No bake-offs.

## 3. Dataset

**Hillstrom** — 64,000 past purchasers, randomized 1/3 men's email, 1/3 women's email, 1/3
holdout; 14-day `visit`, `conversion`, `spend`.

Because it is an **RCT**, a difference in means is an unbiased causal effect. We use that as a
**known ground truth**: M3 estimates it directly; M5 injects selection bias and shows PSM
recovers it. Treatment is collapsed to **binary** (any email vs control). Primary outcome is
**`visit`** (well-powered); `conversion` (~0.6%) is reported with honest, wider uncertainty;
`spend` is described, not modeled.

## 4. Folder structure

```
causal-inference/
├── conf/config.yaml            # one typed config
├── src/mdip/
│   ├── config.py  logging_setup.py
│   ├── data/     # download, validate, load to DuckDB, selection-bias generator (M5)
│   ├── eda/      # profiling + chart helpers
│   ├── stats/    # z-tests, bootstrap CIs, power/MDE, Benjamini–Hochberg
│   ├── causal/   # DAG notes, propensity scores, PSM, IPW, ATE, recovery demo
│   ├── uplift/   # two-model T-learner, subgroup ATE, segmentation, targeting
│   └── viz/      # shared plotting style
├── sql/          # committed analytical queries (M1)
├── notebooks/    # thin narrative notebooks (import src/)
├── tests/        # pytest — the statistics & metrics that must be correct
├── artifacts/    # computed results + figures (gitignored)
├── scripts/download_data.py
├── ROADMAP.md  DESIGN.md  README.md
```

## 5. Data flow

```
download → validate → DuckDB + SQL analytics (M1)
        → EDA + charts + narrative (M2)
        → experiment readout: SRM, z-tests, bootstrap CIs, power, BH (M3)     ── answers Q1
        → features (RFM) + response model + calibration + SHAP (M4)           ── the baseline to beat
        → propensity scores → PSM + IPW + ATE + recovery demo (M5)            ── confirms Q1 causally
        → subgroup ATEs + T-learner uplift + segmentation + targeting (M6)    ── answers Q2–Q5
        → README + business memo (M7)
```

Analysis invariants (enforced cheaply in code):
- **No post-treatment variable** (`visit`) used as a feature when estimating the effect on
  `conversion` — `visit` is a mediator; conditioning on it is post-treatment bias.
- **One global seed** propagated for reproducibility.
- The **matched/adjusted comparison** is always shown next to its covariate-balance check.

## 6. Methods, and why each is in v1.0

| Stage | Method | Why (interview-ready) |
|---|---|---|
| M3 | Two-proportion z-test | On an RCT it *is* the ATE test for a binary outcome |
| M3 | Bootstrap CI | Distribution-free interval; easy to explain and defend |
| M3 | Power / MDE | "What lift could we even have detected?" — experiment literacy |
| M3 | Benjamini–Hochberg | We run a family of tests; controls false-discovery rate |
| M4 | Logistic regression | Interpretable baseline; odds ratios tell the business story |
| M4 | LightGBM + PR-AUC + calibration | Stronger model, right metric for imbalance, usable probabilities |
| M5 | Propensity Score Matching | The canonical way to remove selection bias on observables |
| M5 | IPW | A second estimator as a cross-check on PSM |
| M5 | Recovery demo | *Proves* PSM recovers the RCT truth instead of asserting it |
| M6 | Subgroup ATEs | Heterogeneity, explained with pure statistics |
| M6 | Two-model T-learner + Qini | One well-understood uplift model, honestly evaluated |

Each method ships with a "why / assumptions / how to explain it" note in code and in its
milestone write-up.

## 7. Out of scope for Version 1.0 (on purpose)

Policy learning / off-policy evaluation; multiple meta-learners and bake-offs; causal forests /
DML; advanced refutation (placebo, random common cause, E-value); Streamlit; DiD / synthetic
control; multi-arm creative assignment; Docker / CI / serving. Listing these keeps v1.0 honest
and complete rather than half-built.

## 8. Reproducibility & quality (lightweight, proportionate)

Single seed in config; `requirements.txt`; `scripts/download_data.py` regenerates the data; unit
tests cover the code where a silent bug would produce a confidently wrong business number (the
statistics and the uplift metrics); `ruff` for lint. That is the entire quality surface — matched
to the size of the project.

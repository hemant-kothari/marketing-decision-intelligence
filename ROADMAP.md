# Roadmap — Marketing Decision Intelligence Platform (Version 1.0)

**A focused, self-contained Data Science portfolio project.** It answers one question a business
actually pays for — *"who should we email, and does emailing them even work?"* — using a real
marketing dataset, and it goes **deep on a small number of techniques** rather than wide on many.

> **Design philosophy (v1.0):** depth over breadth. Every technique in this repo is something I
> can explain from first principles in an interview. This is complete as it stands — not a
> framework waiting for more algorithms. Anything cut is listed under *Future enhancements* and
> would arrive as a separate, clearly-scoped commit later.

---

## Working principles (the contract for this repo)

1. Simplest implementation that correctly demonstrates the concept. No over-engineering.
2. Optimize for **interview quality and depth of understanding**, not architecture.
3. Every milestone produces something I can **understand and explain**.
4. **Before implementing any milestone**, I write it up in six parts:
   **(1)** concept in plain language · **(2)** why businesses care · **(3)** why we use it here ·
   **(4)** key assumptions · **(5)** likely interview questions · **(6)** then the code.
5. Stop at each milestone for review → understand → run locally → commit → **push** (I confirm
   "pushed" before the next milestone begins).
6. Readable over clever. I am the primary author; every design decision is explained to me.

**Tie-breaker for any tradeoff, in priority order:**
`(1) statistical reasoning → (2) business decision → (3) visualization → (4) interview story → (5) results-focused README → (6) then software structure.`

---

## The business problem (framing)

A retailer can email past customers, but emailing has a cost and can annoy people. Two questions:

1. **Does the email campaign actually cause more visits and purchases** — or would those customers
   have come back anyway? (An *experiment / causal* question.)
2. **Which customers should we email** to get the most benefit? (A *targeting* question.)

Most analysts answer a different, easier question — "who is *likely to buy*?" — and target those
people. But some of them would have bought **without** the email, so the spend is wasted. The
Data-Scientist move is to target by **causal uplift**: who buys *because* of the email. This
project builds the full argument for that, end to end, and ends in a concrete targeting
recommendation.

## Dataset

**Hillstrom Email Marketing** — 64,000 past purchasers, **randomized** into 1/3 men's-email,
1/3 women's-email, 1/3 no-email (holdout). Outcomes over 14 days: `visit`, `conversion`, `spend`.

**Why this dataset is perfect for the story:** it is a **randomized controlled trial (RCT)**.
Because treatment was randomized, a simple difference in means is already an *unbiased* causal
effect. That gives us a **known correct answer**, which we exploit twice:
- the experiment readout (M3) estimates the true effect directly, and
- the causal milestone (M5) deliberately *breaks* randomness to create bias, then shows
  Propensity Score Matching **recovers** the known answer. We *prove* our causal method works
  instead of just asserting it.

We collapse the two email arms into **treated (1) vs control (0)** for a clean binary analysis.
Primary outcome is **`visit`** (well-powered, ~10.6% vs ~15%). `conversion` (~0.6%) is rare and
we report its wider uncertainty honestly. `spend` is summarized descriptively, not modeled
directly (a direct model on 99%-zero data is noise).

---

## The five questions this project answers

| # | Question | Milestone |
|---|---|---|
| Q1 | Does emailing cause more visits/purchases, and by how much (with a CI)? | M3, confirmed by M5 |
| Q2 | Which customers are **persuadable** (positive uplift)? | M6 |
| Q3 | Which are **sure things** (would act anyway)? | M6 |
| Q4 | Which are **lost causes / sleeping dogs** (no effect / harmed)? | M6 |
| Q5 | Given all this, **who should we email**? | M6 (targeting recommendation) |

---

## Milestones (Version 1.0)

Seven content milestones. Each is a stop point: review → run → commit → push → continue.

### M0 — Project setup ✅
Venv, dependencies, flat package (`data/ eda/ stats/ causal/ uplift/ viz/`), one typed YAML
config, logging, `.gitignore`, planning docs. *No analysis yet — a clean floor.*

### M1 — Business framing + **SQL analysis**
Load Hillstrom, validate it, put it in **DuckDB**, and answer the opening business questions with
committed `.sql` queries: arm sizes and balance, base rates by customer segment, spend
distribution, first look at raw lift.
- **Concepts:** SQL for analytics (GROUP BY, CTEs, window functions), data validation.
- **Resume value:** puts *evidenced* SQL on the CV (currently SQL is only listed, never shown).
- **Business insight:** the raw shape of the campaign and a first, uncorrected lift number.

### M2 — **Exploratory Data Analysis + visualization**
Distributions, response rates across every customer attribute, correlation structure, and a
**written narrative** of what the data says — in a notebook, with publication-quality charts.
- **Concepts:** EDA, distributions, data visualization, customer profiling, communicating findings.
- **Resume value:** EDA and visualization appear **zero times** on the current resume; this is
  core DS vocabulary.
- **Business insight:** which customers look most engaged (correlation only — flagged as such).

### M3 — **Hypothesis testing + confidence intervals (experiment readout)** ⭐
The statistics centerpiece and the answer to Q1. Sample-Ratio-Mismatch check, **two-proportion
z-test** for visit and conversion lift, **bootstrap confidence intervals**, a **power / minimum
detectable effect** analysis, and a **Benjamini–Hochberg** correction because we run a family of
tests.
- **Concepts:** hypothesis testing, p-values (done correctly), confidence intervals, bootstrap,
  statistical power / MDE, multiple-comparison correction, experiment integrity (SRM), A/B testing.
- **Resume value:** **highest** — hypothesis testing and experimentation are the most-asked DS
  interview topics and are entirely absent today.
- **Business insight:** *email causes a statistically significant +X.X pp lift in visits* (with a
  CI); the conversion signal is weaker and we say exactly how uncertain it is.

### M4 — **Feature engineering + predictive response model**
Engineer marketing features (**RFM**: recency / frequency-proxy / monetary, tenure, channel), then
build a **response-propensity model**: interpretable **logistic regression** first (report **odds
ratios**), then **LightGBM** as the stronger model — justified by a real lift in **PR-AUC** (the
right metric under class imbalance) — with a **calibration** curve and **SHAP** drivers.
- **Concepts:** feature engineering, RFM, predictive modeling, class imbalance, PR-AUC vs ROC-AUC,
  probability calibration, model explainability.
- **Resume value:** turns "feature engineering" from one thin clause into a real competency; also
  builds the **baseline our uplift model must beat**, which motivates the causal half.
- **Business insight:** who is *likely to visit* — and the setup for why that is the **wrong**
  targeting question.

### M5 — **Propensity Score Matching, ATE, and the recovery demo** ⭐
A short written DAG + assumptions ledger, then estimate the causal effect three ways — **naive
difference**, **PSM**, and **IPW** — with a **covariate-balance (love) plot** before/after
matching. Then the flagship demonstration: take the RCT (known answer), **inject selection bias**
to make it observational, show the **naive estimate is now wrong**, and show **PSM recovers the
true effect**.
- **Concepts:** causal inference, potential outcomes, confounding, propensity scores, matching,
  IPW, covariate balance, selection bias.
- **Resume value:** the causal specialization the project exists to signal; the recovery demo is
  the strongest single interview story in the repo.
- **Business insight:** the causal lift **confirms** the experiment readout — and we've *shown*
  the method is trustworthy, not just claimed it.

### M6 — **Heterogeneous effects, segmentation & targeting recommendation** ⭐
Two complementary answers to "who to target":
1. **Subgroup ATEs** — the treatment effect *within* customer segments (recency bands, past
   category, channel), using the same tests from M3/M5. Fully explainable, purely statistical.
2. **One uplift model** — a **two-model T-learner** to score each customer's individual uplift,
   evaluated with a **Qini curve**. Customers are placed on the **baseline-response × uplift**
   grid into **Persuadables / Sure Things / Lost Causes / Sleeping Dogs**, each with a profile.

Then a concrete **targeting recommendation**: which segments/customers to email, and the expected
incremental benefit of doing so vs. emailing everyone.
- **Concepts:** heterogeneous treatment effects, subgroup analysis, uplift modeling (T-learner),
  Qini curve, customer segmentation, the persuadable/sure-thing/lost-cause/sleeping-dog framework.
- **Resume value:** customer-analytics maturity — targeting by *causal uplift*, the distinction
  most candidates miss.
- **Business insight:** the ~X% of customers who are persuadable, the segment actively *harmed*
  by email (sleeping dogs to suppress), and a clear "email these, skip those" recommendation.

### M7 — **Documentation, visuals & business memo**
A **results-first README** (hero chart, TL;DR results box, the five questions, methodology,
honest limitations) and a **one-page memo written for a marketing stakeholder**, not a
statistician. Final pass on chart quality and repo polish.
- **Concepts:** communication of insights, executive storytelling, intellectual honesty.
- **Resume value:** the README is the only artifact guaranteed to be read; the memo proves I can
  talk to non-technical stakeholders.
- **Business insight:** the whole story, on one page, in business language.

---

## When can this honestly go on the resume?

| Checkpoint | Through | Honest claim |
|---|---|---|
| **A — fallback** | M3 | *"Email Campaign Experiment Analysis"* — SQL, EDA, hypothesis testing, experiment readout. Real, but **not yet** a causal / targeting project. |
| **B — the line** | **M6 + M7** | Full title. Causal inference, PSM, uplift, segmentation, predictive modeling, targeting recommendation, communicated clearly. **Cross this before adding it.** |

Do not claim causal inference before M5, or targeting before M6.

---

## Resume gap → milestone map

| Missing competency (from resume review) | Filled by |
|---|---|
| Statistical thinking / hypothesis testing | **M3** |
| Experimentation / A-B testing | **M3** |
| Exploratory Data Analysis | **M2** |
| Data visualization | M2, M6, M7 |
| Feature engineering | **M4** |
| Predictive modeling (DS-framed) | M4 |
| Causal inference | **M5**, M6 |
| Customer / marketing analytics | M2, M6 |
| Model evaluation (right metrics) | M4, M6 |
| Business decision-making | M6, M7 |
| Communication of insights | **M7** |
| SQL (evidenced) | **M1** |

---

## Scope boundary — what is deliberately NOT in Version 1.0

These are **out of scope on purpose**, so the repo reads as complete rather than half-built.
Each could return later as its own commit if there's a reason:

- Policy learning / off-policy evaluation of a targeting policy
- Multiple meta-learners (S / X / DR) and an uplift bake-off
- Causal forests, Double Machine Learning
- Advanced refutation suite (placebo, random-common-cause, E-value, Rosenbaum bounds)
- Streamlit / interactive dashboard
- Difference-in-Differences, synthetic control (not identified on this cross-section anyway)
- Multi-arm creative assignment (men's vs women's email)
- Any MLOps / Docker / CI / serving infrastructure

The v1.0 project stands on its own: framing → SQL → EDA → experiment → prediction → causation →
targeting → communication.

---

## Parallel resume fixes (independent of this project — do this week)

1. Rewrite the three **Airtel** bullets in DS language — baseline beaten, class imbalance, chosen
   threshold and *why* (cost of a false positive vs. a missed churner), retention value at risk,
   who acted on it. Same work, Data-Scientist framing.
2. Rebuild **Skills** around analytics keywords: scikit-learn, statsmodels, scipy, matplotlib,
   seaborn, Plotly, Statistics, Hypothesis Testing, A-B Testing, Causal Inference, Experimentation,
   SQL (analytics). Move SQL out of the "DSA, OOP" line.
3. Fixes: `GPGA` → `CGPA`; remove the duplicated "Achievements:" label.

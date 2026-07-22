-- Q: Collapsing the two email arms into one "treated" group, what is the raw lift?
-- Skill: CTE (WITH), conditional aggregation with FILTER, arithmetic across groups.
-- Business meaning: the single number a stakeholder wants first — "emailing lifted
-- visits by X percentage points". This is the value the statistical test in M3 will
-- put a confidence interval around.
WITH rates AS (
    SELECT
        treatment,                                   -- 1 = any email, 0 = control
        COUNT(*)            AS customers,
        AVG(visit)          AS visit_rate,
        AVG(conversion)     AS conversion_rate,
        AVG(spend)          AS avg_spend
    FROM campaign
    GROUP BY treatment
)
SELECT
    ROUND(MAX(visit_rate)      FILTER (WHERE treatment = 1) * 100, 3) AS treated_visit_pct,
    ROUND(MAX(visit_rate)      FILTER (WHERE treatment = 0) * 100, 3) AS control_visit_pct,
    ROUND((MAX(visit_rate)     FILTER (WHERE treatment = 1)
         - MAX(visit_rate)     FILTER (WHERE treatment = 0)) * 100, 3) AS visit_lift_pp,
    ROUND((MAX(conversion_rate) FILTER (WHERE treatment = 1)
         - MAX(conversion_rate) FILTER (WHERE treatment = 0)) * 100, 3) AS conversion_lift_pp,
    ROUND(MAX(avg_spend)       FILTER (WHERE treatment = 1)
        - MAX(avg_spend)       FILTER (WHERE treatment = 0), 4)        AS spend_lift
FROM rates;

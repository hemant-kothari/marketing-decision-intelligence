-- Q: Which kinds of customers respond most, and does email lift differ across them?
-- Skill: GROUP BY a customer attribute, split treated vs control with FILTER, compute
--        a per-group lift. This is the SQL foundation of the "who to target" question.
-- Business meaning: a first, descriptive look at heterogeneity — do recent buyers or
-- multichannel customers respond more to email? We break visit rate down by recency
-- band here; the same pattern is reused for channel, zip_code, and history in M2/M6.
-- (Descriptive only: subgroup causal effects are estimated properly in M6.)
SELECT
    CASE
        WHEN recency <= 3  THEN '1) 0-3 months'
        WHEN recency <= 6  THEN '2) 4-6 months'
        WHEN recency <= 9  THEN '3) 7-9 months'
        ELSE                    '4) 10+ months'
    END                                                            AS recency_band,
    COUNT(*)                                                       AS customers,
    ROUND(AVG(visit) FILTER (WHERE treatment = 1) * 100, 2)        AS treated_visit_pct,
    ROUND(AVG(visit) FILTER (WHERE treatment = 0) * 100, 2)        AS control_visit_pct,
    ROUND((AVG(visit) FILTER (WHERE treatment = 1)
         - AVG(visit) FILTER (WHERE treatment = 0)) * 100, 2)      AS visit_lift_pp
FROM campaign
GROUP BY recency_band
ORDER BY recency_band;

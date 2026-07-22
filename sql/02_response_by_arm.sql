-- Q: What are the visit, conversion, and spend rates in each arm — and the raw lift?
-- Skill: GROUP BY with AVG() on 0/1 columns (a rate), AVG() on a continuous column.
-- Business meaning: this is the headline "did email work?" table, uncorrected.
-- NOTE: on this dataset the raw treated-vs-control gap IS an unbiased causal effect,
-- because customers were randomised. We still call it "raw lift" here and only claim
-- causality after the experiment readout (M3) and matching analysis (M5) confirm it.
SELECT
    segment                                  AS arm,
    COUNT(*)                                 AS customers,
    ROUND(AVG(visit) * 100, 3)               AS visit_rate_pct,
    ROUND(AVG(conversion) * 100, 3)          AS conversion_rate_pct,
    ROUND(AVG(spend), 4)                     AS avg_spend
FROM campaign
GROUP BY segment
ORDER BY visit_rate_pct DESC;

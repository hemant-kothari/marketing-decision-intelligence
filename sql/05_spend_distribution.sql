-- Q: What does the spend outcome look like — and why won't we model it directly?
-- Skill: aggregate conditional counts, percentiles (quantile_cont), share-of-total.
-- Business meaning: spend is "zero-inflated" — almost everyone spends 0, a tiny few
-- spend a lot. This query makes that visible and justifies a design decision: we
-- analyse visit/conversion (well-behaved 0/1 outcomes) and treat spend descriptively,
-- rather than fitting a noisy model to 99%-zero data.
SELECT
    COUNT(*)                                                   AS customers,
    SUM(CASE WHEN spend = 0 THEN 1 ELSE 0 END)                 AS zero_spend_customers,
    ROUND(100.0 * SUM(CASE WHEN spend = 0 THEN 1 ELSE 0 END)
                / COUNT(*), 2)                                 AS pct_zero_spend,
    ROUND(AVG(spend), 4)                                       AS mean_spend_all,
    ROUND(AVG(spend) FILTER (WHERE spend > 0), 2)              AS mean_spend_spenders,
    ROUND(quantile_cont(spend, 0.99), 2)                       AS p99_spend,
    ROUND(MAX(spend), 2)                                       AS max_spend
FROM campaign;

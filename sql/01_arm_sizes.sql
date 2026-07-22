-- Q: How many customers are in each experiment arm, and does the split look right?
-- Skill: GROUP BY, COUNT, ratio against the total.
-- Business meaning: this is the first integrity check of any A/B test. The design
-- randomised customers 1/3 : 1/3 : 1/3, so each arm should be ~33.3%. A big deviation
-- would be a "Sample Ratio Mismatch" and would make every downstream result suspect.
SELECT
    segment                                             AS arm,
    COUNT(*)                                            AS customers,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)  AS pct_of_total
FROM campaign
GROUP BY segment
ORDER BY customers DESC;

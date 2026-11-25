-- Example queries against the POC schema (see docs/schema.md).
-- Written in portable SQL with two named bind parameters: run_id, and true_value (bind Python True).
-- Comparing the boolean column line_result.binding with the bound true_value makes the same text work on
-- SQLite / PostgreSQL (BOOLEAN) and SQL Server (BIT). Run with sqlalchemy.text(), e.g.
--   pd.read_sql(text(stmt), conn, params={"run_id": 1, "true_value": True})

-- 1. Monte Carlo congestion probability per line, straight from the scenario results
SELECT lr.line_id,
       AVG(CASE WHEN lr.binding = :true_value THEN 1.0 ELSE 0.0 END) AS p_binding,
       COUNT(*)                                                     AS n_scenarios
FROM line_result lr
WHERE lr.run_id = :run_id AND lr.study = 'mc_test'
GROUP BY lr.line_id
ORDER BY p_binding DESC;

-- 2. Mean LMP by bus, split by whether ANY line is out of service in the scenario
SELECT br.bus_id,
       CASE WHEN s.n_lines_out > 0 THEN 'line(s) out' ELSE 'all lines in' END AS topology,
       AVG(br.lmp) AS mean_lmp,
       COUNT(*)    AS n
FROM bus_result br
JOIN scenario s ON s.run_id = br.run_id AND s.study = br.study AND s.scenario_id = br.scenario_id
WHERE br.run_id = :run_id AND br.study = 'mc_test'
GROUP BY br.bus_id, CASE WHEN s.n_lines_out > 0 THEN 'line(s) out' ELSE 'all lines in' END
ORDER BY br.bus_id, topology;

-- 3. Surrogate vs OPF: mean absolute error per model and bus on congested scenarios only
SELECT sl.model, sl.bus_id, AVG(ABS(sl.lmp_pred - br.lmp)) AS mae_congested, COUNT(*) AS n
FROM surrogate_lmp sl
JOIN bus_result br ON br.run_id = sl.run_id AND br.study = sl.study
                  AND br.scenario_id = sl.scenario_id AND br.bus_id = sl.bus_id
WHERE sl.run_id = :run_id AND sl.study = 'mc_test'
  AND EXISTS (SELECT 1 FROM line_result lr
              WHERE lr.run_id = sl.run_id AND lr.study = sl.study AND lr.scenario_id = sl.scenario_id
                AND lr.binding = :true_value)
GROUP BY sl.model, sl.bus_id
ORDER BY sl.model, sl.bus_id;

-- 4. Hours per month with SOUTH-CAL at its limit (deterministic hourly study)
SELECT s.month, SUM(CASE WHEN lr.binding = :true_value THEN 1 ELSE 0 END) AS hours_at_limit, COUNT(*) AS hours
FROM line_result lr
JOIN scenario s ON s.run_id = lr.run_id AND s.study = lr.study AND s.scenario_id = lr.scenario_id
WHERE lr.run_id = :run_id AND lr.study = 'hourly' AND lr.line_id = 'SOUTH-CAL'
GROUP BY s.month
ORDER BY s.month;

-- 5. N-1 ranking (security first, then economic impact)
SELECT outaged_line, share_hours_overload, max_post_loading, worst_line, delta_cost_musd_per_year, rank_security, rank_economic
FROM n1_result
WHERE run_id = :run_id
ORDER BY rank_security, rank_economic;

# Database schema

Generated DDL for each dialect: [`sql/schema_sqlite.sql`](../sql/schema_sqlite.sql),
[`sql/schema_postgresql.sql`](../sql/schema_postgresql.sql), [`sql/schema_mssql.sql`](../sql/schema_mssql.sql)
(`python -m src.db --ddl`). Defined once in [`src/db.py`](../src/db.py) with SQLAlchemy Core.

```mermaid
erDiagram
    bus ||--o{ line : "bus0 / bus1"
    bus ||--o{ generator : hosts
    run ||--o{ scenario : contains
    scenario ||--o{ scenario_bus_input : "load, temperature"
    scenario ||--o{ scenario_line_outage : "lines out (sparse)"
    scenario ||--o{ scenario_gen_availability : "derates (sparse)"
    scenario ||--o{ bus_result : "LMP, unserved"
    scenario ||--o{ line_result : "flow, loading, binding"
    scenario ||--o{ gen_dispatch : "MW per unit"
    scenario ||--o{ surrogate_lmp : "predicted LMP per model"
    scenario ||--o{ surrogate_congestion : "P(binding) per line"
    run ||--o{ line_congestion_stat : summary
    run ||--o{ lmp_quantile : summary
    run ||--o{ n1_result : summary
    run ||--o{ n1_delta_lmp : summary
    run ||--o{ surrogate_metric : summary
    run ||--o{ runtime : summary

    bus { string bus_id PK  string label  float x  float y  float load_share  float temp_offset_c }
    line { string line_id PK  string bus0 FK  string bus1 FK  float x_ohm  float s_nom_mw  float outage_rate }
    generator { string gen_id PK  string bus_id FK  string carrier  float p_nom_mw  float marginal_cost  string profile }
    run { int run_id PK  datetime created_utc  string label  int seed  text config_json }
    scenario { int run_id PK  string study PK  int scenario_id PK  datetime hour_start  int hour  int dow  int month  float temp_system_c  float cf_wind_south  float cf_wind_central  float cf_solar_south  float pu_hydro  float total_load_mw  float merit_price  int n_lines_out  float gen_mw_on_outage }
    scenario_bus_input { int run_id PK  string study PK  int scenario_id PK  string bus_id PK  float load_mw  float temp_c }
    scenario_line_outage { int run_id PK  string study PK  int scenario_id PK  string line_id PK }
    scenario_gen_availability { int run_id PK  string study PK  int scenario_id PK  string gen_id PK  float available_fraction }
    bus_result { int run_id PK  string study PK  int scenario_id PK  string bus_id PK  float lmp  float unserved_mw }
    line_result { int run_id PK  string study PK  int scenario_id PK  string line_id PK  float flow_mw  float loading  bool binding }
    gen_dispatch { int run_id PK  string study PK  int scenario_id PK  string gen_id PK  float p_mw }
    surrogate_lmp { int run_id PK  string study PK  int scenario_id PK  string model PK  string bus_id PK  float lmp_pred }
    surrogate_congestion { int run_id PK  string study PK  int scenario_id PK  string line_id PK  float p_binding }
    line_congestion_stat { int run_id PK  string source PK  string line_id PK  float p_binding  float ci95_low  float ci95_high  float expected_hours_per_year }
    lmp_quantile { int run_id PK  string source PK  string bus_id PK  string stat PK  float value }
    n1_result { int run_id PK  string outaged_line PK  float share_hours_overload  float max_post_loading  float delta_cost_musd_per_year  int rank_security  int rank_economic }
    n1_delta_lmp { int run_id PK  string outaged_line PK  string bus_id PK  float mean_delta_lmp }
    surrogate_metric { int run_id PK  string task PK  string model PK  string bus_id PK  string metric PK  float value }
    runtime { int run_id PK  string method PK  float seconds }
```

## Design notes

* **`study`** separates the deterministic hourly run (`hourly`) from the Monte Carlo study scenarios (`mc_test`);
  every result table is keyed by `(run_id, study, scenario_id)`, so several runs can live side by side.
* **Long (tidy) result tables** (`bus_result`, `line_result`, ...) instead of one column per bus/line: adding a bus
  or a line needs no schema change, and every SQL engine aggregates them with plain `GROUP BY`.
* **Sparse outage tables**: only lines that are out and generators below full availability are stored
  (1.7k and 51k rows instead of 160k and 260k).
* **Portable types only** (INTEGER, FLOAT, VARCHAR(n), BOOLEAN/BIT, DATETIME, TEXT/NVARCHAR(max)).
  Percentiles are computed in Python because `PERCENTILE_CONT` differs between engines (and SQLite lacks it).
* **Booleans**: `line_result.binding` is BOOLEAN on PostgreSQL/SQLite and BIT on SQL Server. Hand-written SQL should
  compare with `binding = :true` (bound parameter); the dashboard uses SQLAlchemy Core, which compiles the right form.

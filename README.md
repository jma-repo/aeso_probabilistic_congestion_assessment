# Probabilistic congestion assessment for a future nodal market, with ML surrogates for speed

A proof-of-concept on a **simplified, Alberta-inspired 6-bus grid**. Alberta's AESO plans to move from a single
province-wide pool price to **Locational Marginal Prices (LMPs)** around 2028, so there is no historical Alberta
LMP data to learn from. This project therefore:

1. **simulates** hourly LMPs and congestion with a DC optimal power flow (PyPSA + the open-source HiGHS solver),
2. runs **probabilistic (Monte Carlo) and N-1 contingency studies** on top of it *(Phase 2)*,
3. trains **ML surrogates** that approximate the OPF much faster, and checks honestly where they fail *(Phase 3)*,
4. stores everything in **SQL** and shows it in a **Streamlit** dashboard *(Phase 4)*,
5. calibrates load and wind/solar shapes against **real AESO data** *(Phase 5)*.

> **Status:** All five phases complete. Phase 5 calibrates the toy model against real AESO data for 2024–2025.
> An optional Phase 6 benchmarks Microsoft's pretrained GridSFM AC-OPF graph model against a full AC-OPF solver.

> ⚠️ **This is a DC approximation.** DC-OPF assumes lossless lines, flat voltage magnitudes and small angle
> differences, and ignores reactive power and voltage limits entirely. All capacities, costs and line ratings are
> **made up and illustrative**, loosely shaped like Alberta — they are not AESO data.

---

## What is an LMP? (the two-town example)

Picture two towns joined by one transmission line that can carry **100 MW**.

* **Town A** has a wind farm: 300 MW available, costs about **\$0.5/MWh** to run.
* **Town B** has a gas plant costing **\$50/MWh**, and it needs **150 MW**.

The cheapest dispatch is to send as much wind as possible from A to B. But the line fills up at 100 MW, so the
gas plant in B has to supply the remaining 50 MW.

The **LMP** at a location is the cost of serving *one more MW there*:

* One more MW in **B** cannot come over the full line, so the gas plant produces it: **LMP_B = \$50**.
* One more MW in **A** comes from the wind farm, which still has spare capacity: **LMP_A = \$0.5**.

The \$49.5 difference is the **congestion price**. Multiplied by the 100 MW flow it gives a congestion rent of
\$4,950/h, collected by the market operator. If the line could carry 200 MW, wind would serve all of B, both towns
would pay \$0.5, and there would be no congestion.

Two properties follow from this, and both matter later in the project:

* **Prices jump.** While the line is below its limit there is a single price. The moment it hits the limit, prices
  split by tens of dollars. In other words, LMP is a step function of the inputs.
* **In a meshed grid, power splits over every path in inverse proportion to line reactance.** This *loop flow* can
  push an LMP **above every offer in the system**. In the three-bus test in `tests/test_phase1.py` (offers of \$10
  and \$40), the load bus clears at **\$70/MWh**: serving one more MW there means backing down 1 MW of the \$10 unit
  and adding 2 MW of the \$40 unit.

Both examples are executable tests (`pytest -q`) and are shown in `notebooks/01_phase1_walkthrough.ipynb`.

---

## The toy network

![network](reports/figures/network.png)

| Bus | Role (illustrative) | Load share | Generation (MW @ \$/MWh) |
|---|---|---:|---|
| NORTH | Fort McMurray: oil-sands cogeneration, industrial load | 22% | cogen 4000 @ 25, CCGT 1200 @ 40 |
| EDM | Edmonton | 30% | CCGT 2000 @ 45, gas steam 1500 @ 65 |
| RD | Red Deer, central hub | 8% | wind 900 @ 0.6, CCGT 1500 @ 42, peaker 500 @ 120 |
| CAL | Calgary | 30% | hydro 900 @ 8 (seasonal), CCGT 800 @ 50, peaker 1000 @ 150 |
| SOUTH | Wind and solar region | 10% | wind 4200 @ 0.5, solar 1500 @ 0.3 |
| BC | Intertie to BC (import only) | — | import 1000 @ 55, behind an 800 MW intertie |

The network has eight AC circuits with two loops: SOUTH–CAL–RD and CAL–RD–EDM. NORTH connects through a
**double-circuit** corridor. A single circuit outage therefore does not island the North, but it does load the
parallel circuit; this is the case Phase 2's N-1 study will examine. Each load bus also has an unserved-energy
generator priced at the **\$999.99/MWh** price cap (used as the value of lost load, VOLL), so every outage
combination has a feasible solution.

All parameters live in [`config.yaml`](config.yaml), including the random seed, every capacity, cost and rating,
and every weather and load process parameter.

---

## How to run

```bash
pip install -r requirements.txt
```

```bash
python -m src.run
```

```bash
pytest -q
```

`python -m src.run` runs every implemented phase (currently Phases 1–4, ≈11 min on 4 cores). Use
`python -m src.run --phases 2` to run a single phase; Phase 2 reuses Phase 1 outputs, or creates them if they
are missing. `pytest -q` runs 37 correctness tests in about 90 s, with no network access needed.

Phase 5 is opt-in, because it needs an AESO API key and network access:

```bash
export AESO_API_KEY=...   # or put AESO_API_KEY=... in a local .env file (git-ignored)
```

```bash
python -m src.run --phases 5
```

Then open the dashboard (it reads only from the database):

```bash
streamlit run src/dashboard.py
```

To use PostgreSQL or SQL Server instead of the default SQLite file, set `DB_URL` before running:

```bash
DB_URL="postgresql+psycopg://user:password@host:5432/aeso_poc" python -m src.run --phases 4
```

* Outputs go to `data/` as parquet files; this folder is git-ignored and can be regenerated.
* The SQLite database is written to `data/aeso_poc.sqlite`.
* Figures and summaries go to `reports/` and are committed.
* The walkthroughs are `notebooks/01_phase1_walkthrough.ipynb`, `notebooks/02_phase2_probabilistic.ipynb`,
  `notebooks/03_phase3_surrogates.ipynb`, `notebooks/04_phase4_sql.ipynb` and `notebooks/05_phase5_aeso.ipynb`.
* Phase 2 and Phase 3 tables are also written as CSV files to `reports/tables/`.

### Repository layout

```
config.yaml          all parameters (seed, network, generators, weather/load processes)
src/config.py        load + validate config
src/synth.py         synthetic temperature, load, wind, solar, hydro
src/grid.py          PyPSA network builder; PTDF matrix
src/simulate.py      block-wise DC-OPF; solve_many(): group by topology, solve in parallel
src/montecarlo.py    scenario sampling (uncertainty + forced outages), Wilson CIs, sensitivities
src/contingency.py   N-1: PTDF security screen + re-dispatch economic impact, ranking
src/features.py      feature matrix, merit-order (copper-plate) price, leakage-free splits
src/models.py        baseline, LightGBM per bus, PyTorch MLP, congestion classifier, hybrid wrapper
src/evaluate.py      metrics by bus/regime, MC comparison, worst cases, 1-D step sweeps
src/surrogate_study.py  Phase 3 pipeline
src/db.py            SQLAlchemy schema, writers/readers, DDL export (python -m src.db --ddl)
src/dashboard.py     Streamlit dashboard (reads only from the database)
src/aeso_api.py      AESO API client: retries, polite delay, parquet cache, typed parsers
src/calibrate.py     Phase 5: coverage checks, real shapes, price calibration, real-shape OPF
src/checks.py        sanity checks incl. an independent angle-based DC-OPF
src/plots.py         figures
src/run.py           one-command pipeline
tests/               pytest suite (two-town, loop flow, outages vs zero rating, islanding, PTDF screen, ...)
notebooks/           walkthrough
reports/             generated summaries and figures (committed)
sql/                 generated DDL (SQLite / PostgreSQL / SQL Server) + portable example queries
docs/schema.md       ER diagram of the database
docs/aeso_api_probe.md  response shapes and quirks of the AESO endpoints (Phase 5 probe)
experiments/gridsfm/ Phase 6: GridSFM vs PowerModels/Ipopt benchmark (separate environment, see its README)
```

---

## Phase 1 — synthetic grid and LMP simulator

### Synthetic inputs

The inputs cover two years of hourly data (17,520 hours) on a naive calendar with no DST, generated from a fixed
seed. Each stochastic process uses its own child random stream, so changing one parameter does not reshuffle the
others.

* **Temperature (°C):** an annual cosine (coldest around 15 January), a daily cosine (warmest at 15:00) and a
  persistent AR(1) weather anomaly (φ = 0.985/h), plus small offsets and noise per bus (the North is 3 °C colder).
* **Load per bus:** a share of system load × hour-of-day shape × weekend factor (0.94) × a heating term
  (+1.1% per °C below 15 °C) and a cooling term (+1.5% per °C above 20 °C) × 1.5% noise. The North is industrial
  and keeps only 25% of that variation. The result is a mean load of 9.8 GW and a peak of 12.8 GW.
* **Wind:** CF = sigmoid(seasonal mean + AR(1) anomaly). It is windier in winter, the two sites are correlated
  (ρ = 0.7), and the mean CF is about 0.38.
* **Solar:** clear-sky output from solar elevation at 50°N × (1 − an AR(1) cloud factor). It is zero at night and
  low in winter.
* **Hydro:** monthly availability with a June–July freshet.

![inputs](reports/figures/inputs_overview.png)

### Results (from [`reports/phase1_summary.md`](reports/phase1_summary.md))

| Metric | Value |
|---|---:|
| Hours simulated | 17,520 |
| DC-OPF wall time, all hours | **≈40–50 s** (24 monthly LPs, ≈2 s each) |
| Hours with ≥1 binding line | **27.0%** (target 10–30%); internal lines only: 23.7% |
| Hours with load shedding | 0 |
| Mean / P95 max−min LMP spread across Alberta buses | \$10.5 / \$49.5 per MWh |
| Same spread with no line limits | 5e-15 (a single price, as it should be) |

| Line | Share of hours at limit |
|---|---:|
| SOUTH–CAL | 22.1% |
| BC–CAL (intertie) | 3.4% |
| EDM–NORTH (each circuit) | 1.5% |
| SOUTH–RD, CAL–RD, RD–EDM, CAL–EDM | 0.0% |

| Bus | Mean LMP with line limits | Mean LMP with no limits |
|---|---:|---:|
| NORTH | 45.26 | 44.51 |
| EDM | 45.31 | 44.51 |
| RD | 44.36 | 44.51 |
| CAL | 47.22 | 44.51 |
| SOUTH | **36.75** | 44.51 |
| BC | 46.85 | 44.51 |

Most congestion comes from **wind in the south**. When SOUTH–CAL is full, SOUTH's price collapses to the wind offer
(\$0.5) and Calgary's price rises. CAL and RD prices are often *mixtures* of offers, set through the PTDF weights of
the binding line, which is loop flow at work.

![prices separate only at the limit](reports/figures/spread_vs_loading.png)

The scatter above is the step behaviour in one picture. The price spread is exactly zero at every loading below
100%, and it is between \$2 and \$55 at the limit.

![week](reports/figures/lmp_week.png)
![duration](reports/figures/price_duration.png)
![binding by line](reports/figures/binding_by_line.png)

### Sanity checks (all run automatically by `python -m src.run`)

| Check | Result |
|---|---|
| **Copper plate:** with no line limits every bus has the same price | PASS (max spread 1.4e-14 \$/MWh) |
| **Independent re-solve:** 200 random hours (half of them congested) re-solved with an angle-based DC-OPF written directly in `scipy.optimize.linprog` (HiGHS), without PyPSA | PASS (max LMP difference 2e-13 \$/MWh, max flow difference 2e-11 MW) |
| **Physics:** power balance, Kirchhoff's current law at every bus, no flow above its limit | PASS (all errors < 1e-11 MW) |
| **Congestion target:** 10–30% of hours with ≥1 binding line | PASS (27.0%) |
| **Unit tests:** two-town prices (\$0.5 / \$50), three-bus loop flow (\$70 above both offers), PTDF, determinism, input ranges, config validation | 8 / 8 pass |

PyPSA writes the KVL constraints as cycle constraints, while the independent check uses bus voltage angles. Their
agreement therefore tests both the formulation and the way results are extracted.

### Engineering notes

* **Hours are solved in blocks.** With no storage, unit commitment or ramping limits, each hour's OPF is independent
  of the others. Solving a month of hours in a single LP gives exactly the same answer as solving them one at a time,
  and pays the model-building overhead 24 times instead of 17,520 times.
* **A line outage removes the line.** In DC-OPF, setting the line's rating to 0 is *wrong*: the KVL constraint stays
  in the model and forces the angle difference across the open line to zero. Generator outages are different and can
  be applied exactly through `p_max_pu`. This matters for Phase 2, where scenarios are grouped by topology.
* **Wind and solar costs are \$0.3–0.6, not \$0.** Small, distinct costs avoid ties and degenerate (non-unique) duals.
  6 of 17,520 hours still show a line exactly at its limit with zero price separation, a degenerate vertex. Phase 3
  defines the congestion label as "at the limit" and documents these hours.
* **Line ratings were tuned by hand** to land in the 10–30% band. Lowering the North corridor rating by just 2%
  (1650 → 1620 MW per circuit) raises its binding share from 1.5% to 11.6%, and the system-wide share from 27% to 37%. Because northern load is flat industrial load, whether that corridor binds is close
  to a yes/no function of its rating. This is a real sensitivity, and Phase 2 will treat it as uncertainty rather
  than hide it.

### Honest observations and limitations of the Phase 1 model

* **Prices are very discrete.** Each bus takes only 10–13 distinct LMP values over two years. A pure cost-based
  merit order without bidding behaviour produces flat price plateaus and **almost no scarcity pricing**: peakers run
  in only 9 hours, and the price never goes above \$120. Real Alberta pool prices are far spikier, because of offer
  behaviour, outages, ramping and the intertie. Phase 2 adds outages and Phase 5 compares against real pool price.
  This model should not be read as a price forecast.
* **No losses, no reactive power, no voltage limits** (DC approximation). Line ratings are static, with no
  seasonal or dynamic ratings.
* **No unit commitment, minimum stable generation, ramping or storage**, so hours are independent. Cogeneration is
  treated as fully dispatchable, although in reality much of it is must-run.
* **The BC intertie is import-only, at a fixed \$55**. There is no export, no Montana/Saskatchewan tie and no BC
  price dynamics.
* **The weather is synthetic and shared** across Alberta (one anomaly plus bus offsets). Real Alberta has regional
  weather differences, and wind/load correlation is only partly represented, through season.
* **Risk of circularity.** The Phase 3 surrogates will learn a data-generating process that I designed. Their
  accuracy measures how well they learn *this simulator*, not the real market.

---

## Phase 2 — probabilistic studies

Full auto-generated output: [`reports/phase2_summary.md`](reports/phase2_summary.md) (CSV files in `reports/tables/`).

### Monte Carlo set-up

Each of the **20,000 scenarios** is one independent "possible hour", drawn in three steps:

1. **Pick a base hour** uniformly from the two synthetic years. This keeps temperature, calendar, load and
   renewables consistent with each other.
2. **Perturb it.** Apply a lognormal system load multiplier (σ = 3%) and a per-bus multiplier (σ = 1%), shift wind
   in logit space (σ = 0.5), and add multiplicative solar noise (σ = 10%).
3. **Sample forced outages (state sampling).**
   * Each line is out with probability *q*, set between 0.5% and 2% per line (illustrative, at the pessimistic end).
   * Each thermal, hydro or import plant is split into identical units. The number of available units is
     ~ Binomial(n, 1 − FOR), with FOR from 3% to 8% by technology. This takes about 740 MW of generation out on
     average.

Scenarios are drawn i.i.d., so every probability carries a **95% Wilson interval**. Scenarios that share a topology
(the same set of lines out) are solved together in one LP. The 20,000 scenarios fall into **33 distinct
topologies**, solved as 51 LPs on 4 worker processes.

**Solve time for the full Monte Carlo study: ≈21–26 s wall-clock** (≈1.1–1.3 ms per scenario on 4 cores; the
range is run-to-run variation, including process start-up). The sum of the individual LP times is 76–88 s, about
4 ms per scenario on a single core. This is the benchmark the Phase 3
surrogates have to beat.

An independent check re-solved 100 scenarios (half with a line outage, all with generator derates) using the angle
formulation in scipy. It matches PyPSA to 2e-13 \$/MWh.

### Results

![MC congestion probability](reports/figures/mc_congestion_probability.png)

| Line | Deterministic share (no outages) | **MC P(at limit)** [95% CI] | Expected h/yr | P(at limit \| any line out) |
|---|---:|---:|---:|---:|
| SOUTH–CAL | 22.1% | **22.3%** [21.7, 22.9] | 1,954 | 23.6% |
| BC–CAL (intertie) | 3.4% | **7.0%** [6.7, 7.4] | 617 | 10.6% |
| EDM–NORTH-1 | 1.5% | **3.2%** [2.9, 3.4] | 277 | 16.1% |
| EDM–NORTH-2 | 1.5% | **3.0%** [2.8, 3.3] | 266 | 14.5% |
| SOUTH–RD | 0.0% | **0.6%** [0.5, 0.7] | 51 | 7.1% |
| CAL–EDM | 0.0% | **0.3%** [0.2, 0.4] | 26 | 3.6% |
| RD–EDM, CAL–RD | 0.0% | ≈0% | <1 | ≈0% |

* **P(≥1 internal line at limit) = 26.7%** [26.1, 27.3], which is 2,337 h/yr. Including the BC intertie it is
  33.1% (2,898 h/yr). The deterministic run gave 23.7% and 27.0%.
* **SOUTH–CAL congestion is driven by weather, not reliability.** Adding outages and uncertainty barely changes it
  (22.1% → 22.3%): it is set by how often the south is windy.
* **Outages double congestion on the BC intertie and the North corridor.**
  * The intertie doubles because thermal outages push Alberta prices above the \$55 import offer.
  * The North corridor doubles because one circuit out leaves the other at its limit. P(binding) is 2% with all
    lines in service and 16% with a line out.
* **LMP distribution.** Outages create a price tail. P90 rises from \$50 to \$55, P(LMP ≥ \$100) from 0.1% to
  about 1.2%, and mean prices rise by about \$2. SOUTH is the most dispersed bus: P10 \$0.5, P50 \$45, P90 \$55.
  The tail is still far thinner than real Alberta prices (see the Phase 1 limitations).
* **Adequacy (a by-product of the study).**
  * LOLE is **4.8 h/yr** [2.7, 8.6] and expected unserved energy (EUE) is about 1,400 MWh/yr.
  * Both rest on only 11 events in 20,000 scenarios, so the intervals are wide. Rare-event methods such as
    importance sampling or cross-entropy would tighten them, and are listed as a next step.
  * About 80% of the unserved energy is in **SOUTH**, which has no local thermal generation. It happens when a
    southern line is out on a winter evening with no wind or sun.

![LMP intervals](reports/figures/mc_lmp_intervals.png)
![LMP ECDF](reports/figures/mc_lmp_ecdf.png)

**Convergence.** The plot below shows how many samples are enough. SOUTH–CAL's estimate settles within about
±0.6 percentage points after 20,000 scenarios. Rare lines such as CAL–EDM (0.3%) still have a relative
uncertainty of about 30%.

![convergence](reports/figures/mc_convergence.png)

### Rating sensitivity (same scenarios at every rating)

The first 5,000 scenarios are re-solved with one corridor's rating scaled. Because every rating uses the same
scenarios (common random numbers), the *differences* between ratings contain no sampling noise.

![rating sensitivity](reports/figures/rating_sensitivity.png)

* **SOUTH–CAL responds smoothly.** Each +5% (80 MW) of rating removes about 2 percentage points of congestion
  probability (about 180 h/yr) and saves about \$10M/yr in production cost. This is the kind of number a
  transmission-upgrade case would use.
* **The EDM–NORTH corridor sits on a cliff.** −5% of rating raises congestion from 4% to 24%, and −10% raises it to
  49%. Above the base rating, the only congestion left (2%) comes from single-circuit outages.
  * The North exports a nearly constant surplus, so whether the corridor binds is close to a yes/no function of its
    rating.
  * Real ratings depend on ambient temperature. A deterministic rating assumption could therefore badly mis-state
    congestion on this kind of corridor, which argues for probabilistic or seasonal ratings.

### N-1 contingency analysis

Each of the 8 lines is removed in turn, over every one of the 17,520 hours, and each case is viewed two ways:

* **Security screen:** the pre-contingency market dispatch is kept, and the new flows are computed with the PTDF
  of the reduced network. This answers "does anything overload the instant the line trips?"
* **Economic impact:** a full DC-OPF re-dispatch without the line, compared with the intact case. This answers
  "what does it cost, and how do LMPs move, if the line is out for the hour?"

Together that is 140,160 re-dispatch OPFs, solved in 82 s.

![N-1 security](reports/figures/n1_security.png)

| Rank | Outaged line | Hours with post-contingency overload | Worst overload | Δ production cost (M\$/yr) | Mean \|ΔLMP\| (\$/MWh) | Unserved energy after re-dispatch (MWh/yr) |
|---:|---|---:|---|---:|---:|---:|
| 1 | EDM–NORTH-1 (or -2) | **99%** | other circuit 200% | **+154** | **10.3** | 486 |
| 3 | CAL–RD | 64% | SOUTH–CAL 142% | +25 | 3.5 | 0 |
| 4 | RD–EDM | 55% | CAL–EDM 175% | +15 | 3.8 | 0 |
| 5 | SOUTH–CAL | 40% | SOUTH–RD **241%** | +141 | 3.1 | **1,600** |
| 6 | SOUTH–RD | 32% | SOUTH–CAL 166% | +72 | 1.8 | 0 |
| 7 | CAL–EDM | 22% | SOUTH–CAL 110% | **−2.4** | 0.8 | 0 |
| — | BC–CAL | islands BC: loses up to 800 MW of import | — | +5 | 1.0 | 0 |

![ΔLMP heatmap](reports/figures/n1_dlmp_heatmap.png)

How to read this:

* **The base case is not N-1 secure, by construction.** The market dispatch uses thermal ratings and no
  contingency constraints, which is not security-constrained OPF (SCOPF). On the North double circuit, losing one
  circuit doubles the flow on the other in almost every hour.
  * In real operations, the North corridor would be dispatched to an N-1-secure operating limit, close to one
    circuit's rating, rather than the sum of both.
  * This is the single most important result of the screen. Adding these contingency constraints (SCOPF) would
    raise congestion sharply, and it is the first modelling next step.
* **Security ranking and economic ranking differ.**
  * CAL–RD causes frequent overloads (security rank 3) but is cheap to re-dispatch around (economic rank 5).
  * SOUTH–CAL is the reverse: fewer overload hours, but the largest overload (241%), large cost (+\$141M/yr) and
    **unserved energy in the South**, because the remaining 1,100 MW SOUTH–RD line cannot carry the South's
    winter-evening load when there is no wind.
  * Both views are needed. The ranked table includes both ranks.
* **Removing a line can *lower* cost** (a Braess / transmission-switching effect).
  * CAL–EDM out is cheaper than the intact network in 18.7% of hours and −\$2.4M/yr overall. RD–EDM and CAL–RD
    show the same effect in some hours.
  * In a meshed DC network a line also adds a KVL constraint. Removing it can relieve loop flow on a binding line.
  * This is not a numerical artefact: the best hour for each case was re-solved independently and confirmed
    (CAL–EDM saves \$4,344/h). Operators use the same effect deliberately (optimal transmission switching), but only
    within N-1 security limits.
* **Modelling note.** Outaged lines are *removed* from the network. A test shows that setting the rating to zero
  gives a different (and wrong) LMP, because the KVL constraint stays in the model.

### Additional limitations of Phase 2

* Outages are sampled independently, hour by hour. There are no repair times, common-mode failures or weather-driven
  correlated outages, and no planned maintenance.
* Outage rates and FORs are illustrative. Wind and solar have no separate outage model.
* N-1 covers lines only, not generators or the 800 MW intertie import.
* Base-case flows are not secured against contingencies (see the SCOPF note above). Re-dispatch is economic and
  costless to execute, with no ramp limits.
* Unserved energy estimates rest on few events and have wide intervals.

---

## Phase 3 — ML surrogates

Full auto-generated output: [`reports/phase3_summary.md`](reports/phase3_summary.md) (CSV files in `reports/tables/`).

### Set-up

* **Inputs (39 features).** All of them are known *before* solving the OPF:
  * load per bus and total load, and system temperature;
  * wind, solar and hydro availability, both as capacity factors and in MW;
  * net load, available thermal capacity and reserve margin;
  * South and North surplus (local available generation minus local load);
  * the copper-plate merit-order price (computed analytically; see below);
  * hour, day of week and month;
  * one 0/1 outage indicator per line, and the available fraction of every generator.
* **Targets.** LMP at all six buses, and a binding flag for each of the eight lines.
* **Models.**

  | # | Model | What it is |
  |---|---|---|
  | 1 | **Baseline** | A single system price that ignores the network: the merit-order (copper-plate) price, computed analytically. It equals the copper-plate LP on all 17,520 hours, and matches it to 1e-6 \$/MWh in a test with outages. |
  | 2 | **LightGBM** | One model per bus, with exact TreeSHAP values computed inside LightGBM (`pred_contrib`). |
  | 3 | **PyTorch MLP** | One multi-output network for all buses: 256-256-128 ReLU, Adam, early stopping on validation loss. Undefined LMPs (islanded BC) are masked out of the loss. |
  | 4 | **Congestion classifier** | LightGBM, one binary model per line. Lines with fewer than 5 binding hours in training fall back to the base rate. |

* **Splits. Nothing is shuffled across time, and the test set is never used for fitting or early stopping.**
  * **MC task (main result).** Train on a *separate* set of 20,000 Monte Carlo scenarios with its own seed. The last
    15% of that set (3,000 scenarios) is used for validation and early stopping. Test on the 20,000 Phase 2 study
    scenarios.
  * **Hourly task.** Train on 2025, validate on 2026 H1, test on 2026 H2, with a 7-day embargo at each boundary
    because weather is autocorrelated. This data has no outages.

### Two design choices that mattered more than the model family

Both choices are backed by a full ablation in `reports/phase3_summary.md`:

| MC test MAE (\$/MWh, mean over buses) | LightGBM | MLP |
|---|---:|---:|
| raw LMP target, L2 loss (the textbook set-up) | 1.61 | 2.26 |
| raw LMP target, Huber loss | 1.06 | 1.48 |
| **hybrid target (LMP − merit price), Huber loss** | **0.84** | **1.19** |

1. **Physics-informed "hybrid" target.**
   * When no line is at its limit, LMP equals the copper-plate merit-order price exactly, and that price costs
     microseconds to compute. So the models receive it as an input and learn only the **congestion component**
     (LMP − merit price).
   * Every merit-order step (gas unit *n* becomes marginal) is then exact by construction. The ML only has to learn
     the network steps.
2. **Robust (Huber) loss.**
   * About 0.05% of scenarios hit the \$999.99 price cap. Under an L2 loss these few points dominate training: in
     the first experiment with L2 and a raw target, the MLP's best validation epoch was epoch 13.
   * With Huber loss (5 \$/MWh transition for LightGBM; 1 σ on standardised targets for the MLP), both models fit
     the bulk of the distribution much better.

### Accuracy (MC test set: 20,000 scenarios the models never saw)

| Model | MAE | RMSE | MAE, congested rows | MAE, uncongested rows | P99 \|error\| |
|---|---:|---:|---:|---:|---:|
| Baseline (copper plate) | 3.50 | 13.9 | 10.42 | **0.07** | 23.7 |
| **LightGBM (hybrid, Huber)** | **0.84** | 11.0 | **2.08** | 0.22 | **11.2** |
| MLP (hybrid, Huber) | 1.19 | **9.7** | 2.73 | 0.42 | 13.3 |

* **The naive baseline is perfect without congestion and badly wrong with it.** Its MAE is \$0.07 in uncongested
  rows and \$10.42 in congested ones. In other words, all of the surrogate's value is in the 33% of hours with a
  binding line.
* **LightGBM wins on MAE and in every regime except the North corridor and line outages, where the two models are
  close.** The MLP has a lower RMSE, but only because it extrapolates more aggressively on the few \$999.99
  scarcity points (it predicts about \$150 where LightGBM predicts about \$90). Both models are still wrong there
  by more than \$800.
* **The hourly task is easier.** There are no outages and no scarcity events: LightGBM MAE is \$0.19, the MLP's is
  \$0.27, and the baseline's is \$2.66.
* **Congestion classifier.**
  * For the lines that matter, F1 is 0.97–0.99 and ROC-AUC 1.00: SOUTH–CAL, BC–CAL and both EDM–NORTH circuits.
  * For rare lines, recall drops: 0.75 on SOUTH–RD (117 positives) and 0.61 on CAL–EDM (59 positives).

![error by regime](reports/figures/surrogate_mae_by_regime.png)

### Main result: rerunning the Phase 2 Monte Carlo with the surrogate

![MC congestion: OPF vs surrogate](reports/figures/surrogate_mc_congestion.png)

| Line | OPF P(at limit) [95% CI] | Surrogate (mean predicted prob.) | Error |
|---|---:|---:|---:|
| SOUTH–CAL | 22.31% [21.74, 22.89] | 22.31% | 0.00 pp |
| BC–CAL | 7.04% [6.69, 7.40] | 7.00% | 0.04 pp |
| EDM–NORTH-1 / -2 | 3.16% / 3.03% | 3.14% / 3.04% | ≤0.03 pp |
| SOUTH–RD | 0.58% [0.49, 0.70] | 0.53% | 0.06 pp |
| CAL–EDM | 0.29% [0.23, 0.38] | 0.28% | 0.01 pp |
| P(≥1 internal line) | 26.67% [26.07, 27.29] | 26.59% | 0.08 pp |

* **Congestion probabilities are reproduced to within 0.08 percentage points**, well inside the OPF's own Monte
  Carlo sampling interval for every line with non-trivial congestion. The congestion-hours-per-year numbers match to
  within 5 h/yr.
* **Mean LMPs match within \$0.45/MWh** at every bus. P10 and P90 are within about \$0.6 (LightGBM) and \$0.9 (MLP).
* **One quantile is visibly off: the P50 at CAL and BC** (OPF \$48.78; LightGBM \$46.5, MLP \$46.0). The OPF price
  distribution has *atoms*: 50% of scenarios sit at or below \$45, and the next plateau is a loop-flow mixture price
  of \$48.78. The median falls exactly on that jump, so a tiny shift in the predicted distribution moves the
  quantile by \$2–4.
  * Quantiles of step-shaped distributions are fragile. For decisions, report exceedance probabilities such as
    P(LMP ≥ x), or means.
  * The ECDF below shows the predicted distributions tracking every plateau of the OPF distribution, with the MLP
    smoothing the corners.

![LMP ECDF: OPF vs surrogates](reports/figures/surrogate_lmp_ecdf.png)

| Runtime for the 20,000-scenario study (range over three full runs) | Seconds | Speed-up |
|---|---:|---:|
| DC-OPF, 4 worker processes (Phase 2) | 22–26 | 1× |
| DC-OPF, 1 core (sum of LP times) | 76–88 | — |
| LightGBM + classifier, 4 threads | 1.8–2.2 | **12–13×** |
| LightGBM + classifier, 1 thread | 6.0–6.3 | ≈13–14× vs 1-core OPF |
| MLP + classifier, 4 threads | 0.35–0.37 | **≈65–70×** |
| Merit-order baseline | 0.04 | ≈550–630× |

![runtime](reports/figures/surrogate_runtime.png)

**The speed-up only pays off for repeated studies.**

* The surrogate's up-front cost is about 27 s of OPF to generate the training data plus 12–13 s of fitting. For a *single*
  20,000-scenario study, that is more than running the OPF Monte Carlo directly.
* **Break-even is about 33,000–34,000 scenarios.** Beyond that, every additional study comes almost free: sensitivity
  sweeps, 10⁶-scenario runs for rare events, and real-time what-if screening.
* The learning curve shows how much training data is needed. With 5,000 training OPFs, congestion probabilities
  are already within 0.3 pp and MAE is \$1.11. With 17,000, they are within 0.06 pp and MAE is \$0.84.

![learning curve](reports/figures/surrogate_learning_curve.png)

### LMP is a step function: where each model fails

LMP is piecewise constant in its inputs. It jumps when the marginal unit changes (a merit-order step) and when a
line reaches its limit (a network step). The plots below take one test-period hour, vary a single input, and
re-solve the OPF at 101 points.

![wind sweep](reports/figures/surrogate_step_wind.png)
![load sweep](reports/figures/surrogate_step_load.png)

* **Merit-order steps** (wind CF ≈ 9% and 56%, and the load steps): both surrogates are almost exact, because the
  hybrid target hands them these steps through the merit-price feature. The exception is that LightGBM drifts up
  to about \$3 above the true price just before the 120% load step at CAL.
* **Network steps** (South wind above about 80%, when SOUTH–CAL binds):
  * The OPF drops from \$42 to \$0.5 at SOUTH in two sharp steps.
  * **LightGBM** produces steps too, because trees are piecewise-constant, but the threshold is learned from data.
    It starts the drop about 3 percentage points of CF early and the curve is jagged. At CAL it wobbles about
    \$1–1.5 below the true \$50.
  * **The MLP smooths the jump into a ramp** spread over about 10 percentage points of CF. It also slightly
    *undershoots* the \$0.5 wind floor, predicting about −\$1.5, which a physical market cannot clear at here.
* **Out of the training distribution** (load scaled to 80–95% on an evening hour, a combination that is rare in
  training): the MLP extrapolates linearly and predicts \$33–40 at NORTH, where the true price is \$40. LightGBM
  stays flat at the nearest training values. Trees cannot extrapolate, but here that is a virtue.
* **Scarcity** (\$999.99 prices, about 0.05% of scenarios; every one of the largest errors is one of these): both
  models predict \$50–280. With 10–20 examples in training, combinations of multiple outages that lead to load
  shedding are not learnable. A production surrogate should fall back to the OPF when the classifier or the reserve
  margin flags a stressed system.

### What drives the congestion premium? (TreeSHAP)

![SHAP](reports/figures/surrogate_shap_importance.png)

* **South surplus** (available wind + solar − South load) dominates the SOUTH congestion component, with a mean
  \|SHAP\| of \$8.9/MWh. Raw wind CF and renewable MW add about \$2 each. This matches the physics: SOUTH–CAL binds
  when the South has more than it can export.
* For **CAL**, the same South surplus is the top driver (\$2.5), followed by the merit price and the reserve margin.
* The SHAP dependence plot below shows the step directly.
  * The SHAP value falls from about +\$7 to about −\$18 between 2,000 and 2,500 MW of surplus, which is roughly the
    South's combined export capability.
  * With a southern line out (orange), the drop starts at about 1,200 MW, and at low surplus the premium is higher
    (about +\$10), because the South then depends on a single import path.

![SHAP dependence](reports/figures/surrogate_shap_dependence.png)

### Limitations of the surrogates (honest scope)

* **In-distribution only.** The test scenarios come from the same sampling process as the training scenarios. This
  shows the surrogate can replace the OPF *inside a given study design*, not that it generalises.
* **No network what-ifs without retraining.** Line ratings and topology are not inputs, so the surrogate cannot
  answer Phase 2's rating sensitivity: it would predict 22.2% congestion for every SOUTH–CAL rating, while the
  OPF moves from 26% to 18%. Topology-aware models such as GNNs, or ratings as features, are the next step.
* **Scarcity tails are not reproduced.** This makes the surrogate unsuitable for adequacy metrics such as
  LOLE/EUE: keep the OPF, or use a hybrid OPF fallback, for those.
* **The surrogates learn this simulator.** Their accuracy is measured against a DC-OPF on synthetic data, not
  against a real market.

---

## Phase 4 — storage and visualization

### SQL database

All inputs and results are stored through **SQLAlchemy Core**:

* network data;
* every hour of the deterministic study and every Monte Carlo scenario, with its inputs and outages;
* LMPs, line flows and generator dispatch;
* surrogate predictions;
* all summary tables.

The default is a SQLite file (`data/aeso_poc.sqlite`, about 125 MB). Setting `DB_URL` points the same code at
PostgreSQL or SQL Server. One pipeline run writes about **1.8 million rows in about 20 s**.

![dashboard: network map](reports/figures/dashboard_map.png)

* **Schema.** The ER diagram is in [`docs/schema.md`](docs/schema.md). There are 20 tables in five groups:
  * network: `bus`, `line`, `generator`;
  * `run`: one row per pipeline run, with a snapshot of the config;
  * `scenario`: one row per hour or Monte Carlo scenario, per study;
  * long-format inputs and results, keyed by `(run_id, study, scenario_id)`;
  * summary tables: congestion statistics, LMP quantiles, the N-1 ranking, surrogate metrics, runtimes.
* **Portability.** Only portable types are used (INTEGER, FLOAT, VARCHAR(n), BOOLEAN/BIT, DATETIME,
  TEXT/NVARCHAR(max)). `python -m src.db --ddl` compiles the CREATE TABLE scripts for all three dialects without
  needing a server: [`sql/schema_sqlite.sql`](sql/schema_sqlite.sql), [`sql/schema_postgresql.sql`](sql/schema_postgresql.sql)
  and [`sql/schema_mssql.sql`](sql/schema_mssql.sql). Three portability issues came up and were fixed:
  * a column named `timestamp` was renamed to `hour_start`, to avoid keyword clashes;
  * SQL Server's deprecated TEXT type is replaced by NVARCHAR(max);
  * booleans need care: `binding` is BOOLEAN on PostgreSQL and BIT on SQL Server, so a hand-written
    `CASE WHEN binding THEN …` fails on SQL Server. The example queries compare against a bound `True` parameter
    instead.
* **Percentiles are computed in Python, not SQL.** `PERCENTILE_CONT` differs between engines, and SQLite does not
  have it.
* **Example queries.** [`sql/example_queries.sql`](sql/example_queries.sql) holds five portable queries:
  * congestion probability straight from the raw scenario rows;
  * LMP by topology;
  * surrogate MAE on congested scenarios;
  * monthly binding hours;
  * the N-1 ranking.

  They run in the tests and in `notebooks/04_phase4_sql.ipynb`. That notebook also checks that the SOUTH–CAL
  congestion probability computed from raw SQL rows, from the summary table and from the Phase 2 report agree
  exactly.
* **Limitation: only SQLite was tested against a live database.** PostgreSQL and SQL Server were checked only by
  compiling their DDL; no live server of either was available here. Driver packages (`psycopg`, `pyodbc`) are not
  in `requirements.txt`.

### Streamlit dashboard

`streamlit run src/dashboard.py` opens four tabs. Every number is read from the database, never from files.

| Tab | Content |
|---|---|
| **Network & congestion** | Map with lines coloured by P(line at limit). Choose the source: Monte Carlo with full OPF, Monte Carlo with the surrogate, or the deterministic hourly run. Hover a line for its rating and probability, or a bus for its generators and mean LMP. Table with 95% CIs. |
| **LMP distributions** | Empirical CDF per bus for either study, with a P10/P50/P90/P99/mean table and P(LMP ≥ \$100). Optionally overlays the LightGBM, MLP and baseline predictions for one bus. |
| **N-1 critical lines** | The ranked N-1 table (security and economic ranks) and a heatmap of the mean ΔLMP for each outaged line and bus. |
| **OPF vs surrogate** | Runtime and accuracy headline numbers, OPF vs surrogate congestion probability with CIs, accuracy and runtime tables, and a predicted-vs-true scatter for a chosen bus and model. |

![dashboard: N-1](reports/figures/dashboard_n1.png)
![dashboard: OPF vs surrogate](reports/figures/dashboard_surrogate.png)

The dashboard was checked by rendering all four tabs headlessly with Playwright (no exceptions, no page errors),
and by a Streamlit `AppTest` smoke test in `tests/test_phase4.py`.

---

## Phase 5 — calibration with real AESO data (2024–2025)

Full auto-generated output: [`reports/phase5_summary.md`](reports/phase5_summary.md). The probe of the endpoints'
response shapes and quirks is documented in [`docs/aeso_api_probe.md`](docs/aeso_api_probe.md).

### Data pipeline

* **Key handling.**
  * The key is read from `AESO_API_KEY`, or from a git-ignored `.env` file, and sent only as the `API-KEY`
    header.
  * It is never logged, printed, cached or committed; a test checks that it appears neither in logs nor in
    error messages.
  * In the cloud environment used here, the key was injected by the network proxy, so the code only ever saw a
    placeholder.
* **Client.** One shared `requests` session, with exponential backoff on 429/5xx (respecting `Retry-After`) and a
  0.5 s pause between calls. Each request's endpoint, date range and number of assets is logged.
* **Caching.** Raw responses are cached as parquet in `data/raw/`, and chunks already cached are skipped.
  The first pull took 266 calls (about 8 minutes); reruns take 0 calls.
* **Endpoint quirks found and handled.** Each one has a unit test.

  | Endpoint | Quirk | Handling |
  |---|---|---|
  | Pool price | `endDate` is inclusive; at most 1 year per request | yearly chunks |
  | AIL | Drops the last hour when a request *ends* on a 25-hour fall-back day | each chunk requests one extra day, trims and de-duplicates on UTC, checks completeness, and splits into halves on error or truncation |
  | Metered volume | `endDate` is **exclusive** | request the day after the last wanted day |
  | Metered volume | The real limit is **about 10,000 records per call**, not the documented 100,000. The API returns HTTP 400 *"report is available for maximum end date of …"* | one asset-year per call |
  | Asset list | **No fuel type** (only SOURCE/SINK), so name matching would miss 34 of 50 wind and 30 of 49 solar assets | fuel type taken from the Current Supply Demand asset snapshot (`fuel_type`, `maximum_capability`) |

* **Time.**
  * Every series is keyed on the **UTC** field. On the 2025-11-02 fall-back day the local (MPT) hour 01:00 appears
    twice.
  * The tz database (2026e) and AESO's data agree that **Alberta has used UTC−6 year-round since 2026-11-01**:
    2026-03-08 has 23 hours, 2026-11-01 has 24.
  * Calendar features use `zoneinfo`; no time offset is hard-coded.

### Data quality and coverage

* **Pool price and AIL** are complete: 17,544 of 17,544 hours (2024 is a leap year), with no duplicates and no
  empty values.
* **Wind and solar fleet.** The fuel snapshot lists 50 wind assets (5,684 MW) and 49 solar assets (1,892 MW).
  97 of them have metered data in 2024–25; the other 2 were commissioned later. That is 1.63 million metered rows,
  none negative.
* **Assets commissioned mid-period.** 10 assets started during the period, including about 1.2 GW of wind in
  Aug–Dec 2024. The fleet capacity factor therefore counts an asset's capacity only from its first metered hour
  (figure below).
* **Capacity check.** The snapshot's capability is a current value. Checked against history, each asset's peak
  metered output divided by its capability has a median of 0.99 (range 0.61–1.12), so the snapshot is a reasonable
  denominator for 2024–25.
* **Assets missing from the snapshot.** The snapshot only covers assets reporting today. 21 wind/solar-like
  assets in the asset list are missing from it (retired, small or renamed); 5 of them produced energy in 2024–25.
  That is only **0.14% of fleet energy**, a negligible gap.
  * Caveat: assets with uninformative names and no fuel type are invisible to this check, so 0.14% is a lower
    bound.

![coverage](reports/figures/p5_coverage.png)

### Real vs synthetic shapes: what the synthetic data got wrong

![shapes](reports/figures/p5_shapes.png)

| | Real 2024–25 | Synthetic (Phase 1) |
|---|---:|---:|
| Wind CF: mean / std / P95 | 0.30 / 0.19 / 0.63 | 0.39 / **0.33** / **0.96** |
| Wind CF: hours below 5% of max | 8.8% | 16.8% |
| Wind diurnal shape | night peak, minimum around 10:00 | almost flat |
| Solar CF: mean / P95 | 0.21 / 0.81 | 0.12 / 0.59 |
| Load: mean / std (MW) | 10,214 / 818 | 9,799 / 908 |
| Load diurnal shape | flat-topped, peak at 17:00 | morning and evening double peak |
| Correlation of wind with load | −0.06 | +0.10 |

* **Synthetic wind was far too "peaky".**
  * A single AR(1) site swings from 0 to 100%, whereas the real *fleet* is spread over many sites and rarely
    exceeds 63%.
  * This is the most consequential assumption in the whole POC (see congestion below).
* Synthetic solar was too low.
* The synthetic load had a residential double-peak shape. Real AIL is dominated by flat industrial demand.

### Calibration: does the copper-plate price resemble the real pool price?

With line limits removed, the model's price is the merit-order price, computed exactly; a check against the
copper-plate LP with must-run agrees to 7e-15. It is compared with the real pool price hour by hour.

Two transparent parameters were **fitted on 2024 and tested on 2025**. The objective is the mean absolute
difference between the quantile functions over P1–P95, which deliberately excludes the spike tail.

1. **Gas cost multiplier**, applied to cogen, CCGT, steam and peakers. Fitted value: **0.70**, consistent with low
   2024–25 gas prices.
2. **Cogeneration must-run share.** The objective is **flat** in this direction, so the parameter is **not
   identifiable** from prices. With about 5 GW of wind and 1.5 GW of solar against roughly 10 GW of load, must-run
   cogen almost never pushes the flexible residual below the gas units. The fit returns 0%.

| 2025 test year | Real pool price | Model, base costs | Model, calibrated |
|---|---:|---:|---:|
| P10 / P50 / P90 (\$/MWh) | 1.7 / 22.9 / 64.0 | 42.0 / 45.0 / 55.0 | 29.4 / 31.5 / 45.5 |
| Mean / std | 43.7 / 104.1 | 47.1 / 5.3 | 33.8 / 5.6 |
| Hours ≤ \$1 / ≥ \$100 / ≥ \$500 | 9.3% / 5.2% / 1.6% | 0 / 0 / 0 | 0 / 0 / 0 |
| Quantile distance P1–P95 | — | 23.5 | **14.7** |
| Hourly MAE / median absolute error | — | 40.1 / 25.3 | **32.0 / 14.1** |
| Spearman rank correlation with real | — | 0.70 | 0.70 |

![price duration](reports/figures/p5_price_duration.png)
![price profiles](reports/figures/p5_price_profiles.png)

* **What is close.**
  * The calibrated model matches the 2024 median (\$31.5 vs \$30.6). It halves the test-year quantile distance and
    the median absolute error.
  * The *ordering* of hours is partly right (Spearman 0.70): windy hours are cheap in both.
* **What only partly transfers.**
  * Within one year, the model's price sits on a single plateau (\$31.5) for about half of all hours. The real price
    spreads continuously from \$16 to \$42 over P20–P70 in 2024.
  * In 2025, real prices were lower (median \$22.9), so the 2024-fitted multiplier overshoots the 2025 median by
    \$8.6. The gas price *level* changes from year to year, and one static parameter cannot carry it across years.
* **What differs, and why. This is a structural gap, not a tuning problem.**
  1. **No price spikes.** In reality 8% of hours are ≥ \$100 and 1.9% are ≥ \$500. They are concentrated in the
     evening: the mean real price is \$95–110 for each hour from 17:00 to 20:00. They come from offer behaviour, scarcity, and ramping and commitment
     limits. A cost-based merit order with one step per unit cannot produce them: the modelled maximum is the
     \$120 peaker.
  2. **No zero prices.** In reality 7% of hours are ≤ \$1. These come from self-scheduled imports and must-run
     units offered at \$0 and from minimum stable generation; the toy stack has none of these.
  3. **A flat, stepped body.** Thirteen units give about 10 price levels; the real stack has hundreds of offer
     blocks.
* **Implication.** Use this model for *congestion patterns and relative LMP differences*. Do not use it for price
  levels or price risk. Price spikes need an offer-based supply curve (AESO merit-order data) or a scarcity
  adder; see next steps.

![calibration surface](reports/figures/p5_calibration_surface.png)

### Congestion with real shapes: the headline result changes

The network DC-OPF (same toy network and ratings) was re-run on the real 2024–25 shapes.

| Share of hours at limit | Synthetic inputs | Real shapes, base costs | Real shapes, calibrated costs |
|---|---:|---:|---:|
| SOUTH–CAL | 22.1% | **1.2%** | 1.2% |
| EDM–NORTH (each circuit) | 1.5% | 1.5% | 1.5% |
| BC–CAL (intertie) | 3.4% | 5.3% | 0.0% |
| ≥1 line at limit | 27.0% | **7.9%** | 2.7% |

![congestion real vs synthetic](reports/figures/p5_congestion.png)

* **The 22% SOUTH–CAL congestion in Phases 1–3 was mostly an artefact of the peaky synthetic wind.** With the real
  fleet's diversified output, the southern export limit binds in about 1% of hours.
  * The truth is probably in between. Applying one *province-wide* fleet capacity factor to a single southern bus
    understates the variability of a regional sub-fleet, because southern wind farms are closer together and more
    correlated.
  * The public endpoints give no asset locations. Mapping assets to AESO planning regions is the next step.
* **Intertie congestion depends on the gas price.** With calibrated (cheaper) gas, the \$55 BC import is never
  economic, so the intertie stops binding.
* **The North corridor is unchanged**, because its use is driven by the flat industrial load and cogen, not by
  weather.
* **Lesson for a real nodal study.** The *methods* in Phases 2–4 carry over unchanged: Monte Carlo, N-1, surrogates
  and storage. The *numbers* are only as good as the input shapes and ratings. The calibration step exposed the
  most important wrong assumption, the wind process, before any conclusion was drawn from it.

The real-shape study is stored in the database as study `real_2024_2025`, and the observed hourly series (pool
price, AIL, wind and solar CF) as table `aeso_hourly`. The dashboard's *LMP distributions* tab can overlay the
actual pool price. Note that rerunning Phase 4 recreates the schema, so run Phase 5 after Phase 4.

---

## Phase 6 (stretch) — GridSFM: a pretrained AC-OPF graph model vs a full solver

[GridSFM](https://github.com/microsoft/GridSFM) (Microsoft Research, 2026) is a pretrained graph neural network for
**AC** optimal power flow on grids with at least 500 buses. Given a grid and a scenario it predicts voltages, angles,
generator P/Q, branch flows and a **feasibility probability**. Its sample scenarios and released datasets come from
PGLib cases and from open US state grids built from OpenStreetMap.

The released checkpoint (`gridsfm_open_v1.1.pt`) was run **without retraining** and compared with the full
AC-OPF solver on the same scenarios. Code and setup are in [`experiments/gridsfm/`](experiments/gridsfm/); raw
results are in `reports/phase6_gridsfm_*.json`.

### Set-up

* **Grids.**
  * `msr_mississippi`: the smallest released open (OpenStreetMap-based) grid, with 528 buses, 39 generators and
    902 branches.
  * `case500_goc`: a PGLib case with 500 buses and 171 generators, used as a second check.
* **Scenarios.**
  * The unperturbed base case.
  * "Normal" scenarios: system load × 0.85–1.15, and in about three quarters of them 1–2 of the largest units
    forced out.
  * "Stress" scenarios: load × 1.25–1.70.
  * Both ranges are inside GridSFM's own training perturbations (load × 0.8–1.5, unit outages).
* **Reference solver.** Julia 1.11.5 with PowerModels (AC polar formulation, `ACPPowerModel`) and Ipopt, with
  exactly the package versions and Ipopt settings of GridSFM's data pipeline.
  * Each `.pyg.json` scenario is converted to MATPOWER. On both base cases the conversion reproduces the shipped
    solution's objective to 3e-9 (relative) and voltages to 3e-5 pu.
  * Julia's one-off 10 s compile ("JIT warm-up") is excluded from the timings.
* **GridSFM.** CPU only, 4 threads, single-graph `predict()` and batched calls, in a separate Python virtual
  environment, because GridSFM requires torch < 2.9.

### Inference time vs the full solver

| | msr_mississippi (61 scenarios) | case500_goc (31 scenarios) |
|---|---:|---:|
| Ipopt, scenarios with a solution: median per scenario | 1.58 s (n = 25) | 1.50 s (n = 21) |
| Ipopt, scenarios with no feasible point: median / max | 11.2 s / 532 s (n = 36) | 8.6 s / 28.5 s (n = 10) |
| **GridSFM per scenario** (single-graph / batched) | **0.33 s / 0.31 s** | **0.24 s / 0.20 s** |
| Speed-up on solvable scenarios | ≈ 5× | ≈ 6× |
| Speed-up over the whole study | 112× (2,215 s vs 20 s) | 18× (134 s vs 7 s) |

At 500 buses on a CPU, GridSFM is only about **5× faster than Ipopt on solvable scenarios**. Most of the headline
speed-up comes from **infeasible scenarios**, where Ipopt can take minutes to give up and GridSFM still answers in
0.3 s. The gap should grow with grid size and with GPU batching; neither was tested here.

### Accuracy (zero-shot; compared with the full solver)

| | msr_mississippi | case500_goc |
|---|---:|---:|
| Base case: cost error / V MAE / Pg MAE | −0.30% / 0.0014 pu / 3.3 MW | −0.54% / 0.0011 pu / 0.6 MW |
| Load-scaling-only scenarios: mean \|cost error\| (p95) | 7.0% (13.3%) | 7.1% (12.0%) |
| Scenarios with load scaling and unit outages: mean \|cost error\| (p95) | 9.1% (15.7%) | 6.8% (12.8%) |
| Perturbed scenarios: V MAE / Pg MAE per unit | 0.006–0.007 pu / 10–15 MW | 0.005–0.006 pu / 5 MW |
| **Perturbed scenarios: mean \|total generation − Ipopt\|** | **320–390 MW (≈5% of load)** | **680–770 MW (≈4%)** |

![GridSFM balance](reports/figures/p6_gridsfm_balance.png)

* **The base snapshots are near-perfect, but they are probably part of the training data.** They therefore show
  how well GridSFM fits known snapshots, not how well it generalises.
* **On perturbed scenarios, the predicted total generation barely follows system load.**
  * Across the case500 scenarios, the generation error is strongly anti-correlated with the load factor
    (r = −0.93): too much generation at low load, too little at high load.
  * The prediction therefore violates the power balance by hundreds of MW. Hence cost errors of about 7%, against
    0.3–0.5% on the base cases.
* **The model does not copy the generator set-point input.** Each generator's input features include a Pg column,
  which in the open MSR grids almost equals the solved dispatch (correlation 1.000). Setting it to zero, to
  (Pmin + Pmax)/2, or scaling it with load changed the cost error by less than 0.2 percentage points and the
  balance error by a few MW, so there is no leakage.

### Feasibility predictions (GridSFM probability ≥ 0.5 counted as "feasible")

| | Ipopt found a solution | Ipopt found none |
|---|---:|---:|
| msr_mississippi: predicted feasible / infeasible | 25 / 0 | **14** / 22 |
| case500_goc: predicted feasible / infeasible | 13 / **8** | 0 / 10 |

* Clear cases are classified correctly: every stress scenario (load ≥ 1.25×) is flagged infeasible.
* **Near the feasibility boundary, the classifier fails in both directions.**
  * On Mississippi, 14 scenarios with unit outages at normal load are called feasible although Ipopt found no
    solution.
  * On case500, 8 solvable scenarios at load × 1.04–1.14 are flagged infeasible.
* Caveat: Ipopt's `LOCALLY_INFEASIBLE` is a *local* certificate, not a proof of infeasibility.

### How GridSFM compares with this POC

| | This POC (Phases 1–3) | GridSFM-Open |
|---|---|---|
| Physics | **DC** OPF: linear, lossless, no voltage or reactive power. Exact and fast to solve (about 4 ms per scenario on one core here). | **AC** OPF: non-convex, with losses, voltages, reactive power and transformer taps. Seconds per solve, sometimes minutes, and solves can fail. |
| Grid | Toy 6-bus, 8-circuit Alberta-like system with made-up parameters | Real-size PGLib and OpenStreetMap-derived US grids (from 500 to several thousand buses) |
| Learning set-up | **One small model per grid and per output** (LightGBM per bus, an MLP, a classifier per line), trained in seconds on thousands of OPF solves of *that* grid. Grid-specific, and must be retrained when ratings or topology change (Phase 3). | **One pretrained, topology-agnostic graph model.** Its graph representation of buses, branches and network cycles lets it run zero-shot on other grids and on N-1 topologies, and it can be fine-tuned. |
| Outputs | **LMPs per bus** and binding lines, i.e. the market quantities | Primal AC state (V, θ, Pg, Qg, flows) and a feasibility probability. **No prices.** |
| Accuracy on the target task | MAE \$0.84/MWh; congestion probabilities within 0.08 percentage points of the OPF Monte Carlo | Cost within 0.5% on the base snapshots, but about 7% (p95 12–16%) under load and outage perturbations |
| What it is good for | Fast probabilistic congestion and price studies on a fixed network model | Fast screening of AC feasibility and an approximate AC operating point on real-size grids, or a warm start for the solver |

### Why GridSFM does not output LMPs

* **An LMP is not part of the dispatch.** It is the **dual variable** (Lagrange multiplier) of each bus's
  power-balance constraint: the change in total cost if one more MW were consumed at that bus. GridSFM is trained
  to predict primal quantities (voltages, dispatch, flows) plus feasibility, so its outputs contain no
  multipliers.
* **Recovering prices from a predicted dispatch is not reliable.** It requires knowing which constraints bind and
  their sensitivities. On top of that, the predicted dispatch is not exactly optimal or balanced (see the 300–770 MW
  balance errors above), and small primal errors near a binding limit produce large price errors. This is the same
  step-function issue seen in Phase 3.
* **Practical options.**
  1. Use GridSFM's prediction as a warm start and run the OPF once, to obtain exact duals.
  2. Add a dual (price) output to the model, trained on the multipliers the solver already exports. This was out
     of scope here: no retraining.
  3. For market prices, follow this POC and learn LMPs directly from a market-style OPF, here DC.
* **A note on the shipped duals.** The per-bus dual values (`solution.duals.bus`) shipped with the GridSFM samples
  could not be matched bus by bus to a clean PowerModels re-solve: the means agree, the per-bus values do not. All
  LMPs quoted here come from this project's own solves.

### Could this POC build a 500+-bus Alberta grid from open data and solve it with GridSFM?

In principle, yes: GridSFM's `power_grid/` pipeline does exactly this for US states. For Alberta it would need:

1. **Topology.** OpenStreetMap transmission lines and substations (138/144, 240 and 500 kV). The 240/500 kV
   backbone alone is probably under 500 buses, the model's minimum supported size, so the 138/144 kV network would
   be needed too.
2. **Generators.** Plant locations from OSM `power=plant` or NRCan data, joined to the AESO asset list and the CSD
   fuel and capability data from Phase 5.
3. **Load.** AIL allocated to substations by AESO planning area.
4. **Electrical parameters and solvability.** Line impedances and ratings estimated from voltage class and
   length, then GridSFM's Julia relaxation step to make the AC-OPF solvable.

The result would be a *plausible*, open-data approximation, not AESO's confidential planning model. GridSFM would
give fast AC screening on it, but per Phase 6 its zero-shot accuracy under load changes must be checked against a
solver, and it would still give **no LMPs**. This is listed as a next step rather than done.

---

## Roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Synthetic grid, hourly DC-OPF/LMP simulator, sanity checks | ✅ done |
| 2 | Monte Carlo (load, wind, line and generator outages); N-1 ranking; solve-time budget | ✅ done |
| 3 | Surrogates: copper-plate baseline, per-bus LightGBM + SHAP, multi-output MLP, line-congestion classifier; surrogate MC versus full OPF | ✅ done |
| 4 | SQLAlchemy store (SQLite by default; Postgres/SQL Server through the connection string); Streamlit dashboard | ✅ done |
| 5 | AESO API: real AIL load shape, wind/solar shapes, calibration against pool price (needs `AESO_API_KEY`) | ✅ done |
| 6 | (stretch) GridSFM pretrained AC-OPF graph model vs a full AC-OPF solver (Julia PowerModels + Ipopt) | ✅ done |

### Next steps beyond this POC

* AC power flow and AC-OPF, to add losses, voltage and reactive power.
* Graph neural networks that use the network topology (or ratings and topology as features), so surrogates can
  generalise across outages and network what-ifs.
* Surrogates with an OPF fallback: call the OPF when the classifier or the reserve margin flags a stressed system.
* Open-data Alberta grid with at least 500 buses (OSM topology, AESO assets, AIL by planning area), solved with
  AC-OPF. Use GridSFM for fast screening and warm starts, and a solver for duals and LMPs (see Phase 6).
* Calibrating costs from AESO merit-order data: an offer-based supply curve, needed for price spikes and $0
  hours (Phase 5 shows that a cost-based stack cannot produce them).
* Regional wind and solar shapes: map metered assets to AESO planning regions, and re-derive line ratings,
  because Phase 5 shows congestion results hinge on regional wind variability.
* Forced outage rates from AIES generation capacity data.
* Intertie limits taken from AESO intertie reports.
* Security-constrained OPF: add N-1 constraints to the market dispatch, and use probabilistic or seasonal line
  ratings.
* Rare-event Monte Carlo (importance sampling or cross-entropy) for adequacy metrics, and sequential Monte Carlo with
  repair times.

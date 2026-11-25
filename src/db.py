"""SQL storage for network data, scenarios, inputs and results (SQLAlchemy Core).

Connection string: env var ``DB_URL`` if set, else ``db.url`` in config.yaml
(SQLite file under data/ by default). Examples:

    sqlite:///data/aeso_poc.sqlite
    postgresql+psycopg://user:pass@host:5432/aeso_poc
    mssql+pyodbc://user:pass@host/aeso_poc?driver=ODBC+Driver+18+for+SQL+Server

Only portable column types are used (Integer, Float, String(n), Boolean,
DateTime, Text) and every query in this module / the dashboard is plain
ANSI SQL or SQLAlchemy Core, so the same code targets SQLite, PostgreSQL and
SQL Server. ``python -m src.db --ddl`` writes the CREATE TABLE statements for
all three dialects to sql/ without needing a server.

Layout (see docs/schema.md for the diagram):
  network   : bus, line, generator
  run       : one row per pipeline run (config snapshot)
  scenario  : one row per hour / Monte Carlo scenario, per study
  inputs    : scenario_bus_input, scenario_line_outage, scenario_gen_availability
  results   : bus_result (LMP), line_result (flow/loading/binding), gen_dispatch,
              surrogate_lmp, surrogate_congestion
  summaries : line_congestion_stat, lmp_quantile, n1_result, n1_delta_lmp,
              surrogate_metric, runtime
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import (Boolean, Column, DateTime, Float, ForeignKey, ForeignKeyConstraint, Integer, MetaData,
                        String, Table, Text, create_engine, func, select)
from sqlalchemy.dialects import mssql
from sqlalchemy.engine import Engine, make_url

from .config import ROOT, data_dir, load_config

metadata = MetaData()
LongText = Text().with_variant(mssql.NVARCHAR(None), "mssql")  # NVARCHAR(max) on SQL Server (TEXT is deprecated)
S = String(64)

bus = Table("bus", metadata,
            Column("bus_id", S, primary_key=True), Column("label", String(128)),
            Column("x", Float), Column("y", Float), Column("load_share", Float), Column("temp_offset_c", Float))
line = Table("line", metadata,
             Column("line_id", S, primary_key=True),
             Column("bus0", S, ForeignKey("bus.bus_id"), nullable=False),
             Column("bus1", S, ForeignKey("bus.bus_id"), nullable=False),
             Column("x_ohm", Float), Column("s_nom_mw", Float), Column("outage_rate", Float))
generator = Table("generator", metadata,
                  Column("gen_id", S, primary_key=True), Column("bus_id", S, ForeignKey("bus.bus_id"), nullable=False),
                  Column("carrier", S), Column("p_nom_mw", Float), Column("marginal_cost", Float), Column("profile", S))

run = Table("run", metadata,
            Column("run_id", Integer, primary_key=True, autoincrement=True),
            Column("created_utc", DateTime, nullable=False), Column("label", String(128)),
            Column("seed", Integer), Column("config_json", LongText))

scenario = Table("scenario", metadata,
                 Column("run_id", Integer, ForeignKey("run.run_id"), primary_key=True),
                 Column("study", String(32), primary_key=True),        # hourly | mc_test | mc_train
                 Column("scenario_id", Integer, primary_key=True),
                 Column("hour_start", DateTime),                        # hourly: the hour; MC: sampled base hour
                 Column("hour", Integer), Column("dow", Integer), Column("month", Integer),
                 Column("temp_system_c", Float), Column("cf_wind_south", Float), Column("cf_wind_central", Float),
                 Column("cf_solar_south", Float), Column("pu_hydro", Float), Column("total_load_mw", Float),
                 Column("merit_price", Float), Column("n_lines_out", Integer), Column("gen_mw_on_outage", Float))

_scen_fk = lambda: ForeignKeyConstraint(["run_id", "study", "scenario_id"],  # noqa: E731
                                        ["scenario.run_id", "scenario.study", "scenario.scenario_id"])


def _scen_cols():
    return [Column("run_id", Integer, primary_key=True), Column("study", String(32), primary_key=True),
            Column("scenario_id", Integer, primary_key=True)]


scenario_bus_input = Table("scenario_bus_input", metadata, *_scen_cols(),
                           Column("bus_id", S, ForeignKey("bus.bus_id"), primary_key=True),
                           Column("load_mw", Float), Column("temp_c", Float), _scen_fk())
scenario_line_outage = Table("scenario_line_outage", metadata, *_scen_cols(),
                             Column("line_id", S, ForeignKey("line.line_id"), primary_key=True), _scen_fk())
scenario_gen_availability = Table("scenario_gen_availability", metadata, *_scen_cols(),
                                  Column("gen_id", S, ForeignKey("generator.gen_id"), primary_key=True),
                                  Column("available_fraction", Float), _scen_fk())

bus_result = Table("bus_result", metadata, *_scen_cols(),
                   Column("bus_id", S, ForeignKey("bus.bus_id"), primary_key=True),
                   Column("lmp", Float), Column("unserved_mw", Float), _scen_fk())
line_result = Table("line_result", metadata, *_scen_cols(),
                    Column("line_id", S, ForeignKey("line.line_id"), primary_key=True),
                    Column("flow_mw", Float), Column("loading", Float), Column("binding", Boolean), _scen_fk())
gen_dispatch = Table("gen_dispatch", metadata, *_scen_cols(),
                     Column("gen_id", S, ForeignKey("generator.gen_id"), primary_key=True),
                     Column("p_mw", Float), _scen_fk())
surrogate_lmp = Table("surrogate_lmp", metadata, *_scen_cols(),
                      Column("model", S, primary_key=True),
                      Column("bus_id", S, ForeignKey("bus.bus_id"), primary_key=True),
                      Column("lmp_pred", Float), _scen_fk())
surrogate_congestion = Table("surrogate_congestion", metadata, *_scen_cols(),
                             Column("line_id", S, ForeignKey("line.line_id"), primary_key=True),
                             Column("p_binding", Float), _scen_fk())

line_congestion_stat = Table("line_congestion_stat", metadata,
                             Column("run_id", Integer, ForeignKey("run.run_id"), primary_key=True),
                             Column("source", S, primary_key=True),       # hourly_opf | mc_opf | mc_surrogate
                             Column("line_id", S, ForeignKey("line.line_id"), primary_key=True),
                             Column("p_binding", Float), Column("ci95_low", Float), Column("ci95_high", Float),
                             Column("expected_hours_per_year", Float))
lmp_quantile = Table("lmp_quantile", metadata,
                     Column("run_id", Integer, ForeignKey("run.run_id"), primary_key=True),
                     Column("source", S, primary_key=True),
                     Column("bus_id", S, ForeignKey("bus.bus_id"), primary_key=True),
                     Column("stat", String(16), primary_key=True), Column("value", Float))
n1_result = Table("n1_result", metadata,
                  Column("run_id", Integer, ForeignKey("run.run_id"), primary_key=True),
                  Column("outaged_line", S, ForeignKey("line.line_id"), primary_key=True),
                  Column("islands", S), Column("share_hours_overload", Float), Column("max_post_loading", Float),
                  Column("worst_line", S), Column("overload_mwh_per_year", Float),
                  Column("islanded_injection_max_mw", Float), Column("delta_cost_musd_per_year", Float),
                  Column("delta_congestion_rent_musd_per_year", Float), Column("mean_abs_delta_lmp", Float),
                  Column("share_hours_lmp_changed_gt_1usd", Float), Column("unserved_mwh_per_year", Float),
                  Column("rank_security", Integer), Column("rank_economic", Integer))
n1_delta_lmp = Table("n1_delta_lmp", metadata,
                     Column("run_id", Integer, ForeignKey("run.run_id"), primary_key=True),
                     Column("outaged_line", S, ForeignKey("line.line_id"), primary_key=True),
                     Column("bus_id", S, ForeignKey("bus.bus_id"), primary_key=True),
                     Column("mean_delta_lmp", Float))
surrogate_metric = Table("surrogate_metric", metadata,
                         Column("run_id", Integer, ForeignKey("run.run_id"), primary_key=True),
                         Column("task", String(16), primary_key=True),     # mc | hourly
                         Column("model", S, primary_key=True),
                         Column("bus_id", S, ForeignKey("bus.bus_id"), primary_key=True),
                         Column("metric", S, primary_key=True), Column("value", Float))
runtime = Table("runtime", metadata,
                Column("run_id", Integer, ForeignKey("run.run_id"), primary_key=True),
                Column("method", String(128), primary_key=True), Column("seconds", Float))


# ---------------------------------------------------------------------------
# connection
# ---------------------------------------------------------------------------
def db_url(cfg: dict) -> str:
    url = os.environ.get("DB_URL") or cfg["db"]["url"]
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        rel = url.removeprefix("sqlite:///")
        url = f"sqlite:///{(ROOT / rel).as_posix()}"   # relative paths are relative to the repo root
        (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
    return url


def safe_url(url: str) -> str:
    """Connection string with the password masked (for logs and the dashboard)."""
    return make_url(url).render_as_string(hide_password=True)


def get_engine(cfg: dict) -> Engine:
    return create_engine(db_url(cfg), future=True)


def create_schema(engine: Engine, drop: bool = False) -> None:
    if drop:
        metadata.drop_all(engine)
    metadata.create_all(engine)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _append(engine: Engine, table: Table, df: pd.DataFrame, chunksize: int = 20000) -> int:
    if df.empty:
        return 0
    df = df[[c.name for c in table.columns if c.name in df.columns]]
    df = df.replace({np.nan: None})
    with engine.begin() as conn:
        for start in range(0, len(df), chunksize):
            conn.execute(table.insert(), df.iloc[start:start + chunksize].to_dict(orient="records"))
    return len(df)


def write_network(engine: Engine, cfg: dict) -> None:
    """Insert the network once. (Re-writing it under existing results would break
    foreign keys on PostgreSQL / SQL Server; use a fresh schema to change it.)"""
    with engine.connect() as conn:
        if conn.execute(select(func.count()).select_from(bus)).scalar():
            return
    _append(engine, bus, pd.DataFrame([{"bus_id": b, "label": v["label"], "x": v["x"], "y": v["y"],
                                        "load_share": v["load_share"], "temp_offset_c": v["temp_offset"]}
                                       for b, v in cfg["buses"].items()]))
    _append(engine, line, pd.DataFrame([{"line_id": l, "bus0": v["bus0"], "bus1": v["bus1"], "x_ohm": v["x"],
                                         "s_nom_mw": v["s_nom"], "outage_rate": cfg["montecarlo"]["line_outage_rate"][l]}
                                        for l, v in cfg["lines"].items()]))
    _append(engine, generator, pd.DataFrame([{"gen_id": g, "bus_id": v["bus"], "carrier": v["carrier"], "p_nom_mw": v["p_nom"],
                                              "marginal_cost": v["marginal_cost"], "profile": v["profile"]}
                                             for g, v in cfg["generators"].items()]))


def new_run(engine: Engine, cfg: dict, label: str) -> int:
    with engine.begin() as conn:
        r = conn.execute(run.insert().values(created_utc=datetime.now(timezone.utc).replace(tzinfo=None), label=label,
                                             seed=cfg["seed"], config_json=json.dumps(cfg, default=str)))
        return int(r.inserted_primary_key[0])


def _long(df: pd.DataFrame, key_cols: dict, var: str, val: str, prefix: str = "") -> pd.DataFrame:
    w = df.copy()
    w.columns = [c.removeprefix(prefix) for c in w.columns]
    w.index = pd.RangeIndex(len(w), name="scenario_id") if not isinstance(w.index, pd.RangeIndex) else w.index.rename("scenario_id")
    out = w.reset_index().melt(id_vars="scenario_id", var_name=var, value_name=val)
    for k, v in key_cols.items():
        out[k] = v
    return out


def write_study(engine: Engine, cfg: dict, run_id: int, study: str, inputs: pd.DataFrame, res,
                line_out: pd.DataFrame | None = None, gen_avail: pd.DataFrame | None = None,
                store_dispatch: bool = True) -> dict:
    """Write one study (hourly or a Monte Carlo set): scenarios, inputs and OPF results."""
    from .features import merit_order_price

    n = len(inputs)
    ids = np.arange(n)
    loads = inputs[[c for c in inputs if c.startswith("load_") and not c.startswith("load_mult")]]
    pnom = pd.Series({g: v["p_nom"] for g, v in cfg["generators"].items()})
    ts = inputs.index if isinstance(inputs.index, pd.DatetimeIndex) else pd.to_datetime(inputs["base_timestamp"])
    scen = pd.DataFrame({
        "run_id": run_id, "study": study, "scenario_id": ids, "hour_start": pd.DatetimeIndex(ts).to_pydatetime(),
        "hour": inputs["hour"].to_numpy(), "dow": inputs["dow"].to_numpy(), "month": inputs["month"].to_numpy(),
        "temp_system_c": inputs["temp_CAL"].to_numpy(), "cf_wind_south": inputs["cf_wind_south"].to_numpy(),
        "cf_wind_central": inputs["cf_wind_central"].to_numpy(), "cf_solar_south": inputs["cf_solar_south"].to_numpy(),
        "pu_hydro": inputs["pu_hydro"].to_numpy(), "total_load_mw": loads.sum(axis=1).to_numpy(),
        "merit_price": merit_order_price(cfg, inputs, gen_avail).to_numpy(),
        "n_lines_out": 0 if line_out is None else line_out.sum(axis=1).to_numpy(),
        "gen_mw_on_outage": 0.0 if gen_avail is None else ((1 - gen_avail) * pnom).sum(axis=1).to_numpy(),
    })
    key = {"run_id": run_id, "study": study}
    counts = {"scenario": _append(engine, scenario, scen)}

    def reindexed(df):
        d = df.copy()
        d.index = pd.RangeIndex(n, name="scenario_id")
        return d

    bi = _long(reindexed(loads), key, "bus_id", "load_mw", "load_")
    temps = _long(reindexed(inputs[[f"temp_{b}" for b in loads.columns.str.removeprefix("load_")]]), key, "bus_id", "temp_c", "temp_")
    counts["scenario_bus_input"] = _append(engine, scenario_bus_input, bi.merge(temps, on=["scenario_id", "bus_id", "run_id", "study"]))
    if line_out is not None:
        lo = _long(reindexed(line_out.astype(bool)), key, "line_id", "out")
        counts["scenario_line_outage"] = _append(engine, scenario_line_outage, lo[lo["out"]])
    if gen_avail is not None:
        ga = _long(reindexed(gen_avail), key, "gen_id", "available_fraction")
        counts["scenario_gen_availability"] = _append(engine, scenario_gen_availability, ga[ga["available_fraction"] < 1])

    lmp = _long(reindexed(res.lmp), key, "bus_id", "lmp")
    shed = reindexed(res.dispatch[[c for c in res.dispatch if c.startswith("shed_")]])
    shed_l = _long(shed, key, "bus_id", "unserved_mw", "shed_")
    br = lmp.merge(shed_l, on=["scenario_id", "bus_id", "run_id", "study"], how="left")
    counts["bus_result"] = _append(engine, bus_result, br)
    lr = _long(reindexed(res.flow), key, "line_id", "flow_mw")
    lr["loading"] = _long(reindexed(res.loading), key, "line_id", "loading")["loading"].to_numpy()
    lr["binding"] = _long(reindexed(res.binding.astype(bool)), key, "line_id", "binding")["binding"].to_numpy()
    counts["line_result"] = _append(engine, line_result, lr)
    if store_dispatch:
        gd = _long(reindexed(res.dispatch[[c for c in res.dispatch if not c.startswith("shed_")]]), key, "gen_id", "p_mw")
        counts["gen_dispatch"] = _append(engine, gen_dispatch, gd)
    return counts


def write_surrogate(engine: Engine, run_id: int, study: str, preds: dict[str, pd.DataFrame], proba: pd.DataFrame) -> dict:
    key = {"run_id": run_id, "study": study}
    frames = []
    for model, P in preds.items():
        d = P.copy()
        d.index = pd.RangeIndex(len(d), name="scenario_id")
        f = _long(d, key, "bus_id", "lmp_pred", "lmp_")
        f["model"] = model
        frames.append(f)
    p = proba.copy()
    p.index = pd.RangeIndex(len(p), name="scenario_id")
    return {"surrogate_lmp": _append(engine, surrogate_lmp, pd.concat(frames)),
            "surrogate_congestion": _append(engine, surrogate_congestion, _long(p, key, "line_id", "p_binding"))}


def write_summaries(engine: Engine, run_id: int, rep_dir: Path, hourly_binding: pd.DataFrame) -> dict:
    t = rep_dir / "tables"
    counts = {}
    # congestion statistics from three sources
    from .montecarlo import wilson
    n = len(hourly_binding)
    k = hourly_binding.sum()
    lo, hi = wilson(k, n)
    rows = [pd.DataFrame({"source": "hourly_opf", "line_id": k.index, "p_binding": (k / n).to_numpy(),
                          "ci95_low": lo, "ci95_high": hi, "expected_hours_per_year": (k / n * 8760).to_numpy()})]
    mc = pd.read_csv(t / "mc_line_congestion.csv", index_col=0)
    rows.append(pd.DataFrame({"source": "mc_opf", "line_id": mc.index, "p_binding": mc["p_congested"],
                              "ci95_low": mc["ci95_low"], "ci95_high": mc["ci95_high"],
                              "expected_hours_per_year": mc["expected_hours_per_year"]}))
    sc = t / "surrogate_mc_congestion.csv"
    if sc.exists():
        s = pd.read_csv(sc, index_col=0)
        rows.append(pd.DataFrame({"source": "mc_surrogate", "line_id": s.index, "p_binding": s["surrogate_p_mean_prob"],
                                  "ci95_low": np.nan, "ci95_high": np.nan,
                                  "expected_hours_per_year": s["surrogate_expected_h_per_year"]}))
    lcs = pd.concat(rows)
    lcs["run_id"] = run_id
    counts["line_congestion_stat"] = _append(engine, line_congestion_stat, lcs)

    q = pd.read_csv(t / "mc_lmp_quantiles.csv", index_col=0)
    ql = q[["P10", "P50", "P90", "P99", "mean"]].reset_index(names="bus_id").melt(id_vars="bus_id", var_name="stat")
    ql["source"] = "mc_opf"
    frames = [ql]
    sq = t / "surrogate_mc_quantiles.csv"
    if sq.exists():
        s = pd.read_csv(sq).rename(columns={"bus": "bus_id"})
        s = s.melt(id_vars=["bus_id", "stat"], var_name="source")
        s.loc[s["source"] == "OPF", "source"] = "mc_opf_check"
        frames.append(s[s["source"] != "mc_opf_check"])
    lq = pd.concat(frames)
    lq["run_id"] = run_id
    counts["lmp_quantile"] = _append(engine, lmp_quantile, lq)

    n1 = pd.read_csv(t / "n1_ranking.csv", index_col=0).reset_index(names="outaged_line")
    n1["islands"] = n1["islands"].fillna("")
    n1["run_id"] = run_id
    counts["n1_result"] = _append(engine, n1_result, n1)
    dl = pd.read_csv(t / "n1_mean_delta_lmp.csv", index_col=0).reset_index(names="outaged_line")
    dl = dl.melt(id_vars="outaged_line", var_name="bus_id", value_name="mean_delta_lmp")
    dl["run_id"] = run_id
    counts["n1_delta_lmp"] = _append(engine, n1_delta_lmp, dl)

    mets = []
    for task, fn in (("mc", "surrogate_mc_regression_by_bus.csv"), ("hourly", "surrogate_hourly_regression_by_bus.csv")):
        if (t / fn).exists():
            r = pd.read_csv(t / fn).rename(columns={"bus": "bus_id"}).melt(id_vars=["model", "bus_id"], var_name="metric")
            r["task"] = task
            mets.append(r)
    if mets:
        m = pd.concat(mets)
        m["run_id"] = run_id
        counts["surrogate_metric"] = _append(engine, surrogate_metric, m)
    rt = t / "surrogate_runtime.csv"
    if rt.exists():
        r = pd.read_csv(rt, index_col=0).reset_index(names="method")[["method", "seconds"]]
        r["run_id"] = run_id
        counts["runtime"] = _append(engine, runtime, r)
    return counts


def store_all(cfg: dict, label: str = "pipeline", fresh: bool | None = None) -> dict:
    """Phase 4 entry point: load all phase outputs from data/ + reports/ into the database."""
    from .config import reports_dir
    from .montecarlo import Scenarios
    from .run import load_result

    t0 = time.perf_counter()
    d = data_dir(cfg)
    engine = get_engine(cfg)
    fresh = cfg["db"].get("recreate", True) if fresh is None else fresh
    create_schema(engine, drop=fresh)
    write_network(engine, cfg)
    run_id = new_run(engine, cfg, label)
    counts = {}
    hourly_inputs = pd.read_parquet(d / "phase1" / "inputs.parquet")
    hourly = load_result(d / "phase1")
    for k, v in write_study(engine, cfg, run_id, "hourly", hourly_inputs, hourly,
                            store_dispatch=cfg["db"].get("store_dispatch", True)).items():
        counts[f"hourly.{k}"] = v
    mc_s = Scenarios(*(pd.read_parquet(d / "phase2" / f"mc_{k}.parquet") for k in ("inputs", "line_out", "gen_avail")))
    mc_r = load_result(d / "phase2", "mc_")
    for k, v in write_study(engine, cfg, run_id, "mc_test", mc_s.inputs, mc_r, mc_s.line_out, mc_s.gen_avail,
                            store_dispatch=cfg["db"].get("store_dispatch", True)).items():
        counts[f"mc_test.{k}"] = v
    p3 = d / "phase3"
    if (p3 / "mc_test_congestion_proba.parquet").exists():
        preds = {name: pd.read_parquet(p3 / f"mc_test_pred_{fn}.parquet")
                 for name, fn in (("LightGBM", "lightgbm"), ("MLP", "mlp"), ("baseline", "baseline"))}
        for k, v in write_surrogate(engine, run_id, "mc_test", preds,
                                    pd.read_parquet(p3 / "mc_test_congestion_proba.parquet")).items():
            counts[f"mc_test.{k}"] = v
    counts.update(write_summaries(engine, run_id, reports_dir(cfg), hourly.binding))
    return {"run_id": run_id, "url": safe_url(db_url(cfg)), "rows": counts, "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# reading (used by the dashboard)
# ---------------------------------------------------------------------------
def latest_run_id(engine: Engine) -> int | None:
    with engine.connect() as c:
        return c.execute(select(func.max(run.c.run_id))).scalar()


def read_sql(engine: Engine, query, **params) -> pd.DataFrame:
    with engine.connect() as c:
        return pd.read_sql(query, c, params=params or None)


# ---------------------------------------------------------------------------
# DDL export (no server needed)
# ---------------------------------------------------------------------------
def export_ddl(out_dir: Path) -> list[Path]:
    from sqlalchemy.dialects import postgresql, sqlite
    from sqlalchemy.schema import CreateTable

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, dialect in (("sqlite", sqlite.dialect()), ("postgresql", postgresql.dialect()), ("mssql", mssql.dialect())):
        ddl = "\n".join(f"{str(CreateTable(t).compile(dialect=dialect)).strip()};\n" for t in metadata.sorted_tables)
        p = out_dir / f"schema_{name}.sql"
        p.write_text(f"-- generated by `python -m src.db --ddl` ({name})\n\n{ddl}")
        paths.append(p)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description="Database utilities")
    ap.add_argument("--ddl", action="store_true", help="write CREATE TABLE scripts for sqlite/postgresql/mssql to sql/")
    ap.add_argument("--load", action="store_true", help="load all phase outputs into DB_URL / config db.url")
    args = ap.parse_args(argv)
    if args.ddl:
        for p in export_ddl(ROOT / "sql"):
            print("wrote", p.relative_to(ROOT))
    if args.load:
        print(store_all(load_config()))


if __name__ == "__main__":
    main()

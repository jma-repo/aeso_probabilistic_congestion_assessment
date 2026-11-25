"""Phase 4 tests: portable schema, round-trip storage, example SQL, dashboard smoke test."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text

from src import db
from src.config import ROOT, load_config
from src.montecarlo import sample_scenarios
from src.simulate import solve_many
from src.synth import generate_inputs


def test_ddl_compiles_for_all_dialects(tmp_path):
    paths = db.export_ddl(tmp_path)
    txt = {p.stem: p.read_text() for p in paths}
    assert "IDENTITY" in txt["schema_mssql"] and "BIT" in txt["schema_mssql"] and "NVARCHAR(max)" in txt["schema_mssql"]
    assert "BOOLEAN" in txt["schema_postgresql"]
    for t in ("bus_result", "line_result", "scenario", "n1_result"):
        assert all(f"CREATE TABLE {t}" in v for v in txt.values())


def test_safe_url_masks_password_and_env_override(monkeypatch):
    assert "secret" not in db.safe_url("postgresql+psycopg://user:secret@host/db")
    monkeypatch.setenv("DB_URL", "sqlite:////tmp/x.sqlite")
    assert db.db_url(load_config()) == "sqlite:////tmp/x.sqlite"


@pytest.fixture(scope="module")
def small_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("db") / "t.sqlite"
    cfg = load_config(overrides={"time": {"hours": 24 * 7}, "db": {"url": f"sqlite:///{path}"}})
    base = generate_inputs(cfg)
    hourly = solve_many(cfg, base)
    scen = sample_scenarios(cfg, base, 300, seed=5)
    scen.line_out.iloc[:20, 0] = True  # make sure some outages are present
    mc = solve_many(cfg, scen.inputs, scen.out_lines, scen.gen_avail)
    eng = db.get_engine(cfg)
    db.create_schema(eng, drop=True)
    db.write_network(eng, cfg)
    db.write_network(eng, cfg)  # idempotent
    rid = db.new_run(eng, cfg, "test")
    db.write_study(eng, cfg, rid, "hourly", base, hourly)
    db.write_study(eng, cfg, rid, "mc_test", scen.inputs, mc, scen.line_out, scen.gen_avail)
    return cfg, eng, rid, base, hourly, scen, mc


def test_round_trip_lmp_flow_and_outages(small_db):
    cfg, eng, rid, base, hourly, scen, mc = small_db
    lmp = db.read_sql(eng, text("SELECT scenario_id, bus_id, lmp FROM bus_result WHERE run_id=:r AND study='mc_test'"), r=rid)
    piv = lmp.pivot(index="scenario_id", columns="bus_id", values="lmp")
    np.testing.assert_allclose(piv[mc.lmp.columns].to_numpy(), mc.lmp.to_numpy(), equal_nan=True)
    lo = db.read_sql(eng, text("SELECT scenario_id, line_id FROM scenario_line_outage WHERE run_id=:r"), r=rid)
    assert len(lo) == int(scen.line_out.to_numpy().sum())
    n = db.read_sql(eng, text("SELECT COUNT(*) AS n FROM bus WHERE 1=1"))["n"].iloc[0]
    assert n == len(cfg["buses"])


def test_example_queries_run_and_match_pandas(small_db):
    cfg, eng, rid, base, hourly, scen, mc = small_db
    sql = (ROOT / "sql" / "example_queries.sql").read_text()
    stmts = [s.strip() for s in sql.split(";") if "SELECT" in s]
    assert len(stmts) == 5
    with eng.connect() as c:
        results = [pd.read_sql(text(s), c, params={"run_id": rid, "true_value": True}) for s in stmts]
    p = results[0].set_index("line_id")["p_binding"]
    np.testing.assert_allclose(p[mc.binding.columns].to_numpy(), mc.binding.mean().to_numpy())
    monthly = results[3]
    assert monthly["hours"].sum() == len(base)


def test_dashboard_smoke():
    """Render the dashboard against the pipeline database, if it has been built."""
    url = db.db_url(load_config())
    if url.startswith("sqlite") and not Path(url.split("sqlite:///")[-1]).exists():
        pytest.skip("pipeline database not built (run python -m src.run)")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "src" / "dashboard.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 4

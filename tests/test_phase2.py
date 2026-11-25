"""Phase 2 correctness tests: outage modelling, grouping, statistics, N-1 screen."""
import numpy as np
import pandas as pd
import pytest

from src import contingency, montecarlo
from src.config import load_config
from src.grid import islanded_buses, ptdf
from src.simulate import solve_block, solve_many
from src.synth import generate_inputs
from tests.test_phase1 import three_bus_loop_cfg


def test_wilson_interval_known_values():
    lo, hi = montecarlo.wilson(20, 100)
    assert lo == pytest.approx(0.1333, abs=1e-3) and hi == pytest.approx(0.2888, abs=1e-3)
    lo0, hi0 = montecarlo.wilson(0, 1000)
    assert lo0 == pytest.approx(0.0, abs=1e-12) and 0 < hi0 < 0.005


def test_removing_a_line_differs_from_zero_rating():
    """Why outages remove lines: a zero rating keeps KVL and is a different (wrong) network."""
    cfg = three_bus_loop_cfg()
    idx = pd.RangeIndex(1, name="scenario")
    x = pd.DataFrame({"load_C": [600.0]}, index=idx)
    removed = solve_block(cfg, x, out_lines=("A-B",))
    zeroed = solve_block(cfg, x, line_rating_scale={"A-B": 1e-9})
    # with A-B removed, A and B each feed C radially: A-C limit 300 -> a=300, b=300, LMP_C = 40
    assert removed.lmp.iloc[0]["C"] == pytest.approx(40.0)
    # with A-B 'zero rated', KVL forces theta_A = theta_B, so flows split differently
    assert not np.isclose(zeroed.lmp.iloc[0]["C"], removed.lmp.iloc[0]["C"])


def test_islanding_detection():
    cfg = load_config()
    assert islanded_buses(cfg, ("BC-CAL",)) == {"BC"}
    assert islanded_buses(cfg, ("EDM-NORTH-1",)) == set()
    assert islanded_buses(cfg, ("EDM-NORTH-1", "EDM-NORTH-2")) == {"NORTH"}


@pytest.fixture(scope="module")
def small():
    cfg = load_config(overrides={"time": {"hours": 24 * 30}, "montecarlo": {"workers": 1, "chunk": 200}})
    base = generate_inputs(cfg)
    return cfg, base


def test_scenarios_deterministic_and_outage_rates(small):
    cfg, base = small
    a = montecarlo.sample_scenarios(cfg, base, 3000, seed=7)
    b = montecarlo.sample_scenarios(cfg, base, 3000, seed=7)
    pd.testing.assert_frame_equal(a.inputs, b.inputs)
    pd.testing.assert_frame_equal(a.line_out, b.line_out)
    # empirical outage rate within 4 binomial standard errors of the configured rate
    for l, q in cfg["montecarlo"]["line_outage_rate"].items():
        se = np.sqrt(q * (1 - q) / 3000)
        assert abs(a.line_out[l].mean() - q) < 4 * se + 1e-9
    assert ((a.gen_avail >= 0) & (a.gen_avail <= 1)).all().all()
    assert (a.gen_avail["SOUTH_wind"] == 1).all()  # no separate wind outage model


def test_grouped_solve_matches_per_topology_solve(small):
    cfg, base = small
    scen = montecarlo.sample_scenarios(cfg, base, 400, seed=3)
    # force a few rows to share a non-trivial topology
    scen.line_out.iloc[:30] = False
    scen.line_out.iloc[:30, list(cfg["lines"]).index("SOUTH-CAL")] = True
    res = solve_many(cfg, scen.inputs, scen.out_lines, scen.gen_avail, chunk=150)
    rows = scen.inputs.index[:30]
    direct = solve_block(cfg, scen.inputs.loc[rows], out_lines=("SOUTH-CAL",), gen_availability=scen.gen_avail.loc[rows])
    np.testing.assert_allclose(res.lmp.loc[rows, direct.lmp.columns], direct.lmp, atol=1e-6)
    assert (res.flow.loc[rows, "SOUTH-CAL"] == 0).all()
    assert not res.binding.loc[rows, "SOUTH-CAL"].any()
    assert list(res.lmp.index) == list(scen.inputs.index)


def test_islanded_zero_load_bus_has_undefined_lmp(small):
    cfg, base = small
    x = base.iloc[:5].copy()
    res = solve_many(cfg, x, pd.Series([("BC-CAL",)] * 5, index=x.index))
    assert res.lmp["BC"].isna().all()
    assert res.lmp.drop(columns="BC").notna().all().all()


def test_ptdf_screen_matches_pypsa_power_flow(small):
    """Post-contingency flows from PTDF == PyPSA linear power flow with fixed injections."""
    cfg, base = small
    import pypsa  # noqa: F401
    from src.grid import build_network

    x = base.iloc[:3]
    b = solve_block(cfg, x)
    inj = contingency.bus_injections(cfg, x, b)
    H = ptdf(cfg, out_lines=("CAL-RD",))
    f_ptdf = inj[H.columns].to_numpy() @ H.to_numpy().T
    n = build_network(cfg, x, out_lines=("CAL-RD",))
    n.generators_t.p_set = b.dispatch.reindex(columns=n.generators.index)
    n.lpf()
    np.testing.assert_allclose(n.lines_t.p0[H.index].to_numpy(), f_ptdf, atol=1e-6)

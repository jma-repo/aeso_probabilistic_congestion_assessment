"""Phase 1 correctness tests.

Run: pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src import checks
from src.config import load_config, validate
from src.grid import ptdf
from src.simulate import solve_dispatch
from src.synth import generate_inputs


def two_town_cfg(line_mw=100.0):
    """README example: cheap wind in town A, expensive gas in town B, one line."""
    return {
        "seed": 0,
        "voll": 999.99,
        "binding_tolerance": 1e-4,
        "time": {"solve_chunk_hours": 100},
        "solver": {"name": "highs", "options": {"output_flag": False}},
        "buses": {"A": {"x": 0, "y": 0, "load_share": 0.0}, "B": {"x": 1, "y": 0, "load_share": 1.0}},
        "lines": {"A-B": {"bus0": "A", "bus1": "B", "x": 10.0, "s_nom": line_mw}},
        "generators": {
            "wind_A": {"bus": "A", "carrier": "wind", "p_nom": 300, "marginal_cost": 0.5, "profile": "wind_south"},
            "gas_B": {"bus": "B", "carrier": "ccgt", "p_nom": 300, "marginal_cost": 50.0, "profile": "none"},
        },
    }


def frame(load_b, wind_cf):
    idx = pd.date_range("2025-01-01", periods=len(load_b), freq="h")
    return pd.DataFrame({"load_B": load_b, "cf_wind_south": wind_cf}, index=idx)


def test_two_town_congested_and_uncongested():
    cfg = two_town_cfg(line_mw=100)
    # hour 0: load 150 > line 100 -> congested; hour 1: load 80 < line -> one price
    res = solve_dispatch(cfg, frame([150.0, 80.0], [1.0, 1.0]))
    lmp = res.lmp
    assert lmp.iloc[0]["A"] == pytest.approx(0.5)
    assert lmp.iloc[0]["B"] == pytest.approx(50.0)
    assert res.flow.iloc[0]["A-B"] == pytest.approx(100.0)
    assert bool(res.binding.iloc[0]["A-B"])
    assert lmp.iloc[1]["A"] == pytest.approx(0.5) and lmp.iloc[1]["B"] == pytest.approx(0.5)
    assert not bool(res.binding.iloc[1]["A-B"])
    # congestion rent = price difference x flow
    assert (lmp.iloc[0]["B"] - lmp.iloc[0]["A"]) * res.flow.iloc[0]["A-B"] == pytest.approx(4950.0)


def test_two_town_no_wind_means_single_gas_price():
    cfg = two_town_cfg()
    res = solve_dispatch(cfg, frame([150.0], [0.0]))
    assert res.lmp.iloc[0]["A"] == pytest.approx(50.0) and res.lmp.iloc[0]["B"] == pytest.approx(50.0)


def three_bus_loop_cfg():
    """Triangle with equal reactances: the classic loop-flow example.

    A: cheap gen ($10), B: dear gen ($40), C: 600 MW load, A-C limited to 300 MW.
    With equal x, a transfer A->C sends 2/3 on A-C and 1/3 via B; a transfer
    B->C sends 1/3 on A-C. So flow(A-C) = 2a/3 + b/3 with a + b = 600.
    Binding A-C at 300 MW gives a = b = 300. One more MW at C must keep A-C
    unchanged: 2da + db = 0 and da + db = 1 -> da = -1, db = +2, so
    LMP_C = -1*10 + 2*40 = $70/MWh: higher than ANY offer in the system.
    """
    c = two_town_cfg()
    c["buses"] = {
        "A": {"x": 0, "y": 0, "load_share": 0.0},
        "B": {"x": 1, "y": 0, "load_share": 0.0},
        "C": {"x": 0.5, "y": 1, "load_share": 1.0},
    }
    c["lines"] = {
        "A-B": {"bus0": "A", "bus1": "B", "x": 10.0, "s_nom": 1000},
        "A-C": {"bus0": "A", "bus1": "C", "x": 10.0, "s_nom": 300},
        "B-C": {"bus0": "B", "bus1": "C", "x": 10.0, "s_nom": 1000},
    }
    c["generators"] = {
        "cheap_A": {"bus": "A", "carrier": "x", "p_nom": 1000, "marginal_cost": 10.0, "profile": "none"},
        "dear_B": {"bus": "B", "carrier": "x", "p_nom": 1000, "marginal_cost": 40.0, "profile": "none"},
    }
    return c


def test_loop_flow_lmp_exceeds_every_offer():
    cfg = three_bus_loop_cfg()
    idx = pd.date_range("2025-01-01", periods=1, freq="h")
    res = solve_dispatch(cfg, pd.DataFrame({"load_C": [600.0]}, index=idx))
    lmp = res.lmp.iloc[0]
    assert lmp["A"] == pytest.approx(10.0)
    assert lmp["B"] == pytest.approx(40.0)
    assert lmp["C"] == pytest.approx(70.0)
    assert res.dispatch.iloc[0]["cheap_A"] == pytest.approx(300.0)
    assert res.dispatch.iloc[0]["dear_B"] == pytest.approx(300.0)
    # independent angle-based LP agrees
    ref_lmp, ref_flow, _ = checks.angle_dcopf_hour(cfg, {"C": 600.0}, {"cheap_A": 1000, "dear_B": 1000, "shed_C": 1200.0})
    np.testing.assert_allclose(res.lmp.iloc[0][ref_lmp.index], ref_lmp, atol=1e-6)
    np.testing.assert_allclose(res.flow.iloc[0][ref_flow.index], ref_flow, atol=1e-6)


def test_ptdf_rows_sum_and_slack():
    cfg = three_bus_loop_cfg()
    H = ptdf(cfg, slack="A")
    assert np.allclose(H["A"], 0.0)
    # injecting at B, withdrawing at slack A: 2/3 goes directly B->A, 1/3 via C
    assert H.loc["A-B", "B"] == pytest.approx(-2 / 3)


@pytest.fixture(scope="module")
def full_cfg():
    return load_config(overrides={"time": {"hours": 24 * 21}})


def test_synthetic_inputs_are_deterministic(full_cfg):
    a, b = generate_inputs(full_cfg), generate_inputs(full_cfg)
    pd.testing.assert_frame_equal(a, b)
    c = generate_inputs({**full_cfg, "seed": full_cfg["seed"] + 1})
    assert not np.allclose(a["load_CAL"], c["load_CAL"])


def test_synthetic_inputs_ranges(full_cfg):
    x = generate_inputs(full_cfg)
    for col in ("cf_wind_south", "cf_wind_central", "cf_solar_south", "pu_hydro"):
        assert x[col].between(0, 1).all()
    assert (x.filter(like="load_") > 0).all().all()
    # solar is zero at night
    assert (x.loc[x["hour"].isin([0, 1, 2, 23]), "cf_solar_south"] == 0).all()


def test_copper_plate_and_physics_on_real_config(full_cfg):
    x = generate_inputs(full_cfg)
    copper = solve_dispatch(full_cfg, x, line_limits=False)
    assert checks.check_copper_plate(copper)["passed"]
    res = solve_dispatch(full_cfg, x)
    assert checks.check_physics(full_cfg, x, res)["passed"]


def test_config_validation_catches_errors(full_cfg):
    bad = {**full_cfg, "lines": {**full_cfg["lines"], "X": {"bus0": "NOPE", "bus1": "CAL", "x": 1, "s_nom": 1}}}
    with pytest.raises(ValueError, match="unknown bus0"):
        validate(bad)

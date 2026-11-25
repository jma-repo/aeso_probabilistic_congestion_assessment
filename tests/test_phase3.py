"""Phase 3 tests: baseline exactness, leakage-free splits, model wrappers."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.features import build_features, hourly_split, merit_order_price, scenario_split, targets
from src.models import LGBMCongestion, MeritOrderBaseline, MLPSurrogate, PriceSurrogate
from src.montecarlo import sample_scenarios
from src.simulate import solve_many
from src.synth import generate_inputs


@pytest.fixture(scope="module")
def data():
    cfg = load_config(overrides={"time": {"hours": 24 * 20}})
    base = generate_inputs(cfg)
    scen = sample_scenarios(cfg, base, 600, seed=11)
    return cfg, base, scen


def test_merit_order_price_equals_copper_plate_lp_with_outages(data):
    cfg, _, scen = data
    copper = solve_many(cfg, scen.inputs, None, scen.gen_avail, line_limits=False)
    p = merit_order_price(cfg, scen.inputs, scen.gen_avail)
    np.testing.assert_allclose(p.to_numpy(), copper.lmp["CAL"].to_numpy(), atol=1e-6)


def test_hourly_split_is_ordered_disjoint_with_embargo():
    idx = pd.date_range("2025-01-01", periods=17520, freq="h")
    s = hourly_split(idx, embargo_days=7)
    assert not (s["train"] & s["val"]).any() and not (s["val"] & s["test"]).any()
    assert idx[s["train"]].max() < idx[s["val"]].min() - pd.Timedelta(days=6)
    assert idx[s["val"]].max() < idx[s["test"]].min() - pd.Timedelta(days=6)


def test_scenario_split_is_contiguous():
    s = scenario_split(100, 0.2)
    assert s["train"][:80].all() and s["val"][80:].all()


def test_features_have_no_target_leakage(data):
    cfg, _, scen = data
    X = build_features(cfg, scen.inputs, scen.line_out, scen.gen_avail)
    assert not any(c.startswith(("lmp", "flow", "binding")) for c in X.columns)
    assert X.notna().all().all()


def test_hybrid_wrapper_and_mlp_handle_undefined_lmps(data):
    cfg, _, scen = data
    scen.line_out.iloc[:40] = False
    scen.line_out.iloc[:40, list(cfg["lines"]).index("BC-CAL")] = True  # BC islanded -> NaN LMP
    res = solve_many(cfg, scen.inputs, scen.out_lines, scen.gen_avail)
    X = build_features(cfg, scen.inputs, scen.line_out, scen.gen_avail)
    Y, B = targets(res)
    assert Y["lmp_BC"].isna().sum() >= 40
    m = PriceSurrogate(MLPSurrogate(hidden=(32,), max_epochs=5, patience=5, threads=1), hybrid=True)
    m.fit(X.iloc[:500], Y.iloc[:500], X.iloc[500:], Y.iloc[500:])
    P = m.predict(X)
    assert P.notna().all().all()
    # with a zero residual model the hybrid wrapper returns exactly the merit price
    base = MeritOrderBaseline(list(cfg["buses"])).predict(X)
    np.testing.assert_allclose(base["lmp_CAL"], X["merit_price"])


def test_classifier_falls_back_to_base_rate_for_never_binding_lines(data):
    cfg, _, scen = data
    res = solve_many(cfg, scen.inputs, scen.out_lines, scen.gen_avail)
    X = build_features(cfg, scen.inputs, scen.line_out, scen.gen_avail)
    _, B = targets(res)
    clf = LGBMCongestion({"num_leaves": 7, "seed": 0}, num_boost_round=20).fit(X.iloc[:500], B.iloc[:500], X.iloc[500:], B.iloc[500:])
    p = clf.predict_proba(X)
    assert ((p >= 0) & (p <= 1)).all().all()
    assert isinstance(clf.boosters["CAL-RD"], float)

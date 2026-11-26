"""Feature matrices, targets and splits for the ML surrogates.

Inputs (what an operator would know before solving the OPF):
  * load per bus and total load, temperature (system reference, Calgary)
  * wind / solar / hydro availability, as capacity factors and as MW
  * physically motivated aggregates: net load, South surplus, North surplus
  * calendar: hour, day of week, month
  * outage indicators: one 0/1 column per line, available fraction per generator
Targets: LMP at every bus; binding (at-limit) flag per line.

Splits are never shuffled across time:
  * hourly data: train = 2025, validation = 2026 H1, test = 2026 H2, with a
    7-day embargo at each boundary (weather is autocorrelated);
  * Monte Carlo data: a separate training set (different seed), validation =
    its last 15 % of scenarios, test = the Phase 2 study scenarios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .grid import load_buses
from .synth import PROFILE_COLUMNS

CALENDAR = ["hour", "dow", "month"]


def _available_mw(cfg: dict, inputs: pd.DataFrame, gen_avail: pd.DataFrame | None) -> pd.DataFrame:
    cols = {}
    for g, spec in cfg["generators"].items():
        pu = 1.0 if spec["profile"] == "none" else inputs[PROFILE_COLUMNS[spec["profile"]]].clip(0, 1)
        mw = spec["p_nom"] * pu * pd.Series(1.0, index=inputs.index)
        if gen_avail is not None and g in gen_avail:
            mw = mw * gen_avail[g]
        cols[g] = mw
    return pd.DataFrame(cols, index=inputs.index)


def build_features(cfg: dict, inputs: pd.DataFrame, line_out: pd.DataFrame | None = None,
                   gen_avail: pd.DataFrame | None = None) -> pd.DataFrame:
    buses = load_buses(cfg)
    X = pd.DataFrame(index=inputs.index)
    for b in buses:
        X[f"load_{b}"] = inputs[f"load_{b}"]
    X["total_load"] = X[[f"load_{b}" for b in buses]].sum(axis=1)
    X["temp_system"] = inputs["temp_CAL"]
    for c in ("cf_wind_south", "cf_wind_central", "cf_solar_south", "pu_hydro"):
        X[c] = inputs[c]
    avail = _available_mw(cfg, inputs, gen_avail)
    carrier = {g: s["carrier"] for g, s in cfg["generators"].items()}
    renew = [g for g in avail if carrier[g] in ("wind", "solar")]
    X["renewable_mw"] = avail[renew].sum(axis=1)
    X["net_load"] = X["total_load"] - X["renewable_mw"]
    X["thermal_avail_mw"] = avail[[g for g in avail if carrier[g] not in ("wind", "solar")]].sum(axis=1)
    X["reserve_margin_mw"] = X["thermal_avail_mw"] - X["net_load"]
    for bus in ("SOUTH", "NORTH"):
        gens = [g for g, s in cfg["generators"].items() if s["bus"] == bus]
        X[f"surplus_{bus}"] = avail[gens].sum(axis=1) - X[f"load_{bus}"]
    # copper-plate (merit-order) system price: exact LMP whenever no line binds
    X["merit_price"] = merit_order_price(cfg, inputs, gen_avail)
    for c in CALENDAR:
        X[c] = inputs[c].astype(int)
    for l in cfg["lines"]:
        X[f"out_{l}"] = 0 if line_out is None else line_out[l].astype(int)
    for g, spec in cfg["generators"].items():
        if spec["carrier"] in cfg["montecarlo"]["generator_outage"]:
            X[f"avail_{g}"] = 1.0 if gen_avail is None else gen_avail[g]
    return X.astype(float)


def targets(res) -> tuple[pd.DataFrame, pd.DataFrame]:
    Y = res.lmp.add_prefix("lmp_")
    B = res.binding.astype(int)
    return Y, B


def merit_order_price(cfg: dict, inputs: pd.DataFrame, gen_avail: pd.DataFrame | None = None) -> pd.Series:
    """Naive baseline: system price with NO network (copper plate), computed
    analytically from the merit order: the cost of the cheapest unit that is
    still needed after stacking all cheaper available capacity.

    Optional must-run (``p_min_pu`` per generator) is dispatched first; the
    rest of the load is served from the flexible remainder in cost order.
    Equals the copper-plate DC-OPF price (tested)."""
    avail = _available_mw(cfg, inputs, gen_avail)
    order = sorted(cfg["generators"], key=lambda g: cfg["generators"][g]["marginal_cost"])
    cost = np.array([cfg["generators"][g]["marginal_cost"] for g in order])
    pnom = np.array([cfg["generators"][g]["p_nom"] for g in order])
    pmin = np.array([cfg["generators"][g].get("p_min_pu", 0.0) for g in order])
    av = avail[order].to_numpy()
    must = np.minimum(pmin * pnom, av)
    cum = (av - must).cumsum(axis=1)
    load = inputs[[f"load_{b}" for b in load_buses(cfg)]].sum(axis=1).to_numpy()
    residual = (load - must.sum(axis=1))[:, None]
    covered = cum >= residual - 1e-6
    idx = covered.argmax(axis=1)
    price = cost[idx]
    price[~covered.any(axis=1)] = cfg["voll"]
    return pd.Series(price, index=inputs.index, name="merit_order_price")


def hourly_split(index: pd.DatetimeIndex, embargo_days: int = 7) -> dict[str, np.ndarray]:
    """Boolean masks: train = first year, val = next 6 months, test = rest."""
    t0 = index[0]
    y1 = t0 + pd.DateOffset(years=1)
    h = y1 + pd.DateOffset(months=6)
    e = pd.Timedelta(days=embargo_days)
    return {
        "train": np.asarray(index < y1 - e),
        "val": np.asarray((index >= y1) & (index < h - e)),
        "test": np.asarray(index >= h),
    }


def scenario_split(n: int, val_share: float = 0.15) -> dict[str, np.ndarray]:
    cut = int(n * (1 - val_share))
    m = np.zeros(n, bool)
    m[:cut] = True
    return {"train": m, "val": ~m}

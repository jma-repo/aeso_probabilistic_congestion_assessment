"""Sanity checks for the Phase 1 simulator.

1. Copper plate: with line limits removed, every bus must have the same price.
2. Independent re-solve: an angle-based DC-OPF written directly in scipy/HiGHS
   (no PyPSA) must reproduce PyPSA's LMPs and flows on randomly sampled hours.
   PyPSA formulates KVL with cycle constraints, this one with bus angles, so
   agreement checks both the formulation and our result extraction.
3. Physics: power balance per hour, Kirchhoff current law per bus, flows within
   limits, binding lines have non-zero price separation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

from .grid import load_buses, susceptance_matrices
from .simulate import DispatchResult, total_load
from .synth import PROFILE_COLUMNS


def generator_pmax(cfg: dict, inputs: pd.DataFrame) -> pd.DataFrame:
    """Available MW per generator per hour (incl. shed generators)."""
    cols = {}
    for name, g in cfg["generators"].items():
        pu = 1.0 if g["profile"] == "none" else inputs[PROFILE_COLUMNS[g["profile"]]].clip(0, 1)
        cols[name] = g["p_nom"] * pu * pd.Series(1.0, index=inputs.index)
    for bus in load_buses(cfg):
        cols[f"shed_{bus}"] = pd.Series(2.0 * inputs[f"load_{bus}"].max(), index=inputs.index)
    return pd.DataFrame(cols)


def angle_dcopf_hour(cfg: dict, loads: dict[str, float], pmax: dict[str, float], line_limits: bool = True,
                     out_lines: tuple[str, ...] = ()):
    """Solve one hour of DC-OPF with bus voltage angles as variables.

    min  sum_g c_g p_g
    s.t. A_g p - B_bus theta = d          (nodal balance, dual = LMP)
         theta_slack = 0
        -F <= B_f theta <= F              (line limits)
         0 <= p <= pmax
    """
    buses, lines, A, b = susceptance_matrices(cfg, out_lines)
    gens = list(pmax)
    G, N, L = len(gens), len(buses), len(lines)
    cost = []
    Ag = np.zeros((N, G))
    for j, g in enumerate(gens):
        if g.startswith("shed_"):
            bus, c = g[len("shed_") :], cfg["voll"]
        else:
            bus, c = cfg["generators"][g]["bus"], cfg["generators"][g]["marginal_cost"]
        Ag[buses.index(bus), j] = 1.0
        cost.append(c)
    Bf = np.diag(b) @ A           # lines x buses: flow = Bf @ theta
    Bbus = A.T @ Bf               # buses x buses
    d = np.array([loads.get(bus, 0.0) for bus in buses])

    c = np.r_[np.asarray(cost), np.zeros(N)]
    A_eq = np.zeros((N + 1, G + N))
    A_eq[:N, :G] = Ag
    A_eq[:N, G:] = -Bbus
    A_eq[N, G] = 1.0  # theta of first bus = 0 (slack)
    b_eq = np.r_[d, 0.0]
    bounds = [(0.0, pmax[g]) for g in gens] + [(None, None)] * N
    if line_limits:
        F = np.array([cfg["lines"][l]["s_nom"] for l in lines])
        A_ub = np.zeros((2 * L, G + N))
        A_ub[:L, G:] = Bf
        A_ub[L:, G:] = -Bf
        b_ub = np.r_[F, F]
        res = linprog(c, A_ub=csr_matrix(A_ub), b_ub=b_ub, A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds, method="highs")
    else:
        res = linprog(c, A_eq=csr_matrix(A_eq), b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(res.message)
    theta = res.x[G:]
    return (
        pd.Series(res.eqlin.marginals[:N], index=buses),
        pd.Series(Bf @ theta, index=lines),
        pd.Series(res.x[:G], index=gens),
    )


def check_copper_plate(copper: DispatchResult) -> dict:
    spread = copper.lmp.max(axis=1) - copper.lmp.min(axis=1)
    return {"max_price_spread_without_limits": float(spread.max()), "passed": bool(spread.max() < 1e-6)}


def check_independent_resolve(cfg: dict, inputs: pd.DataFrame, res: DispatchResult, n_hours: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # oversample congested hours: those are where the formulations could disagree
    congested = np.flatnonzero(res.binding.any(axis=1).to_numpy())
    free = np.flatnonzero(~res.binding.any(axis=1).to_numpy())
    k = min(n_hours // 2, len(congested))
    idx = np.r_[rng.choice(congested, k, replace=False), rng.choice(free, n_hours - k, replace=False)]
    pmax = generator_pmax(cfg, inputs)
    lmp_err, flow_err, mismatched = [], [], []
    for i in idx:
        ts = inputs.index[i]
        loads = {bus: inputs.at[ts, f"load_{bus}"] for bus in load_buses(cfg)}
        lmp, flow, _ = angle_dcopf_hour(cfg, loads, pmax.loc[ts].to_dict())
        e_l = float((lmp - res.lmp.loc[ts, lmp.index]).abs().max())
        e_f = float((flow - res.flow.loc[ts, flow.index]).abs().max())
        lmp_err.append(e_l)
        flow_err.append(e_f)
        if e_l > 1e-3:
            mismatched.append(str(ts))
    return {
        "hours_checked": int(len(idx)),
        "congested_hours_checked": int(k),
        "max_abs_lmp_diff": float(max(lmp_err)),
        "max_abs_flow_diff_mw": float(max(flow_err)),
        "hours_lmp_mismatch_gt_1e-3": mismatched[:10],
        "share_hours_lmp_match": float(1 - len(mismatched) / len(idx)),
        "passed": bool(max(flow_err) < 1e-2 and len(mismatched) == 0),
    }


def check_physics(cfg: dict, inputs: pd.DataFrame, res: DispatchResult) -> dict:
    gen_total = res.dispatch.sum(axis=1)
    balance = (gen_total - total_load(cfg, inputs)).abs().max()

    # KCL: injection at each bus = sum of flows leaving it
    buses, lines, A, _ = susceptance_matrices(cfg)
    inj = pd.DataFrame(0.0, index=res.lmp.index, columns=buses)
    for g in res.dispatch:
        bus = g[len("shed_") :] if g.startswith("shed_") else cfg["generators"][g]["bus"]
        inj[bus] += res.dispatch[g]
    for bus in load_buses(cfg):
        inj[bus] -= inputs[f"load_{bus}"]
    kcl = (inj.to_numpy() - res.flow[lines].to_numpy() @ A).__abs__().max()

    s_nom = pd.Series({l: cfg["lines"][l]["s_nom"] for l in res.flow})
    overload = float((res.flow.abs() - s_nom).max().max())

    # On a binding line the two end prices must differ (congestion rent > 0)
    sep = []
    for l in res.flow:
        b0, b1 = cfg["lines"][l]["bus0"], cfg["lines"][l]["bus1"]
        mask = res.binding[l]
        if mask.any():
            sep.append(float((res.lmp.loc[mask, b0] - res.lmp.loc[mask, b1]).abs().min()))
    return {
        "max_power_balance_error_mw": float(balance),
        "max_kcl_error_mw": float(kcl),
        "max_overload_mw": overload,
        "min_price_separation_on_binding_lines": float(min(sep)) if sep else None,
        "passed": bool(balance < 1e-3 and kcl < 1e-3 and overload < 1e-3),
    }


def check_congestion_target(cfg: dict, res: DispatchResult) -> dict:
    share = float(res.binding.any(axis=1).mean())
    t = cfg["congestion_target"]
    return {"share_hours_any_binding": share, "target": [t["min_share"], t["max_share"]],
            "passed": bool(t["min_share"] <= share <= t["max_share"])}


def check_mc_resolve(cfg: dict, scen, res: DispatchResult, n: int = 100, seed: int = 0) -> dict:
    """Re-solve MC scenarios (half with a line outage) with the angle-based LP,
    including generator derates, and compare LMPs (where defined) and flows."""
    rng = np.random.default_rng(seed)
    with_out = np.flatnonzero(scen.line_out.any(axis=1).to_numpy())
    without = np.flatnonzero(~scen.line_out.any(axis=1).to_numpy())
    k = min(n // 2, len(with_out))
    idx = np.r_[rng.choice(with_out, k, replace=False), rng.choice(without, n - k, replace=False)]
    pmax = generator_pmax(cfg, scen.inputs)
    pmax[scen.gen_avail.columns] = pmax[scen.gen_avail.columns] * scen.gen_avail
    outs = scen.out_lines
    lmp_err, flow_err = [], []
    for i in idx:
        sid = scen.inputs.index[i]
        loads = {bus: scen.inputs.at[sid, f"load_{bus}"] for bus in load_buses(cfg)}
        lmp, flow, _ = angle_dcopf_hour(cfg, loads, pmax.loc[sid].to_dict(), out_lines=outs.loc[sid])
        ref = res.lmp.loc[sid, lmp.index]
        ok = ref.notna()
        lmp_err.append(float((lmp[ok] - ref[ok]).abs().max()))
        flow_err.append(float((flow - res.flow.loc[sid, flow.index]).abs().max()))
    return {"scenarios_checked": int(len(idx)), "with_line_outage": int(k),
            "max_abs_lmp_diff": max(lmp_err), "max_abs_flow_diff_mw": max(flow_err),
            "passed": bool(max(lmp_err) < 1e-3 and max(flow_err) < 1e-2)}


def check_switching_paradox(cfg: dict, inputs: pd.DataFrame, line: str, hour) -> dict:
    """Independently confirm that removing ``line`` lowers cost in ``hour``."""
    pmax = generator_pmax(cfg, inputs).loc[hour].to_dict()
    loads = {bus: inputs.at[hour, f"load_{bus}"] for bus in load_buses(cfg)}
    costs = {}
    for label, out in (("intact", ()), ("without_line", (line,))):
        _, _, p = angle_dcopf_hour(cfg, loads, pmax, out_lines=out)
        c = sum(v * (cfg["voll"] if g.startswith("shed_") else cfg["generators"][g]["marginal_cost"]) for g, v in p.items())
        costs[label] = float(c)
    return {"line": line, "hour": str(hour), **costs, "saving_per_hour": costs["intact"] - costs["without_line"],
            "passed": bool(costs["without_line"] <= costs["intact"] + 1e-6)}

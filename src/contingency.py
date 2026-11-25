"""N-1 contingency analysis: one line out at a time, over every hour.

Two complementary views per contingency:

A. Post-contingency SECURITY screen (instantaneous, before any re-dispatch).
   Keep the pre-contingency injections from the base-case OPF and compute the
   new flows on the reduced network with its PTDF matrix. Any line above its
   rating is an overload the operator would have to relieve. This mirrors a
   classic N-1 screen of the market dispatch (the base case here is NOT
   security-constrained, so overloads are expected).

B. ECONOMIC impact (the line is out for the whole hour and the market
   re-dispatches around it). Re-solve the DC-OPF without the line and compare
   LMPs, production cost, congestion and unserved energy with the base case.

If the outage islands part of the network (the BC intertie here), view A is
not a flow problem: the islanded injection is simply lost. We report the lost
MW instead of flows.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .grid import islanded_buses, load_buses, ptdf
from .montecarlo import congestion_rent
from .simulate import DispatchResult, production_cost, solve_many


def bus_injections(cfg: dict, inputs: pd.DataFrame, res: DispatchResult) -> pd.DataFrame:
    """Net injection (generation - load) per bus per hour, MW."""
    buses = list(cfg["buses"])
    inj = pd.DataFrame(0.0, index=res.dispatch.index, columns=buses)
    for g in res.dispatch:
        bus = g[len("shed_"):] if g.startswith("shed_") else cfg["generators"][g]["bus"]
        inj[bus] += res.dispatch[g]
    for bus in load_buses(cfg):
        inj[bus] -= inputs.loc[res.dispatch.index, f"load_{bus}"]
    return inj


def security_screen(cfg: dict, inputs: pd.DataFrame, base: DispatchResult) -> tuple[pd.DataFrame, dict]:
    """View A for every single-line contingency."""
    inj = bus_injections(cfg, inputs, base)
    # self-check: base-case PTDF flows must reproduce the OPF flows
    H0 = ptdf(cfg)
    f0 = inj[H0.columns].to_numpy() @ H0.to_numpy().T
    base_err = float(np.abs(f0 - base.flow[H0.index].to_numpy()).max())

    rows, post_loading = [], {}
    for l in cfg["lines"]:
        isl = islanded_buses(cfg, (l,))
        row = {"contingency": l, "islands": ",".join(sorted(isl)) or ""}
        if isl:
            lost = inj[list(isl)].sum(axis=1)
            row.update({
                "share_hours_overload": np.nan, "max_post_loading": np.nan, "worst_line": "",
                "overload_mwh_per_year": np.nan,
                "islanded_injection_mean_mw": float(lost.mean()), "islanded_injection_max_mw": float(lost.max()),
            })
            rows.append(row)
            continue
        H = ptdf(cfg, out_lines=(l,))
        f = inj[H.columns].to_numpy() @ H.to_numpy().T
        s_nom = np.array([cfg["lines"][m]["s_nom"] for m in H.index])
        load_pct = np.abs(f) / s_nom
        excess = np.clip(np.abs(f) - s_nom, 0, None)
        over = load_pct > 1 + 1e-6
        worst_idx = load_pct.max(axis=0).argmax()
        post_loading[l] = pd.DataFrame(load_pct, index=inj.index, columns=H.index)
        years = len(inj) / 8760
        row.update({
            "share_hours_overload": float(over.any(axis=1).mean()),
            "max_post_loading": float(load_pct.max()),
            "worst_line": H.index[worst_idx],
            "overload_mwh_per_year": float(excess.sum() / years),
            "islanded_injection_mean_mw": 0.0, "islanded_injection_max_mw": 0.0,
        })
        rows.append(row)
    return pd.DataFrame(rows).set_index("contingency"), {"base_ptdf_flow_error_mw": base_err, "post_loading": post_loading}


def economic_impact(cfg: dict, inputs: pd.DataFrame, base: DispatchResult) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """View B: re-dispatch without each line; compare to the intact base case."""
    lines = list(cfg["lines"])
    H = len(inputs)
    big = pd.concat([inputs] * len(lines), ignore_index=True)
    big.index.name = "row"
    out = pd.Series([(l,) for l in lines for _ in range(H)], index=big.index)
    t0 = time.perf_counter()
    res = solve_many(cfg, big, out, workers=cfg["contingency"]["workers"], chunk=cfg["time"]["solve_chunk_hours"])
    wall = time.perf_counter() - t0

    years = H / 8760
    base_cost = production_cost(cfg, base.dispatch).to_numpy()
    base_rent = congestion_rent(cfg, base).to_numpy()
    alberta = [b for b in cfg["buses"] if cfg["buses"][b]["load_share"] > 0]
    base_lmp = base.lmp[alberta].to_numpy()
    rows, dlmp_rows, cheaper_hours = [], {}, {}
    for i, l in enumerate(lines):
        sl = slice(i * H, (i + 1) * H)
        lmp = res.lmp.iloc[sl][alberta].to_numpy()
        d = lmp - base_lmp
        cost = production_cost(cfg, res.dispatch.iloc[sl]).to_numpy()
        shed = res.shed.iloc[sl].sum(axis=1).to_numpy()
        binding = res.binding.iloc[sl].drop(columns=[l])
        r = DispatchResult(*(getattr(res, k).iloc[sl] for k in ("lmp", "flow", "loading", "binding", "dispatch")))
        rent = congestion_rent(cfg, r).to_numpy()
        dlmp_rows[l] = pd.Series(d.mean(axis=0), index=alberta)
        dc = cost - base_cost
        cheaper_hours[l] = {"share_hours_cheaper": float((dc < -1e-6).mean()),
                            "best_hour": base.dispatch.index[int(dc.argmin())], "best_saving_per_hour": float(-dc.min())}
        rows.append({
            "contingency": l,
            "delta_cost_musd_per_year": float((cost - base_cost).sum() / years / 1e6),
            "delta_congestion_rent_musd_per_year": float((rent - base_rent).sum() / years / 1e6),
            "mean_abs_delta_lmp": float(np.abs(d).mean()),
            "p95_abs_delta_lmp": float(np.percentile(np.abs(d).max(axis=1), 95)),
            "share_hours_lmp_changed_gt_1usd": float((np.abs(d).max(axis=1) > 1.0).mean()),
            "share_hours_any_binding_post": float(binding.any(axis=1).mean()),
            "unserved_mwh_per_year": float(shed.sum() / years),
            "hours_with_shed_per_year": float((shed > 1e-3).sum() / years),
        })
    meta = {"wall_seconds": wall, "n_lps": res.meta["n_lps"], "rows": len(big), "cheaper_hours": cheaper_hours}
    return pd.DataFrame(rows).set_index("contingency"), pd.DataFrame(dlmp_rows).T, meta


def rank(screen: pd.DataFrame, econ: pd.DataFrame) -> pd.DataFrame:
    """Ranked table: security first (share of hours with a post-contingency
    overload), economic impact (re-dispatch cost) as tie-break. Islanding
    contingencies have no flow screen; they sort last on security and are
    flagged in the ``islands`` column with the injection they disconnect."""
    t = screen.join(econ)
    t["rank_security"] = t["share_hours_overload"].rank(ascending=False, method="min", na_option="bottom")
    t["rank_economic"] = t["delta_cost_musd_per_year"].rank(ascending=False, method="min")
    return t.sort_values(["share_hours_overload", "delta_cost_musd_per_year"], ascending=False, na_position="last")

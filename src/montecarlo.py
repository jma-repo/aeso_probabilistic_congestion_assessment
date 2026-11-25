"""Monte Carlo scenario engine and probabilistic congestion statistics.

A scenario is one independent "possible hour":
  1. draw a base hour uniformly from the 2-year synthetic inputs
     (gives a consistent temperature / calendar / load / renewable state),
  2. perturb it: lognormal system + bus load multipliers, wind shifts in
     logit space, multiplicative solar noise,
  3. draw forced outages: each line out with probability q_line; each
     generator split into identical units, available units ~ Binomial(n, 1-FOR).

Every scenario is one DC-OPF. Scenarios that share a topology (same set of
lines out) are solved together in one LP; generator outages enter through
p_max_pu and do not change topology. Sampling is i.i.d., so binomial
(Wilson) confidence intervals apply to all probabilities.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .grid import load_buses
from .simulate import DispatchResult, production_cost, solve_many

SCENARIO_INDEX = "scenario"


@dataclass
class Scenarios:
    inputs: pd.DataFrame       # same columns as Phase 1 inputs + base_timestamp + multipliers
    line_out: pd.DataFrame     # bool, scenarios x lines
    gen_avail: pd.DataFrame    # 0..1, scenarios x generators

    @property
    def out_lines(self) -> pd.Series:
        cols = np.array(self.line_out.columns)
        return pd.Series([tuple(cols[r]) for r in self.line_out.to_numpy()], index=self.line_out.index)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sample_scenarios(cfg: dict, base: pd.DataFrame, n: int, seed: int) -> Scenarios:
    mc = cfg["montecarlo"]
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(base), n)
    x = base.iloc[pick].reset_index().rename(columns={"timestamp": "base_timestamp"})
    x.index = pd.RangeIndex(n, name=SCENARIO_INDEX)

    s_sys, s_bus = mc["load_sigma_system"], mc["load_sigma_bus"]
    sys_mult = np.exp(rng.normal(-s_sys**2 / 2, s_sys, n))
    x["load_mult_system"] = sys_mult
    for bus in load_buses(cfg):
        bus_mult = np.exp(rng.normal(-s_bus**2 / 2, s_bus, n))
        x[f"load_{bus}"] = x[f"load_{bus}"] * sys_mult * bus_mult
    for col in ("cf_wind_south", "cf_wind_central"):
        z = _logit(x[col].to_numpy()) + rng.normal(0, mc["wind_logit_sigma"], n)
        x[col] = 1 / (1 + np.exp(-z))
    x["cf_solar_south"] = np.clip(x["cf_solar_south"] * (1 + rng.normal(0, mc["solar_sigma"], n)), 0, 1)

    scale = mc.get("outage_rate_scale", 1.0)
    line_out = pd.DataFrame(
        {l: rng.random(n) < min(1.0, mc["line_outage_rate"][l] * scale) for l in cfg["lines"]}, index=x.index
    )
    avail = {}
    for g, spec in cfg["generators"].items():
        o = mc["generator_outage"].get(spec["carrier"])
        if o is None:
            avail[g] = np.ones(n)
            continue
        units = max(1, int(round(spec["p_nom"] / o["unit_mw"])))
        avail[g] = rng.binomial(units, 1 - min(1.0, o["for"] * scale), n) / units
    gen_avail = pd.DataFrame(avail, index=x.index)
    return Scenarios(x, line_out, gen_avail)


def run_monte_carlo(cfg: dict, base: pd.DataFrame, n: int | None = None, seed: int | None = None,
                    scen: Scenarios | None = None, **kw) -> tuple[Scenarios, DispatchResult]:
    mc = cfg["montecarlo"]
    n = n or mc["n_scenarios"]
    seed = cfg["seed"] + mc["seed_offset"] if seed is None else seed
    t0 = time.perf_counter()
    scen = scen or sample_scenarios(cfg, base, n, seed)
    t_sample = time.perf_counter() - t0
    res = solve_many(cfg, scen.inputs, scen.out_lines, scen.gen_avail,
                     workers=mc["workers"], chunk=mc["chunk"], **kw)
    res.meta["sample_seconds"] = t_sample
    res.meta["total_seconds"] = time.perf_counter() - t0
    return scen, res


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson(k, n, z: float = 1.96):
    """Wilson score interval for a binomial proportion (vectorised)."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return centre - half, centre + half


def congestion_rent(cfg: dict, res: DispatchResult) -> pd.Series:
    """Sum over lines of flow x (LMP_to - LMP_from), $/h (>= 0 in an OPF)."""
    lmp = res.lmp.fillna(0.0)
    rent = 0.0
    for l, ln in cfg["lines"].items():
        rent = rent + res.flow[l] * (lmp[ln["bus1"]] - lmp[ln["bus0"]])
    return rent


def line_congestion_table(cfg: dict, scen: Scenarios, res: DispatchResult,
                          deterministic_share: pd.Series | None = None) -> pd.DataFrame:
    n = len(res.binding)
    k = res.binding.sum()
    lo, hi = wilson(k, n)
    intact = ~scen.line_out.any(axis=1)
    out = pd.DataFrame({
        "p_congested": k / n,
        "ci95_low": lo,
        "ci95_high": hi,
        "expected_hours_per_year": k / n * 8760,
        "p_congested_all_lines_in": res.binding[intact].mean(),
        "p_congested_any_line_out": res.binding[~intact].mean(),
        "p_out_of_service": scen.line_out.mean(),
    })
    if deterministic_share is not None:
        out.insert(0, "deterministic_share", deterministic_share)
    return out.sort_values("p_congested", ascending=False)


def any_binding_stats(res: DispatchResult) -> dict:
    n = len(res.binding)
    internal = [l for l in res.binding if "BC" not in l]
    out = {}
    for name, cols in (("internal_lines", internal), ("all_lines_incl_intertie", list(res.binding))):
        k = int(res.binding[cols].any(axis=1).sum())
        lo, hi = wilson(k, n)
        out[name] = {"p": k / n, "ci95": [float(lo), float(hi)], "expected_hours_per_year": k / n * 8760}
    return out


def lmp_quantiles(res: DispatchResult) -> pd.DataFrame:
    q = res.lmp.quantile([0.10, 0.50, 0.90, 0.99]).T
    q.columns = ["P10", "P50", "P90", "P99"]
    q["mean"] = res.lmp.mean()
    q["p_lmp_ge_100"] = (res.lmp >= 100).mean()
    q["p_lmp_at_cap"] = (res.lmp >= 999.0).mean()
    q["share_undefined_islanded"] = res.lmp.isna().mean()
    return q


def adequacy_stats(cfg: dict, res: DispatchResult) -> dict:
    shed = res.shed.sum(axis=1)
    n = len(shed)
    k = int((shed > 1e-3).sum())
    lo, hi = wilson(k, n)
    eue = shed.mean() * 8760
    eue_se = shed.std(ddof=1) / np.sqrt(n) * 8760
    return {
        "lolp": k / n, "lolp_ci95": [float(lo), float(hi)],
        "lole_hours_per_year": k / n * 8760,
        "eue_mwh_per_year": float(eue), "eue_ci95": [float(eue - 1.96 * eue_se), float(eue + 1.96 * eue_se)],
    }


def economic_stats(cfg: dict, res: DispatchResult) -> dict:
    cost = production_cost(cfg, res.dispatch)
    rent = congestion_rent(cfg, res)
    return {
        "expected_production_cost_musd_per_year": float(cost.mean() * 8760 / 1e6),
        "expected_congestion_rent_musd_per_year": float(rent.mean() * 8760 / 1e6),
        "min_congestion_rent_per_hour": float(rent.min()),
    }


def convergence(res: DispatchResult, lines: list[str]) -> pd.DataFrame:
    """Running estimate and Wilson CI of P(binding) as scenarios accumulate."""
    n = np.arange(1, len(res.binding) + 1)
    frames = []
    for l in lines:
        k = res.binding[l].to_numpy().cumsum()
        lo, hi = wilson(k, n)
        frames.append(pd.DataFrame({"line": l, "n": n, "p": k / n, "lo": lo, "hi": hi}))
    return pd.concat(frames, ignore_index=True)


def rating_sensitivity(cfg: dict, scen: Scenarios, base_res: DispatchResult) -> pd.DataFrame:
    """Common-random-numbers sensitivity: same scenarios, scaled corridor rating.

    Using the identical scenario set for every rating removes sampling noise
    from the *difference* between ratings.
    """
    rs = cfg["montecarlo"]["rating_sensitivity"]
    m = min(rs["n_scenarios"], len(scen.inputs))
    sub = Scenarios(scen.inputs.iloc[:m], scen.line_out.iloc[:m], scen.gen_avail.iloc[:m])
    rows = []
    for corridor, lines in rs["corridors"].items():
        for s in rs["scales"]:
            if s == 1.0:
                r = DispatchResult(*(getattr(base_res, k).iloc[:m] for k in ("lmp", "flow", "loading", "binding", "dispatch")))
            else:
                r = solve_many(cfg, sub.inputs, sub.out_lines, sub.gen_avail, workers=cfg["montecarlo"]["workers"],
                               chunk=cfg["montecarlo"]["chunk"], line_rating_scale={l: s for l in lines})
            k = int(r.binding[lines].any(axis=1).sum())
            lo, hi = wilson(k, m)
            rows.append({
                "corridor": corridor, "rating_scale": s,
                "rating_mw": sum(cfg["lines"][l]["s_nom"] for l in lines) * s,
                "p_corridor_congested": k / m, "ci95_low": float(lo), "ci95_high": float(hi),
                "p_any_internal_congested": float(r.binding[[l for l in r.binding if "BC" not in l]].any(axis=1).mean()),
                "expected_congestion_rent_musd_per_year": float(congestion_rent(cfg, r).mean() * 8760 / 1e6),
                "expected_production_cost_musd_per_year": float(production_cost(cfg, r.dispatch).mean() * 8760 / 1e6),
            })
    return pd.DataFrame(rows)

"""Hourly DC-OPF dispatch and LMP extraction.

Without storage, unit commitment or ramping constraints, each hour's OPF is
independent of every other hour. We therefore put a block of hours (e.g. one
month) into one LP: the solution is identical to solving hour by hour, but
model-building overhead is paid once per block instead of once per hour.

LMP = dual of each bus's nodal power-balance constraint ($/MWh).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from .grid import build_network, islanded_buses, load_buses


@dataclass
class DispatchResult:
    lmp: pd.DataFrame          # snapshots x buses, $/MWh
    flow: pd.DataFrame         # snapshots x lines, MW (positive = bus0 -> bus1)
    loading: pd.DataFrame      # |flow| / s_nom
    binding: pd.DataFrame      # bool, line at its limit
    dispatch: pd.DataFrame     # snapshots x generators (incl. shed_*), MW
    solve_seconds: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def shed(self) -> pd.DataFrame:
        return self.dispatch[[c for c in self.dispatch if c.startswith("shed_")]]

    def frames(self) -> dict[str, pd.DataFrame]:
        return {k: getattr(self, k) for k in ("lmp", "flow", "loading", "binding", "dispatch")}

    @staticmethod
    def concat(parts: list["DispatchResult"]) -> "DispatchResult":
        return DispatchResult(
            *(pd.concat([getattr(p, k) for p in parts]) for k in ("lmp", "flow", "loading", "binding", "dispatch")),
            solve_seconds=sum(p.solve_seconds for p in parts),
        )


def solve_block(
    cfg: dict,
    inputs: pd.DataFrame,
    *,
    line_limits: bool = True,
    out_lines: tuple[str, ...] = (),
    gen_availability: pd.DataFrame | None = None,
    line_rating_scale: float = 1.0,
) -> DispatchResult:
    """Solve one LP over all snapshots in ``inputs``."""
    t0 = time.perf_counter()
    n = build_network(
        cfg,
        inputs,
        line_limits=line_limits,
        out_lines=out_lines,
        gen_availability=gen_availability,
        line_rating_scale=line_rating_scale,
    )
    status, condition = n.optimize(
        solver_name=cfg["solver"]["name"],
        solver_options=cfg["solver"].get("options", {}),
        include_objective_constant=False,
        progress=False,
    )
    if status != "ok":
        raise RuntimeError(f"OPF failed: {status} / {condition}")

    lmp = n.buses_t.marginal_price[list(n.buses.index)].copy()
    flow = n.lines_t.p0[list(n.lines.index)].copy()
    s_nom = n.lines.s_nom
    loading = flow.abs().div(s_nom, axis=1)
    tol = cfg.get("binding_tolerance", 1e-4)
    binding = loading >= (1.0 - tol) if line_limits else loading * False
    dispatch = n.generators_t.p[list(n.generators.index)].copy()
    for df in (lmp, flow, loading, binding, dispatch):
        df.columns.name = None
        df.index.name = inputs.index.name or "timestamp"
    return DispatchResult(lmp, flow, loading, binding.astype(bool), dispatch, time.perf_counter() - t0)


def solve_dispatch(cfg: dict, inputs: pd.DataFrame, *, chunk: int | None = None, verbose: bool = False, **kw) -> DispatchResult:
    """Solve all hours in ``inputs`` in blocks of ``chunk`` snapshots."""
    chunk = chunk or cfg["time"]["solve_chunk_hours"]
    parts = []
    for start in range(0, len(inputs), chunk):
        block = inputs.iloc[start : start + chunk]
        parts.append(solve_block(cfg, block, **kw))
        if verbose:
            print(f"  solved hours {start:>6}-{start + len(block) - 1:<6} in {parts[-1].solve_seconds:5.2f}s")
    res = DispatchResult.concat(parts)
    res.meta["n_hours"] = len(inputs)
    return res


def _solve_task(args):
    cfg, block, out_lines, avail, kw = args
    return out_lines, solve_block(cfg, block, out_lines=out_lines, gen_availability=avail, **kw)


def solve_many(
    cfg: dict,
    inputs: pd.DataFrame,
    out_lines: pd.Series | None = None,
    gen_availability: pd.DataFrame | None = None,
    *,
    workers: int = 1,
    chunk: int | None = None,
    **kw,
) -> DispatchResult:
    """Solve many independent rows (hours or MC scenarios), possibly with outages.

    out_lines: per-row tuple of lines that are out of service. Rows sharing a
        topology are solved together in one LP (as snapshots); different
        topologies need different networks, because an outaged line must be
        removed, not set to zero rating.
    gen_availability: per-row generator availability multipliers (0..1).
    Results are re-aligned to the full line set (an out-of-service line has
    flow 0 and is never binding) and to the original row order. The LMP of an
    islanded bus with no load is undefined and set to NaN.
    """
    chunk = chunk or cfg["time"]["solve_chunk_hours"]
    if out_lines is None:
        out_lines = pd.Series([()] * len(inputs), index=inputs.index)
    out_lines = out_lines.map(lambda t: tuple(sorted(t)))
    tasks = []
    for topo, idx in out_lines.groupby(out_lines, sort=False).groups.items():
        for start in range(0, len(idx), chunk):
            rows = idx[start : start + chunk]
            avail = gen_availability.loc[rows] if gen_availability is not None else None
            tasks.append((cfg, inputs.loc[rows], topo, avail, kw))
    t0 = time.perf_counter()
    if workers > 1 and len(tasks) > 1:
        # "spawn", not fork: forking a process that has already run HiGHS can
        # deadlock the children (inherited solver thread state).
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
            outs = list(ex.map(_solve_task, tasks))
    else:
        outs = [_solve_task(t) for t in tasks]
    wall = time.perf_counter() - t0

    all_lines = list(cfg["lines"])
    parts = []
    for topo, r in outs:
        r.flow = r.flow.reindex(columns=all_lines, fill_value=0.0)
        r.loading = r.loading.reindex(columns=all_lines, fill_value=0.0)
        r.binding = r.binding.reindex(columns=all_lines, fill_value=False).astype(bool)
        for bus in islanded_buses(cfg, topo):
            if cfg["buses"][bus]["load_share"] == 0:
                r.lmp[bus] = np.nan
        parts.append(r)
    res = DispatchResult.concat(parts)
    for k in ("lmp", "flow", "loading", "binding", "dispatch"):
        setattr(res, k, getattr(res, k).loc[inputs.index])
    res.meta.update({"n_lps": len(tasks), "n_topologies": int(out_lines.nunique()), "wall_seconds": wall,
                     "sum_lp_seconds": res.solve_seconds, "rows": len(inputs)})
    return res


def production_cost(cfg: dict, dispatch: pd.DataFrame) -> pd.Series:
    """$/h per row, including unserved energy at VOLL."""
    c = pd.Series({g: (cfg["voll"] if g.startswith("shed_") else cfg["generators"][g]["marginal_cost"]) for g in dispatch})
    return dispatch.mul(c, axis=1).sum(axis=1)


def total_load(cfg: dict, inputs: pd.DataFrame) -> pd.Series:
    return inputs[[f"load_{b}" for b in load_buses(cfg)]].sum(axis=1)


def summarize(cfg: dict, res: DispatchResult, inputs: pd.DataFrame) -> dict:
    """Headline numbers for one dispatch run."""
    any_binding = res.binding.any(axis=1)
    spread = res.lmp.drop(columns=[c for c in ["BC"] if c in res.lmp]).pipe(lambda d: d.max(axis=1) - d.min(axis=1))
    shed = res.shed.sum(axis=1)
    gen = res.dispatch.drop(columns=res.shed.columns)
    by_carrier = gen.T.groupby(lambda g: cfg["generators"][g]["carrier"]).sum().T
    return {
        "hours": int(len(res.lmp)),
        "share_hours_any_binding": float(any_binding.mean()),
        "share_hours_binding_by_line": {k: float(v) for k, v in res.binding.mean().items()},
        "mean_lmp_by_bus": {k: float(v) for k, v in res.lmp.mean().items()},
        "mean_abs_spread_internal_buses": float(spread.mean()),
        "p95_spread_internal_buses": float(np.percentile(spread, 95)),
        "hours_with_load_shed": int((shed > 1e-3).sum()),
        "energy_share_by_carrier": {k: float(v) for k, v in (by_carrier.sum() / by_carrier.sum().sum()).items()},
        "mean_total_load_mw": float(total_load(cfg, inputs).mean()),
        "peak_total_load_mw": float(total_load(cfg, inputs).max()),
        "solve_seconds": float(res.solve_seconds),
    }

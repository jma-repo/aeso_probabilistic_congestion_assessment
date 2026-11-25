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

from .grid import build_network, load_buses


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
        df.index.name = "timestamp"
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

"""Build the toy Alberta-like PyPSA network for a block of snapshots.

Modelling choices (see README "Assumptions"):
* Linear DC power flow: lossless, flat voltage magnitude, no reactive power.
* Every bus with load gets an unserved-energy generator priced at VOLL
  (= the $999.99/MWh price cap) so any outage combination is feasible.
* A line outage REMOVES the line from the network. Setting its limit to zero
  would be wrong in DC-OPF: the KVL constraint would remain and force the
  angle difference across the open line to zero.
* Generator outages are applied through ``p_max_pu`` (per snapshot), which is
  exact because a generator has no network-coupling constraint.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pypsa

from .synth import PROFILE_COLUMNS

logging.getLogger("pypsa").setLevel(logging.ERROR)
logging.getLogger("linopy").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"pypsa|linopy")
try:  # PyPSA >= 1.0 string-dtype notice
    pypsa.options.api.legacy_string_dtype = True
except Exception:  # pragma: no cover
    pass

UNLIMITED_MW = 1.0e5  # "infinite" line rating for the copper-plate run
V_NOM_KV = 240.0


def load_buses(cfg: dict) -> list[str]:
    return [b for b, v in cfg["buses"].items() if v["load_share"] > 0]


def build_network(
    cfg: dict,
    inputs: pd.DataFrame,
    *,
    line_limits: bool = True,
    out_lines: tuple[str, ...] = (),
    gen_availability: pd.DataFrame | None = None,
    line_rating_scale: float | dict[str, float] = 1.0,
) -> pypsa.Network:
    """Return a PyPSA network over ``inputs.index`` snapshots.

    gen_availability: optional DataFrame (snapshots x generator names) of 0/1
        multipliers on p_max_pu (forced outages).
    line_rating_scale: multiply line ratings: one float for all lines, or a
        {line: scale} dict (unlisted lines keep 1.0). Used for sensitivities.
    """
    n = pypsa.Network()
    n.set_snapshots(inputs.index)

    for bus, b in cfg["buses"].items():
        n.add("Bus", bus, v_nom=V_NOM_KV, x=b["x"], y=b["y"], carrier="AC")

    for name, ln in cfg["lines"].items():
        if name in out_lines:
            continue
        scale = line_rating_scale.get(name, 1.0) if isinstance(line_rating_scale, dict) else line_rating_scale
        s_nom = ln["s_nom"] * scale if line_limits else UNLIMITED_MW
        n.add("Line", name, bus0=ln["bus0"], bus1=ln["bus1"], x=ln["x"], r=0.0, s_nom=s_nom)

    for bus in load_buses(cfg):
        n.add("Load", f"load_{bus}", bus=bus, p_set=inputs[f"load_{bus}"])

    for name, g in cfg["generators"].items():
        if g["profile"] == "none":
            pmax = pd.Series(1.0, index=inputs.index)
        else:
            pmax = inputs[PROFILE_COLUMNS[g["profile"]]].clip(0.0, 1.0)
        if gen_availability is not None and name in gen_availability:
            pmax = pmax * gen_availability[name]
        n.add(
            "Generator",
            name,
            bus=g["bus"],
            carrier=g["carrier"],
            p_nom=g["p_nom"],
            marginal_cost=g["marginal_cost"],
            p_max_pu=pmax,
        )

    for bus in load_buses(cfg):
        peak = float(inputs[f"load_{bus}"].max())
        n.add("Generator", f"shed_{bus}", bus=bus, carrier="shed", p_nom=2.0 * peak, marginal_cost=cfg["voll"])
    return n


def susceptance_matrices(cfg: dict, out_lines: tuple[str, ...] = ()):
    """Incidence matrix A (lines x buses) and line susceptances b (1/x).

    Used by the independent PTDF cross-check and later by N-1 screening.
    """
    buses = list(cfg["buses"])
    lines = [l for l in cfg["lines"] if l not in out_lines]
    A = np.zeros((len(lines), len(buses)))
    b = np.zeros(len(lines))
    for i, l in enumerate(lines):
        ln = cfg["lines"][l]
        A[i, buses.index(ln["bus0"])] = 1.0
        A[i, buses.index(ln["bus1"])] = -1.0
        b[i] = 1.0 / ln["x"]
    return buses, lines, A, b


def ptdf(cfg: dict, out_lines: tuple[str, ...] = (), slack: str | None = None) -> pd.DataFrame:
    """Power transfer distribution factors (lines x buses) w.r.t. a slack bus."""
    buses, lines, A, b = susceptance_matrices(cfg, out_lines)
    slack = slack or buses[0]
    s = buses.index(slack)
    Bf = np.diag(b) @ A
    Bbus = A.T @ Bf
    keep = [i for i in range(len(buses)) if i != s]
    H = np.zeros((len(lines), len(buses)))
    H[:, keep] = Bf[:, keep] @ np.linalg.inv(Bbus[np.ix_(keep, keep)])
    return pd.DataFrame(H, index=lines, columns=buses)


def islanded_buses(cfg: dict, out_lines: tuple[str, ...] = ()) -> set[str]:
    """Buses not connected to the largest island once ``out_lines`` are removed."""
    adj: dict[str, set[str]] = {b: set() for b in cfg["buses"]}
    for name, ln in cfg["lines"].items():
        if name not in out_lines:
            adj[ln["bus0"]].add(ln["bus1"])
            adj[ln["bus1"]].add(ln["bus0"])
    seen: set[str] = set()
    islands = []
    for start in adj:
        if start in seen:
            continue
        comp, stack = set(), [start]
        while stack:
            b = stack.pop()
            if b not in comp:
                comp.add(b)
                stack.extend(adj[b] - comp)
        seen |= comp
        islands.append(comp)
    main = max(islands, key=len)
    return set(cfg["buses"]) - main

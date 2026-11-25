"""Load and validate the single YAML config file.

Everything tunable lives in ``config.yaml`` at the repo root. Modules receive
the parsed dict (``cfg``) rather than reading globals, so tests can build
small configs in memory.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Read the YAML config, apply optional nested overrides, validate."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_update(cfg, overrides)
    validate(cfg)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def validate(cfg: dict) -> None:
    """Fail fast with readable messages on inconsistent configs."""
    buses = cfg["buses"]
    for name, ln in cfg["lines"].items():
        for end in ("bus0", "bus1"):
            if ln[end] not in buses:
                raise ValueError(f"line {name}: unknown {end} '{ln[end]}'")
        if ln["s_nom"] <= 0 or ln["x"] <= 0:
            raise ValueError(f"line {name}: s_nom and x must be positive")
    profiles = {"none", "wind_south", "wind_central", "solar_south", "hydro"}
    for name, g in cfg["generators"].items():
        if g["bus"] not in buses:
            raise ValueError(f"generator {name}: unknown bus '{g['bus']}'")
        if g["profile"] not in profiles:
            raise ValueError(f"generator {name}: unknown profile '{g['profile']}'")
        if g["marginal_cost"] >= cfg["voll"]:
            raise ValueError(f"generator {name}: marginal cost must be below VOLL")
    share = sum(b["load_share"] for b in buses.values())
    if abs(share - 1.0) > 1e-6:
        raise ValueError(f"bus load_share must sum to 1, got {share}")
    if len(cfg["load"]["hourly_shape"]) != 24:
        raise ValueError("load.hourly_shape must have 24 values")
    if len(cfg["hydro"]["monthly_pmax_pu"]) != 12:
        raise ValueError("hydro.monthly_pmax_pu must have 12 values")
    mc = cfg.get("montecarlo")
    if mc:
        missing = set(cfg["lines"]) - set(mc["line_outage_rate"])
        if missing:
            raise ValueError(f"montecarlo.line_outage_rate missing lines: {sorted(missing)}")
        carriers = {g["carrier"] for g in cfg["generators"].values()}
        unknown = set(mc["generator_outage"]) - carriers
        if unknown:
            raise ValueError(f"montecarlo.generator_outage has unknown carriers: {sorted(unknown)}")


def data_dir(cfg: dict) -> Path:
    p = ROOT / cfg["paths"]["data_dir"]
    p.mkdir(parents=True, exist_ok=True)
    return p


def reports_dir(cfg: dict) -> Path:
    p = ROOT / cfg["paths"]["reports_dir"]
    (p / "figures").mkdir(parents=True, exist_ok=True)
    return p

"""Static figures (matplotlib) for the README / reports.

Colour rules: fixed categorical order per bus (colour follows the bus, never
its rank), one sequential hue for magnitudes, recessive grid, one y-axis per
chart.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .simulate import DispatchResult, total_load  # noqa: E402

# Validated reference categorical palette, fixed order
SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
BUS_ORDER = ["NORTH", "EDM", "RD", "CAL", "SOUTH", "BC"]
BUS_COLOR = dict(zip(BUS_ORDER, SERIES))
TEXT, TEXT2, GRID = "#191919", "#666666", "#e4e4e4"
SEQ = ["#cce3fd", "#84b9f3", "#338ced", "#1b64be", "#0e3d7e"]  # blue ramp
CRITICAL = "#d03b3b"


def style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "axes.titlecolor": TEXT,
        "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "legend.frameon": False,
        "lines.linewidth": 2.0,
    })


def _save(fig, out: Path, name: str) -> Path:
    p = out / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def network_diagram(cfg: dict, out: Path, line_values: pd.Series | None = None, value_label: str = "") -> Path:
    """Bus/line map. Optionally colour lines by a value in [0,1] (e.g. share of hours binding)."""
    style()
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    ax.set_aspect("equal")
    ax.grid(False)
    ax.axis("off")
    pos = {b: (v["x"], v["y"]) for b, v in cfg["buses"].items()}
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    vmax = max(float(line_values.max()), 1e-9) if line_values is not None else 1
    parallel = {}
    for name, ln in cfg["lines"].items():
        key = tuple(sorted((ln["bus0"], ln["bus1"])))
        parallel.setdefault(key, []).append(name)
    for key, names in parallel.items():
        (x0, y0), (x1, y1) = pos[key[0]], pos[key[1]]
        dx, dy = x1 - x0, y1 - y0
        nrm = np.hypot(dx, dy)
        ox, oy = -dy / nrm, dx / nrm
        for k, name in enumerate(names):
            off = (k - (len(names) - 1) / 2) * 0.07
            c = cmap(float(line_values[name]) / vmax) if line_values is not None else "#9c9c9c"
            ax.plot([x0 + ox * off, x1 + ox * off], [y0 + oy * off, y1 + oy * off], color=c, lw=4, solid_capstyle="round", zorder=1)
        rating = "+".join(str(cfg["lines"][n]["s_nom"]) for n in names)
        lab = f"{rating} MW"
        if line_values is not None:
            lab += "\n" + ", ".join(f"{line_values[n]:.0%}" for n in names)
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, lab, ha="center", va="center", fontsize=7.5, color=TEXT2,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GRID, lw=0.6), zorder=2)

    gens_by_bus: dict[str, list[str]] = {}
    for g, v in cfg["generators"].items():
        gens_by_bus.setdefault(v["bus"], []).append(f"{v['carrier']} {v['p_nom']} MW @ ${v['marginal_cost']:g}")
    for b, (x, y) in pos.items():
        ax.scatter([x], [y], s=420, color=BUS_COLOR.get(b, "#666"), edgecolor="white", linewidth=2, zorder=3)
        share = cfg["buses"][b]["load_share"]
        title = f"{b}  ({cfg['buses'][b]['label']})"
        if share:
            title += f"\nload share {share:.0%}"
        txt = title + "\n" + "\n".join(gens_by_bus.get(b, []))
        side = cfg["buses"][b].get("label_pos", "right")
        dx, dy, ha, va = {"right": (0.2, 0, "left", "center"), "left": (-0.2, 0, "right", "center"),
                          "below": (0, -0.25, "center", "top"), "below_left": (-0.1, -0.22, "right", "top"), "above": (0, 0.25, "center", "bottom")}[side]
        ax.text(x + dx, y + dy, txt, ha=ha, va=va, fontsize=7.5, color=TEXT)
    ttl = "Toy Alberta-like network (illustrative capacities & costs)"
    if line_values is not None:
        ttl += f"\nline colour = {value_label}"
        sm = matplotlib.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, vmax))
        cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
        cb.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        cb.outline.set_visible(False)
    ax.set_title(ttl)
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-1.3, 4.9)
    return _save(fig, out, "network" if line_values is None else "network_congestion")


def inputs_overview(cfg: dict, inputs: pd.DataFrame, out: Path) -> Path:
    """Small multiples: daily means of temperature, total load, wind CF, solar CF."""
    style()
    d = pd.DataFrame({
        "System temperature (°C, Calgary)": inputs["temp_CAL"],
        "Total load (MW)": total_load(cfg, inputs),
        "Wind capacity factor (South)": inputs["cf_wind_south"],
        "Solar capacity factor (South)": inputs["cf_solar_south"],
    })
    daily = d.resample("D").mean()
    fig, axes = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    for ax, col in zip(axes, d.columns):
        ax.plot(d.index, d[col], color=SEQ[1], lw=0.4, label="hourly")
        ax.plot(daily.index, daily[col], color=SERIES[0], lw=1.6, label="daily mean")
        ax.set_title(col, loc="left", fontsize=10)
    axes[0].legend(loc="upper right", ncol=2)
    fig.suptitle("Synthetic hourly inputs (2 years, fixed seed)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "inputs_overview")


def spread_vs_loading(cfg: dict, res: DispatchResult, out: Path) -> Path:
    """Max internal LMP spread vs the highest internal line loading, per hour."""
    style()
    internal_lines = [l for l in res.loading if "BC" not in l]
    internal_buses = [b for b in res.lmp if b != "BC"]
    x = res.loading[internal_lines].max(axis=1)
    y = res.lmp[internal_buses].max(axis=1) - res.lmp[internal_buses].min(axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    binding = res.binding[internal_lines].any(axis=1)
    ax.scatter(x[~binding], y[~binding], s=8, color=SEQ[1], alpha=0.5, linewidths=0, label="no internal line binding")
    ax.scatter(x[binding], y[binding], s=8, color=SERIES[1], alpha=0.5, linewidths=0, label="≥1 internal line binding")
    ax.axvline(1.0, color=TEXT2, lw=1, ls="--")
    ax.text(0.99, ax.get_ylim()[1] * 0.95, "line limit", ha="right", va="top", color=TEXT2, fontsize=8)
    ax.set_xlabel("Highest internal line loading (|flow| / limit)")
    ax.set_ylabel("Max − min LMP across Alberta buses ($/MWh)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Prices separate only when a line hits its limit (step behaviour)")
    ax.legend(loc="upper left", markerscale=2.5)
    return _save(fig, out, "spread_vs_loading")


def lmp_week(cfg: dict, res: DispatchResult, out: Path, buses=("SOUTH", "CAL", "NORTH")) -> Path:
    """One week with the most congested hours: LMP per bus with binding hours shaded."""
    style()
    weekly = res.binding.any(axis=1).resample("W").sum()
    wk_end = weekly.idxmax()
    sl = slice(wk_end - pd.Timedelta(days=6, hours=23), wk_end + pd.Timedelta(hours=23))
    lmp = res.lmp.loc[sl]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bnd = res.binding.loc[sl].any(axis=1)
    for t in bnd.index[bnd]:
        ax.axvspan(t, t + pd.Timedelta(hours=1), color="#efefef", lw=0, zorder=0)
    for b in buses:
        ax.step(lmp.index, lmp[b], where="post", color=BUS_COLOR[b], label=b, lw=1.6)
    ax.set_ylabel("LMP ($/MWh)")
    ax.set_title(f"Week ending {wk_end:%Y-%m-%d}: LMPs (grey = hours with ≥1 binding line)", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(buses))
    fig.autofmt_xdate()
    return _save(fig, out, "lmp_week")


def price_duration(cfg: dict, res: DispatchResult, out: Path, copper: DispatchResult | None = None) -> Path:
    style()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    pct = np.linspace(0, 100, len(res.lmp))
    for b in [b for b in BUS_ORDER if b in res.lmp]:
        ax.plot(pct, np.sort(res.lmp[b].to_numpy())[::-1], color=BUS_COLOR[b], label=b, lw=1.6)
    if copper is not None:
        ax.plot(pct, np.sort(copper.lmp.iloc[:, 0].to_numpy())[::-1], color=TEXT, ls="--", lw=1.2, label="no line limits")
    ax.set_xlabel("% of hours price is at or above")
    ax.set_ylabel("LMP ($/MWh)")
    ax.legend(ncol=7, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8.5)
    ax.set_title("Price duration curves by bus", pad=28)
    return _save(fig, out, "price_duration")


def binding_by_line(cfg: dict, res: DispatchResult, out: Path) -> Path:
    style()
    share = res.binding.mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.barh(share.index, share.values, color=SERIES[0], height=0.6)
    for i, v in enumerate(share.values):
        ax.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=8, color=TEXT)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of hours at limit")
    any_share = res.binding.any(axis=1).mean()
    ax.set_title(f"Binding frequency by line (≥1 line binding in {any_share:.1%} of hours)", loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "binding_by_line")

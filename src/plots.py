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


def network_diagram(cfg: dict, out: Path, line_values: pd.Series | None = None, value_label: str = "",
                    name: str | None = None) -> Path:
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
    for line_name, ln in cfg["lines"].items():
        key = tuple(sorted((ln["bus0"], ln["bus1"])))
        parallel.setdefault(key, []).append(line_name)
    for key, names in parallel.items():
        (x0, y0), (x1, y1) = pos[key[0]], pos[key[1]]
        dx, dy = x1 - x0, y1 - y0
        nrm = np.hypot(dx, dy)
        ox, oy = -dy / nrm, dx / nrm
        for k, line in enumerate(names):
            off = (k - (len(names) - 1) / 2) * 0.07
            c = cmap(float(line_values[line]) / vmax) if line_values is not None else "#9c9c9c"
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
    return _save(fig, out, name or ("network" if line_values is None else "network_congestion"))


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


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------
DIVERGING = ["#1757ac", "#519ced", "#b6d5f8", "#efefef", "#f7c4c4", "#eb6263", "#b02a2b"]


def mc_congestion_probability(table: pd.DataFrame, out: Path) -> Path:
    """Deterministic 2-year share vs Monte Carlo probability (95% Wilson CI) per line."""
    style()
    t = table.sort_values("p_congested")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    h = 0.36
    ax.barh(y + h / 2, t["deterministic_share"], height=h, color=SERIES[0], label="deterministic (2 yr, no outages)")
    err = np.vstack([t["p_congested"] - t["ci95_low"], t["ci95_high"] - t["p_congested"]])
    ax.barh(y - h / 2, t["p_congested"], height=h, color=SERIES[1], label="Monte Carlo (outages + uncertainty)",
            xerr=err, error_kw=dict(ecolor=TEXT, lw=1, capsize=2))
    for i, (d, p) in enumerate(zip(t["deterministic_share"], t["p_congested"])):
        ax.text(max(d, p) + 0.012, i, f"{p:.1%}", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(y, t.index)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Probability a line is at its limit in a given hour")
    ax.set_title("Per-line congestion probability", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "mc_congestion_probability")


def mc_lmp_intervals(q_det: pd.DataFrame, q_mc: pd.DataFrame, out: Path) -> Path:
    """P10-P90 range with P50 marker per bus: deterministic vs Monte Carlo."""
    style()
    buses = [b for b in BUS_ORDER if b in q_mc.index]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k, (q, lab, col) in enumerate(((q_det, "deterministic", SERIES[0]), (q_mc, "Monte Carlo", SERIES[1]))):
        x = np.arange(len(buses)) + (k - 0.5) * 0.28
        ax.vlines(x, q.loc[buses, "P10"], q.loc[buses, "P90"], color=col, lw=5, label=f"{lab}: P10–P90")
        ax.scatter(x, q.loc[buses, "P50"], s=46, color="white", edgecolor=col, linewidth=2, zorder=3,
                   label=f"{lab}: P50")
    ax.set_xticks(np.arange(len(buses)), buses)
    ax.set_ylabel("LMP ($/MWh)")
    ax.set_title("LMP distribution by bus (P10 / P50 / P90)", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8)
    ax.grid(axis="x", visible=False)
    return _save(fig, out, "mc_lmp_intervals")


def mc_lmp_ecdf(res: DispatchResult, out: Path, xmax: float = 160) -> Path:
    style()
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for b in [b for b in BUS_ORDER if b in res.lmp]:
        v = np.sort(res.lmp[b].dropna().to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=BUS_COLOR[b], label=b, lw=1.6)
    tail = float((res.lmp.drop(columns="BC", errors="ignore") > xmax).any(axis=1).mean())
    ax.set_xlim(-2, xmax)
    ax.set_xlabel(f"LMP ($/MWh); {tail:.2%} of scenarios exceed ${xmax:.0f} at some bus (up to the $999.99 cap)")
    ax.set_ylabel("Share of scenarios ≤ x")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Monte Carlo LMP distribution (empirical CDF)", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=6, fontsize=8.5)
    return _save(fig, out, "mc_lmp_ecdf")


def mc_convergence(conv: pd.DataFrame, out: Path) -> Path:
    style()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for k, (line, d) in enumerate(conv.groupby("line", sort=False)):
        d = d[d["n"] >= 50]
        ax.fill_between(d["n"], d["lo"], d["hi"], color=SERIES[k], alpha=0.18, lw=0)
        ax.plot(d["n"], d["p"], color=SERIES[k], lw=1.6, label=line)
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Number of Monte Carlo scenarios (log scale)")
    ax.set_ylabel("Estimated P(line at limit)")
    ax.set_title("Convergence of congestion probability (bands = 95% Wilson CI)", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=8.5)
    return _save(fig, out, "mc_convergence")


def n1_security(ranked: pd.DataFrame, out: Path) -> Path:
    style()
    t = ranked.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    vals = t["share_hours_overload"].fillna(0)
    ax.barh(t.index, vals, color=[SERIES[0] if not isl else "#c8c8c8" for isl in t["islands"]], height=0.6)
    for i, (idx, r) in enumerate(t.iterrows()):
        if r["islands"]:
            txt = f"islands {r['islands']}: loses up to {r['islanded_injection_max_mw']:.0f} MW"
            ax.text(0.01, i, txt, va="center", fontsize=8, color=TEXT)
        else:
            ax.text(vals.iloc[i] + 0.01, i, f"{vals.iloc[i]:.0%}  (worst: {r['worst_line']} {r['max_post_loading']:.0%})",
                    va="center", fontsize=8, color=TEXT)
    ax.set_xlim(0, 1.45)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Share of hours with ≥1 post-contingency overload (pre-contingency dispatch)")
    ax.set_title("N-1 security screen by outaged line", loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "n1_security")


def n1_dlmp_heatmap(dlmp: pd.DataFrame, out: Path) -> Path:
    style()
    cols = [b for b in BUS_ORDER if b in dlmp.columns]
    d = dlmp[cols]
    vmax = float(np.abs(d.to_numpy()).max())
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("div", DIVERGING)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    im = ax.imshow(d.to_numpy(), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(d)), d.index)
    ax.grid(False)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.iat[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 * vmax else TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.set_label("$/MWh")
    ax.set_title("Mean LMP change after re-dispatch, by outaged line", loc="left")
    return _save(fig, out, "n1_dlmp_heatmap")


def rating_sensitivity_plot(sens: pd.DataFrame, out: Path) -> Path:
    style()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for k, (corr, d) in enumerate(sens.groupby("corridor", sort=False)):
        ax.fill_between(d["rating_scale"], d["ci95_low"], d["ci95_high"], color=SERIES[k], alpha=0.18, lw=0)
        ax.plot(d["rating_scale"], d["p_corridor_congested"], color=SERIES[k], marker="o", ms=6, label=corr)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Corridor rating relative to base")
    ax.set_ylabel("P(corridor at limit)")
    ax.set_title("Rating sensitivity (same scenarios for every rating)", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    return _save(fig, out, "rating_sensitivity")


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------
MODEL_COLOR = {"OPF": TEXT, "baseline": "#9c9c9c", "LightGBM": SERIES[0], "MLP": SERIES[1]}


def _mcolor(name: str) -> str:
    for k, c in MODEL_COLOR.items():
        if name.startswith(k):
            return c
    return SERIES[2]


def p3_regime_mae(reg: pd.DataFrame, out: Path, name: str = "surrogate_mae_by_regime", title: str = "") -> Path:
    style()
    models = [c for c in reg.columns if c != "share_of_rows"]
    t = reg.iloc[::-1]
    y = np.arange(len(t))
    h = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8.2, 0.55 * len(t) + 1.6))
    for k, m in enumerate(models):
        ax.barh(y + (len(models) / 2 - k - 0.5) * h, t[m], height=h * 0.92, color=_mcolor(m), label=m)
    ax.set_yticks(y, [f"{r}  ({s:.1%})" for r, s in zip(t.index, t["share_of_rows"])])
    ax.set_xlabel("Mean absolute LMP error, averaged over buses ($/MWh)")
    ax.set_title(title or "Where the surrogates fail: error by operating regime (test set)", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(models), fontsize=8.5)
    ax.grid(axis="y", visible=False)
    return _save(fig, out, name)


def p3_mc_congestion(cmp: pd.DataFrame, out: Path) -> Path:
    style()
    t = cmp.sort_values("opf_p")
    y = np.arange(len(t))
    h = 0.36
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    err = np.vstack([t["opf_p"] - t["opf_ci95_low"], t["opf_ci95_high"] - t["opf_p"]])
    ax.barh(y + h / 2, t["opf_p"], height=h, color=TEXT, label="full DC-OPF (95% CI)", xerr=err,
            error_kw=dict(ecolor="#9c9c9c", lw=1, capsize=2))
    ax.barh(y - h / 2, t["surrogate_p_mean_prob"], height=h, color=SERIES[0], label="LightGBM classifier (mean prob.)")
    for i, (a, b) in enumerate(zip(t["opf_p"], t["surrogate_p_mean_prob"])):
        ax.text(max(a, b) + 0.008, i, f"{a:.2%} vs {b:.2%}", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(y, t.index)
    ax.set_xlim(0, t["opf_p"].max() * 1.35)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("P(line at limit) over the 20,000 Phase 2 scenarios")
    ax.set_title("Monte Carlo congestion probability: OPF vs surrogate", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "surrogate_mc_congestion")


def p3_ecdf_compare(Y: pd.DataFrame, preds: dict, out: Path, buses=("SOUTH", "CAL"), xlim=(-5, 90)) -> Path:
    style()
    fig, axes = plt.subplots(1, len(buses), figsize=(10, 4.2))
    for ax, b in zip(np.atleast_1d(axes), buses):
        col = f"lmp_{b}"
        series = {"OPF": Y[col], **{k: v[col] for k, v in preds.items()}}
        for name, s in series.items():
            v = np.sort(s.dropna().to_numpy())
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=_mcolor(name), label=name,
                    lw=2.2 if name == "OPF" else 1.5, ls="--" if name.startswith("baseline") else "-")
        ax.set_xlim(*xlim)
        ax.set_title(b, loc="left")
        ax.set_xlabel("LMP ($/MWh)")
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    np.atleast_1d(axes)[0].set_ylabel("Share of scenarios ≤ x")
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("LMP distribution over the Monte Carlo test set: OPF vs surrogates", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "surrogate_lmp_ecdf")


def p3_step_sweep(sw: pd.DataFrame, models: list[str], buses: list[str], xlabel: str, out: Path, name: str,
                  title: str, pct: bool = False) -> Path:
    style()
    fig, axes = plt.subplots(1, len(buses), figsize=(10, 4.2))
    for ax, b in zip(np.atleast_1d(axes), buses):
        x = sw["value"]
        bind = sw["any_binding"].to_numpy()
        for i in range(len(x) - 1):
            if bind[i]:
                ax.axvspan(x.iloc[i], x.iloc[i + 1], color="#efefef", lw=0, zorder=0)
        ax.step(x, sw[f"OPF_{b}"], where="mid", color=TEXT, lw=2.4, label="DC-OPF (truth)")
        for m in models:
            ax.plot(x, sw[f"{m}_{b}"], color=_mcolor(m), lw=1.6, ls="--" if m.startswith("baseline") else "-", label=m)
        ax.set_title(b, loc="left")
        ax.set_xlabel(xlabel)
        if pct:
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    np.atleast_1d(axes)[0].set_ylabel("LMP ($/MWh)")
    np.atleast_1d(axes)[0].legend(loc="best", fontsize=8)
    fig.suptitle(title + "  (grey = a line is at its limit)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, name)


def p3_shap(shap_abs: dict[str, pd.Series], out: Path, top: int = 10) -> Path:
    style()
    fig, axes = plt.subplots(1, len(shap_abs), figsize=(10.5, 4.4))
    for ax, (bus, s) in zip(np.atleast_1d(axes), shap_abs.items()):
        t = s.sort_values(ascending=False).head(top).iloc[::-1]
        ax.barh(t.index, t.values, color=SERIES[0], height=0.6)
        ax.set_title(f"{bus}: congestion component of LMP", loc="left", fontsize=10)
        ax.set_xlabel("mean |SHAP| ($/MWh)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("What drives the congestion premium? TreeSHAP, LightGBM hybrid models (test set)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "surrogate_shap_importance")


def p3_shap_dependence(x: pd.Series, sv: pd.Series, color: pd.Series, out: Path, xlabel: str, clabel: str) -> Path:
    style()
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    c = color.astype(bool).to_numpy()
    ax.scatter(x[~c], sv[~c], s=6, color=SEQ[2], alpha=0.5, linewidths=0, label=f"{clabel} = 0")
    ax.scatter(x[c], sv[c], s=10, color=SERIES[1], alpha=0.8, linewidths=0, label=f"{clabel} = 1")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("SHAP value for LMP_SOUTH ($/MWh)")
    ax.set_title("SHAP dependence: South surplus vs its effect on the SOUTH price", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, markerscale=3, fontsize=8.5)
    return _save(fig, out, "surrogate_shap_dependence")


def p3_learning_curve(lc: pd.DataFrame, out: Path) -> Path:
    style()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for k, (m, col) in enumerate((("MAE all rows", "MAE"), ("MAE congested rows", "MAE_congested"))):
        ax.plot(lc["n_train"], lc[col], color=SERIES[k], marker="o", ms=6, label=m)
    ax.set_xscale("log")
    ax.set_xlabel("Training scenarios (OPF solves), log scale")
    ax.set_ylabel("Test MAE, mean over buses ($/MWh)")
    ax.set_title("Learning curve: LightGBM hybrid", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    return _save(fig, out, "surrogate_learning_curve")


def p3_runtime(rt: pd.DataFrame, out: Path) -> Path:
    style()
    t = rt.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(t) + 1.4))
    ax.barh(t.index, t["seconds"], color=[TEXT if i.startswith("DC-OPF") else SERIES[0] for i in t.index], height=0.6)
    for i, v in enumerate(t["seconds"]):
        ax.text(v * 1.15, i, f"{v:.3g} s", va="center", fontsize=8, color=TEXT)
    ax.set_xscale("log")
    ax.set_xlim(t["seconds"].min() / 2, t["seconds"].max() * 8)
    ax.set_xlabel("Seconds for the 20,000-scenario Monte Carlo study (log scale)")
    ax.set_title("Runtime: full OPF vs surrogate", loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, out, "surrogate_runtime")


def p3_pred_vs_true(Y: pd.DataFrame, preds: dict, bus: str, out: Path) -> Path:
    style()
    fig, axes = plt.subplots(1, len(preds), figsize=(10, 4.2), sharex=True, sharey=True)
    col = f"lmp_{bus}"
    for ax, (name, P) in zip(np.atleast_1d(axes), preds.items()):
        ax.scatter(Y[col], P[col], s=5, alpha=0.35, color=_mcolor(name), linewidths=0)
        ax.plot([0, 130], [0, 130], color=TEXT2, lw=1, ls="--")
        ax.set_xlim(-5, 130)
        ax.set_ylim(-5, 130)
        ax.set_title(name, loc="left", fontsize=10)
        ax.set_xlabel(f"OPF LMP {bus} ($/MWh)")
    np.atleast_1d(axes)[0].set_ylabel("Predicted ($/MWh)")
    fig.suptitle(f"Predicted vs true LMP at {bus} (test scenarios; axes clipped at $130)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, f"surrogate_pred_vs_true_{bus}")

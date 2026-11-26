"""Streamlit dashboard — reads everything from the SQL database (never from files).

    streamlit run src/dashboard.py
    DB_URL=postgresql+psycopg://user:pass@host/aeso_poc streamlit run src/dashboard.py

Queries are written with SQLAlchemy Core, so the same code runs on SQLite,
PostgreSQL and SQL Server. Percentiles are computed in pandas because
percentile functions are not portable across those databases.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from sqlalchemy import and_, func, select  # noqa: E402

from src import db  # noqa: E402
from src.config import load_config  # noqa: E402

SERIES = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17"]
BUS_ORDER = ["NORTH", "EDM", "RD", "CAL", "SOUTH", "BC"]
BUS_COLOR = dict(zip(BUS_ORDER, SERIES))
MODEL_COLOR = {"OPF": "#191919", "LightGBM": SERIES[0], "MLP": SERIES[1], "baseline": "#9c9c9c", "(actual)": "#191919"}
SEQ = [[0, "#cce3fd"], [0.25, "#84b9f3"], [0.5, "#338ced"], [0.75, "#1b64be"], [1, "#0e3d7e"]]
SOURCES = {"mc_opf": "Monte Carlo · full DC-OPF", "mc_surrogate": "Monte Carlo · ML surrogate",
           "hourly_opf": "Deterministic 2-year hourly · DC-OPF"}
STUDIES = {"mc_test": "Monte Carlo (20,000 scenarios, outages + uncertainty)", "hourly": "Deterministic hourly (2 years)",
           "real_2024_2025": "Real AESO 2024-25 shapes (calibrated costs)"}

st.set_page_config(page_title="Nodal congestion POC", layout="wide")


# ---------------------------------------------------------------------------
# data access (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def engine():
    return db.get_engine(load_config())


@st.cache_data(ttl=600)
def q(_stmt, key: str) -> pd.DataFrame:
    """Run a SQLAlchemy Core statement; ``key`` makes the cache entry unique."""
    with engine().connect() as c:
        return pd.read_sql(_stmt, c)


def runs() -> pd.DataFrame:
    t = db.run
    return q(select(t.c.run_id, t.c.created_utc, t.c.label).order_by(t.c.run_id.desc()), "runs")


def network():
    return (q(select(db.bus), "bus"), q(select(db.line), "line"), q(select(db.generator), "gen"))


def congestion_stats(run_id: int) -> pd.DataFrame:
    t = db.line_congestion_stat
    return q(select(t).where(t.c.run_id == run_id), f"lcs{run_id}")


def available_studies(run_id: int) -> list[str]:
    t = db.scenario
    have = set(q(select(t.c.study).where(t.c.run_id == run_id).distinct(), f"studies{run_id}")["study"])
    return [s for s in STUDIES if s in have]


def observed_pool_price() -> pd.Series:
    t = db.aeso_hourly
    try:
        return q(select(t.c.pool_price), "aeso_pool")["pool_price"]
    except Exception:  # noqa: BLE001 - table absent before Phase 5
        return pd.Series(dtype=float)


def lmp_values(run_id: int, study: str, buses: tuple[str, ...]) -> pd.DataFrame:
    t = db.bus_result
    s = select(t.c.scenario_id, t.c.bus_id, t.c.lmp).where(
        and_(t.c.run_id == run_id, t.c.study == study, t.c.bus_id.in_(buses)))
    return q(s, f"lmp{run_id}{study}{buses}")


def surrogate_values(run_id: int, buses: tuple[str, ...]) -> pd.DataFrame:
    t = db.surrogate_lmp
    s = select(t.c.scenario_id, t.c.bus_id, t.c.model, t.c.lmp_pred).where(
        and_(t.c.run_id == run_id, t.c.study == "mc_test", t.c.bus_id.in_(buses)))
    return q(s, f"sur{run_id}{buses}")


def mean_lmp_by_bus(run_id: int, study: str) -> pd.Series:
    t = db.bus_result
    s = select(t.c.bus_id, func.avg(t.c.lmp).label("mean_lmp")).where(
        and_(t.c.run_id == run_id, t.c.study == study)).group_by(t.c.bus_id)
    return q(s, f"mean{run_id}{study}").set_index("bus_id")["mean_lmp"]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def network_figure(buses, lines, gens, values: pd.Series, value_label: str, mean_lmp: pd.Series | None = None,
                   fmt="{:.1%}", cmax=None):
    pos = buses.set_index("bus_id")[["x", "y"]]
    label_pos = {b: v.get("label_pos", "above") for b, v in load_config()["buses"].items()}
    fig = go.Figure()
    cmax = cmax or max(float(values.max()), 1e-9)
    groups = lines.groupby(lines.apply(lambda r: tuple(sorted((r.bus0, r.bus1))), axis=1))
    for key, grp in groups:
        (x0, y0), (x1, y1) = pos.loc[key[0]], pos.loc[key[1]]
        dx, dy = x1 - x0, y1 - y0
        nrm = np.hypot(dx, dy)
        ox, oy = -dy / nrm, dx / nrm
        for k, (_, ln) in enumerate(grp.iterrows()):
            off = (k - (len(grp) - 1) / 2) * 0.08
            v = float(values.get(ln.line_id, 0.0))
            colour = _sample_seq(v / cmax)
            xs, ys = [x0 + ox * off, x1 + ox * off], [y0 + oy * off, y1 + oy * off]
            hover = (f"<b>{ln.line_id}</b><br>{ln.bus0} → {ln.bus1}<br>rating {ln.s_nom_mw:,.0f} MW"
                     f"<br>{value_label}: {fmt.format(v)}")
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=colour, width=7),
                                     hoverinfo="skip", showlegend=False))
            # invisible fat hit target along the line for hover
            fig.add_trace(go.Scatter(x=np.linspace(*xs, 12), y=np.linspace(*ys, 12), mode="markers",
                                     marker=dict(size=18, color="rgba(0,0,0,0)"), hovertext=hover,
                                     hoverinfo="text", showlegend=False))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        lab = " / ".join(fmt.format(float(values.get(l, 0.0))) for l in grp.line_id)
        fig.add_annotation(x=mx, y=my, text=lab, showarrow=False, font=dict(size=11, color="#191919"),
                           bgcolor="rgba(255,255,255,0.85)", bordercolor="#e4e4e4", borderwidth=1)
    g = gens.groupby("bus_id").apply(lambda d: "<br>".join(f"{r.carrier} {r.p_nom_mw:,.0f} MW @ ${r.marginal_cost:g}"
                                                           for r in d.itertuples()), include_groups=False)
    for b in buses.itertuples():
        txt = f"<b>{b.bus_id}</b> — {b.label}<br>load share {b.load_share:.0%}<br>{g.get(b.bus_id, '')}"
        if mean_lmp is not None and b.bus_id in mean_lmp:
            txt += f"<br><b>mean LMP ${mean_lmp[b.bus_id]:.2f}</b>"
        tpos = {"right": "middle right", "left": "middle left", "below_left": "bottom left", "above": "top center"}
        fig.add_trace(go.Scatter(x=[b.x], y=[b.y], mode="markers+text", text=[f"  {b.bus_id}  "],
                                 textposition=tpos.get(label_pos.get(b.bus_id), "top center"),
                                 textfont=dict(size=13, color="#191919"),
                                 marker=dict(size=26, color=BUS_COLOR.get(b.bus_id, "#666"), line=dict(color="white", width=2)),
                                 hovertext=txt, hoverinfo="text", showlegend=False))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", showlegend=False, hoverinfo="skip",
                             marker=dict(colorscale=SEQ, cmin=0, cmax=cmax, color=[0], showscale=True,
                                         colorbar=dict(title=value_label, tickformat=".0%" if "%" in fmt else None))))
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                      xaxis=dict(visible=False, range=[-2.8, 2.2]), yaxis=dict(visible=False, scaleanchor="x"))
    return fig


def _sample_seq(t: float) -> str:
    from plotly.colors import sample_colorscale
    return sample_colorscale(SEQ, [min(max(t, 0.0), 1.0)])[0]


def ecdf_figure(series: dict[str, pd.Series], xmax: float):
    fig = go.Figure()
    for name, s in series.items():
        v = np.sort(s.dropna().to_numpy())
        if len(v) > 4000:  # thin for the browser, keep the exact shape at the steps
            idx = np.unique(np.r_[np.linspace(0, len(v) - 1, 4000).astype(int), np.flatnonzero(np.diff(v)) , len(v) - 1])
            v_plot, y = v[idx], (idx + 1) / len(v)
        else:
            v_plot, y = v, np.arange(1, len(v) + 1) / len(v)
        colour = MODEL_COLOR.get(name.split(" ")[-1], BUS_COLOR.get(name.split(" ")[0], "#666"))
        dash = "dash" if "baseline" in name else ("solid")
        fig.add_trace(go.Scatter(x=v_plot, y=y, mode="lines", line=dict(shape="hv", color=colour, width=2, dash=dash),
                                 name=name, hovertemplate="LMP ≤ $%{x:.2f}: %{y:.1%}<extra>" + name + "</extra>"))
    fig.update_layout(height=430, xaxis=dict(title="LMP ($/MWh)", range=[-5, xmax]),
                      yaxis=dict(title="share of scenarios ≤ x", tickformat=".0%"), hovermode="closest",
                      legend=dict(orientation="h", y=1.08), margin=dict(t=40, l=10, r=10, b=10), plot_bgcolor="white")
    fig.update_xaxes(gridcolor="#e4e4e4")
    fig.update_yaxes(gridcolor="#e4e4e4")
    return fig


def quantiles(s: pd.Series) -> dict:
    return {"P10": s.quantile(0.1), "P50": s.quantile(0.5), "P90": s.quantile(0.9), "P99": s.quantile(0.99),
            "mean": s.mean(), "P(LMP ≥ $100)": (s >= 100).mean()}


# ---------------------------------------------------------------------------
# layout
# ---------------------------------------------------------------------------
try:
    rr = runs()
except Exception as exc:  # noqa: BLE001
    st.error(f"Cannot read the database at `{db.safe_url(db.db_url(load_config()))}`: {exc}\n\n"
             "Run `python -m src.run` (or `python -m src.db --load`) first.")
    st.stop()
if rr.empty:
    st.warning("The database has no runs yet. Run `python -m src.run`.")
    st.stop()

with st.sidebar:
    st.markdown("### Probabilistic congestion POC")
    st.caption("Toy Alberta-like 6-bus grid · DC-OPF (PyPSA + HiGHS) · **synthetic, illustrative data**")
    run_id = st.selectbox("Pipeline run", rr["run_id"], format_func=lambda r: f"run {r} · "
                          f"{rr.set_index('run_id').loc[r, 'created_utc']:%Y-%m-%d %H:%M} UTC")
    st.caption(f"Database: `{db.safe_url(db.db_url(load_config()))}`")
    st.caption("DC approximation: lossless, no voltage or reactive power. Capacities, costs and ratings are made up.")

buses, lines, gens = network()
cs = congestion_stats(run_id)
tab_map, tab_lmp, tab_n1, tab_sur = st.tabs(["Network & congestion", "LMP distributions", "N-1 critical lines",
                                             "OPF vs surrogate"])

with tab_map:
    c1, c2 = st.columns([3, 1])
    with c2:
        src = st.radio("Congestion probability from", list(SOURCES), format_func=SOURCES.get)
        sub = cs[cs["source"] == src].set_index("line_id")
        study = "hourly" if src == "hourly_opf" else "mc_test"
        st.metric("Lines with P(at limit) > 5%", int((sub["p_binding"] > 0.05).sum()), help="of 8 circuits")
        show = sub[["p_binding", "ci95_low", "ci95_high", "expected_hours_per_year"]].sort_values("p_binding", ascending=False)
        st.dataframe(show.style.format({"p_binding": "{:.2%}", "ci95_low": "{:.2%}", "ci95_high": "{:.2%}",
                                        "expected_hours_per_year": "{:,.0f}"}, na_rep="—"), width="stretch")
        st.caption("95% Wilson intervals (OPF sources). Surrogate = mean predicted probability of the line classifier.")
    with c1:
        st.markdown(f"**P(line at its limit) — {SOURCES[src]}**  ·  hover a line or bus for details")
        st.plotly_chart(network_figure(buses, lines, gens, sub["p_binding"], "P(at limit)",
                                       mean_lmp_by_bus(run_id, study)), width="stretch")

with tab_lmp:
    c1, c2 = st.columns([1, 3])
    with c1:
        study = st.radio("Study", available_studies(run_id), format_func=STUDIES.get, key="lmp_study")
        sel = st.multiselect("Buses", BUS_ORDER, default=["SOUTH", "CAL", "NORTH"])
        add_sur = study == "mc_test" and st.checkbox("Overlay surrogate predictions (one bus)", value=False)
        xmax = st.slider("x-axis max ($/MWh)", 60, 1000, 130, step=10)
    if not sel:
        st.info("Pick at least one bus.")
    else:
        v = lmp_values(run_id, study, tuple(sel))
        series = {b: v.loc[v.bus_id == b, "lmp"] for b in sel}
        if study == "real_2024_2025":
            obs = observed_pool_price()
            if not obs.empty:
                series["AESO pool price (actual)"] = obs
        if add_sur:
            b0 = sel[0]
            sv = surrogate_values(run_id, (b0,))
            series = {f"{b0} OPF": series[b0]}
            for m in ("LightGBM", "MLP", "baseline"):
                series[f"{b0} {m}"] = sv.loc[sv.model == m, "lmp_pred"]
        with c2:
            st.plotly_chart(ecdf_figure(series, xmax), width="stretch")
            qt = pd.DataFrame({k: quantiles(s) for k, s in series.items()}).T
            st.dataframe(qt.style.format({"P(LMP ≥ $100)": "{:.2%}"} | {c: "${:.2f}" for c in ("P10", "P50", "P90", "P99", "mean")}),
                         width="stretch")
            st.caption("LMP is step-shaped: most hours sit on a few price plateaus (the marginal unit's offer, or a "
                       "loop-flow mixture of two offers when a line binds). Quantiles can jump between plateaus.")

with tab_n1:
    t = db.n1_result
    n1 = q(select(t).where(t.c.run_id == run_id), f"n1{run_id}").drop(columns="run_id")
    n1 = n1.sort_values(["rank_security", "rank_economic"]).set_index("outaged_line")
    num = [c for c in n1.columns if c not in ("islands", "worst_line")]
    n1[num] = n1[num].apply(pd.to_numeric, errors="coerce")       # NULL -> NaN (islanding rows)
    n1[["islands", "worst_line"]] = n1[["islands", "worst_line"]].fillna("").replace("", "—")
    st.markdown("**Single-line outages over all 17,520 hours.** Security screen = flows with the *pre-contingency* "
                "dispatch on the reduced network (PTDF); economic view = DC-OPF re-dispatch without the line.")
    cols = {"islands": "islands", "share_hours_overload": "hours with overload", "max_post_loading": "worst loading",
            "worst_line": "worst line", "delta_cost_musd_per_year": "Δ cost (M$/yr)",
            "mean_abs_delta_lmp": "mean |ΔLMP|", "unserved_mwh_per_year": "unserved MWh/yr",
            "rank_security": "rank (security)", "rank_economic": "rank (economic)"}
    fmts = {"share_hours_overload": "{:.1%}", "max_post_loading": "{:.0%}", "delta_cost_musd_per_year": "{:+,.1f}",
            "mean_abs_delta_lmp": "${:.2f}", "unserved_mwh_per_year": "{:,.0f}", "rank_security": "{:.0f}",
            "rank_economic": "{:.0f}"}
    show = n1[list(cols)].copy()
    for c, f in fmts.items():  # format explicitly: NULL (islanding rows) -> "—"
        show[c] = show[c].map(lambda v, f=f: "—" if pd.isna(v) else f.format(v))
    st.dataframe(show.rename(columns=cols), width="stretch")
    t = db.n1_delta_lmp
    dl = q(select(t).where(t.c.run_id == run_id), f"dl{run_id}").pivot(index="outaged_line", columns="bus_id",
                                                                        values="mean_delta_lmp")
    dl = dl.loc[n1.index, [b for b in BUS_ORDER if b in dl.columns]]
    vmax = float(np.abs(dl.to_numpy()).max())
    fig = go.Figure(go.Heatmap(z=dl.to_numpy(), x=dl.columns, y=dl.index, zmin=-vmax, zmax=vmax, text=dl.round(1).to_numpy(),
                               texttemplate="%{text:+.1f}", colorscale=[[0, "#1757ac"], [0.5, "#efefef"], [1, "#b02a2b"]],
                               colorbar=dict(title="$/MWh"), hovertemplate="%{y} out → %{x}: %{z:+.2f} $/MWh<extra></extra>"))
    fig.update_layout(height=420, margin=dict(t=30, l=10, r=10, b=10), yaxis=dict(autorange="reversed"),
                      title="Mean LMP change after re-dispatch")
    st.plotly_chart(fig, width="stretch")
    st.caption("The base dispatch is not security-constrained, so N-1 overloads are expected (most of all on the "
               "North double circuit). Negative Δ cost = removing the line relieves a loop-flow constraint.")

with tab_sur:
    opf = cs[cs.source == "mc_opf"].set_index("line_id")
    sur = cs[cs.source == "mc_surrogate"].set_index("line_id")
    if sur.empty:
        st.info("No surrogate results in this run (Phase 3 not run).")
    else:
        rt = q(select(db.runtime).where(db.runtime.c.run_id == run_id), f"rt{run_id}").set_index("method")
        k1, k2, k3 = st.columns(3)
        o, s_ = rt.filter(like="4 workers", axis=0)["seconds"], rt.filter(like="LightGBM + classifier, 4", axis=0)["seconds"]
        k1.metric("OPF Monte Carlo (4 workers)", f"{o.iloc[0]:.1f} s")
        k2.metric("Surrogate Monte Carlo (LightGBM)", f"{s_.iloc[0]:.2f} s", f"{o.iloc[0] / s_.iloc[0]:.0f}× faster")
        k3.metric("Max |Δ congestion probability|", f"{(sur.p_binding - opf.p_binding).abs().max() * 100:.2f} pp")
        order = opf.sort_values("p_binding").index
        fig = go.Figure()
        fig.add_trace(go.Bar(y=order, x=opf.loc[order, "p_binding"], orientation="h", name="full DC-OPF (95% CI)",
                             marker_color="#191919", error_x=dict(type="data", symmetric=False,
                                                                  array=opf.loc[order, "ci95_high"] - opf.loc[order, "p_binding"],
                                                                  arrayminus=opf.loc[order, "p_binding"] - opf.loc[order, "ci95_low"],
                                                                  color="#9c9c9c"),
                             hovertemplate="%{y}: %{x:.2%}<extra>OPF</extra>"))
        fig.add_trace(go.Bar(y=order, x=sur.loc[order, "p_binding"], orientation="h", name="surrogate classifier",
                             marker_color=SERIES[0], hovertemplate="%{y}: %{x:.2%}<extra>surrogate</extra>"))
        fig.update_layout(barmode="group", height=440, xaxis=dict(tickformat=".0%", title="P(line at limit)"),
                          legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0), margin=dict(t=80, l=10, r=10, b=10),
                          plot_bgcolor="white", title=dict(text="Monte Carlo congestion probability", y=0.97))
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns([1, 2])
        with c1:
            t = db.surrogate_metric
            m = q(select(t).where(and_(t.c.run_id == run_id, t.c.task == "mc")), f"met{run_id}")
            piv = m[m.metric.isin(["MAE", "RMSE", "MAE_congested", "MAE_uncongested"])].pivot_table(
                index="model", columns="metric", values="value", aggfunc="mean")
            st.markdown("**Accuracy on the 20,000 held-out scenarios** (mean over buses, $/MWh)")
            st.dataframe(piv.style.format("{:.2f}"), width="stretch")
            st.dataframe(rt[["seconds"]].style.format("{:.3g}"), width="stretch")
            bus_sel = st.selectbox("Scatter: bus", BUS_ORDER, index=4)
            model_sel = st.selectbox("Scatter: model", ["LightGBM", "MLP", "baseline"])
        with c2:
            truth = lmp_values(run_id, "mc_test", (bus_sel,)).set_index("scenario_id")["lmp"]
            pred = surrogate_values(run_id, (bus_sel,))
            pred = pred[pred.model == model_sel].set_index("scenario_id")["lmp_pred"]
            jj = pd.concat({"OPF": truth, "pred": pred}, axis=1).dropna().sample(min(6000, len(truth)), random_state=0)
            fig = go.Figure(go.Scattergl(x=jj["OPF"], y=jj["pred"], mode="markers",
                                         marker=dict(size=5, color=MODEL_COLOR[model_sel], opacity=0.4),
                                         hovertemplate="OPF $%{x:.2f} → predicted $%{y:.2f}<extra></extra>"))
            fig.add_trace(go.Scatter(x=[0, 140], y=[0, 140], mode="lines", line=dict(color="#666", dash="dash"),
                                     showlegend=False, hoverinfo="skip"))
            fig.update_layout(height=470, xaxis=dict(title=f"OPF LMP {bus_sel} ($/MWh)", range=[-5, 140]),
                              yaxis=dict(title="predicted ($/MWh)", range=[-5, 140]), plot_bgcolor="white",
                              margin=dict(t=30, l=10, r=10, b=10), showlegend=False,
                              title=f"{model_sel} vs OPF at {bus_sel} (6,000 random test scenarios; axes clipped at $140)")
            st.plotly_chart(fig, width="stretch")

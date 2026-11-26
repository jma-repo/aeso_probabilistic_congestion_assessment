"""Phase 5: calibrate the toy model with real AESO data (2024-2025).

Steps
-----
1. Pull (or read from cache) pool price, Alberta Internal Load (AIL), the asset
   list, the Current Supply Demand asset snapshot (fuel types) and hourly
   metered volumes of every WIND / SOLAR asset.
2. Data quality + coverage: hourly completeness on the UTC key, commissioning
   dates, wind/solar-like assets missing from the fuel snapshot, monthly fleet
   capacity and energy.
3. Real shapes: AIL (scaled across buses by the configured load shares), fleet
   wind and solar capacity factors (energy / capacity of assets in service in
   that hour).
4. Calibration test: with NO line limits (copper plate) the simulated system
   price should resemble the real pool price. Compare distributions, timing and
   shapes; then fit two transparent parameters on 2024 and test on 2025:
     * gas cost multiplier (fuel price level),
     * cogeneration must-run share (industrial steam keeps cogen online).
5. Re-run the network DC-OPF (with line limits) on the real shapes and compare
   congestion with the synthetic Phase 1 study.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from datetime import date

import numpy as np
import pandas as pd

from . import aeso_api as A
from . import plots
from .config import data_dir, reports_dir
from .features import merit_order_price
from .grid import load_buses
from .simulate import solve_dispatch, solve_many, summarize

log = logging.getLogger("aeso")
GAS = ("cogen", "ccgt", "gas_steam", "peaker")


# ---------------------------------------------------------------------------
# 1. data
# ---------------------------------------------------------------------------
def pull(cfg: dict, snapshot: str) -> dict:
    a = cfg["aeso"]
    client = A.AESOClient(cfg)
    cache = A.cache_for(cfg)
    t0 = time.perf_counter()
    out = {
        "pool": A.fetch_pool_price(client, cache, a["start"], a["end"]),
        "ail": A.fetch_ail(client, cache, a["start"], a["end"], a["tz"], a["ail_chunk_days"]),
        "assets": A.fetch_asset_list(client, cache, snapshot),
        "csd": A.fetch_csd_assets(client, cache, snapshot),
    }
    fleet = out["csd"][out["csd"].fuel_type.isin(a["metered_fuels"])].asset_ID.tolist()
    src = out["assets"][out["assets"].asset_type == "SOURCE"]
    # wind/solar-looking assets that the current fuel snapshot does not know (retired, small, renamed)
    cand = src[~src.asset_ID.isin(out["csd"].asset_ID) & src.asset_name.str.contains("wind|solar|pv", case=False)]
    out["candidates"] = cand
    out["metered"] = A.fetch_metered(client, cache, fleet, a["start"], a["end"], a["metered_max_records"])
    out["metered_candidates"] = A.fetch_metered(client, cache, cand.asset_ID.tolist(), a["start"], a["end"],
                                                a["metered_max_records"])
    out["api_calls"] = client.calls
    out["pull_seconds"] = time.perf_counter() - t0
    return out


# ---------------------------------------------------------------------------
# 2-3. quality, coverage, shapes
# ---------------------------------------------------------------------------
def hourly_index(cfg: dict) -> pd.DatetimeIndex:
    a = cfg["aeso"]
    return A.expected_hours(a["start"], a["end"], a["tz"])


def fleet_cf(metered: pd.DataFrame, csd: pd.DataFrame, fuel: str, idx: pd.DatetimeIndex) -> tuple[pd.Series, pd.DataFrame]:
    """Fleet capacity factor = metered energy / capacity of assets in service.

    An asset counts as in service from its first to its last hour with output
    > 0.01 MWh inside the period (so assets commissioned mid-period do not dilute
    the factor). Capacity = current CSD maximum capability (checked against the
    observed maximum output in the coverage report)."""
    ids = csd.loc[csd.fuel_type == fuel, "asset_ID"]
    cap = csd.set_index("asset_ID").maximum_capability_mw
    m = metered[metered.asset_ID.isin(ids)]
    gen = m.pivot_table(index="ts_utc", columns="asset_ID", values="metered_volume_mwh", aggfunc="sum").reindex(idx)
    nz = m[m.metered_volume_mwh > 0.01].groupby("asset_ID").ts_utc.agg(["min", "max"])
    online = pd.DataFrame(False, index=idx, columns=gen.columns)
    for aid, (lo, hi) in nz.iterrows():
        if aid in online:
            online.loc[(idx >= lo) & (idx <= hi), aid] = True
    cap_online = online.mul(cap.reindex(gen.columns).fillna(0), axis=1).sum(axis=1)
    energy = gen.where(online).sum(axis=1)
    cf = (energy / cap_online.replace(0, np.nan)).clip(0, 1).fillna(0.0)
    info = pd.DataFrame({"cap_online_mw": cap_online, "energy_mwh": energy, "assets_online": online.sum(axis=1)})
    return cf.rename(f"cf_{fuel.lower()}"), info


def coverage_report(cfg: dict, raw: dict, idx: pd.DatetimeIndex, info: dict[str, pd.DataFrame]) -> dict:
    a = cfg["aeso"]
    csd = raw["csd"]
    m = raw["metered"]
    cap = csd.set_index("asset_ID").maximum_capability_mw
    peak_ratio = (m.groupby("asset_ID").metered_volume_mwh.max() / cap).dropna()
    nz = m[m.metered_volume_mwh > 0.01].groupby("asset_ID").ts_utc.agg(["min", "max"])
    fuel = csd.set_index("asset_ID").fuel_type
    late = nz[nz["min"] > idx[0] + pd.Timedelta(days=14)].copy()
    late["fuel"] = late.index.map(fuel)
    late["capability_mw"] = late.index.map(cap)
    fleet_ids = csd[csd.fuel_type.isin(a["metered_fuels"])].asset_ID
    mc = raw["metered_candidates"]
    cand_energy = mc.groupby("asset_ID").metered_volume_mwh.sum()
    # group by LOCAL calendar month (UTC months would split local days and add a 7-hour stub month)
    monthly = pd.concat({f: i.tz_convert(a["tz"]).groupby(lambda t: t.strftime("%Y-%m"))
                         .agg({"cap_online_mw": "mean", "energy_mwh": "sum", "assets_online": "max"})
                         for f, i in info.items()}, axis=1)
    return {
        "pool_price": A.completeness(raw["pool"].ts_utc, a["start"], a["end"], a["tz"]),
        "ail": A.completeness(raw["ail"].ts_utc, a["start"], a["end"], a["tz"]),
        "pool_price_nan": int(raw["pool"].pool_price.isna().sum()),
        "ail_nan": int(raw["ail"].ail_mw.isna().sum()),
        "asset_list_rows": int(len(raw["assets"])),
        "csd_snapshot_utc": str(csd.snapshot_utc.iloc[0]),
        "fleet_assets": {f: int((csd.fuel_type == f).sum()) for f in a["metered_fuels"]},
        "fleet_capability_mw": {f: float(csd.loc[csd.fuel_type == f, "maximum_capability_mw"].sum()) for f in a["metered_fuels"]},
        "fleet_assets_with_data": int(m.asset_ID.nunique()),
        "fleet_assets_without_data": sorted(set(fleet_ids) - set(m.asset_ID)),
        "metered_rows": int(len(m)), "metered_negative": int((m.metered_volume_mwh < 0).sum()),
        "peak_output_to_capability": {"median": float(peak_ratio.median()), "min": float(peak_ratio.min()),
                                      "max": float(peak_ratio.max()),
                                      "above_1.05": {k: round(float(v), 2) for k, v in peak_ratio[peak_ratio > 1.05].items()}},
        "commissioned_during_period": late.assign(first=late["min"].astype(str))[["fuel", "capability_mw", "first"]]
                                       .reset_index().to_dict(orient="records"),
        "candidates_not_in_fuel_snapshot": int(len(raw["candidates"])),
        "candidates_with_output": {k: round(float(v) / 1e3, 1) for k, v in cand_energy[cand_energy > 0].items()},
        "candidates_energy_share_of_fleet": float(cand_energy.sum() / m.metered_volume_mwh.sum()),
        "monthly": monthly,
    }


def real_inputs(cfg: dict, raw: dict, idx: pd.DatetimeIndex, cf_wind: pd.Series, cf_solar: pd.Series) -> pd.DataFrame:
    """Model input frame on the real 2024-2025 hours (same columns as the synthetic one).

    Index: UTC hour-beginning (naive, named timestamp_utc). Calendar features use
    local America/Edmonton time from the tz database. Load = AIL x bus share.
    Wind: one fleet shape for both wind buses (no locations in the public data).
    Temperature is not available from these endpoints: columns are NaN and are
    not used by the OPF."""
    a = cfg["aeso"]
    ail = raw["ail"].set_index("ts_utc").ail_mw.reindex(idx)
    if ail.isna().any():
        ail = ail.interpolate(limit=3)
    local = idx.tz_convert(a["tz"])
    x = pd.DataFrame(index=idx.tz_convert("UTC").tz_localize(None))
    x.index.name = "timestamp_utc"
    x["hour"], x["dow"], x["month"] = local.hour, local.dayofweek, local.month
    x["doy"], x["is_weekend"] = local.dayofyear, (local.dayofweek >= 5).astype(int)
    for b in load_buses(cfg):
        x[f"temp_{b}"] = np.nan
        x[f"load_{b}"] = ail.to_numpy() * cfg["buses"][b]["load_share"]
    x["cf_wind_south"] = cf_wind.to_numpy()
    x["cf_wind_central"] = cf_wind.to_numpy()
    x["cf_solar_south"] = cf_solar.to_numpy()
    x["pu_hydro"] = np.asarray(cfg["hydro"]["monthly_pmax_pu"])[x["month"].to_numpy() - 1]
    return x


def shape_stats(name: str, s: pd.Series, hour: pd.Series) -> dict:
    v = s.to_numpy()
    return {"series": name, "mean": v.mean(), "std": v.std(), "p05": np.percentile(v, 5), "p95": np.percentile(v, 95),
            "lag1_autocorr": pd.Series(v).autocorr(1), "lag24_autocorr": pd.Series(v).autocorr(24),
            "share_below_5pct_of_max": float((v < 0.05 * v.max()).mean()),
            "peak_hour": int(pd.Series(v).groupby(hour.to_numpy()).mean().idxmax())}


# ---------------------------------------------------------------------------
# 4. calibration
# ---------------------------------------------------------------------------
def with_params(cfg: dict, gas_mult: float, cogen_must_run: float) -> dict:
    c = copy.deepcopy(cfg)
    for g in c["generators"].values():
        if g["carrier"] in GAS:
            g["marginal_cost"] = round(g["marginal_cost"] * gas_mult, 4)
        if g["carrier"] == "cogen":
            g["p_min_pu"] = cogen_must_run
    return c


QS = np.arange(0.01, 0.96, 0.01)  # fit the body of the distribution, not the offer-driven tail


def quantile_distance(sim: np.ndarray, real: np.ndarray, qs=QS) -> float:
    """Mean absolute difference of the quantile functions (≈ Wasserstein-1 on P1..P95)."""
    return float(np.mean(np.abs(np.quantile(sim, qs) - np.quantile(real, qs))))


def price_stats(name: str, p: pd.Series) -> dict:
    p = p.dropna()
    return {"series": name, "mean": p.mean(), "std": p.std(), "P10": p.quantile(0.1), "P50": p.median(),
            "P90": p.quantile(0.9), "P99": p.quantile(0.99), "max": p.max(),
            "share_le_1": (p <= 1).mean(), "share_ge_100": (p >= 100).mean(), "share_ge_500": (p >= 500).mean(),
            "share_ge_999": (p >= 999).mean()}


def calibrate_grid(cfg: dict, x: pd.DataFrame, real: pd.Series, fit_mask: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows = []
    for k in np.round(np.arange(0.30, 1.21, 0.05), 2):
        for m in np.round(np.arange(0.0, 0.91, 0.1), 1):
            sim = merit_order_price(with_params(cfg, k, m), x).to_numpy()
            rows.append({"gas_mult": k, "cogen_must_run": m,
                         "dist_fit": quantile_distance(sim[fit_mask], real[fit_mask]),
                         "dist_test": quantile_distance(sim[~fit_mask], real[~fit_mask])})
    g = pd.DataFrame(rows)
    best = g.loc[g.dist_fit.idxmin()].to_dict()
    return g, best


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
def phase5(cfg: dict, snapshot: str | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("aeso").setLevel(logging.WARNING)  # keep the console readable; requests stay logged at INFO in file mode
    snapshot = snapshot or cfg["aeso"].get("snapshot_date") or date.today().isoformat()
    rep = reports_dir(cfg)
    fig_dir, tab_dir = rep / "figures", rep / "tables"
    out_dir = data_dir(cfg) / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[phase 5] pulling AESO data {cfg['aeso']['start']} → {cfg['aeso']['end']} (cached chunks are skipped)")
    raw = pull(cfg, snapshot)
    print(f"   {raw['api_calls']} API calls in {raw['pull_seconds']:.1f}s")
    idx = hourly_index(cfg)
    cf_w, info_w = fleet_cf(raw["metered"], raw["csd"], "WIND", idx)
    cf_s, info_s = fleet_cf(raw["metered"], raw["csd"], "SOLAR", idx)
    cov = coverage_report(cfg, raw, idx, {"wind": info_w, "solar": info_s})
    x = real_inputs(cfg, raw, idx, cf_w, cf_s)
    x.to_parquet(out_dir / "real_inputs.parquet")
    real = raw["pool"].set_index("ts_utc").pool_price.reindex(idx)
    real.index = x.index
    print(f"   coverage: pool {cov['pool_price']['present_hours']}/{cov['pool_price']['expected_hours']} h, "
          f"AIL {cov['ail']['present_hours']}/{cov['ail']['expected_hours']} h, "
          f"{cov['fleet_assets_with_data']} wind/solar assets with data")

    # synthetic counterparts (Phase 1)
    syn = pd.read_parquet(data_dir(cfg) / "phase1" / "inputs.parquet")
    syn_load = syn[[f"load_{b}" for b in load_buses(cfg)]].sum(axis=1)
    real_load = x[[f"load_{b}" for b in load_buses(cfg)]].sum(axis=1)
    shapes = pd.DataFrame([
        shape_stats("load MW · real AIL", real_load, x.hour), shape_stats("load MW · synthetic", syn_load, syn.hour),
        shape_stats("wind CF · real fleet", x.cf_wind_south, x.hour), shape_stats("wind CF · synthetic", syn.cf_wind_south, syn.hour),
        shape_stats("solar CF · real fleet", x.cf_solar_south, x.hour), shape_stats("solar CF · synthetic", syn.cf_solar_south, syn.hour),
    ]).set_index("series")
    corr = {"wind_vs_load_real": float(np.corrcoef(x.cf_wind_south, real_load)[0, 1]),
            "wind_vs_load_synthetic": float(np.corrcoef(syn.cf_wind_south, syn_load)[0, 1])}

    # calibration test: copper plate (merit order == copper-plate DC-OPF, exact)
    print("[phase 5] copper-plate price vs real pool price; 2-parameter calibration (fit 2024, test 2025)")
    year = x.index.tz_localize("UTC").tz_convert(cfg["aeso"]["tz"]).year
    fit = np.asarray(year == 2024)
    p_base = merit_order_price(cfg, x).rename("model · base costs")
    grid, best = calibrate_grid(cfg, x, real.to_numpy(), fit)
    ccfg = with_params(cfg, best["gas_mult"], best["cogen_must_run"])
    p_cal = merit_order_price(ccfg, x).rename("model · calibrated")
    syn_copper = pd.read_parquet(data_dir(cfg) / "phase1" / "copper_lmp.parquet").iloc[:, 0].rename("model · synthetic inputs")
    # the analytic merit order must equal the copper-plate LP, also with must-run: check 336 hours
    chk = solve_many(ccfg, x.iloc[:336], line_limits=False)
    copper_check = float((chk.lmp.iloc[:, 0] - p_cal.iloc[:336]).abs().max())

    ps = []
    for nm, p, m in (("real pool price", real, None), ("model · base costs", p_base, None), ("model · calibrated", p_cal, None)):
        for period, mask in (("2024 (fit)", fit), ("2025 (test)", ~fit), ("2024-2025", np.ones(len(x), bool))):
            ps.append({**price_stats(nm, p[mask]), "period": period})
    ps.append({**price_stats("model · synthetic inputs", syn_copper), "period": "synthetic 2 yr"})
    price_tab = pd.DataFrame(ps).set_index(["series", "period"])
    timing = {}
    for nm, p in (("base", p_base), ("calibrated", p_cal)):
        timing[nm] = {
            "pearson_all": float(np.corrcoef(p, real)[0, 1]),
            "spearman_all": float(pd.Series(p.to_numpy()).corr(pd.Series(real.to_numpy()), method="spearman")),
            "spearman_test_2025": float(pd.Series(p[~fit].to_numpy()).corr(pd.Series(real[~fit].to_numpy()), method="spearman")),
            "mae_test_2025": float(np.abs(p[~fit] - real[~fit]).mean()),
            "median_abs_err_test_2025": float(np.abs(p[~fit] - real[~fit]).median()),
            "quantile_distance_test_2025": quantile_distance(p[~fit].to_numpy(), real[~fit].to_numpy()),
        }
    hod = pd.DataFrame({"real pool price": real.groupby(x.hour).mean(), "model · base costs": p_base.groupby(x.hour).mean(),
                        "model · calibrated": p_cal.groupby(x.hour).mean()})
    mon_idx = x.index.tz_localize("UTC").tz_convert(cfg["aeso"]["tz"]).strftime("%Y-%m")
    monthly = pd.DataFrame({"real pool price": real.groupby(mon_idx).mean(), "model · calibrated": p_cal.groupby(mon_idx).mean(),
                            "model · base costs": p_base.groupby(mon_idx).mean()})

    # network OPF with real shapes (base costs = the Phase 1-4 model; calibrated = with fitted parameters)
    print("[phase 5] network DC-OPF on real shapes (base and calibrated costs)")
    syn_bind = pd.read_parquet(data_dir(cfg) / "phase1" / "binding.parquet")
    cong = {"synthetic inputs (Phase 1)": syn_bind.mean(), }
    opf = {}
    for nm, c in (("real shapes · base costs", cfg), ("real shapes · calibrated", ccfg)):
        r = solve_dispatch(c, x)
        opf[nm] = r
        cong[nm] = r.binding.mean()
    cong_tab = pd.DataFrame(cong)
    any_bind = {k: float(r.binding.any(axis=1).mean()) for k, r in opf.items()}
    any_bind["synthetic inputs (Phase 1)"] = float(syn_bind.any(axis=1).mean())
    for nm, r in opf.items():
        slug = "base" if "base" in nm else "calibrated"
        for k, df in r.frames().items():
            df.to_parquet(out_dir / f"opf_{slug}_{k}.parquet")
    lmp_real_cal = opf["real shapes · calibrated"].lmp

    # ---------------------------------------------------------------- outputs
    for nm, t in (("phase5_price_stats", price_tab), ("phase5_shapes", shapes), ("phase5_calibration_grid", grid),
                  ("phase5_congestion", cong_tab), ("phase5_price_by_hour", hod), ("phase5_price_by_month", monthly),
                  ("phase5_coverage_monthly", cov["monthly"])):
        t.to_csv(tab_dir / f"{nm}.csv")
    pd.DataFrame({"pool_price": real, "ail_mw": real_load, "cf_wind": x.cf_wind_south, "cf_solar": x.cf_solar_south,
                  "model_base": p_base, "model_calibrated": p_cal}).to_parquet(out_dir / "hourly_compare.parquet")

    print("[phase 5] figures")
    figs = [
        plots.p5_coverage(cov["monthly"], fig_dir),
        plots.p5_shapes(x, syn, real_load, syn_load, fig_dir),
        plots.p5_price_duration({"real pool price": real, "model · base costs": p_base, "model · calibrated": p_cal}, fig_dir),
        plots.p5_price_profiles(hod, monthly, fig_dir),
        plots.p5_calibration_surface(grid, best, fig_dir),
        plots.p5_congestion(cong_tab, fig_dir),
    ]
    summary = {
        "period": [cfg["aeso"]["start"], cfg["aeso"]["end"]], "snapshot": snapshot,
        "api_calls_this_run": raw["api_calls"], "pull_seconds": raw["pull_seconds"],
        "coverage": {k: v for k, v in cov.items() if k != "monthly"},
        "calibration": {"best": best, "fit_period": "2024", "test_period": "2025",
                        "objective": "mean |quantile difference| over P1..P95 of hourly prices",
                        "copper_lp_check_max_abs_diff": copper_check},
        "timing": timing, "correlations": corr, "any_binding": any_bind,
        "lmp_mean_real_shapes_calibrated": lmp_real_cal.mean().to_dict(),
        "figures": [str(p.relative_to(rep.parent)) for p in figs],
    }
    with open(rep / "phase5_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    (rep / "phase5_summary.md").write_text(markdown(summary, price_tab, shapes, cong_tab, hod, cov, grid))
    store_db(cfg, x, real, raw, opf["real shapes · calibrated"])
    print(f"[phase 5] done -> {rep / 'phase5_summary.md'}")
    return summary


def store_db(cfg: dict, x: pd.DataFrame, real: pd.Series, raw: dict, res) -> None:
    """Add the real-shape study and the observed hourly AESO series to the existing database."""
    from . import db
    eng = db.get_engine(cfg)
    db.create_schema(eng)  # creates aeso_hourly if missing, keeps existing data
    rid = db.latest_run_id(eng)
    if rid is None:
        rid = db.new_run(eng, cfg, "phase5 only")
        db.write_network(eng, cfg)
    with eng.begin() as c:
        for t in (db.scenario_bus_input, db.bus_result, db.line_result, db.gen_dispatch, db.scenario):
            c.execute(t.delete().where((t.c.run_id == rid) & (t.c.study == "real_2024_2025")))
        c.execute(db.aeso_hourly.delete())
    db.write_study(eng, cfg, rid, "real_2024_2025", x, res, store_dispatch=False)
    ail = raw["ail"].set_index("ts_utc")
    obs = pd.DataFrame({"hour_start_utc": x.index.to_pydatetime(), "pool_price": real.to_numpy(),
                        "forecast_pool_price": raw["pool"].set_index("ts_utc").forecast_pool_price.reindex(
                            x.index.tz_localize("UTC")).to_numpy(),
                        "ail_mw": ail.ail_mw.reindex(x.index.tz_localize("UTC")).to_numpy(),
                        "wind_cf": x.cf_wind_south.to_numpy(), "solar_cf": x.cf_solar_south.to_numpy()})
    db._append(eng, db.aeso_hourly, obs)


def _md(df: pd.DataFrame, fmt: dict | None = None, default="{:,.2f}") -> str:
    fmt = fmt or {}
    names = list(df.index.names) if isinstance(df.index, pd.MultiIndex) else [df.index.name or ""]
    head = names + list(df.columns)
    L = ["| " + " | ".join(map(str, head)) + " |", "|" + "---|" * len(head)]
    for idx, r in df.iterrows():
        cells = list(idx) if isinstance(idx, tuple) else [idx]
        for c in df.columns:
            v = r[c]
            cells.append("—" if pd.isna(v) else (fmt.get(c, default).format(v) if not isinstance(v, str) else v))
        L.append("| " + " | ".join(map(str, cells)) + " |")
    return "\n".join(L)


def markdown(s, price_tab, shapes, cong_tab, hod, cov, grid) -> str:
    c = s["coverage"]
    b = s["calibration"]["best"]
    pct = "{:.1%}"
    L = ["# Phase 5 summary (auto-generated by `python -m src.run --phases 5`)", "",
         f"Period {s['period'][0]} → {s['period'][1]} (local dates), keyed on UTC hours. Asset/fuel snapshot {s['snapshot']}.", "",
         "## Data quality and coverage", "",
         f"* Pool price: {c['pool_price']['present_hours']:,}/{c['pool_price']['expected_hours']:,} hours, "
         f"{c['pool_price']['duplicate_rows']} duplicates, {c['pool_price_nan']} empty values",
         f"* AIL: {c['ail']['present_hours']:,}/{c['ail']['expected_hours']:,} hours, {c['ail']['duplicate_rows']} duplicates, "
         f"{c['ail_nan']} empty values",
         f"* Fuel snapshot: {c['fleet_assets']} assets, capability {c['fleet_capability_mw']} MW; "
         f"{c['fleet_assets_with_data']} with metered data; without data: {c['fleet_assets_without_data']}",
         f"* Metered rows {c['metered_rows']:,}; negative values {c['metered_negative']}",
         f"* Peak output / CSD capability per asset: median {c['peak_output_to_capability']['median']:.2f}, "
         f"min {c['peak_output_to_capability']['min']:.2f}, max {c['peak_output_to_capability']['max']:.2f}; "
         f"above 1.05: {c['peak_output_to_capability']['above_1.05']}",
         f"* Wind/solar-like assets missing from the fuel snapshot: {c['candidates_not_in_fuel_snapshot']}; with output in the "
         f"period (GWh): {c['candidates_with_output']} = **{c['candidates_energy_share_of_fleet']:.2%}** of fleet energy",
         f"* Assets commissioned during the period: {len(c['commissioned_during_period'])}", "",
         _md(pd.DataFrame(c["commissioned_during_period"]).set_index("asset_ID"), {"capability_mw": "{:,.0f}"}), "",
         "## Real vs synthetic shapes", "",
         _md(shapes, {"share_below_5pct_of_max": pct, "peak_hour": "{:.0f}"}), "",
         f"Correlation wind CF vs load: real {s['correlations']['wind_vs_load_real']:.3f}, "
         f"synthetic {s['correlations']['wind_vs_load_synthetic']:.3f}", "",
         "## Price calibration (copper plate, no line limits)", "",
         f"Two parameters fitted on 2024 by minimising {s['calibration']['objective']}: "
         f"**gas cost multiplier {b['gas_mult']:.2f}**, **cogeneration must-run {b['cogen_must_run']:.0%}** "
         f"(fit distance {b['dist_fit']:.2f} $/MWh, test-2025 distance {b['dist_test']:.2f} $/MWh). "
         f"Analytic merit order vs copper-plate LP with must-run: max diff {s['calibration']['copper_lp_check_max_abs_diff']:.1e} $/MWh.", "",
         _md(price_tab, {k: pct for k in ("share_le_1", "share_ge_100", "share_ge_500", "share_ge_999")}), "",
         "Timing (hour-by-hour agreement):", "",
         _md(pd.DataFrame(s["timing"]).T, {"mae_test_2025": "{:.1f}", "median_abs_err_test_2025": "{:.1f}"}), "",
         "Mean price by local hour of day:", "", _md(hod), "",
         "## Congestion with real shapes (network DC-OPF, share of hours at limit)", "",
         _md(cong_tab, {k: pct for k in cong_tab.columns}), "",
         "Any line at limit: " + ", ".join(f"{k} {v:.1%}" for k, v in s["any_binding"].items()), ""]
    return "\n".join(L) + "\n"

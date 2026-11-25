"""Phase 3 pipeline: train surrogates, evaluate, and rerun the Monte Carlo with them.

Two learning tasks:
  * MC task (the main result): train on a separate Monte Carlo scenario set
    (different seed), early-stop on its last 15 %, and test on the Phase 2
    study scenarios, which the models never see. Then replace the OPF by the
    surrogate in the Phase 2 study and compare congestion probabilities,
    P10/P50/P90 and runtime.
  * Hourly task: train on 2025, validate on 2026 H1, test on 2026 H2 (7-day
    embargo at each boundary). No outages in this data.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import plots
from .config import data_dir, reports_dir
from .features import build_features, hourly_split, scenario_split, targets
from .models import LGBMCongestion, LGBMPerBus, MeritOrderBaseline, MLPSurrogate, PriceSurrogate
from .montecarlo import Scenarios, run_monte_carlo, wilson
from .simulate import DispatchResult


def _load_scen(folder, prefix):
    return Scenarios(*(pd.read_parquet(folder / f"{prefix}{k}.parquet") for k in ("inputs", "line_out", "gen_avail")))


def _make_models(cfg: dict, buses: list[str], *, hybrid=None, lgbm_obj=None, mlp_loss=None, threads=4) -> dict:
    s = cfg["surrogate"]
    hybrid = s["hybrid"] if hybrid is None else hybrid
    lp = dict(s["lgbm"])
    if lgbm_obj is not None:
        lp["objective"] = lgbm_obj
        if lgbm_obj != "huber":
            lp.pop("alpha", None)
    mp_ = dict(s["mlp"])
    if mlp_loss is not None:
        mp_["loss"] = mlp_loss
    return {
        "baseline (copper plate)": MeritOrderBaseline(buses),
        "LightGBM": PriceSurrogate(LGBMPerBus(lp), hybrid, "LightGBM"),
        "MLP": PriceSurrogate(MLPSurrogate(threads=threads, **mp_), hybrid, "MLP"),
    }


def _time(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return out, best


def phase3(cfg: dict, phase2_runner=None) -> dict:
    s = cfg["surrogate"]
    d = data_dir(cfg)
    out_data = d / "phase3"
    out_data.mkdir(parents=True, exist_ok=True)
    rep = reports_dir(cfg)
    fig_dir, tab_dir = rep / "figures", rep / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    from .run import load_result, save_result  # avoid import cycle

    if not (d / "phase2" / "mc_lmp.parquet").exists():
        phase2_runner(cfg)
    buses = list(cfg["buses"])
    base_inputs = pd.read_parquet(d / "phase1" / "inputs.parquet")
    test_s, test_r = _load_scen(d / "phase2", "mc_"), load_result(d / "phase2", "mc_")
    p2 = json.load(open(rep / "phase2_summary.json"))["monte_carlo"]

    # ------------------------------------------------------------------ MC training data
    n_train = s["n_train_scenarios"]
    print(f"[phase 3] generating MC training set: {n_train:,} OPF solves (seed offset {s['train_seed_offset']})")
    t0 = time.perf_counter()
    train_s, train_r = run_monte_carlo(cfg, base_inputs, n=n_train, seed=cfg["seed"] + s["train_seed_offset"])
    train_opf_seconds = time.perf_counter() - t0
    for k in ("inputs", "line_out", "gen_avail"):
        getattr(train_s, k).to_parquet(out_data / f"train_{k}.parquet")
    save_result(train_r, out_data, "train_")
    print(f"   {train_opf_seconds:.1f}s")

    Xa = build_features(cfg, train_s.inputs, train_s.line_out, train_s.gen_avail)
    Ya, Ba = targets(train_r)
    sp = scenario_split(len(Xa), s["val_share"])
    tr, va = sp["train"], sp["val"]
    Xt = build_features(cfg, test_s.inputs, test_s.line_out, test_s.gen_avail)
    Yt, Bt = targets(test_r)
    cong_t = Bt.any(axis=1)

    # ------------------------------------------------------------------ main models
    print("[phase 3] fitting surrogates on MC training data")
    models = _make_models(cfg, buses)
    for name, m in models.items():
        m.fit(Xa[tr], Ya[tr], Xa[va], Ya[va])
        print(f"   {name:<26} fit {m.fit_seconds:6.1f}s")
    clf = LGBMCongestion(s["lgbm_classifier"]).fit(Xa[tr], Ba[tr], Xa[va], Ba[va])
    print(f"   {'congestion classifier':<26} fit {clf.fit_seconds:6.1f}s")

    # ------------------------------------------------------------------ surrogate Monte Carlo
    print("[phase 3] Monte Carlo with surrogates (timed)")
    timings = {}
    feats, timings["features"] = _time(lambda: build_features(cfg, test_s.inputs, test_s.line_out, test_s.gen_avail))
    preds = {}
    for name, m in models.items():
        preds[name], timings[name] = _time(lambda m=m: m.predict(feats))
        if name != "baseline (copper plate)":
            _, timings[name + " 1 thread"] = _time(lambda m=m: m.predict(feats, threads=1))
    proba, timings["classifier"] = _time(lambda: clf.predict_proba(feats))
    _, timings["classifier 1 thread"] = _time(lambda: clf.predict_proba(feats, threads=1))
    import torch
    torch.set_num_threads(4)

    import pickle
    with open(out_data / "models.pkl", "wb") as f:  # local artefact (data/ is git-ignored)
        pickle.dump({"models": models, "classifier": clf, "feature_columns": list(Xa.columns)}, f)
    for name, P in preds.items():
        P.to_parquet(out_data / f"mc_test_pred_{name.split(' ')[0].lower()}.parquet")
    proba.to_parquet(out_data / "mc_test_congestion_proba.parquet")

    runtime = pd.DataFrame({"seconds": {
        "DC-OPF, 4 workers (Phase 2 wall)": p2["wall_seconds"],
        "DC-OPF, 1 core (sum of LP times)": p2["sum_lp_seconds"],
        "LightGBM + classifier, 4 threads": timings["features"] + timings["LightGBM"] + timings["classifier"],
        "LightGBM + classifier, 1 thread": timings["features"] + timings["LightGBM 1 thread"] + timings["classifier 1 thread"],
        "MLP + classifier, 4 threads": timings["features"] + timings["MLP"] + timings["classifier"],
        "baseline (merit order)": timings["features"],
    }})
    runtime["speed-up vs OPF 4 workers"] = p2["wall_seconds"] / runtime["seconds"]
    runtime["ms_per_scenario"] = 1000 * runtime["seconds"] / len(Xt)
    fit_total = sum(m.fit_seconds for m in models.values()) + clf.fit_seconds
    lgbm_cost = models["LightGBM"].fit_seconds + clf.fit_seconds
    per_scen_opf = p2["wall_seconds"] / len(Xt)
    per_scen_sur = runtime.loc["LightGBM + classifier, 4 threads", "seconds"] / len(Xt)
    break_even = (train_opf_seconds + lgbm_cost) / (per_scen_opf - per_scen_sur)

    # ------------------------------------------------------------------ accuracy
    reg = ev.regression_table(Yt, preds, cong_t)
    reg_sum = ev.summary_by_model(reg)
    masks = ev.regimes(Bt, Yt, test_s.line_out)
    regime = ev.regime_table(Yt, preds, masks)
    cls = ev.classification_table(Bt, proba, s["classifier_threshold"])
    mc_cong = ev.mc_congestion_comparison(Bt, proba, s["classifier_threshold"])
    mc_q = ev.mc_quantile_comparison(Yt, preds)
    worst = {n: ev.worst_cases(Yt, preds[n], Xt, Bt) for n in ("LightGBM", "MLP")}
    any_int = [l for l in Bt if l != "BC-CAL"]
    any_opf = Bt[any_int].any(axis=1).mean()
    # P(any internal line binding) from the classifier assuming independence is wrong; use thresholded union
    any_sur = (proba[any_int] >= s["classifier_threshold"]).any(axis=1).mean()

    # ------------------------------------------------------------------ ablation
    ablation = None
    if s.get("ablation"):
        print("[phase 3] ablation: target (raw / hybrid) x loss (L2 / Huber)")
        rows = []
        for hybrid in (False, True):
            for robust in (False, True):
                ms = _make_models(cfg, buses, hybrid=hybrid, lgbm_obj="huber" if robust else "regression",
                                  mlp_loss="huber" if robust else "mse")
                for nm in ("LightGBM", "MLP"):
                    m = ms[nm].fit(Xa[tr], Ya[tr], Xa[va], Ya[va])
                    r = ev.summary_by_model(ev.regression_table(Yt, {nm: m.predict(Xt)}, cong_t)).iloc[0]
                    rows.append({"model": nm, "target": "hybrid (LMP − merit price)" if hybrid else "raw LMP",
                                 "loss": "Huber" if robust else "L2", **r.to_dict(), "fit_seconds": m.fit_seconds})
        ablation = pd.DataFrame(rows)

    # ------------------------------------------------------------------ learning curve
    print("[phase 3] learning curve")
    lc_rows = []
    idx_tr = np.flatnonzero(tr)
    for n in s["learning_curve_sizes"]:
        n = min(n, len(idx_tr))
        sub = Xa.index[idx_tr[:n]]
        m = _make_models(cfg, buses)["LightGBM"].fit(Xa.loc[sub], Ya.loc[sub], Xa[va], Ya[va])
        r = ev.summary_by_model(ev.regression_table(Yt, {"m": m.predict(Xt)}, cong_t)).iloc[0]
        c = LGBMCongestion(s["lgbm_classifier"]).fit(Xa.loc[sub], Ba.loc[sub], Xa[va], Ba[va])
        pc = c.predict_proba(Xt)
        lc_rows.append({"n_train": n, "MAE": r["MAE"], "MAE_congested": r["MAE_congested"],
                        "max_abs_err_congestion_prob_pp": float((pc.mean() - Bt.mean()).abs().max() * 100)})
    lc = pd.DataFrame(lc_rows)

    # ------------------------------------------------------------------ hourly task
    print("[phase 3] hourly task (time split)")
    base_res = load_result(d / "phase1")
    Xh = build_features(cfg, base_inputs)
    Yh, Bh = targets(base_res)
    hs = hourly_split(Xh.index, s["embargo_days"])
    hmodels = _make_models(cfg, buses)
    hpreds = {}
    for name, m in hmodels.items():
        m.fit(Xh[hs["train"]], Yh[hs["train"]], Xh[hs["val"]], Yh[hs["val"]])
        hpreds[name] = m.predict(Xh[hs["test"]])
    hclf = LGBMCongestion(s["lgbm_classifier"]).fit(Xh[hs["train"]], Bh[hs["train"]], Xh[hs["val"]], Bh[hs["val"]])
    Yht, Bht = Yh[hs["test"]], Bh[hs["test"]]
    h_reg = ev.regression_table(Yht, hpreds, Bht.any(axis=1))
    h_sum = ev.summary_by_model(h_reg)
    h_regime = ev.regime_table(Yht, hpreds, ev.regimes(Bht, Yht, None))
    h_cls = ev.classification_table(Bht, hclf.predict_proba(Xh[hs["test"]]), s["classifier_threshold"])
    h_periods = {k: [str(Xh.index[v][0].date()), str(Xh.index[v][-1].date()), int(v.sum())] for k, v in hs.items()}

    # ------------------------------------------------------------------ SHAP + failure analysis
    print("[phase 3] SHAP and step sweeps")
    lgbm = models["LightGBM"].model
    shap_rows = Xt.iloc[:4000]
    shap_abs, shap_vals = {}, {}
    for b in ("SOUTH", "CAL"):
        sv = lgbm.shap_values(shap_rows, f"lmp_{b}")
        shap_vals[b] = sv
        shap_abs[b] = sv.abs().mean()
    pd.DataFrame(shap_abs).to_csv(tab_dir / "surrogate_shap_mean_abs.csv")

    # a test-period hour with no outages, median load, and a non-trivial wind level
    hrs = base_inputs[hs["test"]]
    tot = hrs.filter(like="load_").sum(axis=1)
    cand = hrs[(tot - tot.median()).abs() < 150]
    row = cand.iloc[len(cand) // 2]
    n_pts = s["step_sweep_points"]
    sweep_models = {"baseline (copper plate)": hmodels["baseline (copper plate)"], "LightGBM": models["LightGBM"],
                    "MLP": models["MLP"]}
    sw_wind = ev.step_sweep(cfg, row, "cf_wind_south", np.linspace(0, 1, n_pts), sweep_models, ["SOUTH", "CAL"])
    row_low_wind = row.copy()
    row_low_wind["cf_wind_south"] = 0.15
    sw_load = ev.step_sweep(cfg, row_low_wind, "load_scale", np.linspace(0.8, 1.3, n_pts),
                            sweep_models, ["CAL", "NORTH"], scale_loads=True)
    sw_wind.to_csv(tab_dir / "surrogate_sweep_wind.csv", index=False)
    sw_load.to_csv(tab_dir / "surrogate_sweep_load.csv", index=False)
    sweep_err = {}
    for nm, sw in (("wind", sw_wind), ("load", sw_load)):
        bcols = [c.removeprefix("OPF_") for c in sw if c.startswith("OPF_")]
        sweep_err[nm] = {m: float(np.mean([np.abs(sw[f"{m}_{b}"] - sw[f"OPF_{b}"]).mean() for b in bcols]))
                         for m in sweep_models}

    # ------------------------------------------------------------------ figures
    print("[phase 3] figures")
    main_preds = {"LightGBM": preds["LightGBM"], "MLP": preds["MLP"], "baseline (copper plate)": preds["baseline (copper plate)"]}
    figs = [
        plots.p3_mc_congestion(mc_cong, fig_dir),
        plots.p3_ecdf_compare(Yt, main_preds, fig_dir),
        plots.p3_regime_mae(regime, fig_dir),
        plots.p3_regime_mae(h_regime, fig_dir, name="surrogate_mae_by_regime_hourly",
                            title="Hourly task (2026 H2 test): error by operating regime"),
        plots.p3_step_sweep(sw_wind, list(sweep_models), ["SOUTH", "CAL"], "South wind capacity factor", fig_dir,
                            "surrogate_step_wind", f"1-D sweep of South wind, hour {row.name:%Y-%m-%d %H:00}", pct=True),
        plots.p3_step_sweep(sw_load, list(sweep_models), ["CAL", "NORTH"], "System load (× actual)", fig_dir,
                            "surrogate_step_load", f"1-D sweep of system load, same hour, wind CF 15%", pct=True),
        plots.p3_shap(shap_abs, fig_dir),
        plots.p3_shap_dependence(shap_rows["surplus_SOUTH"], shap_vals["SOUTH"]["surplus_SOUTH"],
                                 shap_rows["out_SOUTH-RD"] + shap_rows["out_SOUTH-CAL"], fig_dir,
                                 "South surplus: available wind + solar − South load (MW)", "southern line out"),
        plots.p3_learning_curve(lc, fig_dir),
        plots.p3_runtime(runtime[["seconds"]], fig_dir),
        plots.p3_pred_vs_true(Yt, {"LightGBM": preds["LightGBM"], "MLP": preds["MLP"]}, "SOUTH", fig_dir),
    ]

    # ------------------------------------------------------------------ outputs
    reg.to_csv(tab_dir / "surrogate_mc_regression_by_bus.csv", index=False)
    regime.to_csv(tab_dir / "surrogate_mc_regimes.csv")
    cls.to_csv(tab_dir / "surrogate_mc_classifier.csv")
    mc_cong.to_csv(tab_dir / "surrogate_mc_congestion.csv")
    mc_q.to_csv(tab_dir / "surrogate_mc_quantiles.csv", index=False)
    runtime.to_csv(tab_dir / "surrogate_runtime.csv")
    lc.to_csv(tab_dir / "surrogate_learning_curve.csv", index=False)
    h_reg.to_csv(tab_dir / "surrogate_hourly_regression_by_bus.csv", index=False)
    if ablation is not None:
        ablation.to_csv(tab_dir / "surrogate_ablation.csv", index=False)

    lo, hi = wilson(Bt[any_int].any(axis=1).sum(), len(Bt))
    summary = {
        "mc_task": {
            "n_train_scenarios": int(tr.sum()), "n_val_scenarios": int(va.sum()), "n_test_scenarios": len(Xt),
            "n_features": Xt.shape[1], "train_opf_seconds": train_opf_seconds,
            "fit_seconds": {**{k: m.fit_seconds for k, m in models.items()}, "classifier": clf.fit_seconds},
            "mlp_best_epoch": models["MLP"].model.best_epoch,
            "summary_by_model": reg_sum.reset_index().to_dict(orient="records"),
            "any_internal_binding": {"opf": float(any_opf), "opf_ci95": [float(lo), float(hi)], "surrogate_thresholded_union": float(any_sur)},
            "break_even_scenarios": float(break_even),
        },
        "runtime": runtime.reset_index(names="method").to_dict(orient="records"),
        "hourly_task": {"periods": h_periods, "summary_by_model": h_sum.reset_index().to_dict(orient="records")},
        "sweep_mae": sweep_err,
        "sweep_hour": str(row.name),
        "figures": [str(p.relative_to(rep.parent)) for p in figs],
    }
    with open(rep / "phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    (rep / "phase3_summary.md").write_text(markdown(summary, reg_sum, regime, cls, mc_cong, mc_q, runtime, ablation, lc,
                                                     h_sum, h_regime, h_cls, worst, pd.DataFrame(shap_abs)))
    print(f"[phase 3] done -> {rep / 'phase3_summary.md'}")
    return summary


def _md(df: pd.DataFrame, fmt: dict | None = None, default: str = "{:,.2f}", index=True) -> str:
    fmt = fmt or {}
    df = df.copy()
    cols = list(df.columns)
    head = ([df.index.name or ""] if index else []) + cols
    lines = ["| " + " | ".join(map(str, head)) + " |", "|" + "---|" * len(head)]
    for idx, r in df.iterrows():
        cells = [str(idx)] if index else []
        for c in cols:
            v = r[c]
            if isinstance(v, (str, bool, np.bool_)):
                cells.append(str(v))
            elif pd.isna(v):
                cells.append("—")
            else:
                cells.append(fmt.get(c, default).format(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def markdown(s, reg_sum, regime, cls, mc_cong, mc_q, runtime, ablation, lc, h_sum, h_regime, h_cls, worst, shap_abs) -> str:
    mc = s["mc_task"]
    pct = "{:.1%}"
    L = ["# Phase 3 summary (auto-generated by `python -m src.run`)", "",
         "## Monte Carlo task (main result)", "",
         f"* Training data: {mc['n_train_scenarios']:,} + {mc['n_val_scenarios']:,} validation MC scenarios from a "
         f"separate seed ({mc['train_opf_seconds']:.1f} s of OPF). Test: the {mc['n_test_scenarios']:,} Phase 2 study "
         f"scenarios, never seen in training. {mc['n_features']} features.",
         f"* Fit times (s): " + ", ".join(f"{k} {v:.1f}" for k, v in mc["fit_seconds"].items()),
         "", "### Accuracy on the test scenarios (mean over buses, $/MWh)", "",
         _md(reg_sum),
         "", "### Error by operating regime (MAE, mean over buses, $/MWh)", "",
         _md(regime, {"share_of_rows": pct}),
         "", "### Monte Carlo statistics: surrogate vs full OPF", "",
         "Per-line congestion probability (classifier; mean predicted probability and thresholded at 0.5):", "",
         _md(mc_cong, {c: "{:.2%}" for c in ("opf_p", "opf_ci95_low", "opf_ci95_high", "surrogate_p_mean_prob", "surrogate_p_thresholded")}
             | {"abs_err_pp": "{:.2f}", "opf_expected_h_per_year": "{:,.0f}", "surrogate_expected_h_per_year": "{:,.0f}"}),
         "",
         f"* P(≥1 internal line at limit): OPF {mc['any_internal_binding']['opf']:.2%} "
         f"[{mc['any_internal_binding']['opf_ci95'][0]:.2%}, {mc['any_internal_binding']['opf_ci95'][1]:.2%}]; "
         f"surrogate (union of thresholded classifiers) {mc['any_internal_binding']['surrogate_thresholded_union']:.2%}",
         "", "LMP quantiles ($/MWh):", "",
         _md(mc_q.set_index(["bus", "stat"])),
         "", "### Classifier quality per line", "",
         _md(cls, {"base_rate": "{:.2%}", "mean_predicted_prob": "{:.2%}", "n_positive": "{:,.0f}", "Brier": "{:.4f}"}),
         "", "### Runtime for the 20,000-scenario study", "",
         _md(runtime, {"seconds": "{:.3g}", "speed-up vs OPF 4 workers": "{:,.0f}×", "ms_per_scenario": "{:.4f}"}),
         "",
         f"* Up-front cost of the surrogate: {mc['train_opf_seconds']:.0f} s of training OPFs + "
         f"{mc['fit_seconds']['LightGBM'] + mc['fit_seconds']['classifier']:.0f} s fitting (LightGBM + classifier). "
         f"Break-even vs re-running the OPF Monte Carlo: **≈{mc['break_even_scenarios']:,.0f} scenarios**.",
         ""]
    if ablation is not None:
        L += ["### Ablation: target and loss (MC test set, $/MWh)", "",
              _md(ablation.set_index(["model", "target", "loss"]), {"fit_seconds": "{:.1f}"}), ""]
    L += ["### Learning curve (LightGBM hybrid + classifier)", "",
          _md(lc.set_index("n_train"), {"max_abs_err_congestion_prob_pp": "{:.2f}"}), "",
          "### Largest errors", ""]
    for n, w in worst.items():
        L += [f"{n}:", "", _md(w.set_index("row")), ""]
    L += ["### TreeSHAP: mean |SHAP| of the congestion component ($/MWh), top 10", "",
          _md(shap_abs.sort_values("SOUTH", ascending=False).head(10)), "",
          "## Hourly task (time split, no outages)", "",
          "Periods: " + "; ".join(f"{k} {v[0]} → {v[1]} ({v[2]:,} h)" for k, v in s["hourly_task"]["periods"].items()), "",
          _md(h_sum), "", _md(h_regime, {"share_of_rows": pct}), "",
          _md(h_cls, {"base_rate": "{:.2%}", "mean_predicted_prob": "{:.2%}", "n_positive": "{:,.0f}", "Brier": "{:.4f}"}), "",
          "## 1-D step sweeps (MAE vs OPF along the sweep, $/MWh)", "",
          f"Base hour {s['sweep_hour']}.", "",
          _md(pd.DataFrame(s["sweep_mae"]).T), ""]
    return "\n".join(L) + "\n"

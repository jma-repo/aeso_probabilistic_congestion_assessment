"""Metrics and failure analysis for the surrogates.

Everything here compares surrogate outputs with the DC-OPF "truth" on data the
models never saw during fitting or early stopping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score

from .features import build_features
from .montecarlo import wilson
from .simulate import solve_block

INTERTIE = "BC-CAL"


def regression_table(Y: pd.DataFrame, preds: dict[str, pd.DataFrame], congested: pd.Series) -> pd.DataFrame:
    """Per model x bus: MAE/RMSE overall and split by congested / uncongested rows."""
    rows = []
    c = congested.to_numpy()
    for name, P in preds.items():
        for col in Y:
            e = (P[col] - Y[col]).to_numpy()
            ok = ~np.isnan(e)
            ae = np.abs(e)
            rows.append({
                "model": name, "bus": col.removeprefix("lmp_"),
                "MAE": ae[ok].mean(), "RMSE": np.sqrt((e[ok] ** 2).mean()),
                "MAE_congested": ae[ok & c].mean(), "MAE_uncongested": ae[ok & ~c].mean(),
                "RMSE_congested": np.sqrt((e[ok & c] ** 2).mean()), "RMSE_uncongested": np.sqrt((e[ok & ~c] ** 2).mean()),
                "P99_abs_error": np.percentile(ae[ok], 99),
            })
    return pd.DataFrame(rows)


def summary_by_model(reg: pd.DataFrame) -> pd.DataFrame:
    return reg.groupby("model", sort=False)[["MAE", "RMSE", "MAE_congested", "MAE_uncongested", "P99_abs_error"]].mean()


def regimes(B: pd.DataFrame, Y: pd.DataFrame, line_out: pd.DataFrame | None) -> dict[str, np.ndarray]:
    internal = [l for l in B if l != INTERTIE]
    north = [l for l in B if l.startswith("EDM-NORTH")]
    m = {
        "no line at limit": ~B.any(axis=1).to_numpy(),
        "SOUTH-CAL at limit": B["SOUTH-CAL"].to_numpy().astype(bool),
        "North corridor at limit": B[north].any(axis=1).to_numpy(),
        "intertie at limit": B[INTERTIE].to_numpy().astype(bool),
        "other internal line at limit": B[[l for l in internal if l != "SOUTH-CAL" and l not in north]].any(axis=1).to_numpy(),
        "scarcity (any LMP >= $100)": (Y >= 100).any(axis=1).to_numpy(),
    }
    if line_out is not None:
        m["≥1 line out of service"] = line_out.any(axis=1).to_numpy()
    return m


def regime_table(Y: pd.DataFrame, preds: dict[str, pd.DataFrame], masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """Mean absolute error over buses, per regime and model."""
    out = {}
    for name, P in preds.items():
        ae = (P - Y).abs()
        out[name] = {r: float(np.nanmean(ae.to_numpy()[m])) if m.any() else np.nan for r, m in masks.items()}
    t = pd.DataFrame(out)
    t.insert(0, "share_of_rows", {r: float(m.mean()) for r, m in masks.items()})
    return t


def classification_table(B: pd.DataFrame, proba: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    rows = []
    for l in B:
        y, p = B[l].to_numpy(), proba[l].to_numpy()
        yh = (p >= thr).astype(int)
        two = 0 < y.sum() < len(y)
        rows.append({
            "line": l, "base_rate": y.mean(), "mean_predicted_prob": p.mean(),
            "precision": precision_score(y, yh, zero_division=np.nan) if yh.sum() else np.nan,
            "recall": recall_score(y, yh, zero_division=np.nan) if y.sum() else np.nan,
            "F1": f1_score(y, yh, zero_division=np.nan) if (y.sum() or yh.sum()) else np.nan,
            "ROC_AUC": roc_auc_score(y, p) if two else np.nan,
            "Brier": brier_score_loss(y, p),
            "n_positive": int(y.sum()),
        })
    return pd.DataFrame(rows).set_index("line")


def mc_congestion_comparison(B: pd.DataFrame, proba: pd.DataFrame, thr: float = 0.5) -> pd.DataFrame:
    """Congestion probability from OPF vs from the classifier (two estimators)."""
    n = len(B)
    k = B.sum()
    lo, hi = wilson(k, n)
    t = pd.DataFrame({
        "opf_p": k / n, "opf_ci95_low": lo, "opf_ci95_high": hi,
        "surrogate_p_mean_prob": proba.mean(), "surrogate_p_thresholded": (proba >= thr).mean(),
    })
    t["abs_err_pp"] = (t["surrogate_p_mean_prob"] - t["opf_p"]).abs() * 100
    t["within_opf_ci"] = (t["surrogate_p_mean_prob"] >= t["opf_ci95_low"]) & (t["surrogate_p_mean_prob"] <= t["opf_ci95_high"])
    t["opf_expected_h_per_year"] = t["opf_p"] * 8760
    t["surrogate_expected_h_per_year"] = t["surrogate_p_mean_prob"] * 8760
    return t.sort_values("opf_p", ascending=False)


def mc_quantile_comparison(Y: pd.DataFrame, preds: dict[str, pd.DataFrame], qs=(0.10, 0.50, 0.90)) -> pd.DataFrame:
    rows = []
    for col in Y:
        tq = Y[col].quantile(list(qs))
        for q in qs:
            row = {"bus": col.removeprefix("lmp_"), "stat": f"P{int(q * 100)}", "OPF": tq[q]}
            for name, P in preds.items():
                row[name] = P[col].quantile(q)
            rows.append(row)
        row = {"bus": col.removeprefix("lmp_"), "stat": "mean", "OPF": Y[col].mean()}
        row.update({name: P[col].mean() for name, P in preds.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def worst_cases(Y: pd.DataFrame, P: pd.DataFrame, X: pd.DataFrame, B: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    err = (P - Y).abs()
    flat = err.stack().sort_values(ascending=False).head(n)
    rows = []
    for (i, col), e in flat.items():
        rows.append({
            "row": i, "bus": col.removeprefix("lmp_"), "opf_lmp": Y.at[i, col], "predicted": P.at[i, col], "abs_error": e,
            "lines_at_limit": ", ".join(B.columns[B.loc[i].astype(bool)]) or "—",
            "lines_out": ", ".join(c.removeprefix("out_") for c in X.columns if c.startswith("out_") and X.at[i, c] > 0) or "—",
            "net_load_mw": X.at[i, "net_load"], "reserve_margin_mw": X.at[i, "reserve_margin_mw"],
        })
    return pd.DataFrame(rows)


def step_sweep(cfg: dict, row_inputs: pd.Series, var: str, grid: np.ndarray, models: dict, buses: list[str],
               scale_loads: bool = False) -> pd.DataFrame:
    """Vary one input over ``grid`` for a single scenario; OPF truth vs surrogates.

    var = an input column (e.g. 'cf_wind_south'), or 'load_scale' with
    scale_loads=True to multiply every bus load.
    """
    x = pd.DataFrame([row_inputs] * len(grid)).reset_index(drop=True)
    x.index.name = "sweep"
    if scale_loads:
        for c in [c for c in x if c.startswith("load_")]:
            x[c] = x[c].astype(float) * grid
    else:
        x[var] = grid
    for c in ("hour", "dow", "month", "doy", "is_weekend"):
        if c in x:
            x[c] = x[c].astype(int)
    truth = solve_block(cfg, x)
    X = build_features(cfg, x)
    out = pd.DataFrame({"value": grid})
    for b in buses:
        out[f"OPF_{b}"] = truth.lmp[b].to_numpy()
    out["any_binding"] = truth.binding.any(axis=1).to_numpy()
    for name, m in models.items():
        P = m.predict(X)
        for b in buses:
            out[f"{name}_{b}"] = P[f"lmp_{b}"].to_numpy()
    return out

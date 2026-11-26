"""Phase 6: GridSFM-Open (pretrained AC-OPF graph model) vs a full AC-OPF solver.

Runs in a SEPARATE virtual environment, because GridSFM pins torch<2.9 and
torch_geometric (see experiments/gridsfm/README.md):

    python experiments/gridsfm/benchmark.py --gridsfm-root /path/to/GridSFM/model \
        --case msr_mississippi --n-normal 30 --n-stress 10

What it does
------------
1. Converts the GridSFM ``.pyg.json`` case to a MATPOWER-style ``ppc`` and solves
   AC-OPF with pandapower's PIPS interior-point solver (pure Python, no Julia).
   The conversion is validated against the Julia/Ipopt solution shipped with
   the sample (objective and voltages).
2. Builds perturbed scenarios (load scaling, random generator outages; a
   "stress" group with heavy load scaling to provoke infeasibility).
3. For every scenario: AC-OPF with PIPS (truth + feasibility = convergence) and
   GridSFM inference (timed: data preparation + forward pass, single and batched).
4. Writes a JSON summary and a markdown table to reports/.

Nothing is trained or fine-tuned.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# pyg.json -> MATPOWER ppc
# ---------------------------------------------------------------------------
def pyg_to_ppc(obj: dict) -> dict:
    g = obj["grid"]
    base = float(np.asarray(g["context"]).ravel()[0])
    bus_x = np.asarray(g["nodes"]["bus"], float)          # [base_kV, type, Vmin, Vmax]
    gen_x = np.asarray(g["nodes"]["generator"], float)    # [mbase, _, Pmin, Pmax, _, Qmin, Qmax, Vg, c2, c1, c0]
    load_x = np.asarray(g["nodes"]["load"], float)        # [Pd, Qd]
    shunt_x = np.asarray(g["nodes"].get("shunt", []), float).reshape(-1, 2)  # [Bs, Gs]
    e = g["edges"]
    nb = len(bus_x)
    bus = np.zeros((nb, 13))
    bus[:, 0] = np.arange(1, nb + 1)
    bus[:, 1] = bus_x[:, 1]
    bus[:, 6] = 1
    bus[:, 7] = 1.0
    bus[:, 9] = bus_x[:, 0]
    bus[:, 10] = 1
    bus[:, 11] = bus_x[:, 3]
    bus[:, 12] = bus_x[:, 2]
    for li, bi in zip(e["load_link"]["senders"], e["load_link"]["receivers"]):
        bus[bi, 2] += load_x[li, 0] * base
        bus[bi, 3] += load_x[li, 1] * base
    if len(shunt_x):
        for si, bi in zip(e["shunt_link"]["senders"], e["shunt_link"]["receivers"]):
            bus[bi, 5] += shunt_x[si, 0] * base
            bus[bi, 4] += shunt_x[si, 1] * base
    gen_bus = dict(zip(e["generator_link"]["senders"], e["generator_link"]["receivers"]))
    ng = len(gen_x)
    gen = np.zeros((ng, 21))
    for gi in range(ng):
        gen[gi, 0] = gen_bus[gi] + 1
        gen[gi, 3] = gen_x[gi, 6] * base
        gen[gi, 4] = gen_x[gi, 5] * base
        gen[gi, 5] = gen_x[gi, 7]
        gen[gi, 6] = gen_x[gi, 0] if gen_x[gi, 0] > 0 else base
        gen[gi, 7] = 1
        gen[gi, 8] = gen_x[gi, 3] * base
        gen[gi, 9] = gen_x[gi, 2] * base
    gencost = np.zeros((ng, 7))
    gencost[:, 0] = 2
    gencost[:, 3] = 3
    gencost[:, 4] = gen_x[:, 8] / base**2
    gencost[:, 5] = gen_x[:, 9] / base
    gencost[:, 6] = gen_x[:, 10]
    rows = []
    for (s, r, f) in zip(e["ac_line"]["senders"], e["ac_line"]["receivers"], e["ac_line"]["features"]):
        angmin, angmax, bfr, bto, rr, xx, rate = f[:7]
        rows.append([s + 1, r + 1, rr, xx, bfr + bto, rate * base, rate * base, rate * base, 0, 0, 1,
                     math.degrees(angmin), math.degrees(angmax)])
    for (s, r, f) in zip(e.get("transformer", {}).get("senders", []), e.get("transformer", {}).get("receivers", []),
                         e.get("transformer", {}).get("features", [])):
        angmin, angmax, rr, xx, rate, _, _, tap, shift, bfr, bto = f[:11]
        rows.append([s + 1, r + 1, rr, xx, bfr + bto, rate * base, rate * base, rate * base, tap, math.degrees(shift), 1,
                     math.degrees(angmin), math.degrees(angmax)])
    branch = np.asarray(rows, float)
    return {"version": "2", "baseMVA": base, "bus": bus, "gen": gen, "branch": branch, "gencost": gencost}


def solve_acopf(ppc: dict) -> dict:
    import pandapower as pp
    from pandapower.converter.pypower.from_ppc import from_ppc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net = from_ppc(copy.deepcopy(ppc), f_hz=60, validate_conversion=False)
        t0 = time.perf_counter()
        try:
            pp.runopp(net, init="flat", verbose=False, calculate_voltage_angles=True, numba=False)
            ok = bool(net.OPF_converged)
        except Exception:  # noqa: BLE001 - PIPS raises OPFNotConverged
            ok = False
        dt = time.perf_counter() - t0
    out = {"converged": ok, "seconds": dt}
    if ok:
        # pandapower orders gens as (ext_grid, gen); map back to the original gen rows via the bus/P match
        out["cost"] = float(net.res_cost)
        out["vm"] = net.res_bus.vm_pu.to_numpy()
        out["va"] = np.radians(net.res_bus.va_degree.to_numpy())
        out["pg_total_mw"] = float(net.res_gen.p_mw.sum() + net.res_ext_grid.p_mw.sum())
        out["lam_p"] = net.res_bus.lam_p.to_numpy() if "lam_p" in net.res_bus else None
    return out


# ---------------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------------
def perturb(obj: dict, load_scale: float, kill: list[int]) -> dict:
    o = copy.deepcopy(obj)
    o.pop("solution", None)
    g = o["grid"]
    for row in g["nodes"]["load"]:
        row[0] *= load_scale
        row[1] *= load_scale
    if kill:
        keep = [i for i in range(len(g["nodes"]["generator"])) if i not in set(kill)]
        remap = {old: new for new, old in enumerate(keep)}
        g["nodes"]["generator"] = [g["nodes"]["generator"][i] for i in keep]
        gl = g["edges"]["generator_link"]
        pairs = [(remap[s], r) for s, r in zip(gl["senders"], gl["receivers"]) if s in remap]
        gl["senders"], gl["receivers"] = [p[0] for p in pairs], [p[1] for p in pairs]
    o["metadata"] = {k: v for k, v in o.get("metadata", {}).items() if not k.endswith("_map")}
    o["metadata"].update({"system_load_factor": load_scale, "n_gens_killed": len(kill)})
    return o


def total_cost(gen_x: np.ndarray, pg: np.ndarray) -> float:
    return float((gen_x[:, 8] * pg**2 + gen_x[:, 9] * pg + gen_x[:, 10]).sum())


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gridsfm-root", required=True, help="path to the cloned GridSFM/model directory")
    ap.add_argument("--case", default="msr_mississippi")
    ap.add_argument("--n-normal", type=int, default=30)
    ap.add_argument("--n-stress", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(REPO / "reports"))
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    root = Path(args.gridsfm_root)
    sys.path.insert(0, str(root))
    from gridsfm import batch_data_list, load_model, load_pyg_json, predict, prepare_for_inference
    from torch_geometric.utils import unbatch

    sample = root / "samples" / f"{args.case}.pyg.json"
    base_obj = json.load(open(sample))
    gen_x0 = np.asarray(base_obj["grid"]["nodes"]["generator"], float)
    nb, ng = len(base_obj["grid"]["nodes"]["bus"]), len(gen_x0)
    work = Path(args.out) / "gridsfm_scenarios"
    work.mkdir(parents=True, exist_ok=True)

    # ---- 1. validate the conversion on the shipped base case (Julia/Ipopt solution) ----
    ref_cost = base_obj["metadata"]["objective"]
    ref_v = np.asarray(base_obj["solution"]["nodes"]["bus"], float)[:, 1]
    ref_lmp = np.asarray(base_obj["solution"]["duals"]["bus"], float)[:, 0] / base_obj["grid"]["context"][0][0][0]
    base_sol = solve_acopf(pyg_to_ppc(base_obj))
    validation = {"reference_objective_julia_ipopt": ref_cost, "pips_converged": base_sol["converged"],
                  "pips_seconds": base_sol["seconds"]}
    if base_sol["converged"]:
        validation.update({
            "pips_objective": base_sol["cost"], "objective_rel_diff": (base_sol["cost"] - ref_cost) / ref_cost,
            "max_abs_v_diff_pu": float(np.abs(base_sol["vm"] - ref_v).max()),
            "reference_lmp_usd_per_mwh": {"min": float(ref_lmp.min()), "mean": float(ref_lmp.mean()), "max": float(ref_lmp.max())},
        })
    print("conversion check:", json.dumps(validation, indent=1, default=float))

    # ---- 2. scenarios ----
    rng = np.random.default_rng(args.seed)
    scen = []
    for i in range(args.n_normal + args.n_stress):
        stress = i >= args.n_normal
        s = float(rng.uniform(1.25, 1.70) if stress else rng.uniform(0.85, 1.15))
        k = int(rng.integers(0, 3))
        big = np.argsort(-gen_x0[:, 3])[: max(3, ng // 4)]          # outages among the larger units
        kill = sorted(rng.choice(big, size=k, replace=False).tolist()) if k else []
        o = perturb(base_obj, s, kill)
        p = work / f"{args.case}_s{i:02d}.pyg.json"
        p.write_text(json.dumps(o))
        scen.append({"id": i, "group": "stress" if stress else "normal", "load_scale": s, "gens_out": len(kill),
                     "path": p, "gen_x": np.asarray(o["grid"]["nodes"]["generator"], float)})

    # ---- 3a. full solver ----
    for sc in scen:
        r = solve_acopf(pyg_to_ppc(json.load(open(sc["path"]))))
        sc["solver"] = r
        print(f"  PIPS s{sc['id']:02d} ({sc['group']}, load×{sc['load_scale']:.2f}, {sc['gens_out']} out): "
              f"{'ok ' if r['converged'] else 'FAIL'} {r['seconds']:.2f}s")

    # ---- 3b. GridSFM ----
    model = load_model(str(root / "checkpoints" / "gridsfm_open_v1.1.pt"), device="cpu")
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = predict(model, str(scen[0]["path"]))                         # cold: builds per-topology caches
        cold = time.perf_counter() - t0
        single = []
        for sc in scen:
            t0 = time.perf_counter()
            out = predict(model, str(sc["path"]))
            single.append(time.perf_counter() - t0)
            sc["gridsfm"] = {"feas": float(out["feas"]), "pg": out["Pg"].numpy().astype(float),
                             "v": out["V"].numpy().astype(float), "theta": out["theta"].numpy().astype(float)}
        t0 = time.perf_counter()
        prepared = [prepare_for_inference(load_pyg_json(sc["path"])) for sc in scen]
        t_prep = time.perf_counter() - t0
        batch = batch_data_list(prepared)
        t0 = time.perf_counter()
        bo = model(batch)
        t_fwd = time.perf_counter() - t0
        feas_b = torch.sigmoid(bo.feas_logit).numpy()

    # ---- 4. compare ----
    rows = []
    for sc, fb in zip(scen, feas_b):
        s, gsf = sc["solver"], sc["gridsfm"]
        row = {"id": sc["id"], "group": sc["group"], "load_scale": round(sc["load_scale"], 3), "gens_out": sc["gens_out"],
               "solver_converged": s["converged"], "solver_seconds": s["seconds"],
               "gridsfm_feas_prob": gsf["feas"], "gridsfm_feas_prob_batched": float(fb)}
        if s["converged"]:
            c_pred = total_cost(sc["gen_x"], gsf["pg"])
            row.update({"cost_solver": s["cost"], "cost_gridsfm": c_pred, "cost_err_pct": 100 * (c_pred - s["cost"]) / s["cost"],
                        "v_mae_pu": float(np.abs(gsf["v"] - s["vm"]).mean()),
                        "pg_total_err_pct": 100 * (gsf["pg"].sum() * base_obj["grid"]["context"][0][0][0] - s["pg_total_mw"]) / s["pg_total_mw"]})
        rows.append(row)
    conv = np.array([r["solver_converged"] for r in rows])
    pred = np.array([r["gridsfm_feas_prob"] >= 0.5 for r in rows])
    ok_rows = [r for r in rows if r["solver_converged"]]
    summary = {
        "case": args.case, "buses": nb, "generators": ng, "scenarios": len(rows), "threads": args.threads,
        "conversion_validation": validation,
        "solver": {"name": "pandapower PIPS (AC-OPF, interior point, pure Python)",
                   "converged": int(conv.sum()), "failed": int((~conv).sum()),
                   "seconds_total": float(sum(r["solver_seconds"] for r in rows)),
                   "seconds_mean": float(np.mean([r["solver_seconds"] for r in rows]))},
        "gridsfm": {"checkpoint": "microsoft/GridSFM_Open gridsfm_open_v1.1.pt",
                    "cold_first_call_s": cold, "single_mean_s": float(np.mean(single)), "single_total_s": float(np.sum(single)),
                    "batched_prepare_s": t_prep, "batched_forward_s": t_fwd, "batched_total_s": t_prep + t_fwd},
        "accuracy_on_converged": {
            "n": len(ok_rows),
            "cost_abs_err_pct_mean": float(np.mean([abs(r["cost_err_pct"]) for r in ok_rows])) if ok_rows else None,
            "cost_abs_err_pct_max": float(np.max([abs(r["cost_err_pct"]) for r in ok_rows])) if ok_rows else None,
            "v_mae_pu_mean": float(np.mean([r["v_mae_pu"] for r in ok_rows])) if ok_rows else None,
            "pg_total_err_pct_mean": float(np.mean([r["pg_total_err_pct"] for r in ok_rows])) if ok_rows else None,
        },
        "feasibility": {
            "solver_converged_and_gridsfm_feasible": int((conv & pred).sum()),
            "solver_converged_but_gridsfm_infeasible": int((conv & ~pred).sum()),
            "solver_failed_but_gridsfm_feasible": int((~conv & pred).sum()),
            "solver_failed_and_gridsfm_infeasible": int((~conv & ~pred).sum()),
        },
        "rows": rows,
    }
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    with open(out / "phase6_gridsfm_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1, default=float))


if __name__ == "__main__":
    main()

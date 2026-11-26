"""Phase 6: GridSFM-Open (pretrained AC-OPF graph model) vs a full AC-OPF solver.

Runs in a SEPARATE Python virtual environment (GridSFM pins torch<2.9 and needs
torch_geometric), with Julia + PowerModels + Ipopt for the reference solves.
See experiments/gridsfm/README.md for the exact setup.

    python experiments/gridsfm/benchmark.py --gridsfm-root <GridSFM/model> \
        --julia <julia binary> --julia-project <env with PowerModels+Ipopt>

Steps
-----
1. Convert the GridSFM ``.pyg.json`` case to a MATPOWER file. The conversion is
   validated against the Julia/Ipopt solution shipped with the sample.
2. Build perturbed scenarios: load scaling and random outages of large units,
   plus a "stress" group with heavy load scaling to provoke infeasibility.
3. Solve every scenario with the full AC-OPF (PowerModels ACP + Ipopt, the same
   formulation and settings GridSFM's data pipeline uses), in one Julia process
   after a JIT warm-up.
4. Run GridSFM on the same scenarios (single-graph and batched, CPU) and compare
   cost, voltages, dispatch and feasibility predictions.

Nothing is trained or fine-tuned.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# pyg.json -> MATPOWER
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
    gen = np.zeros((ng, 10))
    for gi in range(ng):
        gen[gi] = [gen_bus[gi] + 1, 0, 0, gen_x[gi, 6] * base, gen_x[gi, 5] * base, gen_x[gi, 7],
                   gen_x[gi, 0] if gen_x[gi, 0] > 0 else base, 1, gen_x[gi, 3] * base, gen_x[gi, 2] * base]
    gencost = np.column_stack([np.full(ng, 2), np.zeros(ng), np.zeros(ng), np.full(ng, 3),
                               gen_x[:, 8] / base**2, gen_x[:, 9] / base, gen_x[:, 10]])
    rows = []
    for s, r, f in zip(e["ac_line"]["senders"], e["ac_line"]["receivers"], e["ac_line"]["features"]):
        angmin, angmax, bfr, bto, rr, xx, rate = f[:7]
        rows.append([s + 1, r + 1, rr, xx, bfr + bto, rate * base, rate * base, rate * base, 0, 0, 1,
                     math.degrees(angmin), math.degrees(angmax)])
    tr = e.get("transformer", {})
    for s, r, f in zip(tr.get("senders", []), tr.get("receivers", []), tr.get("features", [])):
        angmin, angmax, rr, xx, rate, _, _, tap, shift, bfr, bto = f[:11]
        rows.append([s + 1, r + 1, rr, xx, bfr + bto, rate * base, rate * base, rate * base, tap,
                     math.degrees(shift), 1, math.degrees(angmin), math.degrees(angmax)])
    return {"baseMVA": base, "bus": bus, "gen": gen, "branch": np.asarray(rows, float), "gencost": gencost}


def write_matpower(ppc: dict, path: Path, name: str) -> None:
    def block(a):
        return "\n".join("\t" + "\t".join(f"{v:.12g}" for v in row) + ";" for row in a)
    path.write_text(
        f"function mpc = {name}\nmpc.version = '2';\nmpc.baseMVA = {ppc['baseMVA']:g};\n"
        f"mpc.bus = [\n{block(ppc['bus'])}\n];\nmpc.gen = [\n{block(ppc['gen'])}\n];\n"
        f"mpc.branch = [\n{block(ppc['branch'])}\n];\nmpc.gencost = [\n{block(ppc['gencost'])}\n];\n")


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


def total_cost(gen_x: np.ndarray, pg_pu: np.ndarray) -> float:
    return float((gen_x[:, 8] * pg_pu**2 + gen_x[:, 9] * pg_pu + gen_x[:, 10]).sum())


def run_julia(julia: str, project: str, mfiles: list[Path], out: Path) -> dict:
    cases = out.with_suffix(".cases.json")
    cases.write_text(json.dumps([str(p) for p in mfiles]))
    t0 = time.perf_counter()
    subprocess.run([julia, f"--project={project}", str(HERE / "solve_acopf.jl"), str(cases), str(out)],
                   check=True, env=os.environ.copy())
    res = json.loads(out.read_text())
    res["process_wall_seconds"] = time.perf_counter() - t0
    return res


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gridsfm-root", required=True, help="cloned GridSFM/model directory (with checkpoints/)")
    ap.add_argument("--julia", required=True)
    ap.add_argument("--julia-project", required=True, help="Julia project with PowerModels + Ipopt")
    ap.add_argument("--case", default="msr_mississippi")
    ap.add_argument("--n-normal", type=int, default=40)
    ap.add_argument("--n-stress", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workdir", default=str(REPO / "data" / "phase6"))
    ap.add_argument("--out", default=str(REPO / "reports"))
    args = ap.parse_args(argv)

    import torch
    torch.set_num_threads(args.threads)
    root = Path(args.gridsfm_root)
    sys.path.insert(0, str(root))
    from gridsfm import batch_data_list, load_model, load_pyg_json, predict, prepare_for_inference

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    base_obj = json.load(open(root / "samples" / f"{args.case}.pyg.json"))
    base = float(np.asarray(base_obj["grid"]["context"]).ravel()[0])
    gen_x0 = np.asarray(base_obj["grid"]["nodes"]["generator"], float)
    nb, ng = len(base_obj["grid"]["nodes"]["bus"]), len(gen_x0)

    # ---- scenarios (scenario 0 = the unperturbed base case, used to validate the conversion) ----
    rng = np.random.default_rng(args.seed)
    big = np.argsort(-gen_x0[:, 3])[: max(3, ng // 4)]     # outages drawn among the larger units
    scen = [{"id": 0, "group": "base", "load_scale": 1.0, "gens_out": 0, "obj": copy.deepcopy(base_obj)}]
    for i in range(1, args.n_normal + args.n_stress + 1):
        stress = i > args.n_normal
        s = float(rng.uniform(1.25, 1.70) if stress else rng.uniform(0.85, 1.15))
        k = int(rng.integers(0, 3))
        kill = sorted(rng.choice(big, size=k, replace=False).tolist()) if k else []
        scen.append({"id": i, "group": "stress" if stress else "normal", "load_scale": s, "gens_out": len(kill),
                     "obj": perturb(base_obj, s, kill)})
    for sc in scen:
        sc["pyg"] = work / f"{args.case}_s{sc['id']:03d}.pyg.json"
        sc["pyg"].write_text(json.dumps(sc["obj"]))
        sc["m"] = work / f"{args.case}_s{sc['id']:03d}.m"
        write_matpower(pyg_to_ppc(sc["obj"]), sc["m"], f"s{sc['id']:03d}")
        sc["gen_x"] = np.asarray(sc["obj"]["grid"]["nodes"]["generator"], float)

    # ---- full AC-OPF (Julia) ----
    print(f"[phase 6] solving {len(scen)} scenarios with PowerModels + Ipopt ...")
    jl = run_julia(args.julia, args.julia_project, [sc["m"] for sc in scen], work / "julia_results.json")
    for sc, r in zip(scen, jl["results"]):
        sc["solver"] = r

    # ---- GridSFM ----
    print(f"[phase 6] GridSFM inference on {len(scen)} scenarios (CPU, {args.threads} threads) ...")
    model = load_model(str(root / "checkpoints" / "gridsfm_open_v1.1.pt"), device="cpu")
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        predict(model, str(scen[0]["pyg"]))
        cold = time.perf_counter() - t0
        single = []
        for sc in scen:
            t0 = time.perf_counter()
            out = predict(model, str(sc["pyg"]))
            single.append(time.perf_counter() - t0)
            sc["gsf"] = {"feas": float(out["feas"]), "pg": out["Pg"].numpy().astype(float),
                         "v": out["V"].numpy().astype(float)}
        t0 = time.perf_counter()
        prepared = [prepare_for_inference(load_pyg_json(sc["pyg"])) for sc in scen]
        t_prep = time.perf_counter() - t0
        t0 = time.perf_counter()
        model(batch_data_list(prepared))
        t_fwd = time.perf_counter() - t0

    # ---- compare ----
    rows = []
    for sc, ts in zip(scen, single):
        s, g = sc["solver"], sc["gsf"]
        load_mw = float(np.asarray(sc["obj"]["grid"]["nodes"]["load"], float)[:, 0].sum() * base)
        row = {"id": sc["id"], "group": sc["group"], "load_scale": round(sc["load_scale"], 3), "gens_out": sc["gens_out"],
               "load_mw": load_mw, "gridsfm_total_pg_mw": float(g["pg"].sum() * base),
               "solver_status": s["status"], "solver_converged": s["converged"],
               "solver_ipopt_s": s["ipopt_seconds"], "solver_build_and_solve_s": s["build_and_solve_seconds"],
               "gridsfm_s": ts, "gridsfm_feas_prob": g["feas"]}
        if s["converged"]:
            pg_ref = np.asarray(s["pg"], float)
            c_ref, c_pred = s["objective"], total_cost(sc["gen_x"], g["pg"])
            lmp = np.asarray(s["lmp"], float)
            row.update({"solver_total_pg_mw": float(np.nansum(pg_ref) * base),
                        "balance_error_mw": float((g["pg"].sum() - np.nansum(pg_ref)) * base),
                        "cost_solver": c_ref, "cost_gridsfm": c_pred, "cost_err_pct": 100 * (c_pred - c_ref) / c_ref,
                        "v_mae_pu": float(np.abs(g["v"] - np.asarray(s["vm"])).mean()),
                        "pg_mae_mw": float(np.abs(g["pg"] - pg_ref).mean() * base),
                        "pg_max_err_mw": float(np.abs(g["pg"] - pg_ref).max() * base),
                        "lmp_min": float(lmp.min()), "lmp_mean": float(lmp.mean()), "lmp_max": float(lmp.max())})
        rows.append(row)

    b0 = scen[0]["solver"]
    ref_obj = base_obj["metadata"]["objective"]
    ref_v = np.asarray(base_obj["solution"]["nodes"]["bus"], float)[:, 1]
    ref_lmp = np.asarray(base_obj["solution"]["duals"]["bus"], float)[:, 0] / base
    validation = {"reference_objective": ref_obj, "julia_status": b0["status"]}
    if b0["converged"]:
        validation.update({"julia_objective": b0["objective"], "objective_rel_diff": (b0["objective"] - ref_obj) / ref_obj,
                           "max_abs_v_diff_pu": float(np.abs(np.asarray(b0["vm"]) - ref_v).max()),
                           "max_abs_lmp_diff_usd_mwh": float(np.abs(np.asarray(b0["lmp"]) - ref_lmp).max())})

    conv = np.array([r["solver_converged"] for r in rows])
    pred = np.array([r["gridsfm_feas_prob"] >= 0.5 for r in rows])
    ok = [r for r in rows if r["solver_converged"]]
    by_group = {}
    groups = {"base": lambda r: r["group"] == "base",
              "normal: load scaling only": lambda r: r["group"] == "normal" and r["gens_out"] == 0,
              "normal: load scaling + unit outages": lambda r: r["group"] == "normal" and r["gens_out"] > 0,
              "stress": lambda r: r["group"] == "stress"}
    for grp, sel in groups.items():
        rr = [r for r in ok if sel(r)]
        allg = [r for r in rows if sel(r)]
        by_group[grp] = {"n": len(allg), "converged": len(rr),
                         "cost_abs_err_pct_mean": float(np.mean([abs(r["cost_err_pct"]) for r in rr])) if rr else None,
                         "cost_abs_err_pct_p95": float(np.percentile([abs(r["cost_err_pct"]) for r in rr], 95)) if rr else None,
                         "v_mae_pu_mean": float(np.mean([r["v_mae_pu"] for r in rr])) if rr else None,
                         "pg_mae_mw_mean": float(np.mean([r["pg_mae_mw"] for r in rr])) if rr else None,
                         "abs_balance_error_mw_mean": float(np.mean([abs(r["balance_error_mw"]) for r in rr])) if rr else None}
    ipopt = np.array([r["solver_ipopt_s"] for r in rows])
    build = np.array([r["solver_build_and_solve_s"] for r in rows])
    summary = {
        "case": args.case, "buses": nb, "generators": ng, "branches": int(len(pyg_to_ppc(base_obj)["branch"])),
        "scenarios": len(rows), "threads": args.threads,
        "conversion_validation": validation,
        "solver": {"name": "PowerModels ACPPowerModel + Ipopt (Julia), same settings as GridSFM's pipeline",
                   "jit_warmup_s": jl["warmup_seconds"], "process_wall_s": jl["process_wall_seconds"],
                   "ipopt_s_mean": float(ipopt.mean()), "ipopt_s_median": float(np.median(ipopt)),
                   "build_and_solve_s_mean": float(build.mean()), "build_and_solve_s_total": float(build.sum()),
                   "status_counts": {k: int(v) for k, v in zip(*np.unique([r["solver_status"] for r in rows], return_counts=True))}},
        "gridsfm": {"checkpoint": "microsoft/GridSFM_Open gridsfm_open_v1.1.pt", "cold_first_call_s": cold,
                    "single_s_mean": float(np.mean(single)), "single_s_total": float(np.sum(single)),
                    "batched_prepare_s": t_prep, "batched_forward_s": t_fwd,
                    "batched_s_per_scenario": (t_prep + t_fwd) / len(rows)},
        "speedup": {"single_vs_build_and_solve": float(build.mean() / np.mean(single)),
                    "single_vs_ipopt_only": float(ipopt.mean() / np.mean(single)),
                    "batched_vs_build_and_solve": float(build.mean() / ((t_prep + t_fwd) / len(rows)))},
        "accuracy_by_group": by_group,
        "feasibility_confusion": {"solver_ok_gridsfm_feasible": int((conv & pred).sum()),
                                  "solver_ok_gridsfm_infeasible": int((conv & ~pred).sum()),
                                  "solver_failed_gridsfm_feasible": int((~conv & pred).sum()),
                                  "solver_failed_gridsfm_infeasible": int((~conv & ~pred).sum())},
        "rows": rows,
    }
    out = Path(args.out)
    with open(out / f"phase6_gridsfm_{args.case}.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1, default=float))


if __name__ == "__main__":
    main()

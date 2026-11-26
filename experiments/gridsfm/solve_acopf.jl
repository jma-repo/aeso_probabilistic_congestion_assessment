#!/usr/bin/env julia
# Full AC-OPF reference solver for the Phase 6 benchmark.
#
# Same formulation and solver settings as GridSFM's own data pipeline
# (power_grid/US/topology_solver_pipeline/gen_perturbed_data.jl):
#   PowerModels ACPPowerModel + build_opf, Ipopt tol=1e-6, acceptable_tol=1e-4, max_iter=3000.
#
# Usage:  julia --project=<env with PowerModels+Ipopt> solve_acopf.jl <cases.json> <results.json>
#   cases.json: ["path/to/case_0.m", "path/to/case_1.m", ...]  (MATPOWER files)
# The first case is solved once extra as a JIT warm-up and that time is reported separately.
using PowerModels, Ipopt, JuMP, JSON3
PowerModels.silence()

const SOLVER = optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0, "sb" => "yes",
                                         "max_iter" => 3000, "tol" => 1e-6, "acceptable_tol" => 1e-4)
const OK = ("LOCALLY_SOLVED", "OPTIMAL", "ALMOST_LOCALLY_SOLVED", "ALMOST_OPTIMAL")

function solve_one(path)
    t0 = time()
    data = PowerModels.parse_file(path; validate = true)
    t_parse = time() - t0
    t1 = time()
    res = PowerModels.solve_model(data, ACPPowerModel, SOLVER, PowerModels.build_opf;
                                  setting = Dict("output" => Dict("duals" => true)))
    t_build_solve = time() - t1
    status = string(res["termination_status"])
    out = Dict{String,Any}("path" => path, "status" => status, "converged" => status in OK,
                           "objective" => res["objective"], "ipopt_seconds" => res["solve_time"],
                           "build_and_solve_seconds" => t_build_solve, "parse_seconds" => t_parse)
    if out["converged"]
        sol = res["solution"]; base = data["baseMVA"]
        nb = length(data["bus"]); ng = length(data["gen"])
        out["vm"] = [sol["bus"][string(i)]["vm"] for i in 1:nb]
        out["va"] = [sol["bus"][string(i)]["va"] for i in 1:nb]
        out["pg"] = [get(get(sol["gen"], string(i), Dict()), "pg", NaN) for i in 1:ng]   # per-unit
        # PowerModels' KCL dual has the opposite sign to the LMP convention -> negate; per-unit -> $/MWh
        out["lmp"] = [-sol["bus"][string(i)]["lam_kcl_r"] / base for i in 1:nb]
    end
    return out
end

function main(args)
    paths = JSON3.read(read(args[1], String), Vector{String})
    warm = solve_one(paths[1])                       # JIT compilation happens here
    results = [solve_one(p) for p in paths]
    open(args[2], "w") do io
        JSON3.write(io, Dict("warmup_seconds" => warm["build_and_solve_seconds"] + warm["parse_seconds"],
                             "results" => results))
    end
    n_ok = count(r -> r["converged"], results)
    println("solved $(length(results)) cases, $(n_ok) converged")
end

main(ARGS)

# Phase 6 (stretch): GridSFM-Open vs a full AC-OPF solver

Results and discussion are in the main [README](../../README.md#phase-6-stretch--gridsfm-a-pretrained-ac-opf-graph-model-vs-a-full-solver).
Raw results are in `reports/phase6_gridsfm_<case>.json`, and the figure is `reports/figures/p6_gridsfm_balance.png`.

Nothing here is installed into the main environment. GridSFM pins `torch>=2.6,<2.9` and needs `torch_geometric`;
the reference solver needs Julia.

## Files

| File | Purpose |
|---|---|
| `benchmark.py` | Builds the scenarios, converts `.pyg.json` to MATPOWER, calls the Julia solver, runs GridSFM, and writes the JSON results |
| `solve_acopf.jl` | Reference AC-OPF: PowerModels `ACPPowerModel` + `build_opf` + Ipopt (tol 1e-6, acceptable_tol 1e-4, max_iter 3000), the same as GridSFM's data pipeline. It solves every case in one process, after a JIT warm-up, and returns status, objective, timings, voltages, dispatch and LMPs. |
| `plot_results.py` | Draws the figure. Runs in the main environment. |

## Setup (the steps actually used)

```bash
git clone --depth 1 https://github.com/microsoft/GridSFM.git
```

Python environment for GridSFM (torch 2.8.0, torch_geometric 2.8.1, gridsfm 1.1.0):

```bash
python -m venv gridsfm-venv
```

```bash
gridsfm-venv/bin/pip install "torch>=2.6,<2.9" "torch_geometric>=2.5,<3" "numpy<3" "scipy<2" "huggingface_hub>=0.34,<1"
```

```bash
gridsfm-venv/bin/pip install -e GridSFM/model
```

Checkpoint (61 MB; `load_model` verifies its SHA-256). `HF_HUB_DISABLE_XET=1` avoids the Xet transfer hosts:

```bash
cd GridSFM/model && HF_HUB_DISABLE_XET=1 ../../gridsfm-venv/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('microsoft/GridSFM_Open', 'gridsfm_open_v1.1.pt', local_dir='checkpoints')"
```

Julia 1.11.5 with GridSFM's own `Project.toml` and `Manifest.toml`, which gives PowerModels 0.21.5, Ipopt 1.14.0 and
JuMP 1.29.4:

```bash
curl -fLO https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-1.11.5-linux-x86_64.tar.gz && mkdir julia && tar -xzf julia-1.11.5-linux-x86_64.tar.gz -C julia --strip-components=1
```

```bash
mkdir jlproj && cp GridSFM/power_grid/US/topology_solver_pipeline/{Project,Manifest}.toml jlproj/
```

```bash
julia/bin/julia --project=jlproj -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Network hosts needed (cloud environment allow-list):

* `huggingface.co` and `us.aws.cdn.hf.co`;
* `julialang-s3.julialang.org`, `pkg.julialang.org`, `us-west.pkg.julialang.org` / `us-east.pkg.julialang.org`,
  and `storage.julialang.net`.

## Run

```bash
gridsfm-venv/bin/python experiments/gridsfm/benchmark.py --gridsfm-root GridSFM/model --julia julia/bin/julia --julia-project jlproj --case msr_mississippi --n-normal 40 --n-stress 20
```

```bash
gridsfm-venv/bin/python experiments/gridsfm/benchmark.py --gridsfm-root GridSFM/model --julia julia/bin/julia --julia-project jlproj --case case500_goc --n-normal 20 --n-stress 10
```

```bash
python experiments/gridsfm/plot_results.py
```

The Mississippi run takes about 40 minutes, mostly Ipopt proving infeasibility; one scenario alone took 9 minutes.

## Things that did not work (kept for the record)

* **pandapower's pure-Python PIPS AC-OPF** does not converge on these 500-bus cases, even with all limits relaxed,
  although the converted network reproduces the reference voltages exactly in an AC power flow.
* **`cyipopt`** fails to build without a system Ipopt library. Julia's `Ipopt_jll` ships its own binaries, which
  is why the Julia route was used.

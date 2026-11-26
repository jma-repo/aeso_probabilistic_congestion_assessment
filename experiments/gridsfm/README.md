# Phase 6 (stretch): GridSFM-Open vs a full AC-OPF solver — work in progress

**Status: blocked on the reference solver; waiting for a decision. See "Blocker" below.**

## Setup used (separate virtual environment, nothing added to the main requirements)

GridSFM pins `torch>=2.6,<2.9` and needs `torch_geometric`, while this repo uses a newer torch. So a separate venv is
used:

```bash
git clone --depth 1 https://github.com/microsoft/GridSFM.git
```

```bash
python -m venv gridsfm-venv
```

```bash
gridsfm-venv/bin/pip install "torch>=2.6,<2.9" "torch_geometric>=2.5,<3" "numpy<3" "scipy<2" "huggingface_hub>=0.34,<1" pandapower numba
```

```bash
gridsfm-venv/bin/pip install -e GridSFM/model
```

```bash
cd GridSFM/model && HF_HUB_DISABLE_XET=1 ../../gridsfm-venv/bin/python -c "from huggingface_hub import hf_hub_download; hf_hub_download('microsoft/GridSFM_Open', 'gridsfm_open_v1.1.pt', local_dir='checkpoints')"
```

This installs torch 2.8.0, torch_geometric 2.8.1 and gridsfm 1.1.0. The checkpoint is 61 MB, and `load_model`
verifies its SHA-256. No Julia was needed for inference.

## Results so far

GridSFM was run on three of the ≥500-bus base cases shipped with GridSFM, including the smallest open "MSR" grid
(`msr_mississippi`, 528 buses). It ran on CPU with 4 threads, after one warm-up call, and was compared with the
Julia/Ipopt AC-OPF solution stored in each sample.

| case | buses | inference | cost error | V MAE | Pg MAE | feasibility prob. |
|---|---:|---:|---:|---:|---:|---:|
| msr_mississippi | 528 | 258 ms | −0.30% | 0.0014 pu | 3.3 MW | 1.000 |
| case500_goc | 500 | 212 ms | −0.54% | 0.0011 pu | 0.6 MW | 0.997 |
| msr_louisiana | 571 | 267 ms | −0.09% | 0.0016 pu | 1.0 MW | 1.000 |

Caveat: these base grids are very likely part of GridSFM's training distribution, so this is **not** an
out-of-sample test.

## Blocker: the reference solver

The task needs the *same* perturbed scenarios solved by a full AC-OPF solver, for timing, and to check the
feasibility predictions. Three routes were tried:

* GridSFM's own data pipeline uses **Julia** (PowerModels + Ipopt). It is not installed.
* `cyipopt` (Python bindings to Ipopt) **fails to build**, because there is no system Ipopt library.
* pandapower's pure-Python **PIPS** interior-point AC-OPF **does not converge** on these cases (528 / 500 buses).
  It fails even with all branch, angle and voltage limits relaxed, and from a power-flow start. It works on
  pandapower's own case30 and case118.
  * This is not a data-conversion problem. `benchmark.py`'s `.pyg.json` → MATPOWER conversion reproduces the
    Julia/Ipopt voltages exactly when an AC power flow is run at the reference dispatch.

`benchmark.py` already contains the full scenario study: conversion, perturbed scenarios, timing, and the
feasibility confusion matrix. It only needs a working AC-OPF solver behind `solve_acopf()`.

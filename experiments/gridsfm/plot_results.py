"""Figure for Phase 6: does GridSFM's dispatch follow system load? (run in the main environment)

    python experiments/gridsfm/plot_results.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src import plots  # noqa: E402

CASES = {"msr_mississippi": "msr_mississippi (528 buses, open OSM-based US grid)",
         "case500_goc": "case500_goc (500 buses, pglib)"}


def main():
    plots.style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, (case, title) in zip(axes, CASES.items()):
        s = json.load(open(REPO / "reports" / f"phase6_gridsfm_{case}.json"))
        rows = s["rows"]
        load = np.array([r["load_mw"] for r in rows])
        gsf = np.array([r["gridsfm_total_pg_mw"] for r in rows])
        conv = np.array([r["solver_converged"] for r in rows])
        ipopt = np.array([r.get("solver_total_pg_mw", np.nan) for r in rows])
        lo, hi = load.min() * 0.97, load.max() * 1.03
        ax.plot([lo, hi], [lo, hi], color=plots.TEXT2, lw=1, ls="--", label="generation = load")
        ax.scatter(load[conv], ipopt[conv], s=40, color=plots.TEXT, label="full AC-OPF (Ipopt), converged", zorder=3)
        ax.scatter(load[conv], gsf[conv], s=34, color=plots.SERIES[0], label="GridSFM, solver converged", zorder=4)
        ax.scatter(load[~conv], gsf[~conv], s=34, facecolor="white", edgecolor=plots.SERIES[1], linewidth=1.6,
                   label="GridSFM, solver found no feasible point", zorder=4)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("Total load in the scenario (MW)")
    axes[0].set_ylabel("Total active generation (MW)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("GridSFM-Open zero-shot: predicted generation barely follows system load", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    out = REPO / "reports" / "figures" / "p6_gridsfm_balance.png"
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()

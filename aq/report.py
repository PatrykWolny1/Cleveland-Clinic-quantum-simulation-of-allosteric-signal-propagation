"""
Stage 9 - save deliverables and interpretability.

For each target:
  <name>_connectivity.npy - N x N connectivity matrix (deliverable #1)
  <name>_hitlist.json/csv - top-5 predicted sites (deliverable #2)
  <name>_heatmap.png - matrix preview (if matplotlib is available)
Globally:
  leaderboard.csv - one row per configuration/target (for iteration)
  report.md - concise methodology report (deliverable #3)
"""
from __future__ import annotations
import os
import json
import csv
import numpy as np


def save_target(outdir, name, M, hits, keys):
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, f"{name}_connectivity.npy"), M)

    with open(os.path.join(outdir, f"{name}_hitlist.json"), "w") as f:
        json.dump(hits, f, indent=2)

    with open(os.path.join(outdir, f"{name}_hitlist.csv"), "w", newline="") as f:
        if hits:
            wcsv = csv.DictWriter(f, fieldnames=list(hits[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(hits)

    _heatmap(outdir, name, M)


def _heatmap(outdir, name, M):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(5, 4))
    plt.imshow(M, cmap="magma", origin="lower")
    plt.colorbar(label="quantum connectivity")
    plt.title(f"{name} - connectivity matrix")
    plt.xlabel("residue j"); plt.ylabel("residue i")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_heatmap.png"), dpi=110)
    plt.close()


def save_leaderboard(outdir, rows):
    if not rows:
        return
    cols = sorted({k for r in rows for k in r})
    path = os.path.join(outdir, "leaderboard.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def save_report(outdir, cfg, rows):
    lines = ["# Phase I - baseline: methodology report\n",
             "## Quantum metric\n",
             "Continuous-time quantum walk in the single-excitation subspace "
             "over the residue contact graph. Propagator U(t)=exp(-iHt); "
             "connectivity metric M_ij = <|<j|U(t)|i>|^2>_t. Proxy for the "
             "biological signal: the probability of quantum excitation transfer "
             "between residues defines a dynamic connectivity that, in allostery, "
             "links the distal pocket to the active site.\n",
             "## Configuration\n",
             "```yaml", json.dumps(cfg, indent=2), "```\n",
             "## Results per target\n"]
    for r in rows:
        lines.append(f"- **{r.get('target')}**: "
                     f"AUC={r.get('roc_auc','-')}, "
                     f"hit@5={r.get('hit@5','-')}, "
                     f"enrichment@5={r.get('enrichment@5','-')}, "
                     f"p={r.get('mwu_p','-')}")
    with open(os.path.join(outdir, "report.md"), "w") as f:
        f.write("\n".join(str(x) for x in lines))

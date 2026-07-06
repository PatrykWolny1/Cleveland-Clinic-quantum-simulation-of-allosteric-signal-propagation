"""
Stage 9 - save deliverables i interpretowalnosc.

For each targetu:
  <name>_connectivity.npy - connectivity matrix N x N (deliverable #1)
  <name>_hitlist.json/csv - top-5 przewidzianych miejsc (deliverable #2)
  <name>_heatmap.png - podglad matrix (if matplotlib)
Globalnie:
  leaderboard.csv - one row in konfiguracje/target (to iteracji)
  report.md - zwiezly raport metodologiczny (deliverable #3)
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
    lines = ["# Phase I - baseline: raport metodologiczny\n",
             "## Metryka quantum\n",
             "Continuous-time quantum walk w single-excitation subspace "
             "in graph kontaktow reszt. Propagator U(t)=exp(-iHt); metric "
             "connectivity M_ij = <|<j|U(t)|i>|^2>_t. Proxy sygnalu biologicznego: "
             "prawdopodobienstwo quantumgo transfer wzbudzenia between residues "
             "wyznacza dynamiczna connectivity, which w allosterii combines distal "
             "pocket z active site.\n",
             "## Konfiguracja\n",
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

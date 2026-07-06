#!/usr/bin/env python3
"""
Visualization 3D: save oceny allosterycznej (LST) w column B-factor structure
apo. Otworz w PyMOL/ChimeraX i pokoloruj: `spectrum b` -> heat map pocket.
Saves also {name}_allosteric.pml (script PyMOL z podswietleniem top-5).
  python viz_run.py
Optimization: metric analytically, MP over proteins, GPU eig under use_gpu.
"""
import os, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aq import data, graph, propagate, rank
from aq.backend import to_numpy, get_backend

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
GPU = CFG.get("backend", {}).get("use_gpu", False)


def _score(coords):
    xp, _ = get_backend(GPU)
    A = graph.build_graph(coords, "cutoff", cutoff=8.0); gd = graph.graph_distance(A)
    Hf = graph.Hamiltonian(A, "logf", sigma=1.0); wf, Vf, _, _ = propagate.eig(Hf, GPU)
    M = to_numpy(propagate.q_interference(wf, Vf, xp, 20.0))
    _, s = rank.rank_fluctuation(M, gd, d_min=3)
    return s


def _worker(arg):
    t, clean, raw, out = arg
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    s = _score(apo.coords)
    fin = s[np.isfinite(s)]
    b = 100.0 * (s - fin.min()) / (fin.max() - fin.min() + 1e-9) # 0-100
    b = np.where(np.isfinite(b), b, 0.0)
    key2b = {(apo.keys[i][0], int(apo.keys[i][1])): float(b[i]) for i in range(len(apo.keys))}
    top5 = [apo.keys[i] for i in np.argsort(-np.where(np.isfinite(s), s, -np.inf))[:5]]
    # przepisz apo PDB z B-factor = ocena
    src = os.path.join(clean, t["apo"]); dst = os.path.join(out, f"{t['name']}_allosteric.pdb")
    with open(src) as f, open(dst, "w") as g:
        for ln in f:
            if ln.startswith(("ATOM", "HETATM")):
                ch = ln[21]; rs = ln[22:26].strip()
                try:
                    val = key2b.get((ch, int(rs)), 0.0)
                except ValueError:
                    val = 0.0
                ln = ln[:60] + f"{val:6.2f}" + ln[66:]
            g.write(ln)
    # script PyMOL
    sel = " or ".join(f"(chain {k[0]} and resi {k[1]})" for k in top5)
    pml = os.path.join(out, f"{t['name']}_allosteric.pml")
    with open(pml, "w") as g:
        g.write(f"load {os.path.basename(dst)}\nhide everything\nshow cartoon\n"
                f"spectrum b, blue_white_red\nselect top5, {sel}\n"
                f"show spheres, top5 and name CA\ncolor yellow, top5\nzoom\n")
    return t["name"], dst, [f"{k[0]}{k[1]}" for k in top5]


def main():
    clean = os.path.join(ROOT, CFG["paths"]["pdb_clean"]); raw = os.path.join(ROOT, CFG["paths"]["pdb_raw"])
    out = os.path.join(ROOT, CFG["paths"]["outputs"]); os.makedirs(out, exist_ok=True)
    T = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    with ProcessPoolExecutor(max_workers=CFG.get("backend", {}).get("n_workers", 3)) as ex:
        for name, dst, top5 in ex.map(_worker, [(t, clean, raw, out) for t in T]):
            print(f" {name:22s} -> {os.path.basename(dst)} (B-factor=ocena; top5: {', '.join(top5)})")
    print(f"\nOtworz w PyMOL: `pymol {out}/<name>_allosteric.pml` (heat map + top5)")
    print(f"or ChimeraX: open <name>_allosteric.pdb; color byattribute bfactor")


if __name__ == "__main__":
    main()

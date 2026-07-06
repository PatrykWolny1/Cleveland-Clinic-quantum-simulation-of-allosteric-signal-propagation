#!/usr/bin/env python3
"""
Noise resilience of the winning detector (LST interference) across 3 proteins.
  python noise_run.py
"""
import os, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from aq import data, graph, noise

ROOT = os.path.dirname(os.path.abspath(__file__))
GPU = False


def _gd(A): return graph.graph_distance(A)


def _worker(arg):
    t, clean = arg
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
    y = data.label_nodes(apo, pocket)
    ps, aucs = noise.dephasing_curve(apo.coords, y, _gd, sigma=1.0, T=20.0, gpu=GPU)
    cn = noise.coordinate_noise(apo.coords, y, _gd, sigmas=(0.5, 1.0, 1.5),
                                n_real=8, res_sigma=1.0, gpu=GPU)
    return t["name"], ps, aucs, cn


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    targets = [t for t in yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
               if t.get("holo")]
    with ProcessPoolExecutor(max_workers=min(3, len(targets))) as ex:
        res = list(ex.map(_worker, [(t, clean) for t in targets]))

    print("=== DECOHERENCE (p=0 quantum -> p=1 classical limit), LST detector AUC ===")
    for name, ps, aucs, _ in res:
        print(f" {name:22s} " + " ".join(f"p{p:.2f}={a:.3f}" for p, a in zip(ps, aucs)))
    print("\n=== STRUCTURAL NOISE (coordinate perturbation), AUC mean+/-std ===")
    for name, _, _, cn in res:
        print(f" {name:22s} " + " ".join(f"{s}A={m:.3f}+/-{sd:.3f}" for s, (m, sd) in cn.items()))
    # save to outputs directory
    out = os.path.join(ROOT, base["paths"]["outputs"]); os.makedirs(out, exist_ok=True)
    import json
    with open(os.path.join(out, "noise_resilience.json"), "w") as f:
        json.dump({name: {"dephasing": dict(zip([f"p{p}" for p in ps], aucs)),
                          "structural": {f"{s}A": m for s, (m, sd) in cn.items()}}
                   for name, ps, aucs, cn in res}, f, indent=2)
    print(f"\n-> {os.path.join(out, 'noise_resilience.json')}")


if __name__ == "__main__":
    main()

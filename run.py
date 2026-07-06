#!/usr/bin/env python3
"""
Orchestrator (single konfiguracja z config.yaml).
  python run.py [nazwa_targetu ...]
To przemiatania wariantow uzyj experiments.py (z cache eigendecomposition).
"""
from __future__ import annotations
import os, sys, time, yaml
from concurrent.futures import ProcessPoolExecutor

from aq import data, graph, propagate, rank, score, report
from aq.backend import capabilities

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_cfg():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "targets.yaml")) as f:
        targets = yaml.safe_load(f)["targets"]
    return cfg, targets


def _raw_path(cfg, tgt):
    raw = tgt.get("raw") or (tgt["name"].split("_")[0] + ".pdb")
    return os.path.join(ROOT, cfg["paths"].get("pdb_raw", "pdb_raw"), raw)


def run_target(tgt, cfg):
    name = tgt["name"]; t0 = time.time()
    clean = os.path.join(ROOT, cfg["paths"]["pdb_clean"])
    outdir = os.path.join(ROOT, cfg["paths"]["outputs"])
    dcfg = cfg.get("data", {})

    apo = data.load_apo(os.path.join(clean, tgt["apo"]), name,
                        tgt.get("domain_range"),
                        tgt.get("chain_select", dcfg.get("chain_select", "all")))
    A = graph.build_graph(apo.coords, cfg["graph"]["variant"],
                          **cfg["graph"].get("params", {}))
    H = graph.Hamiltonian(A, cfg["Hamiltonian"]["variant"],
                          **cfg["Hamiltonian"].get("params", {}))
    L = graph.Hamiltonian(A, "laplacian")

    mcfg = dict(cfg["metric"]); mcfg["use_gpu"] = cfg["backend"]["use_gpu"]
    M, backend_name = propagate.connectivity_matrix(H, mcfg, L=L)

    rk = cfg["ranking"]
    gd = graph.graph_distance(A)
    row = {"target": name, "n_nodes": apo.n, "backend": backend_name}
    if rk["mode"] == "active_site":
        active = tgt.get("active_site")
        active_idx = (_resolve_active(apo, active) if active
                      else data.detect_active_site(_raw_path(cfg, tgt), apo,
                           cfg.get("data", {}).get("cofactor_codes"),
                           cfg.get("data", {}).get("active_cutoff", 6.0))[0])
        _rk = dict(A=A, smooth=rk.get("smooth", 0), beta=rk.get("beta", 0.5),
                   burial=rk.get("burial", 0))
        if active_idx:
            order, sc = rank.rank_active_site(M, active_idx, gd,
                                              rk.get("normalize", "zscore"), **_rk)
            row["n_active"] = len(active_idx)
        else:
            order, sc = rank.rank_fluctuation(M, gd, rk.get("d_min", 3), rk.get("d_max"), **_rk)
            row["note"] = "none active-site -> fallback fluctuation"
    else:
        order, sc = rank.rank_fluctuation(M, gd, rk.get("d_min", 3), rk.get("d_max"),
                                          A=A, smooth=rk.get("smooth", 0),
                                          beta=rk.get("beta", 0.5), burial=rk.get("burial", 0))
    hits = rank.hit_list(order, apo.keys, apo.resnames, sc, rk.get("top", 5))

    if tgt.get("holo"):
        pocket, lig = data.extract_pocket(os.path.join(clean, tgt["holo"]),
                          tgt.get("ligand"), cfg["scoring"]["contact_cutoff"])
        labels = data.label_nodes(apo, pocket)
        row.update(score.evaluate(sc, order, labels, tuple(cfg["scoring"]["ks"])))
        row["ligand"] = lig
    else:
        row["note"] = "none holo - only hit-list"
    row["sec"] = round(time.time() - t0, 2)
    report.save_target(outdir, name, M, hits, apo.keys)
    return row, hits


def _resolve_active(apo, spec):
    want = set()
    for s in spec:
        want.add((s[0], int(s[1])) if isinstance(s, (list, tuple)) else int(s))
    return [i for i, (ch, rs, ic) in enumerate(apo.keys)
            if (ch, rs) in want or rs in want]


def main():
    cfg, targets = load_cfg()
    if len(sys.argv) > 1:
        sel = set(sys.argv[1:]); targets = [t for t in targets if t["name"] in sel]
    outdir = os.path.join(ROOT, cfg["paths"]["outputs"]); os.makedirs(outdir, exist_ok=True)
    print("backend capabilities:", capabilities())
    print("GPU requested:", cfg["backend"]["use_gpu"])
    print(f"targets: {[t['name'] for t in targets]}\n")
    rows = []
    workers = cfg["backend"].get("n_workers", 1)
    if workers > 1 and not cfg["backend"]["use_gpu"]:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in [ex.submit(run_target, t, cfg) for t in targets]:
                row, _ = fut.result(); rows.append(row); print(_fmt(row))
    else:
        for t in targets:
            row, _ = run_target(t, cfg); rows.append(row); print(_fmt(row))
    lb = report.save_leaderboard(outdir, rows)
    report.save_report(outdir, cfg, rows)
    print(f"\nleaderboard -> {lb}\noutputs -> {outdir}")


def _fmt(row):
    keys = ["target", "n_nodes", "roc_auc", "hit@5", "enrichment@5", "mwu_p", "sec"]
    return " ".join(f"{k}={row.get(k,'-')}" for k in keys)


if __name__ == "__main__":
    main()

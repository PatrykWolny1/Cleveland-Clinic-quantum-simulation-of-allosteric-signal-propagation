#!/usr/bin/env python3
"""
Variant sweep + CONSENSUS, with an eigendecomposition cache.

Optimization: eig(H)/eig(L) is computed once per (chain, graph, H) signature and
reused across all metrics (here classical gnm/resistance uses eig(L)).
Parallelism runs over targets; on GPU it runs sequentially.

  python experiments.py
"""
from __future__ import annotations
import os, copy, csv, time, yaml
from concurrent.futures import ProcessPoolExecutor

from aq import data, graph, propagate, rank, score, consensus
from aq.backend import capabilities, to_numpy

ROOT = os.path.dirname(os.path.abspath(__file__))

BAND = {"mode": "fluctuation", "d_min": 3, "d_max": 8}
EXPERIMENTS = [
    # --- quantum (distal) ---
    ("Q_coherent_band", {"metric": {"metric": "coherent"}, "ranking": dict(BAND)}),
    ("Q_adjacency_band", {"Hamiltonian": {"variant": "adjacency", "params": {}},
                          "metric": {"metric": "time_avg"}, "ranking": dict(BAND)}),
    ("Q_coherent_multi", {"metric": {"metric": "coherent_multi"}, "ranking": dict(BAND)}),
    ("Q_communic_band", {"Hamiltonian": {"variant": "adjacency", "params": {}},
                          "metric": {"metric": "q_communicability", "beta": 0.5},
                          "ranking": dict(BAND)}),
    ("C_communic_band", {"Hamiltonian": {"variant": "adjacency", "params": {}},
                          "metric": {"metric": "communicability", "beta": 0.15},
                          "ranking": dict(BAND)}),
    ("Q_communic_multi", {"Hamiltonian": {"variant": "adjacency", "params": {}},
                          "metric": {"metric": "q_communic_multi"},
                          "ranking": dict(BAND)}),
    ("A_qcommunic_zabs", {"Hamiltonian": {"variant": "adjacency", "params": {}},
                          "metric": {"metric": "q_communicability", "beta": 0.5},
                          "ranking": {"mode": "active_site", "normalize": "zabs",
                                      "d_min": 3, "d_max": 8}}),
    ("A_gnm_zabs", {"metric": {"metric": "gnm"},
                          "ranking": {"mode": "active_site", "normalize": "zabs",
                                      "d_min": 3, "d_max": 8}}),
    ("C_yukawa_band", {"metric": {"metric": "yukawa", "m2": 0.5}, "ranking": dict(BAND)}),
    # --- LST components (user documents) ---
    ("LST_interference", {"metric": {"metric": "interference"}, "ranking": dict(BAND)}),
    ("LST_logf_coherent",{"Hamiltonian": {"variant": "logf", "params": {"sigma": 1.0}},
                          "metric": {"metric": "coherent"}, "ranking": dict(BAND)}),
    ("LST_logf_interf", {"Hamiltonian": {"variant": "logf", "params": {"sigma": 1.0}},
                          "metric": {"metric": "interference"}, "ranking": dict(BAND)}),
    ("LST_interf_multi", {"metric": {"metric": "lst_interf_multi"}, "ranking": dict(BAND)}),
    # --- classical reference (challenge requires comparison) ---
    ("C_classical_band", {"metric": {"metric": "classical"}, "ranking": dict(BAND)}),
    ("C_gnm_zscore", {"metric": {"metric": "gnm"},
                          "ranking": {"mode": "active_site", "normalize": "zscore",
                                      "d_min": 3, "d_max": 8}}),
    ("C_resistance_prox",{"metric": {"metric": "resistance"},
                          "ranking": {"mode": "active_site", "normalize": "none",
                                      "d_min": 3, "d_max": 8}}),
    # --- active-site: proximal (KRAS) / two-sided / distal ---
    ("A_active_proximal",{"metric": {"metric": "time_avg"},
                          "ranking": {"mode": "active_site", "normalize": "none",
                                      "d_min": 3, "d_max": 8}}),
    ("A_active_zabs", {"metric": {"metric": "time_avg"},
                          "ranking": {"mode": "active_site", "normalize": "zabs",
                                      "d_min": 3, "d_max": 8}}),
    ("A_active_zscore", {"metric": {"metric": "time_avg"},
                          "ranking": {"mode": "active_site", "normalize": "zscore",
                                      "d_min": 3, "d_max": 8}}),
]
# consensus of complementary detectors (distal quantum + proximal + all-paths)
# strong, complementary members (no leakage): quantum distal +
# classical-limit distal + two-sided anomalous (KRAS). Best-of aggregation.
CONSENSUS_SET = ["LST_logf_interf", "LST_logf_coherent", "A_active_zabs", "Q_communic_band"]


def deep_update(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _gsig(cfg):
    g = cfg["graph"]
    return (cfg["data"]["chain_select"], g["variant"], tuple(sorted(g.get("params", {}).items())))


def process_target(tgt, base_cfg, exps):
    clean = os.path.join(ROOT, base_cfg["paths"]["pdb_clean"])
    rawdir = os.path.join(ROOT, base_cfg["paths"].get("pdb_raw", "pdb_raw"))
    # GPU auto: large N -> GPU, small -> CPU (lower overhead)
    bk = base_cfg["backend"]
    rows, orders = [], {}
    labels = None; n_nodes = 0; apo_ref = None; cons_orders = {}; has_active = False

    cfgs = [(nm, deep_update(base_cfg, ov)) for nm, ov in exps]
    ggroups = {}
    for nm, cfg in cfgs:
        ggroups.setdefault(_gsig(cfg), []).append((nm, cfg))

    for gsig, items in ggroups.items():
        cs = gsig[0]
        apo = data.load_apo(os.path.join(clean, tgt["apo"]), tgt["name"],
                            tgt.get("domain_range"), cs)
        apo_ref = apo; n_nodes = apo.n
        gpu = bool(bk.get("use_gpu")) or (bk.get("auto_gpu") and apo.n >= bk.get("n_gpu_min", 700))

        A = graph.build_graph(apo.coords, gsig[1], **dict(gsig[2]))
        L = graph.Hamiltonian(A, "laplacian")
        gd = graph.graph_distance(A)
        # eig once per role: L always; A lazily
        wL, VL, xp, bname = propagate.eig(L, gpu)
        eigL = (wL, VL); eig_ham = {} # cache eig per Hamiltonian variant

        if tgt.get("holo"):
            pocket, _ = data.extract_pocket(os.path.join(clean, tgt["holo"]),
                            tgt.get("ligand"), base_cfg["scoring"]["contact_cutoff"])
            labels = data.label_nodes(apo, pocket)
        raw = os.path.join(rawdir, tgt.get("raw") or tgt["name"].split("_")[0] + ".pdb")
        active_idx, cof = ([], None)
        if os.path.exists(raw):
            active_idx, cof = data.detect_active_site(raw, apo,
                                base_cfg.get("data", {}).get("cofactor_codes"),
                                base_cfg.get("data", {}).get("active_cutoff", 6.0))

        for nm, cfg in items:
            t0 = time.time()
            ham = cfg["Hamiltonian"]["variant"]
            hpar = cfg["Hamiltonian"].get("params", {})
            if ham == "laplacian":
                cache = {"H": eigL, "L": eigL}
            else:
                hkey = (ham, tuple(sorted(hpar.items())))
                if hkey not in eig_ham:
                    Hh = graph.Hamiltonian(A, ham, **hpar)
                    wA, VA, _, _ = propagate.eig(Hh, gpu); eig_ham[hkey] = (wA, VA)
                cache = {"H": eig_ham[hkey], "L": eigL}

            mcfg = dict(cfg["metric"]); mcfg["use_gpu"] = gpu
            rk = cfg["ranking"]
            if mcfg["metric"] == "lst_interf_multi":
                ords = []
                for sig in (0.5, 1.0, 2.0):
                    Hs = graph.Hamiltonian(A, "logf", sigma=sig)
                    ws, Vs, _, _ = propagate.eig(Hs, gpu)
                    Mk = to_numpy(propagate.q_interference(ws, Vs, xp, mcfg.get("t_max", 20.0)))
                    ok, _ = rank.rank_fluctuation(Mk, gd, rk.get("d_min", 3), rk.get("d_max"),
                                                  A=A, burial=rk.get("burial", 0))
                    ords.append(ok)
                order, sc = consensus.aggregate(ords, apo.n, "mean")
                orders[nm] = (order, sc)
                row = {"exp": nm, "target": tgt["name"], "n_nodes": apo.n,
                       "backend": bname, "active_site": "", "sec": round(time.time() - t0, 3)}
                if labels is not None:
                    row.update(score.evaluate(sc, order, labels, tuple(base_cfg["scoring"]["ks"])))
                rows.append(row); continue
            if mcfg["metric"] == "q_communic_multi":
                wA, VA = cache["H"]; ords = []
                for beta in (0.25, 0.5, 1.0, 2.0):
                    Mk = to_numpy(propagate.q_communicability(wA, VA, xp, beta))
                    ok, _ = rank.rank_fluctuation(Mk, gd, rk.get("d_min", 3), rk.get("d_max"),
                                                  A=A, burial=rk.get("burial", 0))
                    ords.append(ok)
                order, sc = consensus.aggregate(ords, apo.n, "mean")
                orders[nm] = (order, sc)
                row = {"exp": nm, "target": tgt["name"], "n_nodes": apo.n,
                       "backend": bname, "active_site": "", "sec": round(time.time() - t0, 3)}
                if labels is not None:
                    row.update(score.evaluate(sc, order, labels, tuple(base_cfg["scoring"]["ks"])))
                rows.append(row); continue
            if mcfg["metric"] == "coherent_multi":
                wH, VH = cache["H"]
                ws = to_numpy(xp.sort(wH)); import numpy as _np
                gap = float(_np.median(_np.diff(ws))); base = 1.0 / max(abs(gap), 1e-6)
                ords = []
                for fct in (0.5, 1.0, 2.0, 4.0):
                    Mk = to_numpy(propagate.q_coherent(wH, VH, xp, t_star=base * fct))
                    ok, _ = rank.rank_fluctuation(Mk, gd, rk.get("d_min", 3), rk.get("d_max"),
                                                  A=A, smooth=rk.get("smooth", 0),
                                                  beta=rk.get("beta", 0.5), burial=rk.get("burial", 0))
                    ords.append(ok)
                order, sc = consensus.aggregate(ords, apo.n, "mean")
                orders[nm] = (order, sc)
                row = {"exp": nm, "target": tgt["name"], "n_nodes": apo.n,
                       "backend": bname, "active_site": "", "sec": round(time.time() - t0, 3)}
                if labels is not None:
                    row.update(score.evaluate(sc, order, labels, tuple(base_cfg["scoring"]["ks"])))
                rows.append(row); continue

            M = propagate.metric_from_cache(cache, mcfg, xp)
            if rk["mode"] == "active_site" and active_idx:
                has_active = True
                order, sc = rank.rank_active_site(M, active_idx, gd, rk.get("normalize", "zscore"),
                                    A=A, smooth=rk.get("smooth", 0), beta=rk.get("beta", 0.5),
                                    burial=rk.get("burial", 0))
                asite = f"{cof}:{len(active_idx)}"
            else:
                order, sc = rank.rank_fluctuation(M, gd, rk.get("d_min", 3), rk.get("d_max"),
                                    A=A, smooth=rk.get("smooth", 0), beta=rk.get("beta", 0.5),
                                    burial=rk.get("burial", 0))
                asite = "none->fluct" if rk["mode"] == "active_site" else ""
            orders[nm] = (order, sc)
            row = {"exp": nm, "target": tgt["name"], "n_nodes": apo.n,
                   "backend": bname, "active_site": asite, "sec": round(time.time() - t0, 3)}
            if labels is not None:
                row.update(score.evaluate(sc, order, labels, tuple(base_cfg["scoring"]["ks"])))
            rows.append(row)

    if all(k in orders for k in CONSENSUS_SET):
        members = [orders[k][0] for k in CONSENSUS_SET]
        for mode, tag in (("min", "CONSENSUS_bestof"), ("mean", "CONSENSUS_mean"),
                          ("rrf", "CONSENSUS_rrf")):
            c_order, c_score = consensus.aggregate(members, n_nodes, mode)
            cons_orders[tag] = (c_order, c_score)
            row = {"exp": tag, "target": tgt["name"], "n_nodes": n_nodes,
                   "backend": "-", "active_site": "min" if mode == "min" else "mean", "sec": 0.0}
            if labels is not None:
                row.update(score.evaluate(c_score, c_order, labels, tuple(base_cfg["scoring"]["ks"])))
            rows.append(row)
    return rows, apo_ref, cons_orders, orders, has_active


def main():
    with open(os.path.join(ROOT, "config.yaml")) as f:
        base_cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "targets.yaml")) as f:
        targets = [t for t in yaml.safe_load(f)["targets"] if t.get("holo")]
    base_cfg["data"]["chain_select"] = "largest"
    print("capabilities:", capabilities(), "| GPU:", base_cfg["backend"]["use_gpu"])

    rows = []
    bk = base_cfg["backend"]
    use_any_gpu = bool(bk.get("use_gpu")) or bool(bk.get("auto_gpu"))
    workers = bk.get("n_workers", 1)
    if workers > 1 and not use_any_gpu:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for fut in [ex.submit(process_target, t, base_cfg, EXPERIMENTS) for t in targets]:
                rows.extend(fut.result()[0])
    else:
        for t in targets:
            rows.extend(process_target(t, base_cfg, EXPERIMENTS)[0])

    for r in sorted(rows, key=lambda x: (x["target"], x["exp"])):
        print(f"[{r['exp']:20s}] {r['target']:20s} N={r['n_nodes']:4d} "
              f"AUC={r.get('roc_auc','-')} hit@5={r.get('hit@5','-')} "
              f"enr@5={r.get('enrichment@5','-')} p={r.get('mwu_p','-')} "
              f"{r.get('active_site','')}")

    out = os.path.join(ROOT, base_cfg["paths"]["outputs"]); os.makedirs(out, exist_ok=True)
    pref = ["exp", "target", "n_nodes", "roc_auc", "hit@5", "enrichment@5", "mwu_p", "active_site"]
    cols = pref + [c for c in sorted({k for r in rows for k in r}) if c not in pref]
    with open(os.path.join(out, "leaderboard_experiments.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)

    print("\n=== mean over targets ===")
    by = {}
    for r in rows:
        by.setdefault(r["exp"], []).append(r)
    for exp, rs in sorted(by.items()):
        aucs = [x["roc_auc"] for x in rs if isinstance(x.get("roc_auc"), (int, float))]
        hits = sum(x.get("hit@5", 0) for x in rs)
        m = sum(aucs) / len(aucs) if aucs else float("nan")
        print(f" {exp:20s} mean_AUC={m:.3f} hit@5={hits}/{len(rs)}")
    print(f"\n-> {os.path.join(out, 'leaderboard_experiments.csv')}")


if __name__ == "__main__":
    main()

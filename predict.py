#!/usr/bin/env python3
"""
Final prediction (deliverable #2). Saves Three variants top-5 per target:

  *_hitlist_precision.json : Q_communic_band - best hit@5 (3/3 in walidacji)
                             -> main lista miejsc allosterycznych
  *_hitlist_adaptive.json : regula leakage-free (cofactor->A_active_zabs,
                             inaczej Q_coherent_band) - detector separacji (AUC)
  *_hitlist_consensus.json : best-of komplementarnych detectors

Everything z this samej sciezki co experiments.py (cache eigow).
  python predict.py
"""
import os, json, csv, yaml
from experiments import process_target, EXPERIMENTS
from aq import rank

ROOT = os.path.dirname(os.path.abspath(__file__))


def _save(out, name, hits, suffix):
    with open(os.path.join(out, f"{name}_hitlist_{suffix}.json"), "w") as f:
        json.dump(hits, f, indent=2)
    with open(os.path.join(out, f"{name}_hitlist_{suffix}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hits[0].keys())); w.writeheader(); w.writerows(hits)


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    base["data"]["chain_select"] = "largest"
    targets = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    out = os.path.join(ROOT, base["paths"]["outputs"]); os.makedirs(out, exist_ok=True)

    for t in targets:
        _, apo, cons, orders, has_active = process_target(t, base, EXPERIMENTS)
        K = apo.keys, apo.resnames

        # 1) precyzja (main deliverable)
        o, s = orders["Q_communic_band"]
        prec = rank.hit_list(o, *K, s, top=5)
        _save(out, t["name"], prec, "precision")

        # 2) adaptacyjny detector separacji (AUC):
        # cofactor -> better z {A_active_zabs, A_gnm_zabs} by decisiveness
        # (peakedness score, unsupervised, leakage-free);
        # inaczej -> Q_adjacency_band (lider separacji in ABL/myosin)
        import numpy as _np
        def _peak(o_s):
            v = _np.asarray(o_s[1], float); v = v[_np.isfinite(v)]
            if v.std() < 1e-12: return 0.0
            z = (v - v.mean()) / v.std(); return float((z ** 4).mean()) # kurtosis = decisiveness
        # pula candidates: LST (ABL/myosin) + active-site (KRAS) if detected
        pool = ["LST_logf_interf", "LST_logf_coherent"]
        if has_active:
            pool += [k for k in ("A_active_zabs", "A_gnm_zabs") if k in orders]
        pool = [k for k in pool if k in orders]
        key = max(pool, key=lambda k: _peak(orders[k])) # most decisive
        o2, s2 = orders[key]
        _save(out, t["name"], rank.hit_list(o2, *K, s2, top=5), "adaptive")

        # 3) consensus (best-of + rrf)
        for tag, suf in (("CONSENSUS_bestof", "consensus"), ("CONSENSUS_rrf", "consensus_rrf")):
            if tag in cons:
                co, cs = cons[tag]
                _save(out, t["name"], rank.hit_list(co, *K, cs, top=5), suf)

        top = ", ".join(f"{h['chain']}{h['resseq']}{h['resname']}" for h in prec)
        print(f"{t['name']:22s} [precision] {top}")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()

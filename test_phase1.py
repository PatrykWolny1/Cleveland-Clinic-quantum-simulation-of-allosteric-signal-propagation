"""Test jakosci Phase I - plumbing + poprawnosc scoringu."""
import os, numpy as np
from aq import data, graph, propagate, rank, score

ROOT = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(ROOT, "pdb_clean")

def build(metric="time_avg", mode="fluctuation"):
    apo = data.load_apo(os.path.join(CLEAN, "SYN_apo.pdb"), "SYN")
    A = graph.build_graph(apo.coords, "cutoff", cutoff=8.0)
    H = graph.Hamiltonian(A, "laplacian")
    M, bk = propagate.connectivity_matrix(
        H, {"metric": metric, "t_max": 20.0, "n_t": 64, "use_gpu": False})
    gd = graph.graph_distance(A)
    if mode == "active_site":
        order, sc = rank.rank_active_site(M, list(range(20)), exclude_idx=range(20))
    else:
        order, sc = rank.rank_fluctuation(M, gd, d_min=3)
    pocket, lig = data.extract_pocket(os.path.join(CLEAN, "SYN_holo.pdb"),
                                      None, 5.0)
    labels = data.label_nodes(apo, pocket)
    ev = score.evaluate(sc, order, labels, (5, 10, 20))
    return apo, A, M, order, sc, labels, ev, bk

print("=== 1) Sanity: structure graph i matrix ===")
apo, A, M, order, sc, labels, ev, bk = build()
print(f"backend={bk} N={apo.n} edges={int(A.sum()//2)} "
      f"meana degree={A.sum(1).mean():.1f}")
print(f"M symetryczna? {np.allclose(M, M.T)} M>=0? {(M>=-1e-12).all()} "
      f"ground-truth residues pocket={int(labels.sum())}")

print("\n=== 2) Result metric CTQW (fluctuation) ===")
print({k: ev[k] for k in ev})
top = order[:5]
print("top-5 indeksy:", list(top), "| which this pocket (32..45):",
      [int(i) for i in top if 32 <= i <= 45])

print("\n=== 3) Tryb active_site (core 0..19 as active site) ===")
*_, ev2, _ = build(mode="active_site")
print({k: ev2[k] for k in ev2})

print("\n=== 4) Walidacja scorera (niedependent from metric) ===")
lab = labels.astype(float)
auc_perfect = score.roc_auc(lab, labels) # labels as score
rng = np.random.default_rng(0)
aucs = [score.roc_auc(rng.random(len(labels)), labels) for _ in range(200)]
print(f"AUC(perfect scores) = {auc_perfect:.3f} (oczek. 1.000)")
print(f"AUC(random) mean = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} (oczek. ~0.5)")

print("\n=== 5) Comparison: metric quantum vs losowy baseline ===")
auc_q = ev["roc_auc"]
print(f"AUC CTQW = {auc_q:.3f} vs losowo ~0.5 -> "
      f"{'sygnal obecny' if auc_q>0.55 else 'none przeweights'}")

"""Test Phase II on SYN: new variants + distal fix."""
import os, numpy as np
from aq import data, graph, propagate, rank, score
CLEAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdb_clean")

def pipe(chain="all", gvar="cutoff", gp=None, hvar="laplacian",
         metric="time_avg", mode="fluctuation", d_min=3, d_max=None, wdist=True):
    apo = data.load_apo(os.path.join(CLEAN,"SYN_apo.pdb"),"SYN",None,chain)
    A = graph.build_graph(apo.coords, gvar, **(gp or {"cutoff":8.0}))
    H = graph.Hamiltonian(A, hvar)
    M,_ = propagate.connectivity_matrix(H,{"metric":metric,"t_max":20.0,"n_t":64,"use_gpu":False})
    gd = graph.graph_distance(A)
    if mode=="active_site":
        order,sc = rank.rank_active_site(M, list(range(20)), gd, wdist, exclude_idx=range(20))
    else:
        order,sc = rank.rank_fluctuation(M, gd, d_min, d_max)
    pocket,_ = data.extract_pocket(os.path.join(CLEAN,"SYN_holo.pdb"),None,5.0)
    labels = data.label_nodes(apo, pocket)
    ev = score.evaluate(sc, order, labels,(5,10,20))
    intop5 = [int(i) for i in order[:5] if 32<=i<=45]
    return ev, intop5

print("graph distance (csgraph) - sanity:")
apo = data.load_apo(os.path.join(CLEAN,"SYN_apo.pdb"),"SYN")
A = graph.build_graph(apo.coords,"cutoff",cutoff=8.0)
D = graph.graph_distance(A)
print(f" D shape={D.shape} finite%={100*np.isfinite(D).mean():.0f} max={np.nanmax(D[np.isfinite(D)]):.0f}")

print("\nvariants (SYN, pocket=32..45):")
for tag,kw in [
    ("baseline time_avg/laplacian/fluct", {}),
    ("knn tau k=10", {"gvar":"knn","gp":{"k":10}}),
    ("tight_binding", {"hvar":"tight_binding"}),
    ("coherent metric", {"metric":"coherent"}),
    ("dist band d[3,8]", {"d_min":3,"d_max":8}),
    ("active_site RAW (wdist=False)", {"mode":"active_site","wdist":False}),
    ("active_site DIST-WEIGHTED", {"mode":"active_site","wdist":True}),
]:
    ev,intop5 = pipe(**kw)
    print(f" {tag:36s} AUC={ev.get('roc_auc'):.3f} hit@5={ev.get('hit@5')} "
          f"enr@5={ev.get('enrichment@5')} pocket_in_top5={intop5}")

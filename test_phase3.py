"""Phase III in SYN: kontrast + z-score per-powloka. active-site = core 0..19."""
import os, numpy as np
from aq import data, graph, propagate, rank, score
CLEAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pdb_clean")

apo = data.load_apo(os.path.join(CLEAN,"SYN_apo.pdb"),"SYN",None,"all")
A = graph.build_graph(apo.coords,"cutoff",cutoff=8.0)
gd = graph.graph_distance(A)
pocket,_ = data.extract_pocket(os.path.join(CLEAN,"SYN_holo.pdb"),None,5.0)
labels = data.label_nodes(apo, pocket)
active = list(range(20)) # core = active site

def run(metric, hvar, mode, normalize="zscore", d_min=3, d_max=None):
    H = graph.Hamiltonian(A, hvar)
    L = graph.Hamiltonian(A, "laplacian")
    M,_ = propagate.connectivity_matrix(H,{"metric":metric,"t_max":20.0,
                        "n_t":64,"use_gpu":False}, L=L)
    if mode=="active_site":
        order,sc = rank.rank_active_site(M, active, gd, normalize, exclude_idx=active)
    else:
        order,sc = rank.rank_fluctuation(M, gd, d_min, d_max)
    ev = score.evaluate(sc, order, labels,(5,10,20))
    intop5 = [int(i) for i in order[:5] if 32<=i<=45]
    return ev, intop5

print(f"pocket=32..45 ({int(labels.sum())} reszt), active=0..19\n")
rows = [
    ("time_avg | active RAW", ("time_avg","laplacian","active_site","none")),
    ("time_avg | active DIST", ("time_avg","laplacian","active_site","distance")),
    ("time_avg | active ZSCORE", ("time_avg","laplacian","active_site","zscore")),
    ("contrast | active ZSCORE", ("contrast","laplacian","active_site","zscore")),
    ("classical | active ZSCORE", ("classical","laplacian","active_site","zscore")),
    ("contrast | fluctuation band", ("contrast","laplacian","fluctuation","zscore")),
]
for tag,args in rows:
    ev,intop5 = run(*args)
    print(f" {tag:32s} AUC={ev.get('roc_auc'):.3f} hit@5={ev.get('hit@5')} "
          f"enr@5={ev.get('enrichment@5')} pocket_in_top5={intop5}")

# detection active-site z mini-raw (cofactor GDP at rdzeniu)
print("\n--- test detection active-site z raw ---")
raw = os.path.join(CLEAN,"SYN_raw.pdb")
with open(os.path.join(CLEAN,"SYN_apo.pdb")) as f: base=f.read().replace("END\n","")
core_c = apo.coords[:20].mean(0)
with open(raw,"w") as f:
    f.write(base)
    for j,(dx,dy,dz) in enumerate(np.random.RandomState(1).randn(8,3)*1.5,1):
        x,y,z = core_c+[dx,dy,dz]
        f.write(f"HETATM{9000+j:5d} P{j:<2d} GDP A 800 {x:8.3f}{y:8.3f}{z:8.3f} 1.00 0.00 P\n")
    f.write("END\n")
idx,code = data.detect_active_site(raw, apo, cutoff=6.0)
print(f" detected cofactor={code}, residues active-site={len(idx)}, "
      f"whether w rdzeniu 0..19: {all(i<20 for i in idx)}")

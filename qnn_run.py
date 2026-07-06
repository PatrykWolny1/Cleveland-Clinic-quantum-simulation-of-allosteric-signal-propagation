#!/usr/bin/env python3
"""
Quantum neural network (VQC) - a learned combinator of quantum features.

LEAVE-ONE-PROTEIN-OUT evaluation (leakage-free): train on 2 proteins,
predict the 3rd without its labels. Comparison against:
  - best single feature (unsupervised),
  - classical logistic regression on the same features (Quantum vs classical AI).

  python qnn_run.py
"""
import os, yaml, numpy as np
from aq import data, graph, propagate, qnn, score
from aq.backend import to_numpy, get_backend

ROOT = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["LST_interf", "coherent", "q_communic", "classical", "degree", "centrality"]
N_Q = len(FEATURES)


def _band(M, gd, d_min=3):
    W = M.astype(float).copy(); np.fill_diagonal(W, 0.0)
    b = np.isfinite(gd) & (gd >= d_min)
    return (W * b).sum(1)


def extract_features(apo, A, gd, gpu=False):
    L = graph.Hamiltonian(A, "laplacian")
    Hf = graph.Hamiltonian(A, "logf", sigma=1.0)
    wL, VL, xp, _ = propagate.eig(L, gpu)
    wA, VA, _, _ = propagate.eig(A, gpu)
    wF, VF, _, _ = propagate.eig(Hf, gpu)
    f = {
        "LST_interf": _band(to_numpy(propagate.q_interference(wF, VF, xp, 20.0)), gd),
        "coherent": _band(to_numpy(propagate.q_coherent(wL, VL, xp, None)), gd),
        "q_communic": _band(to_numpy(propagate.q_communicability(wA, VA, xp, 0.5)), gd),
        "classical": _band(to_numpy(propagate.heat_avg_analytic(wL, VL, xp, 20.0)), gd),
        "degree": A.sum(1),
        "centrality": np.array([np.mean(gd[i][np.isfinite(gd[i])]) for i in range(len(A))]),
    }
    X = np.column_stack([f[k] for k in FEATURES]).astype(float)
    # standardize per protein + squash to [-1,1] (angular encoding)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    return np.tanh(X)


def load_all(targets, clean):
    data_by = {}
    for t in targets:
        apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
        A = graph.build_graph(apo.coords, "cutoff", cutoff=8.0)
        gd = graph.graph_distance(A)
        X = extract_features(apo, A, gd)
        pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
        y = data.label_nodes(apo, pocket)
        data_by[t["name"]] = (X, y)
    return data_by


def logistic_np(Xtr, ytr, Xte, iters=500, lr=0.3):
    w = np.zeros(Xtr.shape[1]); b = 0.0
    pos = max(1.0, ytr.sum()); neg = max(1.0, (1 - ytr).sum())
    sw = np.where(ytr > 0.5, neg / pos, 1.0)
    for _ in range(iters):
        z = Xtr @ w + b; p = 1 / (1 + np.exp(-z)); g = sw * (p - ytr)
        w -= lr * (Xtr.T @ g) / len(ytr); b -= lr * g.mean()
    return 1 / (1 + np.exp(-(Xte @ w + b)))


def auc(pred, y):
    o = np.argsort(-pred)
    return score.evaluate(pred, o, y, (5,)).get("roc_auc")


def run(data_by, GPU=False):
    names = list(data_by)
    print(f"Leave-one-protein-out ({N_Q} quantum features, {N_Q} qubits, VQC L=2)\n")
    q_aucs, k_aucs, c_aucs = [], [], []
    for test in names:
        Xtr = np.vstack([data_by[n][0] for n in names if n != test])
        ytr = np.concatenate([data_by[n][1] for n in names if n != test])
        Xte, yte = data_by[test]
        # 1) VQC (variational) - SPSA
        theta = qnn.train_spsa(Xtr, ytr, N_Q, L=2, iters=150, seed=0, use_gpu=GPU)
        pq = qnn.predict(Xte, theta, N_Q, L=2, use_gpu=GPU)
        # 2) quantum kernel (QSVM) - convex training, no barren plateau
        xp, _ = get_backend(GPU)
        Ptr = qnn.zz_feature_states(Xtr, 2, xp); Pte = qnn.zz_feature_states(Xte, 2, xp)
        Ktr = qnn.quantum_kernel(Ptr, Ptr, xp); Kte = qnn.quantum_kernel(Pte, Ptr, xp)
        pk = qnn.kernel_logistic(Ktr, ytr.astype(float), Kte)
        # 3) classical logistic regression (baseline)
        pc = logistic_np(Xtr, ytr, Xte)
        aq_, ak_, ac_ = auc(pq, yte), auc(pk, yte), auc(pc, yte)
        best_feat = max(range(N_Q), key=lambda j: max(auc(Xte[:, j], yte), auc(-Xte[:, j], yte)))
        bf = max(auc(Xte[:, best_feat], yte), auc(-Xte[:, best_feat], yte))
        q_aucs.append(aq_); k_aucs.append(ak_); c_aucs.append(ac_)
        print(f" test={test:22s} VQC={aq_:.3f} QKernel={ak_:.3f} logistic={ac_:.3f} "
              f"best_single={bf:.3f} ({FEATURES[best_feat]})")
    print(f"\n mean: VQC={np.mean(q_aucs):.3f} QKernel={np.mean(k_aucs):.3f} "
          f"logistic={np.mean(c_aucs):.3f}")


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    targets = [t for t in yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
               if t.get("holo")]
    run(load_all(targets, clean))


if __name__ == "__main__":
    main()

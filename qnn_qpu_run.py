#!/usr/bin/env python3
"""
Quantum neural network for the QPU (data re-uploading) on ALL features.
PCA(full feature set)->6 components -> QNN 3 qubits -> leave-one-out + c-Myc.
Export circuit (Braket/Classiq).
  python qnn_qpu_run.py
"""
import os, yaml, numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.decomposition import PCA
from aq import data, spaces, qnn_qpu, score

ROOT = os.path.dirname(os.path.abspath(__file__)); GPU = False
N_QUBITS = 3; N_PCA = 2*N_QUBITS; L = 3


def auc(p, y): return score.evaluate(p, np.argsort(-p), y, (5,)).get("roc_auc")


def _worker(arg):
    t, clean = arg
    apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
    D, names = spaces.all_features(apo.coords, GPU)
    D = spaces.rank_gauss(D) # align distributions (transfer)
    y = None
    if t.get("holo"):
        pocket, _ = data.extract_pocket(os.path.join(clean, t["holo"]), None, 5.0)
        y = data.label_nodes(apo, pocket)
    return t["name"], D, y, apo.keys, apo.resnames, names


def _prep(Xtr, Xte, pca):
    Ztr = pca.transform(Xtr); Zte = pca.transform(Xte)
    mu, sd = Ztr.mean(0), Ztr.std(0)+1e-9
    return np.tanh((Ztr-mu)/sd), np.tanh((Zte-mu)/sd)


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    clean = os.path.join(ROOT, base["paths"]["pdb_clean"])
    out = os.path.join(ROOT, base["paths"]["outputs"]); os.makedirs(out, exist_ok=True)
    T = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    train = [t for t in T if t.get("holo")]; test = [t for t in T if not t.get("holo")]
    with ProcessPoolExecutor(max_workers=min(4, len(T))) as ex:
        res = list(ex.map(_worker, [(t, clean) for t in (train+test)]))
    D = {r[0]: (r[1], r[2], r[3], r[4]) for r in res}; names = res[0][5]
    tr = [t["name"] for t in train]
    print(f"QNN for QPU (data re-uploading, {N_QUBITS} qubits, L={L}) on {len(names)} features\n")

    aucs = []
    for these in tr:
        Xtr = np.vstack([D[n][0] for n in tr if n != these])
        ytr = np.concatenate([D[n][1] for n in tr if n != these]).astype(float)
        pca = PCA(n_components=N_PCA).fit(Xtr)
        Ztr, Zte = _prep(Xtr, D[these][0], pca)
        theta = qnn_qpu.train(Ztr, ytr, N_QUBITS, L=L, steps=120)
        p = np.asarray(qnn_qpu.predict(Zte, theta, N_QUBITS, L))
        a = auc(p, D[these][1]); aucs.append(a)
        print(f" test={these:22s} QNN_QPU AUC={a:.3f}")
    print(f"\n mean QNN_QPU={np.mean(aucs):.3f} (variance explained by PCA: "
          f"{PCA(n_components=N_PCA).fit(np.vstack([D[n][0] for n in tr])).explained_variance_ratio_.sum():.2f})")

    # final model on 3 proteins -> c-Myc + export circuit
    Xall = np.vstack([D[n][0] for n in tr]); yall = np.concatenate([D[n][1] for n in tr]).astype(float)
    pca = PCA(n_components=N_PCA).fit(Xall)
    Zall, _ = _prep(Xall, Xall, pca)
    theta = qnn_qpu.train(Zall, yall, N_QUBITS, L=L, steps=120)
    for t in test:
        Ztr, Zte = _prep(Xall, D[t["name"]][0], pca)
        p = np.asarray(qnn_qpu.predict(Zte, theta, N_QUBITS, L))
        order = np.argsort(-p); keys, resn = D[t["name"]][2], D[t["name"]][3]
        print(f"\n QNN_QPU -> {t['name']} top5: " +
              ", ".join(f"{keys[i][0]}{keys[i][1]}{resn[i]}" for i in order[:5]))
    circ = qnn_qpu.export_circuit(theta, N_QUBITS, L)
    open(os.path.join(out, "qnn_qpu_circuit.txt"), "w").write(circ)
    print(f"\n-> QPU circuit: {os.path.join(out, 'qnn_qpu_circuit.txt')} (Braket/Classiq)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Execution on QPU (CTQW): coarse-graining -> real circuit XY-hopping (Qiskit Aer),
agreement z classicalm e^{-iHt} (Trotter error), budzet resources (qubits/depth/
gates 2q) i degradation under noise. Ready under AWS Braket / Classiq / IBM.
  python qpu_run.py
This jest kod Under QPU -> reguła CPU/GPU yields; symulator zastapi hardware.
"""
import os, yaml, numpy as np
from aq import data, graph, coarse, qpu_ctqw as Q

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
M_QUBITS = 12 # qubit budget (coarse-grained nodes)
T = 6.0; STEPS = 12


def main():
    clean = os.path.join(ROOT, CFG["paths"]["pdb_clean"])
    T_ = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    print(f"CTQW on QPU (coarse -> {M_QUBITS} qubits, Trotter steps={STEPS})\n")
    print(f" {'protein':22s} {'qubits':>6s} {'depth':>6s} {'2q':>5s} {'blad_Trot':>9s} {'kor_noise(p=.01)':>15s}")
    for t in T_:
        apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
        A = graph.build_graph(apo.coords, "cutoff", cutoff=8.0)
        labels, Ac, sizes = coarse.spectral_coarsen(A, M_QUBITS)
        Ac = (Ac > 0).astype(float); np.fill_diagonal(Ac, 0.0)
        src = int(np.argmax(Ac.sum(1))) # node-source (najwyzszy degree)
        qc = Q.build_ctqw(Ac, src, t=T, steps=STEPS)
        pv = Q.statevector_probs(qc); cl = Q.classical_ctqw(Ac, src, t=T)
        err = float(np.abs(pv - cl).max())
        res = Q.resources(qc)
        pn = Q.noisy_probs(qc, p_depol=0.01, shots=2048)
        kor = float(np.corrcoef(pv, pn)[0, 1])
        print(f" {t['name']:22s} {res['qubits']:6d} {res['depth']:6d} {res['2q_gates']:5d} "
              f"{err:9.4f} {kor:15.3f}")
    print(f"\nBudzet: 1 qubit / node coarse. Full protein (N={{170..954}}) -> coarse to "
          f"budzetu hardwareu (IBM/IonQ ~100+ qubits). Depth ~ steps x edges.")
    print("Export in hardware: qc.qasm() / Braket / Classiq. Noise -> mitigation + plytsze circuits.")


if __name__ == "__main__":
    main()

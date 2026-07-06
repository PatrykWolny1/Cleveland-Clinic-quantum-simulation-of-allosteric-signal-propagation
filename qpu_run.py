#!/usr/bin/env python3
"""
QPU execution profile (CTQW). Per protein: coarse-grain -> real XY-hopping circuit
(Qiskit Aer), then report the full near-term-hardware feasibility profile:
  - qubit budget (= coarse nodes), native depth and 2-qubit gate count,
  - hardware-basis transpile (cx/rz/sx/x): depth + CX count (the honest numbers),
  - estimated surviving circuit fidelity from CX count at IBM-class error rates,
  - Trotter convergence (accuracy vs steps) and fidelity vs the classical walk,
  - noise degradation sweep (depolarizing) -> correlation with the clean result.
Circuits export to AWS Braket / Classiq / IBM (qc: OpenQASM).
  python qpu_run.py
This is code written FOR QPU -> the CPU/GPU speed rule yields to the device;
the Aer simulator stands in for hardware during development.
"""
import os, yaml, numpy as np
from aq import data, graph, coarse, qpu_ctqw as Q

ROOT = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
M_QUBITS = 10          # qubit budget (coarse-grained nodes)
T = 6.0; STEPS = 12    # walk time and Trotter steps (accuracy/depth trade-off)
NOISE_P = (0.005, 0.01, 0.02)


def main():
    clean = os.path.join(ROOT, CFG["paths"]["pdb_clean"])
    targets = yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
    print(f"CTQW on QPU - feasibility profile (coarse -> {M_QUBITS} qubits, Trotter={STEPS})\n")
    hdr = (f"  {'protein':22s}{'N':>5s}{'qb':>4s}{'depth':>6s}{'2q':>5s} | "
           f"{'hw_dep':>6s}{'CX':>5s}{'fid.5%':>7s}{'fid1%':>7s} | {'Trot':>6s} | noise corr p="
           + "/".join(str(p) for p in NOISE_P))
    print(hdr); print("  " + "-"*len(hdr))
    for t in targets:
        apo = data.load_apo(os.path.join(clean, t["apo"]), t["name"], None, "largest")
        N = len(apo.coords)
        A = graph.build_graph(apo.coords, "cutoff", cutoff=8.0)
        labels, Ac, sizes = coarse.spectral_coarsen(A, M_QUBITS)
        Ac = (Ac > 0).astype(float); np.fill_diagonal(Ac, 0.0)
        if np.triu(Ac, 1).sum() == 0:
            print(f"  {t['name']:22s} coarse graph disconnected - skip"); continue
        src = int(np.argmax(Ac.sum(1)))
        qc = Q.build_ctqw(Ac, src, t=T, steps=STEPS)
        nat, hw = Q.resources_hw(qc)
        f05 = Q.estimated_fidelity(hw["cx"], hw["1q"], 0.005)
        f1 = Q.estimated_fidelity(hw["cx"], hw["1q"], 0.01)
        cl = Q.classical_ctqw(Ac, src, T); pv = Q.statevector_probs(qc)
        trot = float(np.abs(pv - cl).max())
        cors = []
        for p in NOISE_P:
            pn = Q.noisy_probs(qc, p_depol=p, shots=1024)
            cors.append(np.corrcoef(pv, pn)[0, 1])
        cstr = "/".join(f"{c:.2f}" for c in cors)
        print(f"  {t['name']:22s}{N:>5d}{nat['qubits']:>4d}{nat['depth']:>6d}{nat['2q']:>5d} | "
              f"{hw['depth']:>6d}{hw['cx']:>5d}{f05:>7.2f}{f1:>7.2f} | {trot:>6.2f} | {cstr}")
    print(f"\n  Qubit budget = 1 qubit / coarse node; full protein (N shown) -> coarsen to the")
    print(f"  hardware budget (IBM/IonQ 100+ qubits). Trade-off: more Trotter steps lower the")
    print(f"  Trotter error but raise depth+CX -> lower fidelity; coarse-graining + error")
    print(f"  mitigation keep circuits shallow. Export: OpenQASM -> AWS Braket / Classiq / IBM.")


if __name__ == "__main__":
    main()

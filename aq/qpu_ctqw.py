"""
Real circuit QPU: CTQW as XY-hopping w single-excitation subspace.

Node = qubit; excitation (one |1>) propagates as quantum walk.
H = sum_{(i,j)} w_ij (X_iX_j + Y_iY_j)/2 -> e^{-iHt} Trotter over edges,
each term = RXX(theta)*RYY(theta) (commute, exact per edge; Trotter error
only between edges sharing node). Amplitude in node k =
<k|e^{-iHt}|source> -> |.|^2 = metric interference/transition (QPU-native).
Executable in AWS Braket / Classiq / IBM. Here Qiskit Aer (statevector + noise).
"""
from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit


def build_ctqw(Acoarse, source, t=20.0, steps=4):
    M = Acoarse.shape[0]
    qc = QuantumCircuit(M)
    qc.x(int(source)) # excitation in source
    edges = [(i, j, float(Acoarse[i, j])) for i in range(M) for j in range(i+1, M) if Acoarse[i, j] > 0]
    dt = t / steps
    for _ in range(steps):
        for i, j, w in edges:
            th = w * dt
            qc.rxx(th, i, j); qc.ryy(th, i, j)
    return qc


def classical_ctqw(Acoarse, source, t=20.0):
    """Exact |<k|e^{-iHt}|source>|^2 w podspace 1-excitation (H=hopping)."""
    H = Acoarse.astype(float).copy(); np.fill_diagonal(H, 0.0)
    w, V = np.linalg.eigh(H)
    psi0 = np.zeros(H.shape[0]); psi0[int(source)] = 1.0
    psi = V @ (np.exp(-1j*w*t) * (V.T @ psi0))
    return np.abs(psi)**2


def statevector_probs(qc):
    """|amplitude|^2 in nodech (stany one-hot) z simulation statevector."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    M = qc.num_qubits
    qc2 = qc.copy(); qc2.save_statevector()
    sim = AerSimulator(method="statevector")
    sv = sim.run(transpile(qc2, sim)).result().get_statevector()
    data = np.asarray(sv)
    probs = np.zeros(M)
    for k in range(M):
        idx = 1 << k # stan one-hot |...1_k...>
        probs[k] = abs(data[idx])**2
    return probs


def noisy_probs(qc, p_depol=0.01, shots=4096):
    """Prawdopodobienstwa nodes z model noise depolarizing (hardware near-term)."""
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
    from qiskit import transpile
    M = qc.num_qubits
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p_depol, 1), ["rx", "ry", "x", "h"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p_depol*2, 2), ["rxx", "ryy", "cx"])
    qc2 = qc.copy(); qc2.measure_all()
    sim = AerSimulator(noise_model=nm)
    counts = sim.run(transpile(qc2, sim), shots=shots).result().get_counts()
    probs = np.zeros(M)
    for bits, c in counts.items():
        b = bits.replace(" ", "")[::-1] # little-endian
        if b.count("1") == 1: # podspace 1-excitation
            probs[b.index("1")] += c
    s = probs.sum()
    return probs/s if s > 0 else probs


def resources(qc):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    tq = transpile(qc, AerSimulator(), optimization_level=1)
    return {"qubits": qc.num_qubits, "depth": tq.depth(), "gates": sum(tq.count_ops().values()),
            "2q_gates": tq.count_ops().get("rxx", 0)+tq.count_ops().get("ryy", 0)+tq.count_ops().get("cx", 0)}

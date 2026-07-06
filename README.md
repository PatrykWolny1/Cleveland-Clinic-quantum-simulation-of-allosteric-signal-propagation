# GlobalQuantum

Quantum simulation of allosteric signal propagation for predicting hidden
(allosteric / cryptic) drug-binding pockets in "undruggable" proteins, directly
from a static apo structure and its contact topology — no classical molecular
dynamics. Built for the Cleveland Clinic problem, 2026 Global Quantum + AI Challenge.

The core metric is a sign-carrying, phase-driven interference of a continuous-time
quantum walk on the residue contact graph (an original method adapted to this
problem). The final predictor fuses an unsupervised pocket-geometry router with a
learned cross-protein transfer model.

---

## 1. Running order

Prerequisites:
- conda env `torch_gpu` (numpy, scipy, scikit-learn; CuPy/Numba optional for speed).
- `pip install gplearn` (symbolic regression: `mlaw_run.py`, `lsttree_run.py`).
- `pip install qiskit qiskit-aer` (real quantum circuits: `qpu_run.py`).
- Data in place: `pdb_clean/` (apo + holo), `pdb_raw/` (raw PDB for cofactor detection).
- Outputs are written to `./outputs`.

Run these in order to reproduce the full submission and all deliverables:

1. `python submission_run.py`
   Final consolidated submission. Per protein: router + learned transfer + their
   fusion, each with AUC + p-value + 95% CI; picks the best; saves the N×N
   connectivity matrix, top-5 hit-list, and per-protein JSON. Includes c-Myc.
2. `python viz_run.py`
   3D interpretability: writes each apo structure with the allosteric score in the
   B-factor column (`*_allosteric.pdb`) plus a PyMOL script (`*_allosteric.pml`).
3. `python qpu_run.py`
   Real CTQW quantum circuit per protein (Qiskit): coarse-grain → circuit, qubit /
   depth / gate budget, Trotter fidelity vs classical, and noise degradation.
4. `python noise_run.py`
   Noise resilience: decoherence (quantum→classical dephasing curve) and structural
   (coordinate-perturbation) robustness. Saves `outputs/noise_resilience.json`.
5. `python coarsen.py`
   Coarse-graining with a signal-retention proof (AUC retained vs eigensolve speedup).
6. `python audit_run.py`
   Diagnostic: transferable power of each of the 23 feature channels, grouped by type.

Optional / exploratory (not needed for the submission; reproduce individual methods):
`spaces_run.py`, `mlaw_run.py`, `lsttree_run.py`, `qboost_run.py`, `qnn_qpu_run.py`,
`curated_run.py`, `final_run.py`, `experiments.py`, `predict.py`, `eml_run.py`,
`operator_run.py`, `specop_run.py`, `lstnet_run.py`, `qnn_run.py`, `bench.py`.

Best result (fusion, blind, apo→holo): KRAS G12C AUC 0.68 (p=2.8e-3),
BCR-ABL1 0.75 (p=3.1e-6), Cardiac Myosin 0.73 (p=4.7e-5); single interference
detector reaches 0.85 on BCR-ABL1.

---

## 2. Files

### Entry-point runners (top level)

- `submission_run.py` — Final submission: router + transfer + fusion, significance
  (AUC/p/CI), connectivity matrices, hit-lists, c-Myc. Flags: `USE_RESONANCE`,
  `LEAN_K` (feature-selection width; 20 = full bank = best).
- `final_run.py` — Per-protein routed detector with significance, matrix, top-5.
- `viz_run.py` — 3D B-factor PDBs + PyMOL scripts.
- `qpu_run.py` — Real CTQW circuits, resource budget, Trotter fidelity, noise.
- `noise_run.py` — Noise resilience (dephasing + structural).
- `coarsen.py` — Coarse-graining with retention proof.
- `audit_run.py` — Per-channel transferable-power audit (feature diagnostic).
- `curated_run.py` — Feature curation + marginal contribution of phase channels.
- `spaces_run.py` — Representation spaces (zscore/rank_gauss/whiten) + OOF stacking.
- `mlaw_run.py` — ML law discovery: gradient boosting + symbolic regression.
- `lsttree_run.py` — Symbolic tree with physics primitives (sin_log/threshold/interf/reso).
- `qboost_run.py` — Quantum boosting of decision stumps as a QUBO (→ annealer/QAOA).
- `qnn_qpu_run.py` — Data-re-uploading variational quantum classifier (QPU-ready).
- `experiments.py`, `predict.py` — Detector grid + consensus, prediction-only targets.
- `eml_run.py`, `operator_run.py`, `specop_run.py`, `lstnet_run.py`, `qnn_run.py` —
  Additional explored methods (symbolic law, operator transfer, spectral operator,
  network, early VQC).
- `bench.py` — Timing / scaling benchmark.
- `run.py` — Convenience driver.
- `make_synthetic.py` — Builds a synthetic 3-protein harness for plumbing tests.
- `test_phase1.py`, `test_phase2.py`, `test_phase3.py` — Self-tests.

### Core library — `aq/`

- `data.py` — Load apo/holo PDB, extract 5 Å pocket, label residues, detect active
  site from raw-PDB cofactor.
- `graph.py` — Contact graph (cutoff / k-NN), Hamiltonians (Laplacian / adjacency /
  log-degree resonance), graph distance.
- `propagate.py` — Analytic quantum-walk metrics (closed-form time averages):
  interference, quadrature, coherence, phase coherence, communicability, screened
  (Yukawa) propagator, effective mass, resonance overlap (dense anti-aliased sampling).
- `rank.py` — Distal-band scoring, active-site (shell z-score) scoring, hit lists.
- `score.py` — ROC-AUC and evaluation.
- `stats.py` — Bootstrap AUC 95% CI + Mann–Whitney p-value.
- `router.py` — Unsupervised pocket-geometry routing (decisiveness / kurtosis).
- `spaces.py` — Feature spaces + transforms; `all_features` (23-channel bank).
- `coarse.py` — Spectral coarse-graining + spectrum-retention proof.
- `sample.py` — Eigen-free stochastic estimator (Gaussian probes + Chebyshev).
- `gauge.py` — Peierls-phase (chiral) Hamiltonian and phase descriptors.
- `qpu_ctqw.py` — Real CTQW circuit (XY-hopping), statevector + noisy simulation,
  classical check, resource estimate (Qiskit).
- `qboost.py` — QUBO construction + simulated-annealing solve (Numba).
- `qnn_qpu.py`, `qnn.py` — Data-re-uploading VQC / earlier quantum kernel + VQC.
- `mlaw.py`, `lsttree.py`, `eml.py` — Gradient boosting, symbolic trees, symbolic law.
- `operator.py`, `specop.py`, `lstnet.py` — Operator-transfer, spectral operator, network.
- `plateau.py` — Angular packing-equilibrium descriptor (Numba).
- `consensus.py` — Detector consensus (rank fusion / RRF).
- `report.py` — Report/artifact helpers.
- `backend.py` — CPU/CuPy dispatch (`use_gpu` flag), Numba guard.

### Config, data, report

- `config.yaml` — Paths (`pdb_clean`, `pdb_raw`, `outputs`), backend (`use_gpu`,
  `n_workers`), metric options.
- `targets.yaml` — Benchmark set (apo, holo, ligand) + c-Myc (prediction-only).
- `extra_targets.yaml.template` — Drop-in template to add ASD apo–holo pairs.
- `report/methodological_report.md` — Methodological report (deliverable #3).
- `MANIFEST.md` — Short index of entry points.
- `pdb_clean/`, `pdb_raw/` — Input structures (local; not shipped).
- `outputs/` — Generated matrices, hit-lists, visualizations, JSON.

---

## 3. Optimization

Speed follows one rule: CPU multiprocessing / Numba and/or GPU CuPy — unless the
code is written for QPU. In practice: metrics are analytic (no time-stepping);
proteins run in parallel; eigensolves and complex matmuls move to GPU (CuPy) under
`config.yaml: backend.use_gpu`; a Numba stochastic estimator handles large N
eigen-free. The QPU-targeted parts — the CTQW circuit, the QBoost QUBO, and the
data-re-uploading VQC — are written for quantum hardware, where classical CPU/GPU
speedups give way to the device (their classical simulation still uses GPU where it helps).

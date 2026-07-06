# GlobalQuantum - kompletny pakiet (Cleveland Clinic Quantum+AI Challenge 2026)

## Prediction pocket allosterycznych z apo-topologii (quantum walk LST), without MD.

## Main Points WEJSCIA
- `submission_run.py` -> Final Submission: router+transfer GB->fusion, significance
                          (AUC+p+95%CI), connectivity matrices, top-5, c-Myc.
                          Best result: KRAS 0.684, ABL 0.747, myosin 0.733 (significant).
- `final_run.py` -> per protein: routed detector + significance + matrix + top5.
- `viz_run.py` -> visualization 3D: ocena w B-factor PDB + script PyMOL.
- `qpu_run.py` -> real circuit CTQW (Qiskit): coarse->circuit, resources, noise.
- `noise_run.py` -> noise resilience (decoherence + structurelny).
- `coarsen.py` -> coarse-graining z proofem retention sygnalu.
- `audit_run.py` -> transferable moc each channelu (diagnostyka cech).

## EKSPLORACJA (method testowane)
experiments/predict (grid+consensus), spaces_run (spacee+OOF),
mlaw_run (GB+symboliczna), lsttree_run (drzewo LST), qboost_run (quantum boosting),
qnn_qpu_run (data re-uploading VQC), curated_run (curation+phase), eml/operator/specop.

## PAKIET aq/ (core)
data, graph, propagate (metric analytic: interference/coherence/phase/resonance),
rank, score, stats (bootstrap CI+p), router, spaces (all_features: 23 channels),
coarse, sample (stochastic estimator eig-free), gauge (phase Peierlsa),
qpu_ctqw (circuit CTQW), qboost, qnn_qpu, plateau, mlaw, backend (CPU/CuPy).

## Data (not w paczce - lokalne)
pdb_clean/ : 4OBE,1OPL,5TBY apo + 6OIM,5MO4,6C1H holo + 1NKP_cMyc apo
pdb_raw/ : 7 surowych PDB (auto-detection active site z cofactora)
targets.yaml (set), extra_targets.yaml.template (extension ASD)

## RAPORT
report/methodological_report.md - deliverable #3 (metric + rationale + results)

## Usage
conda env torch_gpu; `python submission_run.py`. Results -> ./outputs
Optimization: MP over proteins, metric analytically, GPU eig under config use_gpu;
kod for QPU (qpu_ctqw, qbost QUBO, VQC) - there reguła CPU/GPU yields.

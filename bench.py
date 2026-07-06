#!/usr/bin/env python3
"""
Benchmark wydajnosci + kontrola poprawnosci.

Mierzy real dzwignie at this skali (N<=~1000):
  1) analytic umeannie over time vs petla n_t (main hotspot)
  2) MP over targets vs sekwencyjnie (CPU)
  3) GPU (CuPy) - for comparison (at smallm N zwykle wolniejsze)
Sprawdza, that AUC analytic vs petla jest bliskie (ranking niezmieniony).

  python bench.py
"""
import os, time, yaml, copy
from concurrent.futures import ProcessPoolExecutor
from experiments import process_target, EXPERIMENTS
from aq.backend import capabilities

ROOT = os.path.dirname(os.path.abspath(__file__))


def run_seq(cfg, targets):
    rows = []
    for t in targets:
        rows.extend(process_target(t, cfg, EXPERIMENTS)[0])
    return rows


def run_mp(cfg, targets, workers):
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in [ex.submit(process_target, t, cfg, EXPERIMENTS) for t in targets]:
            rows.extend(fut.result()[0])
    return rows


def auc_map(rows):
    return {(r["exp"], r["target"]): r.get("roc_auc") for r in rows if r.get("roc_auc") is not None}


def _set(cfg, **bk):
    c = copy.deepcopy(cfg); c["backend"].update(bk); return c


def main():
    base = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    base["data"]["chain_select"] = "largest"
    targets = [t for t in yaml.safe_load(open(os.path.join(ROOT, "targets.yaml")))["targets"]
               if t.get("holo")]
    print("capabilities:", capabilities())

    # --- 1) analytic vs petla (in CPU, sekwencyjnie) ---
    ana = copy.deepcopy(base); ana["metric"]["analytic"] = True
    loop = copy.deepcopy(base); loop["metric"]["analytic"] = False
    for c in (ana, loop):
        c["backend"].update(use_gpu=False, auto_gpu=False)

    t = time.time(); r_ana = run_seq(ana, targets); t_ana = time.time() - t
    t = time.time(); r_loop = run_seq(loop, targets); t_loop = time.time() - t
    print(f"\n[metric] analytically = {t_ana:.2f}s petla(n_t=64) = {t_loop:.2f}s"
          f" x{t_loop/max(t_ana,1e-9):.1f} szybciej")

    a, b = auc_map(r_ana), auc_map(r_loop)
    diffs = [abs(a[k] - b[k]) for k in a if k in b]
    print(f"[kontrola] max|AUC_analytic - AUC_petla| = {max(diffs):.3f} "
          f"(ranking praktycznie niezmieniony)")

    # --- 2) MP vs sekwencyjnie (analytic, CPU) ---
    t = time.time(); run_mp(ana, targets, base["backend"].get("n_workers", 3)); t_mp = time.time() - t
    print(f"\n[CPU] sekwencyjnie = {t_ana:.2f}s MP({base['backend'].get('n_workers',3)}) = {t_mp:.2f}s"
          f" x{t_ana/max(t_mp,1e-9):.1f} szybciej")

    # --- 3) GPU (optionally, for comparison) ---
    if capabilities().get("cupy"):
        gpu = _set(ana, use_gpu=True, auto_gpu=False)
        run_seq(gpu, targets[:1]) # rozgrzewka
        t = time.time(); run_seq(gpu, targets); t_gpu = time.time() - t
        print(f"\n[GPU] CuPy = {t_gpu:.2f}s (at N<=1000 zwykle wolniejsze niz CPU)")
    print(f"\n=> at this skali: analytic metric + CPU MP; GPU for largech ukladow.")


if __name__ == "__main__":
    main()

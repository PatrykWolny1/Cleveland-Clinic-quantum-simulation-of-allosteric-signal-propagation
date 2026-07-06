"""
Stage 4/5 - propagation i connectivity matrix, z jawnym API in eigendecomposition
(umozliwia CACHE: computesmy eig(H) once i uzywamy to many metryk/times).

Metryki quantum:
  time_avg : <|U(t)|^2>_t (baseline)
  coherent : |U(t*)|^2 (retains interference)
Analog classical (dyfusion) - to comparison requiredgo via challenge:
  classical: <exp(-Lt)>_t (heat-kernel in Laplacianie)
Metryka kontrastu (in temat hypothesis: kwant vs dyfusion):
  contrast : Q_time_avg - C_classical (gdzie transport quantum beats dyfusion)

Whole algebra in numpy/CuPy. eig(H) i eig(L) can podac z zewnatrz.
"""
from __future__ import annotations
import numpy as np
from .backend import get_backend, to_numpy


def eig(H, use_gpu=False):
    """Eigendecomposition symetrycznego H. Returns (w, V, xp, name)."""
    xp, name = get_backend(use_gpu)
    w, V = xp.linalg.eigh(xp.asarray(H, dtype=xp.float64))
    return w, V, xp, name


# --------- metric liczone z GOTOWEJ eigendecomposition -------------------
def q_time_avg(w, V, xp, t_max=20.0, n_t=64):
    Vc = V.astype(xp.complex128)
    times = xp.linspace(t_max / n_t, t_max, n_t)
    M = xp.zeros(V.shape, dtype=xp.float64)
    for t in times:
        U = (Vc * xp.exp(-1j * w * t)) @ Vc.conj().T
        M += U.real ** 2 + U.imag ** 2
    return M / n_t


def q_coherent(w, V, xp, t_star=None):
    if t_star is None:
        ws = xp.sort(w)
        gap = float(to_numpy(xp.median(xp.diff(ws))))
        t_star = 1.0 / max(abs(gap), 1e-6)
    Vc = V.astype(xp.complex128)
    U = (Vc * xp.exp(-1j * w * t_star)) @ Vc.conj().T
    return U.real ** 2 + U.imag ** 2


def c_heat(wL, VL, xp, t_max=20.0, n_t=64):
    """Classical dyfusion: <exp(-L t)>_t. wL,VL = eig(Laplacianu)."""
    times = xp.linspace(t_max / n_t, t_max, n_t)
    M = xp.zeros(VL.shape, dtype=xp.float64)
    for t in times:
        P = (VL * xp.exp(-wL * t)) @ VL.T
        M += P
    return M / n_t


# --------- wysokopoziomowy dispatcher (without cache) ------------------------
def connectivity_matrix(H, cfg, L=None):
    """Returns (M numpy, backend_name). L (Laplacian) needed dla
    metryk classical/contrast; if None, computed as eig(H) when H=L."""
    m = cfg.get("metric", "time_avg")
    gpu = cfg.get("use_gpu", False)
    t_max, n_t, t_star = cfg.get("t_max", 20.0), cfg.get("n_t", 64), cfg.get("t_star")

    w, V, xp, name = eig(H, gpu)
    if m == "time_avg":
        M = q_time_avg(w, V, xp, t_max, n_t)
    elif m == "coherent":
        M = q_coherent(w, V, xp, t_star)
    elif m in ("classical", "contrast"):
        Lmat = H if L is None else L
        wL, VL, _, _ = eig(Lmat, gpu)
        C = c_heat(wL, VL, xp, t_max, n_t)
        if m == "classical":
            M = C
        else:
            Q = q_time_avg(w, V, xp, t_max, n_t)
            M = Q - C
    elif m == "gnm":
        wL, VL, _, _ = eig(H if L is None else L, gpu); M = gnm_cov(wL, VL, xp)
    elif m == "resistance":
        wL, VL, _, _ = eig(H if L is None else L, gpu); M = resistance(wL, VL, xp)
    else:
        raise ValueError(f"nieznana metric: {m}")
    return to_numpy(M), name


# --------- classical analogi allosterii z eig(Laplacianu) ----------------
def pinv_laplacian(wL, VL, xp, tol=1e-9):
    """Moore-Penrose L^+ z eigendecomposition (skips zero mode)."""
    inv = xp.where(wL > tol, 1.0 / xp.where(wL > tol, wL, 1.0), 0.0)
    return (VL * inv) @ VL.T


def gnm_cov(wL, VL, xp):
    """GNM: kowariancja fluktuacji residues = L^+ (standardowy predictor
    sviaenia allosterycznego w sieci elastycznej)."""
    return pinv_laplacian(wL, VL, xp)


def resistance(wL, VL, xp):
    """Effective resistance (commute-time) - connectivity niedependent from
    geodezyjnej distance (uwzglednia all rownolegle sciezki).
    Returns -R, so wyzej = silniej sviaone."""
    Lp = pinv_laplacian(wL, VL, xp)
    d = xp.diag(Lp)
    R = d[:, None] + d[None, :] - 2.0 * Lp
    return -R


def communicability(wA, VA, xp, beta=0.15):
    """Classical komunikowalnosc Estrady: expm(beta*A). wA,VA=eig(A)."""
    return (VA * xp.exp(beta * wA)) @ VA.T


def q_communicability(wA, VA, xp, beta=0.5):
    """Quantum komunikowalnosc: |expm(i*beta*A)|^2 (retains interference)."""
    Vc = VA.astype(xp.complex128)
    G = (Vc * xp.exp(1j * beta * wA)) @ Vc.conj().T
    return G.real ** 2 + G.imag ** 2


def heat_avg_analytic(wL, VL, xp, T=20.0):
    """<exp(-L t)>_[0,T] analytically: g(l)=(1-e^{-lT})/(lT), g(0)=1."""
    lt = wL * T
    g = xp.where(wL > 1e-12, (1.0 - xp.exp(-lt)) / xp.where(wL > 1e-12, lt, 1.0), 1.0)
    return (VL * g) @ VL.T


def q_interference(w, V, xp, T=20.0):
    """LST rown. 10: transfer z ZACHOWANYM signiem interference.
    Re<U(t)>_ij = sum_a V_ia V_ja cos(l_a t); <cos(l t)>_[0,T] = sinc(lT).
    M_ij = sum_a V_ia V_ja sinc(l_a T) - one O(N^3), sign = konstruktywna(+)
    / destruktywna(-) interference between residues."""
    x = w * T
    sinc = xp.where(xp.abs(x) > 1e-9, xp.sin(x) / xp.where(xp.abs(x) > 1e-9, x, 1.0), 1.0)
    return (V * sinc) @ V.T


def q_ita(w, V, xp):
    """Quantum infinite-time average: M_ij = sum_a V_ia^2 V_ja^2 (one matmul).
    Rownowazna rankingowo <|U(t)|^2>_t, ~64x cheaper niz petla."""
    P = V.real ** 2 if xp.iscomplexobj(V) else V ** 2
    return P @ P.T


def yukawa(wL, VL, xp, m2=0.5):
    """Propagator masywny (coded H = -box + V): G=(L+m^2)^{-1}.
    Ekranowane significance o skali 1/m (LST + coded model)."""
    inv = 1.0 / (wL + m2)
    return (VL * inv) @ VL.T


def effective_mass(wL, VL, xp, T=20.0):
    """Masa efektywna per residue z zaniku autokorelacji (G(t)~exp(-m t)):
    m_i = -log( sum_a V_ia^2 e^{-lambda_a T} ) / T. Low masa = slow zanik
    = trwale significance (channel allosteryczny)."""
    p_ret = (VL ** 2) @ xp.exp(-wL * T) # (e^{-LT})_ii
    p_ret = xp.clip(p_ret, 1e-12, None)
    return -xp.log(p_ret) / T


def q_quadrature(w, V, xp, T=20.0):
    """Component KWADRATUROWA (Im propagatora), analytically:
    <Im U(t)>_ij = sum_a V_ia V_ja <sin(l_a t)> ; <sin(lt)>_[0,T]=(1-cos(lT))/(lT).
    New channel phase (dotad uzywalismy only Re/interference i |U|^2)."""
    x = w * T
    f = xp.where(xp.abs(x) > 1e-9, (1.0 - xp.cos(x)) / xp.where(xp.abs(x) > 1e-9, x, 1.0), 0.0)
    return (V * f) @ V.T


def phase_coherence(w, V, xp, t_star=None):
    """Coherence phase per residue (phase-locking):
    R_i = |sum_j U_ij| / sum_j |U_ij|. High = sviaenia residues i are
    consistent w fazie (konstruktywna interference); low = rozproszone/destruktywne."""
    if t_star is None:
        ws = xp.sort(w); gap = float(to_numpy(xp.median(xp.diff(ws))))
        t_star = 1.0 / max(abs(gap), 1e-6)
    Vc = V.astype(xp.complex128)
    U = (Vc * xp.exp(-1j * w * t_star)) @ Vc.conj().T
    num = xp.abs(U.sum(1)); den = xp.abs(U).sum(1) + 1e-12
    return num / den


def resonance_overlap(w, V, xp, ref_idx, T_max=None, n_times=None, oversample=2.0, gpu=False):
    """Resonance residues z reference: mean over time |<i|e^{-iHt}|ref>|^2
    (CTQW transition prob z superposition ref). GESTE sampling time z krokiem
    below Nyquista (dt < pi/lambda_max) -> ANTY-ALIASING. Batch over times.
    NATYWNIE QPU (test SWAP/CTQW); klasycznie = batched complex overlap (as CUDA).
    """
    N = V.shape[0]
    ref = xp.zeros(N, dtype=xp.float64); ref[xp.asarray(ref_idx, dtype=xp.int64)] = 1.0
    c = (V.T @ ref).astype(xp.complex128) # rzut ref in modes
    wmax = float(to_numpy(w).max())
    if T_max is None:
        wpos = to_numpy(w); wpos = wpos[wpos > 1e-9]
        T_max = float(4.0 * np.pi / wpos.min()) if len(wpos) else 40.0 # several okresow najwolniejszego modu
        T_max = min(T_max, 200.0)
    if n_times is None:
        dt = np.pi / (oversample * (wmax + 1e-9)) # Nyquist z nadsamplingm
        n_times = int(min(2000, max(64, np.ceil(T_max / dt))))
    ts = xp.linspace(0.0, T_max, n_times)
    phases = xp.exp(-1j * xp.outer(ts, w)) # (n_times, N) e^{-i w t}
    psi = (phases * c[None, :]) @ V.T # (n_times, N) U(t)|ref>
    return (psi.real**2 + psi.imag**2).mean(0) # mean |psi_i|^2


def plv_resonance(w, V, xp, ref_idx, T_max=40.0, n_times=None, oversample=2.0, gpu=False):
    """PLV (phase-locking value, z TSR-NUFFT): stability phase odpowiedzi
    timesej residues psi_i(t)=(U(t)|ref>)_i. PLV_i = |<e^{i arg psi_i(t)}>_t|.
    High = residue sviaona z singlem mode (coherent resonance);
    low = many modes (phase rotuje). Dense sampling (anti-aliasing)."""
    N = V.shape[0]
    ref = xp.zeros(N, dtype=xp.float64); ref[xp.asarray(ref_idx, dtype=xp.int64)] = 1.0
    c = (V.T @ ref).astype(xp.complex128)
    wmax = float(to_numpy(w).max())
    if n_times is None:
        dt = np.pi / (oversample * (wmax + 1e-9))
        n_times = int(min(2000, max(64, np.ceil(T_max / dt))))
    ts = xp.linspace(0.0, T_max, n_times)
    psi = (xp.exp(-1j * xp.outer(ts, w)) * c[None, :]) @ V.T # (n_times, N)
    ph = psi / (xp.abs(psi) + 1e-12) # e^{i arg}
    return xp.abs(ph.mean(0)) # PLV per residue


def metric_from_cache(cache, cfg, xp):
    """Computes metryke z readych eigow (cache: dict z 'H'->(w,V), 'L'->(wL,VL)).
    Uzywane w experiments.py by not powtarzac eigendecomposition."""
    m = cfg.get("metric", "time_avg")
    t_max, n_t, t_star = cfg.get("t_max", 20.0), cfg.get("n_t", 64), cfg.get("t_star")
    ana = cfg.get("analytic", True) # analytic umeannie (fast, default)
    w, V = cache["H"]
    if m == "time_avg":
        M = q_ita(w, V, xp) if ana else q_time_avg(w, V, xp, t_max, n_t)
    elif m == "ita":
        M = q_ita(w, V, xp)
    elif m == "interference":
        M = q_interference(w, V, xp, t_max)
    elif m == "coherent":
        M = q_coherent(w, V, xp, t_star)
    elif m == "classical":
        wL, VL = cache["L"]
        M = heat_avg_analytic(wL, VL, xp, t_max) if ana else c_heat(wL, VL, xp, t_max, n_t)
    elif m == "contrast":
        wL, VL = cache["L"]
        if ana:
            M = q_ita(w, V, xp) - heat_avg_analytic(wL, VL, xp, t_max)
        else:
            M = q_time_avg(w, V, xp, t_max, n_t) - c_heat(wL, VL, xp, t_max, n_t)
    elif m == "gnm":
        wL, VL = cache["L"]; M = gnm_cov(wL, VL, xp)
    elif m == "resistance":
        wL, VL = cache["L"]; M = resistance(wL, VL, xp)
    elif m == "yukawa":
        wL, VL = cache["L"]; M = yukawa(wL, VL, xp, cfg.get("m2", 0.5))
    elif m == "communicability":
        wA, VA = cache["H"]; M = communicability(wA, VA, xp, cfg.get("beta", 0.15))
    elif m == "q_communicability":
        wA, VA = cache["H"]; M = q_communicability(wA, VA, xp, cfg.get("beta", 0.5))
    else:
        raise ValueError(f"nieznana metric: {m}")
    return to_numpy(M)

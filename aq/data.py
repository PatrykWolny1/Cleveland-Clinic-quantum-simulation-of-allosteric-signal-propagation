"""
Stage 0 - data.

apo (input) -> nodes = residues domain (atomy CA). Optional selection
                     lancucha (chain_select) i zakresu domain katalitycznej.
holo (walidacja) -> residues pocket (<= cutoff from ligandu-leku), mapowane
                     in numeracje apo BEZ zanieczyszczenia between lancuchami.

Wykluczamy wode i hetero-residues as nodes (hypothesis sieci elastycznej).
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import numpy as np
from Bio.PDB import PDBParser, is_aa

_PARSER = PDBParser(QUIET=True)
_SOLVENT = {"HOH", "WAT", "DOD"}
_IONS = {"In", "CL", "MG", "K", "CA", "ZN", "MN", "SO4", "PO4", "GOL",
         "EDO", "ACT", "HOH", "WAT", "DOD", "FMT", "NO3", "CO3", "GDP",
         "GTP", "GNP", "GSPGCP"}


@dataclass
class Structure:
    keys: list # (chain, resseq, icode)
    resnames: list
    coords: np.ndarray # (N,3) CA
    name: str

    @property
    def n(self):
        return len(self.keys)


def _iter_chain_residues(model, chain):
    for res in chain:
        if not is_aa(res, standard=False) or "CA" not in res:
            continue
        yield res


def load_apo(path, name, domain_range=None, chain_select="all"):
    """Wczytuje nodes CA.

    chain_select: "all" | "largest" | konkretne ID lancucha.
    domain_range: dict {chain:(lo,hi)} ograniczajacy to domain (optional).
    """
    model = next(_PARSER.get_structure(name, path).get_models())

    # policz residues in chain, ew. wybierz najwiekszy
    counts = {c.id: sum(1 for _ in _iter_chain_residues(model, c)) for c in model}
    counts = {k: v for k, v in counts.items() if v > 0}
    if chain_select == "largest":
        keep = {max(counts, key=counts.get)}
    elif chain_select in (None, "all"):
        keep = set(counts)
    else:
        keep = {chain_select}

    keys, resnames, coords = [], [], []
    for chain in model:
        if chain.id not in keep:
            continue
        rng = domain_range.get(chain.id) if domain_range else None
        for res in _iter_chain_residues(model, chain):
            resseq = res.id[1]
            if rng and not (rng[0] <= resseq <= rng[1]):
                continue
            keys.append((chain.id, resseq, res.id[2].strip()))
            resnames.append(res.resname.strip())
            coords.append(res["CA"].coord)
    if not keys:
        raise ValueError(f"{name}: none nodes CA (chain_select={chain_select})")
    return Structure(keys, resnames, np.asarray(coords, float), name)


def extract_pocket(holo_path, ligand_resname=None, contact_cutoff=5.0):
    """Returns (set kluczy residues pocket, kod ligandu)."""
    model = next(_PARSER.get_structure("holo", holo_path).get_models())
    hetero = {}
    for chain in model:
        for res in chain:
            rn = res.resname.strip()
            if res.id[0].strip() == "" or is_aa(res, standard=False):
                continue
            if rn in _SOLVENT:
                continue
            hetero.setdefault(rn, []).append(res)

    if ligand_resname:
        chosen = ligand_resname
    else:
        cand = {rn: sum(len(list(r.get_atoms())) for r in rs)
                for rn, rs in hetero.items() if rn not in _IONS}
        if not cand:
            raise ValueError(f"None ligandu-leku w {holo_path}")
        chosen = max(cand, key=cand.get)

    lig_xyz = np.asarray([a.coord for r in hetero.get(chosen, [])
                          for a in r.get_atoms()], float)
    pocket = set()
    for chain in model:
        for res in chain:
            if not is_aa(res, standard=False):
                continue
            atoms = np.asarray([a.coord for a in res.get_atoms()], float)
            if atoms.size == 0:
                continue
            d = np.linalg.norm(atoms[:, None, :] - lig_xyz[None, :, :], axis=2)
            if d.min() <= contact_cutoff:
                pocket.add((chain.id, res.id[1], res.id[2].strip()))
    return pocket, chosen


def label_nodes(apo, pocket_keys):
    """Wektor 0/1 for nodes apo.

    Redukujemy pocket to DOMINUJACEGO lancucha holo (najsoej reszt),
    then etykietujemy over (resseq,icode). Dzieki selectionowi one lancucha
    apo (chain_select) not has zanieczyszczenia between kopiami dimeru.
    """
    if not pocket_keys:
        return np.zeros(apo.n, int)
    by_chain = {}
    for ch, rs, ic in pocket_keys:
        by_chain.setdefault(ch, set()).add((rs, ic))
    dom = max(by_chain, key=lambda c: len(by_chain[c]))
    pk = by_chain[dom]

    labels = np.array([1 if (k[1], k[2]) in pk else 0 for k in apo.keys], int)
    return labels


# --- detection active site (orthosteric) z surowej structure --------
# Cofactor endogenny (np. GDP w KRAS, nukleotyd w myosin) wyznacza
# active site. This Not jest allosteric pocket (other, ortosteryczny
# site), so uzycie go as point odniesienia for connectivity not jest
# przeciekiem odpowiedzi.
_COFACTORS = {"GDP", "GTP", "GNP", "GCP", "GSP", "GDPNP", "ANP", "ADP", "ATP",
              "AGS", "ACP", "Above", "NAP", "FAD", "Itself", "SAH",
              # analogi stanu przejsciowego / myosin
              "VO4", "BEF", "ALF", "MGF", "AF3", "POP", "PPV"}


def detect_active_site(raw_path, apo, cofactor_codes=None, cutoff=6.0):
    """Returns indeksy nodes apo w poblizu endogennego cofactora w raw.

    apo i raw dziela uklad coordinates (apo this subset raw), so
    fitsmy over distance CA. None cofactora -> empty lista.
    """
    codes = set(cofactor_codes) if cofactor_codes else _COFACTORS
    model = next(_PARSER.get_structure("raw", raw_path).get_models())
    lig = []
    for chain in model:
        for res in chain:
            if res.resname.strip() in codes and res.id[0].strip() != "":
                lig.extend(a.coord for a in res.get_atoms())
    if not lig:
        return [], None
    lig = np.asarray(lig, float)
    d = np.linalg.norm(apo.coords[:, None, :] - lig[None, :, :], axis=2)
    idx = np.nonzero(d.min(1) <= cutoff)[0]
    found = None
    for chain in model:
        for res in chain:
            if res.resname.strip() in codes and res.id[0].strip() != "":
                found = res.resname.strip(); break
        if found:
            break
    return [int(i) for i in idx], found

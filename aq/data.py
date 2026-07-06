"""
Stage 0 - data.

apo (input) -> nodes = domain residues (CA atoms). Optional selection of the
                     chain (chain_select) and of the catalytic domain range.
holo (validation) -> pocket residues (<= cutoff from the drug ligand), mapped
                     onto the apo numbering WITHOUT cross-chain contamination.

I exclude water and hetero-residues as nodes (elastic-network hypothesis).
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import Counter
import numpy as np
from Bio.PDB import PDBParser, is_aa

_PARSER = PDBParser(QUIET=True)
_SOLVENT = {"HOH", "WAT", "DOD"}
_IONS = {"NA", "CL", "MG", "K", "CA", "ZN", "MN", "SO4", "PO4", "GOL",
         "EDO", "ACT", "HOH", "WAT", "DOD", "FMT", "NO3", "CO3", "GDP",
         "GTP", "GNP", "GSP", "GCP"}


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
    """Load CA nodes.

    chain_select: "all" | "largest" | a specific chain ID.
    domain_range: dict {chain:(lo,hi)} restricting to the domain (optional).
    """
    model = next(_PARSER.get_structure(name, path).get_models())

    # count residues per chain, optionally pick the largest
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
        raise ValueError(f"{name}: no CA nodes (chain_select={chain_select})")
    return Structure(keys, resnames, np.asarray(coords, float), name)


def extract_pocket(holo_path, ligand_resname=None, contact_cutoff=5.0):
    """Returns (set of pocket residue keys, ligand code)."""
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
            raise ValueError(f"No drug ligand in {holo_path}")
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
    """0/1 vector over the apo nodes.

    I reduce the pocket to the DOMINANT holo chain (the one with the most
    residues), then label by (resseq, icode). Thanks to selecting a single
    apo chain (chain_select), there is no contamination between dimer copies.
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


# --- active site detection (orthosteric) from the raw structure -----
# The endogenous cofactor (e.g. GDP in KRAS, the nucleotide in myosin)
# marks the active site. This is NOT the allosteric pocket (a different,
# orthosteric site), so using it as a reference point for connectivity
# is not a leak of the answer.
_COFACTORS = {"GDP", "GTP", "GNP", "GCP", "GSP", "GDPNP", "ANP", "ADP", "ATP",
              "AGS", "ACP", "NAD", "NAP", "FAD", "SAM", "SAH",
              # transition-state analogs / myosin
              "VO4", "BEF", "ALF", "MGF", "AF3", "POP", "PPV"}


def detect_active_site(raw_path, apo, cofactor_codes=None, cutoff=6.0):
    """Returns indices of apo nodes near the endogenous cofactor in raw.

    apo and raw share the coordinate frame (apo is a subset of raw), so I
    match by CA distance. No cofactor -> empty list.
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

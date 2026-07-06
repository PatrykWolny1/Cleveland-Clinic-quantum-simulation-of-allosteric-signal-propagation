"""Synthetic target for testing the Phase I plumbing.

Construction: core (active site) -- channel -- distal pocket +
scattered surface residues. In the holo the ligand sits at the pocket,
so the ground truth = pocket residues. The channel connects the pocket to
the core, so the quantum transfer should distinguish the pocket from the
scattered background.
"""
import os
import numpy as np

np.random.seed(0)
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "pdb_clean")
os.makedirs(OUT, exist_ok=True)


def cluster(center, n, spread):
    return np.asarray(center) + np.random.randn(n, 3) * spread


core = cluster([0, 0, 0], 20, 2.2) # active site
channel = np.array([[3.0 + 3.4 * k, 0, 0] for k in range(12)]) # bridge
channel += np.random.randn(*channel.shape) * 0.4
pocket = cluster([3.0 + 3.4 * 12 + 4, 0, 0], 14, 2.2) # distal pocket
surface = cluster([0, 0, 0], 20, 12.0) # scattered background

coords = np.vstack([core, channel, pocket, surface])
N = len(coords)
# segment labels (used to generate the ligand at the pocket)
pocket_lo = len(core) + len(channel)
pocket_hi = pocket_lo + len(pocket)
pocket_centroid = coords[pocket_lo:pocket_hi].mean(0)


def write_pdb(path, xyz, ligand_xyz=None):
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(xyz, 1):
            f.write(f"ATOM {i:5d} CA ALA A{i:4d} "
                    f"{x:8.3f}{y:8.3f}{z:8.3f} 1.00 0.00 C\n")
        if ligand_xyz is not None:
            for j, (x, y, z) in enumerate(ligand_xyz, 1):
                f.write(f"HETATM{9000 + j:5d} C{j:<2d} LIG A 900 "
                        f"{x:8.3f}{y:8.3f}{z:8.3f} 1.00 0.00 C\n")
        f.write("END\n")


# apo: nodes only
write_pdb(os.path.join(OUT, "SYN_apo.pdb"), coords)
# holo: the same coordinates + a small ligand at the pocket
lig = pocket_centroid + np.random.randn(6, 3) * 1.2
write_pdb(os.path.join(OUT, "SYN_holo.pdb"), coords, lig)

print(f"N={N} core=0..{len(core)-1} channel={len(core)}..{pocket_lo-1} "
      f"pocket={pocket_lo}..{pocket_hi-1} surface={pocket_hi}..{N-1}")
print("apo ->", os.path.join(OUT, "SYN_apo.pdb"))
print("holo ->", os.path.join(OUT, "SYN_holo.pdb"))

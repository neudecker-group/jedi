"""
Regenerates all reference files for diethyldisulfide and HCN.
Run this script once after upgrading to a new Python/ASE version
where neighborlist ordering has changed.

Usage: python regenerate_all_refs.py
"""

import numpy as np
import ase.io as io
from ase.vibrations.vibrations import VibrationsData
from strainjedi.jedi import Jedi
from tests.resources import path_to_test_resources


def save_rim_list(path, rim_list):
    """Save rim_list as .npz, handling empty arrays correctly."""
    def safe_array(arr, ncols):
        if arr is None or (hasattr(arr, 'shape') and arr.shape[0] == 0):
            return np.empty((0, ncols), dtype=int)
        return arr.astype(int)

    np.savez(
        path,
        bonds=safe_array(rim_list[0], 2),
        custom=safe_array(rim_list[1], 2),
        angles=safe_array(rim_list[2], 3),
        dihedrals=safe_array(rim_list[3], 4)
    )


# ===========================================================================
#  Diethyldisulfide
# ===========================================================================

print("=" * 60)
print("Regenerating diethyldisulfide reference files...")
print("=" * 60)

deds_path = path_to_test_resources() / "diethyldisulfid"

mol  = io.read(deds_path / "opt.json")
mol2 = io.read(deds_path / "dis.json")
hess = VibrationsData.read(deds_path / "modes.json")

j = Jedi(mol, mol2, hess)
j.run(printout=False)

np.savetxt(deds_path / "ERIMs",               j.E_RIMs)
np.savetxt(deds_path / "procERIMs",           j.proc_E_RIMs)
np.savetxt(deds_path / "delta_q",             j.delta_q)
np.savetxt(deds_path / "energies",            j.energies)
np.savetxt(deds_path / "jediInternalHessian", j.H)
np.savetxt(deds_path / "bmatrix",             j.B)
np.savetxt(deds_path / "indices",             j.indices)
save_rim_list(deds_path / "rim_list.npz",     j.rim_list)

print(f"  ERIMs:               shape {j.E_RIMs.shape}")
print(f"  procERIMs:           shape {j.proc_E_RIMs.shape}")
print(f"  delta_q:             shape {j.delta_q.shape}")
print(f"  energies:            {j.energies}")
print(f"  jediInternalHessian: shape {j.H.shape}")
print(f"  bmatrix:             shape {j.B.shape}")
print(f"  indices:             shape {j.indices.shape}")
print(f"  rim_list.npz:        bonds={j.rim_list[0].shape}, "
      f"angles={j.rim_list[2].shape}, "
      f"dihedrals={j.rim_list[3].shape}")


# ===========================================================================
#  HCN
# ===========================================================================

print()
print("=" * 60)
print("Regenerating HCN reference files...")
print("=" * 60)

hcn_path = path_to_test_resources() / "hcn"

mol  = io.read(hcn_path / "opt.json")
mol2 = io.read(hcn_path / "dis.json")
hess = VibrationsData.read(hcn_path / "modes.json")

# Full run
j = Jedi(mol, mol2, hess)
j.run(printout=False)

np.savetxt(hcn_path / "ERIMs",               j.E_RIMs)
np.savetxt(hcn_path / "delta_q",             j.delta_q)
np.savetxt(hcn_path / "energies",            j.energies)
np.savetxt(hcn_path / "jediInternalHessian", j.H)
np.savetxt(hcn_path / "bmatrix",             j.B)
save_rim_list(hcn_path / "rim_list.npz",     j.rim_list)

print(f"  ERIMs:               shape {j.E_RIMs.shape}")
print(f"  delta_q:             shape {j.delta_q.shape}")
print(f"  energies:            {j.energies}")
print(f"  jediInternalHessian: shape {j.H.shape}")
print(f"  bmatrix:             shape {j.B.shape}")
print(f"  rim_list.npz:        bonds={j.rim_list[0].shape}, "
      f"angles={j.rim_list[2].shape}, "
      f"dihedrals={j.rim_list[3].shape}")

# Partial run
parthess = VibrationsData.from_2d(
    mol,
    np.loadtxt(hcn_path / "parthess"),
    indices=[2, 3, 5, 8, 9, 11]
)
jp = Jedi(mol, mol2, parthess)
jp.partial_analysis(indices=[2, 3, 5, 8, 9, 11])

np.savetxt(hcn_path / "pERIMs", jp.E_RIMs)

print(f"  pERIMs:              shape {jp.E_RIMs.shape}")

print()
print("All reference files regenerated successfully.")

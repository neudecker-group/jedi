import dataclasses
import itertools
import warnings
from typing import Optional

import ase.neighborlist
import numpy as np
from numpy.typing import NDArray

from strainjedi import constants


@dataclasses.dataclass
class RICS:
    bonds: Optional[NDArray]
    custom_bonds: Optional[NDArray]
    angles: Optional[NDArray]
    dihedrals: Optional[NDArray]


# TODO: Set return type to RICS, but beware of subtle breakages everywhere. Not fun.
def calculate(mol, indices, custom_bonds):
    """Gets the redundant internal coordinates"""

    cutoff = ase.neighborlist.natural_cutoffs(mol, mult=constants.COVALENCY_FACTOR)
    bonds = np.vstack(ase.neighborlist.neighbor_list("ij", a=mol, cutoff=cutoff)).T  # determine covalent bonds

    bonds = bonds[bonds[:, 0] < bonds[:, 1]]  # remove double metioned
    bonds, counts = np.unique(bonds, return_counts=True, axis=0)
    if not np.all(counts == 1):
        print(
            "unit cell too small hessian not calculated for interaction \
               jedi analysis for a finite system consisting of the cell will be conducted"
        )
    bonds = np.atleast_2d(bonds)

    if len(indices) != len(mol):
        bonds = bonds[np.all([np.isin(bonds[:, 0], indices), np.isin(bonds[:, 1], indices)], axis=0)]

    rim_list = [bonds]

    # possibility of adding custom bonds like hbonds, long range interactions
    if custom_bonds is not None:
        bonds = np.vstack((bonds, custom_bonds))
        rim_list.append(custom_bonds)
    if custom_bonds is None:
        rim_list.append(np.array([]))

    # compute adjacency
    neighbors = [[] for _ in range(len(mol))]
    for a, b in bonds:
        a = int(a)
        b = int(b)
        neighbors[a].append(b)
        neighbors[b].append(a)

    ########find angles
    # create array containing all angles (ba)
    ba_rows = []
    for x, nbrs in enumerate(neighbors):
        for o1, o2 in itertools.combinations(nbrs, 2):
            ba_rows.append([o1, x, o2])

    ba = np.asarray(ba_rows)
    ba_flag = ba.size > 0

    if ba_flag:
        ba = np.atleast_2d(ba)
        ba = ba[ba[:, 1].argsort(kind="stable")]  # sort by atom2
        ba = ba[ba[:, 0].argsort(kind="stable")]  # sort by atom1

        nan = np.full((len(ba), 1), -1)
        _nan = np.hstack((nan, ba))
        rim_list.append(ba)
    else:
        rim_list.append(np.array([]))

    # degree of each node = how often it appears anywhere in bond list.
    # This is represented as the length of each neighbor[x] = [a, b, ...]
    deg = np.fromiter((len(n) for n in neighbors), dtype=np.int64)

    # A bond is torsionable if both endpoints have degree > 1
    mask = (deg[bonds[:, 0]] > 1) & (deg[bonds[:, 1]] > 1)
    torsionable_bonds = bonds[mask]

    # torsion angles
    da_rows = []
    LINEAR = (0.0, 180.0, 360.0)
    for torsionable_row in torsionable_bonds:
        j, k = map(int, torsionable_row)

        left_atoms = [i for i in neighbors[j] if i != k]
        right_atoms = [l for l in neighbors[k] if l != j]

        for i, l in itertools.product(left_atoms, right_atoms):
            da_pre = np.array([i, j, k, l], dtype=int)

            if len(set(da_pre)) != 4:
                print(
                    "bonds for dihedral angle span over more than one unit cell\n %s will not be taken into account in the further analysis"
                    % (np.atleast_2d(da_pre))
                )
                continue

            try:
                if round(mol.get_angle(i, j, k, mic=True)) in LINEAR:
                    continue
                if round(mol.get_angle(j, k, l, mic=True)) in LINEAR:
                    continue
            except Exception:
                continue

            da_rows.append(da_pre)

    da = np.asarray(da_rows, dtype=int)
    rim_list.append(np.atleast_2d(da) if da.size > 0 else np.array([]))
    rim_list_sorted = [arr if arr.size == 0 else np.sort(arr, axis=1, kind="mergesort") for arr in rim_list]

    return rim_list_sorted


def intersect(rics0, ricsF):
    """Returns the intersection of rics0 and ricsF, i.e. those RICs that are only present in both."""

    if len(rics0[0]) != len(ricsF[0]):
        warnings.warn_explicit(
            f"The distorted structure has a different number of bonds ({len(ricsF[0])})\n"
            f"compared to the relaxed structure ({len(rics0[0])}). "
            f"In this case the JEDI strain analysis can not be applied correctly.",
            UserWarning,
            "",
            0,
        )
    if len(rics0[2]) != len(ricsF[2]):
        warnings.warn_explicit(
            f"The distorted structure has a different number of angles ({len(ricsF[2])})"
            f" compared to the relaxed structure ({len(rics0[2])}). ",
            UserWarning,
            "",
            0,
        )
    if len(rics0[3]) != len(ricsF[3]):
        warnings.warn_explicit(
            f"The distorted structure has a different number of dihedral angles ({len(ricsF[3])})"
            f" compared to the relaxed structure ({len(rics0[3])}).",
            UserWarning,
            "",
            0,
        )

    common_rims = [np.empty(0) for _ in range(4)]
    for i in range(len(rics0)):
        if rics0[i].shape[0] == 0:
            continue
        elif ricsF[i].shape[0] == 0:
            common_rims[i] = np.empty(0)
        else:
            rics0v = rics0[i].view([("", rics0[i].dtype)] * rics0[i].shape[1]).ravel()
            ricsFv = (
                ricsF[i].view([("", ricsF[i].dtype)] * ricsF[i].shape[1]).ravel()
            )  # get a viable input for np.intersect1d()
            rim_l, ind, _ = np.intersect1d(
                rics0v, ricsFv, return_indices=True
            )  # get the rims that exist in both structures
            rim_l = rim_l[ind.argsort(kind="stable")]
            common_rims[i] = rim_l.view(rics0[i].dtype).reshape(-1, rics0[i].shape[1])
    common_rims_sorted = [arr if arr.size == 0 else np.sort(arr, axis=1, kind="mergesort") for arr in common_rims]

    return common_rims_sorted

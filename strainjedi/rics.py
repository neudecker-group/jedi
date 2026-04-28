import dataclasses
import itertools
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


def get_rics(mol, indices, custom_bonds):
    """Gets the redundant internal coordinates"""

    cutoff = ase.neighborlist.natural_cutoffs(mol, mult=constants.COVALENCY_FACTOR)
    bl = np.vstack(ase.neighborlist.neighbor_list("ij", a=mol, cutoff=cutoff)).T  # determine covalent bonds

    bl = bl[bl[:, 0] < bl[:, 1]]  # remove double metioned
    bl, counts = np.unique(bl, return_counts=True, axis=0)
    if ~np.all(counts == 1):
        print(
            "unit cell too small hessian not calculated for interaction \
               jedi analysis for a finite system consisting of the cell will be conducted"
        )
    bl = np.atleast_2d(bl)

    if len(indices) != len(mol):
        bl = bl[np.all([np.isin(bl[:, 0], indices), np.isin(bl[:, 1], indices)], axis=0)]

    rim_list = [bl]

    # possibility of adding custom bonds like hbonds, long range interactions
    if custom_bonds is not None:
        bl = np.vstack((bl, custom_bonds))
        rim_list.append(custom_bonds)
    if custom_bonds is None:
        rim_list.append(np.array([]))

    # compute adjacency
    neighbors = [[] for _ in range(len(mol))]
    for a, b in bl:
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
    mask = (deg[bl[:, 0]] > 1) & (deg[bl[:, 1]] > 1)
    torsionable_bonds = bl[mask]

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

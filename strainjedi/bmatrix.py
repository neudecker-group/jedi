import itertools

import ase.geometry
import ase.neighborlist
import numpy as np
from ase.units import Bohr

from strainjedi import constants


def get_b_matrix(atoms0, rim_list, indices=None):
    """Calculates the derivatives of the RICs with respect to all cartesian coordinates using ase functions"""
    mol = atoms0
    if indices is None:
        indices = np.arange(0, len(mol))

    rim_size = sum([np.shape(length)[0] for length in rim_list])
    b = np.zeros([int(len(indices) * 3), int(rim_size)], dtype=float)  # shape of B-matrix (NCarts,NRIMs)

    # map atom index -> row block start in B (only for selected indices)
    # (so we can write to the right rows without scanning indices)
    idx_pos = {int(a): p for p, a in enumerate(indices)}

    # We define this helper here inside this function to avoid having to pass too many things around.
    def _write_stretch_column(q_i: int, q_j: int, column: int) -> None:
        u = mol.get_distance(q_i, q_j, mic=True, vector=True)

        du = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0]  # (2,3)
        d_i, d_j = du[0], du[1]

        pi = idx_pos.get(q_i)
        if pi is not None:
            r = 3 * pi
            b[r : r + 3, column] = d_i

        pj = idx_pos.get(q_j)
        if pj is not None:
            r = 3 * pj
            b[r : r + 3, column] = d_j

    # get all derivatives
    column = 0  # Initilization of columns to specifiy position in B-Matrix
    for q in rim_list[0]:
        q_i, q_j = int(q[0]), int(q[1])
        _write_stretch_column(q_i, q_j, column)
        column += 1

    for q in rim_list[1]:
        q_i, q_j = int(q[0]), int(q[1])
        _write_stretch_column(q_i, q_j, column)
        column += 1

    #################ba###############################

    for q in rim_list[2]:
        row = 0  # Initilization of rows to specifiy position in B-Matrix

        BA = [int(q[0]), int(q[1]), int(q[2])]  # create list of involved atoms
        q_i, q_j, q_k = BA
        u = mol.get_distance(q_i, q_j, mic=True, vector=True)
        v = mol.get_distance(q_k, q_j, mic=True, vector=True)

        for NAtom in indices:  # for-loop of Number of Atoms
            for q in BA:
                if NAtom != q:
                    continue
                if q == q_j:  # if-Statements for sign-factors
                    b_j = get_B_matrix_angles_derivatives(np.atleast_2d(u), np.atleast_2d(v))[0][1]
                    b[row : row + 3, column] = -b_j
                elif q == q_i:
                    b_j = get_B_matrix_angles_derivatives(np.atleast_2d(u), np.atleast_2d(v))[0][0]
                    b[row : row + 3, column] = -b_j
                elif q == q_k:
                    b_j = get_B_matrix_angles_derivatives(np.atleast_2d(u), np.atleast_2d(v))[0][2]
                    b[row : row + 3, column] = -b_j
            row += 3
        column += 1

    for q in rim_list[3]:
        row = 0  # Initilization of rows to specifiy position in B-Matrix

        DA = [
            int(q[0]),
            int(q[1]),
            int(q[2]),
            int(q[3]),
        ]  # create list of involved atoms
        q_i, q_j, q_k, q_l = DA

        # copy needed because derivative function rewrites vector variable as normed vector
        u = np.copy(np.atleast_2d(mol.get_distance(q_i, q_j, mic=True, vector=True)))
        w = np.copy(np.atleast_2d(mol.get_distance(q_k, q_l, mic=True, vector=True)))
        v = np.copy(np.atleast_2d(mol.get_distance(q_j, q_k, mic=True, vector=True)))

        for NAtom in indices:  # for-loop of Number of Atoms
            for q in DA:
                if NAtom != q:
                    continue
                if q == q_i:  # if-Statements for sign-factors
                    b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][0]) * Bohr
                    b[row : row + 3, column] = b_k
                elif q == q_j:
                    b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][1]) * Bohr
                    b[row : row + 3, column] = b_k
                elif q == q_k:
                    b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][2]) * Bohr
                    b[row : row + 3, column] = b_k
                elif q == q_l:
                    b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][3]) * Bohr
                    b[row : row + 3, column] = b_k

            row += 3
        column += 1

    return np.transpose(b)


def get_B_matrix_angles_derivatives(u, v):
    angle = ase.geometry.get_angles(u, v)  # angle between v and u

    if angle == 180 or angle == 0:  # an auxilliary vector is used if linear angles are existing
        (u, v), (lu, lv) = ase.geometry.conditional_find_mic([u, v], cell=None, pbc=None)
        nu = u / lu
        nv = v / lv
        if (np.arccos(np.dot(nu, (np.array([1, -1, 1]))))) == np.pi:
            w = np.cross(nu, ([-1, 1, 1]))
        else:
            w = np.cross(nu, ([1, -1, 1]))

        nw = w / np.linalg.norm(w)
        d_ba1 = (np.cross(nu, nw)) / np.linalg.norm(u)
        d_ba2 = (-1 * ((np.cross(nu, nw)) / np.linalg.norm(u))) + (
            -1 * ((np.cross(nw, nv)) / np.linalg.norm(v))
        )  # equation to calculate dBA/dx [1]
        d_ba3 = (np.cross(nw, nv)) / np.linalg.norm(v)
        d_ba = np.array([[d_ba1[0], d_ba2[0], d_ba3[0]]])

    else:
        d_ba = np.radians(ase.geometry.get_angles_derivatives(u, v))
    return d_ba * Bohr


def get_rics(mol, indices, custom_bonds):
    """Gets the redundant internal coordinates"""

    cutoff = ase.neighborlist.natural_cutoffs(mol, mult=constants.COVALENT_CUTOFF)
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

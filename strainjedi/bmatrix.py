import ase.geometry
import numpy as np
from ase.units import Bohr


def get_b_matrix(atoms0, rim_list, indices=None):
    """Calculates the derivatives of the RICs with respect to all cartesian coordinates using ase functions"""
    mol = atoms0
    if indices is None:
        indices = np.arange(0, len(mol))

    rim_size = sum([np.shape(length)[0] for length in rim_list])
    b = np.zeros([int(len(indices) * 3), int(rim_size)], dtype=float)  # shape of B-matrix (NCarts,NRIMs)

    # get all derivatives
    column = 0  # Initilization of columns to specifiy position in B-Matrix
    for q in rim_list[0]:
        row = 0  # Initilization of rows to specifiy position in B-Matrix

        ########  Section for stretches  #########

        BL = [int(q[0]), int(q[1])]  # create list of involved atoms
        q_i, q_j = BL

        u = mol.get_distance(q_i, q_j, mic=True, vector=True)
        for NAtom in indices:  # for-loop of Number of Atoms
            for q in BL:
                if (
                    NAtom == q
                ):  # derivative of redundnat internal coordinate w/ respect to cartesian coordinates is not equal zero
                    # if redundant internal coordinate (q) contains the Atomnumber (NAtoms) of the cartesian coordinate (x0_coords) from which is derived from.

                    # if-/elif-statement for the right sign-factor (see [1])
                    if q == q_i:
                        b_i = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0][0]
                        b[row : row + 3, column] = b_i  # change value of zero array at specified position to b_i
                    elif q == q_j:
                        b_i = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0][1]
                        b[row : row + 3, column] = b_i  # change value of zero array at specified position to b_i
            row += 3
        column += 1

    for q in rim_list[1]:
        row = 0  # Initilization of rows to specifiy position in B-Matrix

        ########  Section for custom stretches  #########

        CL = [int(q[0]), int(q[1])]  # create list of involved atoms
        q_i, q_j = CL

        u = mol.get_distance(q_i, q_j, mic=True, vector=True)
        for NAtom in indices:  # for-loop of Number of Atoms
            for q in CL:
                if NAtom == q:
                    # if-/elif-statement for the right sign-factor
                    if q == q_i:
                        b_i = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0][0]
                        b[row : row + 3, column] = b_i  # change value of zero array at specified position to b_i
                    elif q == q_j:
                        b_i = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0][1]
                        b[row : row + 3, column] = b_i  # change value of zero array at specified position to b_i

            row += 3
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

        u = np.copy(
            np.atleast_2d(mol.get_distance(q_i, q_j, mic=True, vector=True))
        )  #####copy needed because derivative function rewrites vector variable as normed vector
        w = np.copy(np.atleast_2d(mol.get_distance(q_k, q_l, mic=True, vector=True)))
        v = np.copy(np.atleast_2d(mol.get_distance(q_j, q_k, mic=True, vector=True)))

        for NAtom in indices:  # for-loop of Number of Atoms
            for q in DA:
                if NAtom == q:
                    if q == q_i:  # if-Statements for sign-factors
                        b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][0]) * Bohr
                        b[row : row + 3, column] = b_k
                        u = np.copy(
                            np.atleast_2d(mol.get_distance(q_i, q_j, mic=True, vector=True))
                        )  #####copy needed because derivative function rewrites vector variable as normed vector
                        w = np.copy(np.atleast_2d(mol.get_distance(q_k, q_l, mic=True, vector=True)))
                        v = np.copy(np.atleast_2d(mol.get_distance(q_j, q_k, mic=True, vector=True)))

                    elif q == q_j:
                        b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][1]) * Bohr
                        b[row : row + 3, column] = b_k
                        u = np.copy(
                            np.atleast_2d(mol.get_distance(q_i, q_j, mic=True, vector=True))
                        )  #####copy needed because derivative function rewrites vector variable as normed vector
                        w = np.copy(np.atleast_2d(mol.get_distance(q_k, q_l, mic=True, vector=True)))
                        v = np.copy(np.atleast_2d(mol.get_distance(q_j, q_k, mic=True, vector=True)))

                    elif q == q_k:
                        b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][2]) * Bohr
                        b[row : row + 3, column] = b_k
                        u = np.copy(
                            np.atleast_2d(mol.get_distance(q_i, q_j, mic=True, vector=True))
                        )  #####copy needed because derivative function rewrites vector variable as normed vector
                        w = np.copy(np.atleast_2d(mol.get_distance(q_k, q_l, mic=True, vector=True)))
                        v = np.copy(np.atleast_2d(mol.get_distance(q_j, q_k, mic=True, vector=True)))

                    elif q == q_l:
                        b_k = np.radians(ase.geometry.get_dihedrals_derivatives(u, v, w)[0][3]) * Bohr
                        b[row : row + 3, column] = b_k
                        u = np.copy(
                            np.atleast_2d(mol.get_distance(q_i, q_j, mic=True, vector=True))
                        )  #####copy needed because derivative function rewrites vector variable as normed vector
                        w = np.copy(np.atleast_2d(mol.get_distance(q_k, q_l, mic=True, vector=True)))
                        v = np.copy(np.atleast_2d(mol.get_distance(q_j, q_k, mic=True, vector=True)))
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

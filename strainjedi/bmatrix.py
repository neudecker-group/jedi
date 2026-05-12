import ase.geometry
import ase.neighborlist
import ase.units
import numpy as np


def calculate(atoms0, rim_list, indices=None):
    """Calculates the derivatives of the RICs with respect to all cartesian coordinates using ase functions"""
    mol = atoms0
    if indices is None:
        indices = np.arange(0, len(mol))

    rim_size = sum([np.shape(length)[0] for length in rim_list])
    b = np.zeros([int(len(indices) * 3), int(rim_size)], dtype=float)  # shape of B-matrix (NCarts,NRIMs)

    # map atom index -> row block start in B (only for selected indices)
    # (so we can write to the right rows without scanning indices)
    idx_pos = {int(a): p for p, a in enumerate(indices)}

    # get all derivatives
    column = 0  # Initilization of columns to specifiy position in B-Matrix
    for q in rim_list[0].tolist() + rim_list[1].tolist():
        q_i, q_j = map(int, q[:2])
        u = mol.get_distance(q_i, q_j, mic=True, vector=True)

        du = ase.geometry.get_distances_derivatives(np.atleast_2d(u))[0]  # (2,3)
        d_i, d_j = du[0], du[1]

        pi = idx_pos.get(q_i)
        if pi is not None:
            _write_row_block(b, pi, column, d_i)

        pj = idx_pos.get(q_j)
        if pj is not None:
            _write_row_block(b, pj, column, d_j)
        column += 1

    for angle in rim_list[2]:
        q_i, q_j, q_k = map(int, angle)

        u = mol.get_distance(q_i, q_j, mic=True, vector=True)
        v = mol.get_distance(q_k, q_j, mic=True, vector=True)

        # Compute derivatives once for this angle.
        # d[0] -> atom i, d[1] -> atom j (center), d[2] -> atom k
        d = _bond_angle_derivatives(np.atleast_2d(u), np.atleast_2d(v))[0]
        dmap = {q_i: -d[0], q_j: -d[1], q_k: -d[2]}

        for pos, atom in enumerate(indices):
            vec = dmap.get(int(atom))
            if vec is not None:
                _write_row_block(b, pos, column, vec)

        column += 1

    for dihedral in rim_list[3]:
        q_i, q_j, q_k, q_l = map(int, dihedral)

        # Copy needed because ASE derivative function might mutate input arrays...
        u = np.copy(np.atleast_2d(mol.get_distance(q_i, q_j, mic=True, vector=True)))
        w = np.copy(np.atleast_2d(mol.get_distance(q_k, q_l, mic=True, vector=True)))
        v = np.copy(np.atleast_2d(mol.get_distance(q_j, q_k, mic=True, vector=True)))

        # Compute derivatives once.
        # d[0] -> atom i, d[1] -> atom j, d[2] -> atom k, d[3] -> atom l
        d = ase.geometry.get_dihedrals_derivatives(u, v, w)[0]
        d = np.radians(d) * ase.units.Bohr
        dmap = {q_i: d[0], q_j: d[1], q_k: d[2], q_l: d[3]}

        for pos, atom in enumerate(indices):
            vec = dmap.get(int(atom))
            if vec is not None:
                _write_row_block(b, pos, column, vec)

        column += 1

    return np.transpose(b)


def hessian_to_ric(B, H_cart):
    """Projects the cartesian Hessian `H_cart` into RIC space using the B-matrix `B`."""
    B_plus = pinv(B)
    B_transp_plus = pinv(B.T)

    if B.ndim == 1:
        return B_transp_plus.dot(H_cart).dot(B_plus)

    P = p_matrix(B, B_plus)
    return P.dot(B_transp_plus).dot(H_cart).dot(B_plus).dot(P)


def p_matrix(B, B_plus):
    """Computes the P-matrix (projection operator in RIC space)."""
    return np.dot(B, B_plus)


def pinv(B, rcond=1e-4):
    """Calculates the pseudoinverse of `B`."""
    if B.ndim == 1:
        return B / 2

    return np.linalg.pinv(B, rcond)


def restrict(B, indices):
    """
    Returns the B-matrix `B` reduced to cartesian coordinates belonging to `indices`.
    """
    if B.shape[1] % 3 != 0:
        raise ValueError("B matrix does not have 3N cartesian columns")
    idx_set = set(map(int, indices))

    # Work on a copy to avoid mutating the caller's matrix.
    Bz = np.array(B, copy=True)

    # zero columns for atoms not in subset
    n_atoms = Bz.shape[1] // 3
    for atom_i in range(n_atoms):
        if atom_i not in idx_set:
            Bz[:, atom_i * 3 : atom_i * 3 + 3] = 0.0

    # slice to subset cartesian coordinates
    col = np.array([[i * 3, i * 3 + 1, i * 3 + 2] for i in indices]).ravel()
    return np.take(Bz, col, axis=1)


# ---- internal helpers ---- #


def _bond_angle_derivatives(u, v):
    """
    Return bond-angle derivatives for B-matrix construction.

    Parameters
    ----------
    u, v : array-like
        Vectors defining the angle (ASE convention: angle between `u` and `v`).

    Returns
    -------
    numpy.ndarray
        Array shaped like ASE's angle-derivative output (per-atom derivative vectors),
        converted to radians and multiplied by `ase.units.Bohr` to match this code's
        historical unit handling.

    Notes
    -----
    Handles linear angles (0°/180°) using an auxiliary vector fallback.
    """

    angle = ase.geometry.get_angles(u, v)  # angle between v and u

    if angle == 180 or angle == 0:  # an auxiliary vector is used if linear angles are existing
        (u, v), (lu, lv) = ase.geometry.conditional_find_mic([u, v], cell=None, pbc=None)
        nu = u / lu
        nv = v / lv
        if (np.arccos(np.dot(nu, (np.array([1, -1, 1]))))) == np.pi:
            w = np.cross(nu, ([-1, 1, 1]))
        else:
            w = np.cross(nu, ([1, -1, 1]))

        nw = w / np.linalg.norm(w)
        d_ba1 = (np.cross(nu, nw)) / np.linalg.norm(u)
        d_ba2 = (-1 * ((np.cross(nu, nw)) / np.linalg.norm(u))) + (-1 * ((np.cross(nw, nv)) / np.linalg.norm(v)))
        d_ba3 = (np.cross(nw, nv)) / np.linalg.norm(v)
        d_ba = np.array([[d_ba1[0], d_ba2[0], d_ba3[0]]])

    else:
        d_ba = np.radians(ase.geometry.get_angles_derivatives(u, v))

    return d_ba * ase.units.Bohr


def _write_row_block(b, row_pos, column, vec) -> None:
    """
    Write a 3-component derivative vector into the B-matrix row block for one atom.

    Parameters
    ----------
    b : numpy.ndarray
        Working B-matrix buffer with shape (3 * len(indices), rim_size) in `calculate()`,
        i.e. Cartesian rows stacked as x,y,z blocks per selected atom.
    row_pos : int
        Position of the atom within the selected `indices` list (0-based), not the
        global atom index. The written rows are [3*row_pos : 3*row_pos+3].
    column : int
        Column index of the current RIC (bond/angle/dihedral) in the B-matrix buffer.
    vec : array-like, shape (3,)
        Derivative components to write into the (x, y, z) rows.
    """
    r = 3 * row_pos
    b[r : r + 3, column] = vec

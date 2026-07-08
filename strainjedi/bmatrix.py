import ase.geometry
import ase.units
import numpy as np
from numpy.typing import NDArray


def calculate(atoms0, rim_list, indices=None) -> NDArray:
    """
    Compute the Wilson B-matrix for a set of redundant internal coordinates (RICs).

    The function evaluates the derivatives of bond lengths, angles, and
    dihedral angles with respect to Cartesian coordinates and assembles them
    into the Wilson B-matrix. Periodic boundary conditions are handled through
    ASE's minimum-image convention (`mic=True`).

    Parameters
    ----------
    atoms0 : ase.Atoms
        Atomic structure for which the B-matrix is evaluated.
    rim_list : sequence of array_like
        Collection of redundant internal coordinates. The expected order is:

        - ``rim_list[0]``: bond-length coordinates
        - ``rim_list[1]``: additional bond-length coordinates
        - ``rim_list[2]``: bond-angle coordinates, shape ``(n_angles, 3)``
        - ``rim_list[3]``: dihedral coordinates, shape ``(n_dihedrals, 4)``

        Bond coordinates are defined by atom pairs ``(i, j)``, angles by
        triplets ``(i, j, k)``, and dihedrals by quadruplets
        ``(i, j, k, l)``.
    indices : array_like of int, optional
        Indices of atoms whose Cartesian coordinates are included in the
        B-matrix. If ``None`` (default), all atoms are included.

    Returns
    -------
    numpy.ndarray
        Wilson B-matrix of shape ``(n_rics, 3 * n_atoms_selected)``, where
        ``n_rics`` is the total number of internal coordinates and
        ``n_atoms_selected`` is ``len(indices)``. Each row contains the
        derivatives of a single internal coordinate with respect to the
        selected Cartesian coordinates.

    Notes
    -----
    The returned matrix is assembled from:

    - Bond-length derivatives computed with
      ``ase.geometry.get_distances_derivatives``.
    - Bond-angle derivatives computed with
      ``_bond_angle_derivatives``.
    - Dihedral-angle derivatives computed with
      ``ase.geometry.get_dihedrals_derivatives``.

    Dihedral derivatives are converted from degrees to radians and scaled by
    ``ase.units.Bohr`` before insertion into the matrix.
    """
    mol = atoms0
    if indices is None:
        indices = np.arange(0, len(mol))

    rim_size = sum(np.shape(length)[0] for length in rim_list)
    b = np.zeros([int(len(indices) * 3), int(rim_size)], dtype=float)  # shape of B-matrix (NCarts,NRIMs)

    # map atom index -> row block start in B (only for selected indices)
    # (so we can write to the right rows without scanning indices)
    idx_pos = {int(a): p for p, a in enumerate(indices)}

    # get all derivatives
    column = 0  # Initialization of columns to specify position in B-Matrix
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
    """
    Transform a Cartesian Hessian into redundant internal coordinate (RIC) space.

    The transformation is performed using the Moore–Penrose pseudoinverse of
    the Wilson B-matrix. For redundant coordinate sets, the projected Hessian
    is additionally constrained to the physically valid RIC subspace using the
    projection matrix returned by ``p_matrix``.

    Parameters
    ----------
    B : numpy.ndarray
        Wilson B-matrix relating Cartesian displacements to internal
        coordinate displacements. For a system with ``n_rics`` redundant
        internal coordinates and ``n_cart`` Cartesian coordinates, the matrix
        typically has shape ``(n_rics, n_cart)``.
    H_cart : numpy.ndarray
        Cartesian Hessian matrix of shape ``(n_cart, n_cart)``.

    Returns
    -------
    numpy.ndarray
        Hessian matrix expressed in redundant internal coordinate space. The
        returned array has shape ``(n_rics, n_rics)`` for a two-dimensional
        B-matrix.

    Notes
    -----
    The transformation is given by

    .. math::

        H_\\mathrm{RIC} = B^{+T} H_\\mathrm{cart} B^+,

    where :math:`B^+` denotes the Moore–Penrose pseudoinverse of the
    B-matrix.

    For redundant internal coordinates, the result is projected into the
    valid RIC subspace using

    .. math::

        H_\\mathrm{RIC} = P B^{+T} H_\\mathrm{cart} B^+ P,

    where :math:`P` is the projector obtained from ``p_matrix(B, B_plus)``.
    """
    B_plus = pinv(B)
    B_transp_plus = pinv(B.T)

    if B.ndim == 1:
        return B_transp_plus.dot(H_cart).dot(B_plus)

    P = p_matrix(B, B_plus)
    return P.dot(B_transp_plus).dot(H_cart).dot(B_plus).dot(P)


def p_matrix(B, B_plus):
    """
    Compute the projection matrix in redundant internal coordinate (RIC) space.

    The projection matrix maps vectors in redundant internal coordinate space
    onto the subspace spanned by the Wilson B-matrix. It is commonly used to
    remove components that do not correspond to physically valid internal
    coordinate displacements.

    Parameters
    ----------
    B : numpy.ndarray
        Wilson B-matrix of shape ``(n_rics, n_cart)``, relating Cartesian
        coordinate displacements to redundant internal coordinate
        displacements.
    B_plus : numpy.ndarray
        Moore–Penrose pseudoinverse of ``B``, typically of shape
        ``(n_cart, n_rics)``.

    Returns
    -------
    numpy.ndarray
        Projection matrix of shape ``(n_rics, n_rics)`` defined as

        .. math::

            P = B B^+.

    Notes
    -----
    For a non-redundant coordinate set, the projection matrix is the identity
    matrix (up to numerical precision). For redundant coordinate sets, it
    projects vectors onto the physically valid RIC subspace.
    """
    return np.dot(B, B_plus)


def pinv(B, rcond=1e-4):
    """
    Compute the pseudoinverse of a Wilson B-matrix.

    For two-dimensional matrices, the Moore–Penrose pseudoinverse is computed
    using :func:`numpy.linalg.pinv`. One-dimensional inputs are treated as a
    special case and scaled by a factor of one-half.

    Parameters
    ----------
    B : numpy.ndarray
        Wilson B-matrix or vector to invert. Typical matrix dimensions are
        ``(n_rics, n_cart)``.
    rcond : float, optional
        Relative cutoff for small singular values passed to
        :func:`numpy.linalg.pinv`. Singular values smaller than
        ``rcond * largest_singular_value`` are treated as zero. The default is
        ``1e-4``.

    Returns
    -------
    numpy.ndarray
        Pseudoinverse of ``B``. For a matrix input with shape
        ``(n_rics, n_cart)``, the returned array has shape
        ``(n_cart, n_rics)``.

    Notes
    -----
    The one-dimensional special case

    .. math::

        B^+ = \\frac{B}{2}

    is included for compatibility with code paths that represent a single
    internal coordinate as a vector rather than a matrix.
    """
    if B.ndim == 1:
        return B / 2

    return np.linalg.pinv(B, rcond)


def restrict(B, indices):
    """
    Restrict a Wilson B-matrix to a subset of Cartesian coordinates.

    The function constructs a reduced B-matrix containing only the Cartesian
    coordinates associated with the specified atoms. Contributions from all
    other atoms are first removed by setting their Cartesian columns to zero,
    after which the matrix is sliced to retain only the selected coordinate
    columns.

    Parameters
    ----------
    B : numpy.ndarray
        Wilson B-matrix of shape ``(n_rics, 3 * n_atoms)``, where rows
        correspond to internal coordinates and columns correspond to
        Cartesian coordinates ordered as
        ``(x_0, y_0, z_0, x_1, y_1, z_1, ...)``.
    indices : array_like of int
        Indices of atoms whose Cartesian coordinates should be retained in the
        reduced matrix.

    Returns
    -------
    numpy.ndarray
        Restricted B-matrix of shape
        ``(n_rics, 3 * len(indices))`` containing only the Cartesian
        coordinates of the selected atoms.

    Raises
    ------
    ValueError
        If the number of columns in ``B`` is not divisible by three, indicating
        that the matrix does not represent a valid set of Cartesian
        coordinates.

    Notes
    -----
    A copy of ``B`` is created internally, so the input matrix is never
    modified in place.

    The ordering of Cartesian coordinates in the returned matrix follows the
    order of atoms given in ``indices``.
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

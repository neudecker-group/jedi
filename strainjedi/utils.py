import ase.units
import numpy as np
from numpy.typing import NDArray

from strainjedi import constants


def get_hbonds(mol, covf=constants.COVALENCY_FACTOR, vdwf=constants.VAN_DER_WAALS_FACTOR):
    """
    Get all hbonds in a structure.
    Hbonds are defined as the HY bond inside X-H···Y where X and Y can be O, N, F and the angle XHY is larger than 90°
    and the distance between HY is shorter than 0.9 times the sum of the vdw radii of H and Y.

    Parameters
    ----------
    mol: class
        Structure of which the hbonds should be determined.
    Returns:
        2D array of indices.

    """
    from ase.data.vdw import vdw_radii
    from ase.neighborlist import natural_cutoffs, neighbor_list

    cutoff = natural_cutoffs(mol, mult=covf)  ## cutoff for covalent bonds see Bakken et al.
    bl = np.vstack(neighbor_list("ij", a=mol, cutoff=cutoff)).T  # determine covalent bonds

    bl = bl[bl[:, 0] < bl[:, 1]]  # remove double mentioned
    bl = np.unique(bl, axis=0)

    hpartner = ["N", "O", "F"]
    hpartner_ls = []
    hcutoff = {
        ("H", "N"): vdwf * (vdw_radii[1] + vdw_radii[7]),
        ("H", "O"): vdwf * (vdw_radii[1] + vdw_radii[8]),
        ("H", "F"): vdwf * (vdw_radii[1] + vdw_radii[9]),
    }  # save the maximum distances for given pairs to be taken account as interactions
    hbond_ls = []  # create a list to store all the bonds
    for i in range(len(mol)):
        if mol.symbols[i] in hpartner:  # check atoms indices of N F O elements
            hpartner_ls.append(i)
    for i in bl:
        if mol.symbols[i[0]] == "H" and mol.symbols[i[1]] in hpartner:
            for j in hpartner_ls:
                if j != i[1] and (
                    mol.get_distance(i[0], j, mic=True) < hcutoff[(mol.symbols[i[0]], mol.symbols[j])]
                    and mol.get_angle(i[1], i[0], j, mic=True) > 90
                ):
                    hbond_ls.append([i[0], j])
        elif mol.symbols[i[0]] in hpartner and mol.symbols[i[1]] == "H":
            for j in hpartner_ls:
                if j != i[0] and (
                    mol.get_distance(i[1], j, mic=True) < hcutoff[(mol.symbols[i[1]], mol.symbols[j])]
                    and mol.get_angle(i[0], i[1], j, mic=True) > 90
                ):
                    hbond_ls.append([i[1], j])
    if len(hbond_ls) > 0:
        hbond_ls = np.array(hbond_ls)
        hbond_ls = np.sort(hbond_ls, axis=1)
        hbond_ls = np.atleast_2d(hbond_ls)
    return hbond_ls


def validate_hessian(modes, atoms0) -> tuple[NDArray, bool]:
    """
    Validate and, if necessary, reorder a Hessian matrix to match an Atoms object.

    This function checks whether the atomic ordering implicit in the Hessian
    (`modes`) is consistent with the ordering of atoms in `atoms0`. If the
    ordering differs, the Hessian is permuted to match `atoms0`.

    Parameters
    ----------
    modes : numpy.ndarray
        Cartesian Hessian matrix or modal representation associated with a
        molecular structure. Typically has shape (3N, 3N), where N is the
        number of atoms.
    atoms0 : ase.Atoms
        Reference atomic structure defining the correct atom ordering.

    Returns
    -------
    hessian : numpy.ndarray
        Hessian reordered (if necessary) to match the atom ordering of
        `atoms0`. If the ordering already matches, the input is returned
        unchanged.
    ok : bool
        True if the original Hessian ordering already matched `atoms0`,
        False if a reordering was performed.

    Notes
    -----
    - The function assumes that the Hessian ordering corresponds directly to
      the atomic ordering in Cartesian blocks (x, y, z per atom).
    - Reordering is performed at the atomic block level, not individual
      Cartesian components.
    - No physical transformation is applied; only index permutation.

    Examples
    --------
    >>> H, ok = validate_hessian(H, atoms)
    >>> if not ok:
    ...     print("Hessian was reordered to match atomic structure.")
    """
    perm = []
    for atom in atoms0:
        match = None
        for i, vib_atom in enumerate(modes._atoms):
            if atom.symbol != vib_atom.symbol:
                continue
            if np.allclose(atom.position, vib_atom.position, rtol=1e-5, atol=1e-5):
                match = i
                break
        perm.append(match)

    perm = np.asarray(perm, dtype=int)

    if np.array_equal(perm, np.arange(len(atoms0))):
        hessian = modes._hessian2d
        return hessian, True

    dof = np.repeat(perm, 3) * 3 + np.tile(np.arange(3), len(perm))
    hessian = modes._hessian2d[np.ix_(dof, dof)]
    return hessian, False


def _convert_units(use_ase_units: bool, rim_list, E_RIMs, deltaE=None, delta_q=None, E_RIMs_total=None):
    """
    Convert RIC-related energies and coordinate displacements between ASE and
    chemical energy units.

    This function performs unit conversions for redundant internal coordinate
    (RIC) analysis outputs, including coordinate displacements and energy
    decompositions. It supports switching between ASE-native units and
    chemistry-friendly units (kcal/mol and Å/degrees).

    Important
    ---------
    This internal API may change at any time; users are encouraged to instead make use of
    the `ase_units` parameters on Jedi.run() and Jedi.visualize().

    Parameters
    ----------
    use_ase_units : bool
        If True, convert outputs to ASE units (Å for lengths, degrees for
        angles, Hartree for energies). If False, convert to chemical units
        (Bohr for lengths, radians for angles, kcal/mol for energies).
    rim_list : sequence of numpy.ndarray
        RIC definition used to determine how many entries correspond to bond
        and custom bond coordinates versus angular coordinates.
    E_RIMs : array_like
        Energies associated with each redundant internal coordinate.
    deltaE : float or array_like, optional
        Total energy difference between geometries.
    delta_q : numpy.ndarray, optional
        Coordinate differences along RICs. Modified in-place if provided.
    E_RIMs_total : float or array_like, optional
        Total summed RIC energy.

    Returns
    -------
    E_RIMs : array_like
        Converted RIC energies.
    deltaE : float or array_like or None
        Converted total energy difference, if provided.
    delta_q : numpy.ndarray or None
        Converted RIC displacements, if provided.
    E_RIMs_total : float or array_like or None
        Converted total RIC energy, if provided.

    Notes
    -----
    Conversion behavior:

    - Length-like coordinates (bonds and custom bonds) use Bohr ↔ Å.
    - Angular coordinates use radians ↔ degrees.
    - Energies are converted between Hartree and kcal/mol using ASE constants.
    """
    if use_ase_units:
        b = rim_list[0].shape[0] + rim_list[1].shape[0]

        if delta_q is not None:
            dq = delta_q.copy()
            dq[:b] = dq[:b] * ase.units.Bohr
            dq[b:] = np.degrees(dq[b:])
            delta_q = dq

        E_RIMs = E_RIMs * ase.units.Hartree
        if E_RIMs_total is not None:
            E_RIMs_total = E_RIMs_total * ase.units.Hartree

    else:
        E_RIMs = E_RIMs / ase.units.kcal * ase.units.mol * ase.units.Hartree
        if E_RIMs_total is not None:
            E_RIMs_total = E_RIMs_total * ase.units.mol / ase.units.kcal * ase.units.Hartree
        if deltaE is not None:
            deltaE = deltaE * ase.units.mol / ase.units.kcal

    return E_RIMs, deltaE, delta_q, E_RIMs_total

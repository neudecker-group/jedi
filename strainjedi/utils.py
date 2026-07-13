import ase.units
import ase.vibrations
import numpy as np
from numpy.typing import NDArray

from strainjedi import constants


def get_hbonds(mol, *, covf=constants.COVALENCY_FACTOR, vdwf=constants.VAN_DER_WAALS_FACTOR, extra_hpartners=()):
    """
    Identify hydrogen bonds in an atomic structure.

    A hydrogen bond is defined as an X–H···Y interaction where X and Y are typically
    N, O, or F (extendable via `extra_hpartners`). The H···Y distance must be smaller
    than `vdwf` times the sum of van der Waals radii, and the X–H···Y angle must exceed 90°.

    The donor X-H bond is identified from ASE covalent neighbor detection (:func:`ase.neighborlist.natural_cutoffs`),
    scaled by `covf`.

    Parameters
    ----------
    mol : ase.Atoms or similar
        Atomic structure containing positions and symbols.
    covf : float, optional
        Scaling factor for covalent bond detection cutoffs (default from constants).
    vdwf : float, optional
        Scaling factor for van der Waals distance criterion (default from constants).
    extra_hpartners : iterable of str, optional
        Additional elements to treat as hydrogen-bond partners.

    Returns
    -------
    ndarray of shape (n, 2), dtype int
        Array of hydrogen bonds as [H_index, acceptor_index] pairs.

    Raises
    ------
    ValueError
        If an element in `extra_hpartners` is not recognized by ASE atomic data.

    Notes
    -----
    - Covalent bonds are reduced to a single representation per bond (i–j instead of both i–j and j–i)
      to avoid double-counting during hydrogen bond donor identification.
    """
    from ase.data import atomic_numbers
    from ase.data.vdw import vdw_radii
    from ase.neighborlist import natural_cutoffs, neighbor_list

    cutoff = natural_cutoffs(mol, mult=covf)  ## cutoff for covalent bonds see Bakken et al.
    bl = np.vstack(neighbor_list("ij", a=mol, cutoff=cutoff)).T  # determine covalent bonds

    bl = bl[bl[:, 0] < bl[:, 1]]  # remove double mentioned
    bl = np.unique(bl, axis=0)

    hpartners = {"N", "O", "F", *extra_hpartners}
    try:
        hcutoff = {
            ("H", elem): vdwf * (vdw_radii[atomic_numbers["H"]] + vdw_radii[atomic_numbers[elem]]) for elem in hpartners
        }  # save the maximum distances for given pairs to be taken account as interactions
    except KeyError as e:
        raise ValueError(f"Unknown element symbol: {e.args[0]!r}") from None

    acceptors = [i for i, symbol in enumerate(mol.symbols) if symbol in hpartners]
    hbonds = []

    for a, b in bl:
        if mol.symbols[a] == "H" and mol.symbols[b] in hpartners:
            h, donor = a, b
        elif mol.symbols[b] == "H" and mol.symbols[a] in hpartners:
            h, donor = b, a
        else:
            continue

        for acceptor in acceptors:
            if acceptor == donor:
                continue

            if (
                mol.get_distance(h, acceptor, mic=True) < hcutoff[("H", mol.symbols[acceptor])]
                and mol.get_angle(donor, h, acceptor, mic=True) > 90
            ):
                hbonds.append([h, acceptor])

    hbonds = np.asarray(hbonds, dtype=int)
    if hbonds.size == 0:
        return np.empty((0, 2), dtype=int)

    hbonds = np.unique(hbonds, axis=0)
    return hbonds


def validate_hessian(modes: ase.vibrations.VibrationsData, atoms0) -> tuple[NDArray, bool]:
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

    try:
        perm = np.asarray(perm, dtype=int)
    except TypeError:
        raise ValueError("hessian does not appear to match atoms") from None

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
    decompositions. It supports switching between ASE-native units (eV, Å and degrees) and
    chemistry-friendly units (kcal/mol, Bohr and radians).

    Important
    ---------
    This internal API may change at any time; users are encouraged to instead make use of
    the `ase_units` parameters on Jedi.run() and Jedi.visualize().

    Parameters
    ----------
    use_ase_units : bool
        If True, convert outputs to ASE units (Å for lengths, degrees for
        angles, eV for energies). If False, convert to chemical units
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
    - Energies are converted between eV and kcal/mol using ASE constants.
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
        if deltaE is not None:
            deltaE = deltaE * ase.units.Hartree

    else:
        E_conversion = ase.units.mol / ase.units.kcal * ase.units.Hartree
        E_RIMs = E_RIMs * E_conversion
        if E_RIMs_total is not None:
            E_RIMs_total = E_RIMs_total * E_conversion
        if deltaE is not None:
            deltaE = deltaE * E_conversion

    return E_RIMs, deltaE, delta_q, E_RIMs_total

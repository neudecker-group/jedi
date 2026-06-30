import ase.units
import numpy as np
from numpy.typing import NDArray


def validate_hessian(modes, atoms0) -> tuple[NDArray, bool]:
    """
    Validates that the order of atoms0 matches the order of the Hessian's elements,
    and reorderes the Hessian to match if that is not the case.

    Returns a tuple of the (reordered) Hessian and a boolean indicating whether the
    Hessian's order matched.

    Parameters
    ----------
    modes:
        The Hessian to validate against atoms0.

    atoms0:
        The Atoms object to validate against.

    Returns
    -------
    hessian : ndarray
        The (possibly) reordered Hessian aligned with `atoms0`.

    ok : bool
        True if the Hessian matched, False if not and the Hessian had to be reordered.

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
    Internal use only.
    Users are encouraged to make use of the `ase_units` parameter of Jedi.visualize() instead.

    Converts the units of `delta_q`, `E_geometries`, `E_RIMs_total`, and `E_RIMs` to
    kcal/mol; if `use_ase_units` is True, use Angstrom and degrees instead.

    Parameters
    ----------
    use_ase_units : bool
        Whether to use ASE units or kcal/mol.
    rim_list
        A list of RICs to determine the bond energies.
    deltaE
        Energy difference between the geometries.
    E_RIMs_total
        The total energy of all RICs.
    E_RIMs
        A nested list (n.b.: dubious?) of the energies for each RIC.
    delta_q: optional
        Array of deformations along the RICs.
    """
    if use_ase_units:
        b = rim_list[0].shape[0] + rim_list[1].shape[0]
        if delta_q is not None:
            delta_q[0:b] *= ase.units.Bohr
            delta_q[b:] = np.degrees(delta_q[b:])
        E_RIMs = E_RIMs * ase.units.Hartree
        if E_RIMs_total is not None:
            E_RIMs_total *= ase.units.Hartree
    else:
        E_RIMs = E_RIMs / ase.units.kcal * ase.units.mol * ase.units.Hartree
        if E_RIMs_total is not None:
            E_RIMs_total *= ase.units.mol / ase.units.kcal * ase.units.Hartree
        if deltaE is not None:
            deltaE *= ase.units.mol / ase.units.kcal

    return E_RIMs, deltaE, delta_q, E_RIMs_total

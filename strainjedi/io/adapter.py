"""Conversion from parser output to ASE objects.

**This is the only module under :mod:`strainjedi.io` that imports ASE**, and the only place
atomic units are converted to ASE's eV/Angstrom. Keeping both facts true of exactly one file
is what makes the parsers reusable without ASE, and means a unit bug has one place to hide.

Import it explicitly -- ``from strainjedi.io.adapter import to_atoms`` -- rather than from
:mod:`strainjedi.io`, so that reaching for ASE is visible at the call site.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.vibrations.data import VibrationsData

from strainjedi.constants import BOHR_ANG, HARTREE_EV, HESSIAN_AU_TO_ASE
from strainjedi.io.types import MissingBlock, ParseError, QCOutput


def to_atoms(out: QCOutput) -> Atoms:
    """Build an :class:`ase.Atoms` from parsed data, with the energy attached.

    The energy is attached through a :class:`SinglePointCalculator` because that is how
    :meth:`strainjedi.jedi.Jedi.get_energies` reads it -- via
    ``atoms.get_potential_energy()``, not via any argument.

    Masses reported by the program are applied when available, so isotope choices made in the
    calculation survive into anything that reasons about them.
    """
    atoms = Atoms(
        numbers=out.numbers,
        positions=out.positions * BOHR_ANG,
        cell=out.cell * BOHR_ANG,
        pbc=out.pbc,
    )

    if out.masses is not None:
        # A partial-Hessian job reports masses only for the atoms it covers, so they are
        # scattered into place and ASE's defaults stand for the rest.
        masses = atoms.get_masses()
        masses[out.hessian_indices if out.hessian_indices is not None else slice(None)] = out.masses
        atoms.set_masses(masses)

    if out.energy is not None:
        atoms.calc = SinglePointCalculator(atoms, energy=out.energy * HARTREE_EV)

    return atoms


def unconstrained_indices(atoms: Atoms) -> np.ndarray:
    """Indices of atoms not held fixed by a ``FixAtoms`` constraint.

    A partial frequency analysis produces a Hessian covering only the moving atoms, and
    ``VibrationsData`` needs to be told which ones those are.
    """
    fixed: list[int] = []
    for constraint in atoms.constraints:
        if constraint.__class__.__name__ == "FixAtoms":
            fixed.extend(constraint.todict()["kwargs"]["indices"])

    return np.delete(np.arange(len(atoms)), fixed)


def to_vibrations(
    out: QCOutput,
    atoms: Atoms | None = None,
    indices: Sequence[int] | None = None,
) -> VibrationsData:
    """Build an :class:`ase.vibrations.VibrationsData` from a parsed Hessian.

    Args:
        out: Parsed output containing a Hessian.
        atoms: Structure to attach it to. Defaults to the one in ``out``; pass the optimised
            geometry explicitly when the Hessian came from a separate file. Its elements must
            appear in the same order as in ``out`` -- see below.
        indices: Which atoms the Hessian covers, for a partial frequency analysis. Taken from
            ``out`` when the file said so, otherwise derived from ``atoms``' ``FixAtoms``
            constraints if the Hessian is too small to be a full one.

    Raises:
        ParseError: If ``atoms`` does not have the same elements in the same order as ``out``.
            Q-Chem reorders the molecule for a partial-Hessian job -- H2O2 comes back as
            O,H,O,H rather than O,O,H,H -- so pairing that Hessian with the structure from the
            optimisation would silently misassign every force constant. Refusing is the point:
            a loud failure beats a plausible wrong answer.
    """
    if out.hessian is None:
        raise MissingBlock("Hessian", source=out.source)

    if atoms is None:
        atoms = to_atoms(out)
    elif not np.array_equal(atoms.numbers, out.numbers):
        raise ParseError(
            f"cannot attach the Hessian from '{out.source}' to this structure: the file has "
            f"elements {[int(z) for z in out.numbers]} but the structure has "
            f"{[int(z) for z in atoms.numbers]}. Programs reorder atoms for partial-Hessian "
            f"jobs, so use the geometry from the same file, or reorder the structure to match."
        )

    if indices is None:
        indices = out.hessian_indices

    if indices is None and out.hessian.shape[0] != 3 * len(atoms):
        indices = unconstrained_indices(atoms)

    return VibrationsData.from_2d(atoms, out.hessian * HESSIAN_AU_TO_ASE, indices)

"""Reading geometries, energies and Hessians out of quantum-chemistry output files.

Everything here returns plain NumPy arrays in **atomic units** -- Hartree, Bohr,
Hartree/Bohr^2 -- because that is what the physics is done in. Nothing in this module imports
ASE. To get :class:`ase.Atoms` or :class:`ase.vibrations.VibrationsData`, import the adapter
explicitly::

    from strainjedi.io import read_output
    from strainjedi.io.adapter import to_atoms, to_vibrations

    opt  = read_output("opt.out")        # program detected from the file
    freq = read_output("freq.hess")

    jedi = Jedi(to_atoms(opt), to_atoms(dist), to_vibrations(freq, to_atoms(opt)))

Which file holds what differs by program, so these functions take a path to the file that
actually contains the data rather than a shared basename:

===========  ===================================  ==============================
Program      Geometry and energy                  Hessian
===========  ===================================  ==============================
ORCA         ``.out``                             ``.hess`` (also has geometry)
Gaussian     ``.log``                             ``.fchk`` if present, else ``.log``
Q-Chem       ``.out``                             same ``.out``
===========  ===================================  ==============================
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from strainjedi.io.registry import detect_program, reader_for
from strainjedi.io.types import (
    MissingBlock,
    NotConverged,
    ParseError,
    ProgramNotDetected,
    QCOutput,
)

__all__ = [
    "MissingBlock",
    "NotConverged",
    "ParseError",
    "ProgramNotDetected",
    "QCOutput",
    "detect_program",
    "read_energy",
    "read_geometry",
    "read_hessian",
    "read_output",
]


def read_output(path: Path | str, *, program: str | None = None) -> QCOutput:
    """Parse whatever a single output file contains.

    Args:
        path: File to read.
        program: Override detection, e.g. ``"orca"``. Rarely needed.

    Returns:
        A :class:`QCOutput` in atomic units. Fields the file does not contain are None.
    """
    return reader_for(path, program).read_output(path)


def read_geometry(path: Path | str, *, program: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Atomic numbers and positions in Bohr."""
    out = read_output(path, program=program)
    return out.numbers, out.positions


def read_energy(path: Path | str, *, program: str | None = None) -> float:
    """Total energy in Hartree."""
    out = read_output(path, program=program)
    if out.energy is None:
        raise MissingBlock("energy", source=path)
    return out.energy


def read_hessian(path: Path | str, *, program: str | None = None) -> np.ndarray:
    """Symmetric (3N, 3N) Cartesian Hessian in Hartree/Bohr^2.

    Only frequency calculations contain one, and some programs need to be asked; the raised
    :class:`MissingBlock` names the keyword that does it.
    """
    out = read_output(path, program=program)
    if out.hessian is None:
        hint = getattr(reader_for(path, program), "HESSIAN_HINT", None)
        raise MissingBlock("Hessian", source=path, hint=hint)
    return out.hessian

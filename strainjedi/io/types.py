"""Return type and error hierarchy for the quantum-chemistry output parsers.

Everything here is plain Python and NumPy: no ASE. See :mod:`strainjedi.io.readers` for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


class ParseError(Exception):
    """Base class for every failure raised while reading a QC output file."""


class ProgramNotDetected(ParseError):
    """The file could not be attributed to a supported program."""


class MissingBlock(ParseError):
    """A block the caller asked for is not present in the file.

    Carries the keyword that makes the program print the block, because "no Hessian found"
    is a support ticket while "no Hessian found, add iop(7/33=1)" is not.
    """

    def __init__(self, what: str, source: Path | str | None = None, hint: str | None = None):
        message = f"No {what} found"
        if source is not None:
            message += f" in '{source}'"
        message += "."
        if hint is not None:
            message += f" {hint}"
        super().__init__(message)
        self.what = what
        self.source = source
        self.hint = hint


class NotConverged(ParseError):
    """The calculation itself did not converge, so its numbers are not usable."""


@dataclass(frozen=True, eq=False)
class QCOutput:
    """Everything JEDI needs from one output file, in atomic units.

    Atomic units throughout -- Hartree, Bohr, Hartree/Bohr^2 -- because that is what the
    physics is done in. Conversion to ASE's eV/Angstrom happens once, in
    :mod:`strainjedi.io.adapter`, and nowhere else.

    Attributes:
        numbers: (N,) atomic numbers. Not chemical symbols: Gaussian prints Z, and ghost
            atoms and isotopes make symbols lossy.
        positions: (N, 3) Cartesian coordinates in Bohr.
        energy: Total energy in Hartree, or None if the file has none.
        hessian: (3N, 3N) symmetric Cartesian Hessian in Hartree/Bohr^2, or None.
        masses: atomic masses in amu *as the program itself reported them*, or None. Needed to
            undo Q-Chem's mass weighting exactly rather than guessing isotopes. One per atom,
            or one per entry of `hessian_indices` when the Hessian is partial -- a program
            that computes a partial Hessian only reports masses for the atoms it covers.
        hessian_indices: which atoms `hessian` covers, indexing into `positions`, or None when
            it covers all of them. A partial frequency analysis produces a Hessian smaller
            than 3N, and this is what says which atoms moved.
        cell: (3, 3) lattice vectors in Bohr. All-zero for an isolated molecule.
        pbc: (3,) periodicity flags. JEDI needs these: `bmatrix` measures distances with the
            minimum-image convention, and the VMD visualiser draws the box.
        program: Lowercase program name, e.g. "orca".
        version: Parsed version tuple, e.g. (5, 0, 0). Diagnostic only -- never branch on it;
            see the anchor tables in the reader modules.
        source: Path the data was read from.
    """

    numbers: np.ndarray
    positions: np.ndarray
    program: str
    source: Path
    energy: float | None = None
    hessian: np.ndarray | None = None
    masses: np.ndarray | None = None
    hessian_indices: np.ndarray | None = None
    cell: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    pbc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=bool))
    version: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.numbers)
        covered = n if self.hessian_indices is None else len(self.hessian_indices)

        if self.positions.shape != (n, 3):
            raise ParseError(f"positions has shape {self.positions.shape}, expected {(n, 3)}")

        if self.hessian_indices is not None:
            out_of_range = [int(i) for i in self.hessian_indices if not 0 <= i < n]
            if out_of_range:
                raise ParseError(f"hessian_indices {out_of_range} are outside a {n}-atom structure")

        if self.hessian is not None and self.hessian.shape != (3 * covered, 3 * covered):
            raise ParseError(f"hessian has shape {self.hessian.shape}, expected {(3 * covered, 3 * covered)}")

        if self.masses is not None and self.masses.shape != (covered,):
            raise ParseError(f"masses has shape {self.masses.shape}, expected {(covered,)}")

        if self.cell.shape != (3, 3):
            raise ParseError(f"cell has shape {self.cell.shape}, expected (3, 3)")

        if self.pbc.shape != (3,):
            raise ParseError(f"pbc has shape {self.pbc.shape}, expected (3,)")

    @property
    def natoms(self) -> int:
        return len(self.numbers)

    @property
    def is_partial_hessian(self) -> bool:
        """Whether the Hessian covers only some of the atoms."""
        return self.hessian_indices is not None

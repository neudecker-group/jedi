"""Reader for Q-Chem output.

Verified against Q-Chem 6.0.0. Q-Chem is the awkward one of the three: it prints the Hessian
mass-weighted and as bare columns with no row or column labels, so the matrix dimension has to
come from the geometry. It does, however, print the masses it actually used, which makes
undoing the weighting exact rather than a guess at which isotopes were meant.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from strainjedi.constants import BOHR_ANG
from strainjedi.io import scan
from strainjedi.io.elements import symbol_to_number
from strainjedi.io.types import MissingBlock, NotConverged, ParseError, QCOutput

MAGIC = (b"A Quantum Leap Into The Future Of Chemistry", b"Q-Chem, Inc.")
"""Deliberately the full banners: a bare b"Q-Chem" would also match citations in other
programs' outputs, the way a bare b"Gaussian" matches ORCA's basis-set notes."""

ANCHORS = {
    "version": [re.compile(r"Q-Chem\s+(\d+)\.(\d+)\.(\d+)"), re.compile(r"Q-Chem\s+(\d+)\.(\d+)")],
    "geometry": ["Standard Nuclear Orientation (Angstroms)"],
    "energy": ["Total energy in the final basis set =", "Total energy ="],
    # Anchored at the start of the line so it cannot also match "Projected Mass-Weighted
    # Hessian Matrix:" or "Eigenvectors of Proj. Mass-Weighted Hessian Matrix:", both of
    # which follow it in the same file and look identical to a substring search.
    "hessian": [re.compile(r"^\s*Mass-Weighted Hessian Matrix:")],
    "masses": [re.compile(r"^\s*Atom\s+\d+\s+Element\s+\S+\s+Has Mass\s")],
    "scf_failure": ["SCF failed to converge", "ERROR: alpha_min"],
}

HESSIAN_HINT = "Add 'vibman_print 7' to the $rem section of a freq job."


def read_version(lines: list[str]) -> tuple[int, ...]:
    for pattern in ANCHORS["version"]:
        for line in lines:
            match = pattern.search(line)
            if match:
                return tuple(int(g) for g in match.groups())
    return ()


def check_converged(lines: list[str]) -> None:
    """Raise if the SCF did not converge, so downstream numbers are never silently trusted."""
    hits = scan.find_anchors(lines, ANCHORS["scf_failure"])
    if hits:
        raise NotConverged(f"SCF did not converge: {lines[hits[0]].strip()}")


def read_energy(lines: list[str]) -> float | None:
    """Last total energy, in Hartree."""
    energy = None
    for i in scan.find_anchors(lines, ANCHORS["energy"]):
        tokens = lines[i].split()
        if tokens and scan.is_float_row(tokens[-1:]):
            energy = scan.to_float(tokens[-1])
    return energy


def read_geometry(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Last standard nuclear orientation, as (numbers, positions in Bohr)."""
    hits = scan.find_anchors(lines, ANCHORS["geometry"])
    if not hits:
        raise MissingBlock("geometry", hint="Expected a 'Standard Nuclear Orientation' block.")

    numbers: list[int] = []
    positions: list[list[float]] = []

    for line in lines[hits[-1] + 1 :]:
        tokens = line.split()
        if len(tokens) != 5 or not scan.is_int_row(tokens[:1]) or not scan.is_float_row(tokens[2:]):
            if numbers:
                break
            continue
        numbers.append(symbol_to_number(tokens[1]))
        positions.append([scan.to_float(t) for t in tokens[2:]])

    if not numbers:
        raise ParseError("found a geometry header but no atom rows under it")

    return np.array(numbers), np.array(positions) / BOHR_ANG


def read_masses(lines: list[str]) -> np.ndarray | None:
    """Masses in amu exactly as Q-Chem used them, from its ``Has Mass`` listing.

    These are what the Hessian was weighted with, so using them to undo the weighting is exact
    even when isotopes were specified. Returns None when the listing is absent, which happens
    for jobs that were not frequency calculations.
    """
    hits = scan.find_anchors(lines, ANCHORS["masses"])
    if not hits:
        return None
    return np.array([scan.to_float(lines[i].split()[-1]) for i in hits])


def read_hessian(lines: list[str], natoms: int, masses: np.ndarray) -> np.ndarray:
    """Cartesian Hessian in Hartree/Bohr^2, with Q-Chem's mass weighting removed.

    Args:
        lines: The whole file.
        natoms: Needed because the block itself carries no dimension information.
        masses: In amu, as reported by the program.
    """
    hits = scan.find_anchors(lines, ANCHORS["hessian"])
    if not hits:
        raise MissingBlock("Hessian", hint=HESSIAN_HINT)

    mass_weighted = scan.parse_bare_block(lines, hits[0], 3 * natoms)

    weights = np.repeat(np.sqrt(masses), 3)
    return scan.symmetrize(mass_weighted * np.outer(weights, weights))


def read_output(path: Path | str) -> QCOutput:
    """Parse a Q-Chem ``.out`` into a :class:`QCOutput`."""
    path = Path(path)
    lines = scan.read_lines(path)
    check_converged(lines)

    numbers, positions = read_geometry(lines)
    masses = read_masses(lines)

    hessian = None
    if scan.find_anchors(lines, ANCHORS["hessian"]):
        if masses is None:
            raise MissingBlock(
                "atomic masses",
                source=path,
                hint="They are needed to undo Q-Chem's mass weighting of the Hessian.",
            )
        if len(masses) != len(numbers):
            raise ParseError(f"parsed {len(masses)} masses for {len(numbers)} atoms")
        hessian = read_hessian(lines, len(numbers), masses)

    return QCOutput(
        numbers=numbers,
        positions=positions,
        energy=read_energy(lines),
        hessian=hessian,
        masses=masses,
        program="qchem",
        version=read_version(lines),
        source=path,
    )

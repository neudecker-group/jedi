"""Reader for Q-Chem output.

Verified against Q-Chem 6.0.0 and 6.3.1. Q-Chem is the awkward one of the three: it prints the
Hessian mass-weighted and as bare columns with no row or column labels, so the matrix
dimension has to come from the geometry. It does, however, print the masses it actually used,
which makes undoing the weighting exact rather than a guess at which isotopes were meant.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from strainjedi.constants import BOHR_ANG
from strainjedi.io import scan
from strainjedi.io.elements import symbol_to_number
from strainjedi.io.report import warn_imaginary_frequencies
from strainjedi.io.types import MissingBlock, NotConverged, ParseError, QCOutput

MAGIC = (b"A Quantum Leap Into The Future Of Chemistry", b"Q-Chem, Inc.")
"""Deliberately the full banners: a bare b"Q-Chem" would also match citations in other
programs' outputs, the way a bare b"Gaussian" matches ORCA's basis-set notes."""

ANCHORS = {
    "version": [re.compile(r"Q-Chem\s+(\d+)\.(\d+)\.(\d+)"), re.compile(r"Q-Chem\s+(\d+)\.(\d+)")],
    "geometry": ["Standard Nuclear Orientation (Angstroms)"],
    "energy": [
        "Total energy in the final basis set =",  # Q-Chem <= 6.0
        "Total energy =",  # Q-Chem >= 6.3
    ],
    # Not an energy to report, but the line that says where the optimisation stopped. See
    # read_energy: two different job types make the naive "last energy wins" rule wrong, in
    # opposite directions, and this marker separates them.
    "optimisation_end": ["Final energy is"],
    # Anchored at the start of the line so it cannot also match "Projected Mass-Weighted
    # Hessian Matrix:" or "Eigenvectors of Proj. Mass-Weighted Hessian Matrix:", both of
    # which follow it in the same file and look identical to a substring search.
    "hessian": [re.compile(r"^\s*Mass-Weighted Hessian Matrix:")],
    "masses": [re.compile(r"^\s*Atom\s+\d+\s+Element\s+\S+\s+Has Mass\s")],
    "scf_failure": ["SCF failed to converge", "ERROR: alpha_min"],
    "imaginary": [re.compile(r"This Molecule has\s+(\d+)\s+Imaginary Frequencies")],
    "partial_hessian": ["Hessian Limited to Following Atoms"],
    "reordering": ["Reordering Atoms for Partial Hessian Frequencies"],
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


def count_imaginary(lines: list[str]) -> int | None:
    """How many imaginary frequencies Q-Chem reported, or None if it ran no frequency analysis.

    Q-Chem states the count outright, including ``0``, so absence of the line means the file
    simply has no vibrational analysis to judge.
    """
    hits = scan.find_anchors(lines, ANCHORS["imaginary"])
    if not hits:
        return None

    match = ANCHORS["imaginary"][0].search(lines[hits[-1]])
    return int(match.group(1)) if match else None


def read_energy(lines: list[str]) -> float | None:
    """Total energy in Hartree: the last plain SCF energy of the optimisation.

    "Last energy in the file" is wrong for two different job types, in opposite directions,
    and neither is rare:

    * A **semi-numerical** frequency job (``IDERIV=1``) differentiates analytic gradients, so
      it prints one energy per *displaced* geometry after the optimisation has finished --
      139 of them in the 6.3 sample. The last is a displaced point, not the structure.
    * An **EFEI** job (``$distort``) reports a final energy that includes the work done by the
      external force. That is a value on the force-modified surface, not the plain one JEDI
      needs; in the 6.4 sample it is 92.8 kcal/mol away from the electronic energy.

    Both are resolved by the same observation: Q-Chem's ``Final energy is`` marks where the
    optimisation ended. Displaced energies come *after* it, and the EFEI-augmented value *is*
    it. So take the last plain energy at or before that marker, and neither can be picked up.
    Validated against nine outputs spanning 6.0, 6.3 and 6.4: identical on eight, correct on
    the EFEI one, where the old rule was not.
    """
    marker = scan.find_anchors(lines, ANCHORS["optimisation_end"])
    cutoff = marker[-1] if marker else len(lines)

    hits = scan.find_anchors(lines, ANCHORS["energy"])
    # A single-point job has no marker and no trailing noise, so it falls back to every hit.
    candidates = [i for i in hits if i <= cutoff] or hits

    energy = None
    for i in candidates:
        tokens = lines[i].split()
        if tokens and scan.is_float_row(tokens[-1:]):
            energy = scan.to_float(tokens[-1])
    return energy


def read_geometry(lines: list[str], before: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Last standard nuclear orientation, as (numbers, positions in Bohr).

    Args:
        lines: The whole file.
        before: Ignore geometries printed after this line. Used to skip the permuted copy a
            partial-Hessian job prints, so the structure comes back in the order it went in.
    """
    hits = scan.find_anchors(lines, ANCHORS["geometry"])
    if before is not None:
        hits = [i for i in hits if i < before] or hits
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


def read_hessian_indices(lines: list[str], positions: np.ndarray) -> np.ndarray | None:
    """Which atoms a partial Hessian covers, or None for an ordinary full one.

    A ``PHESS`` job computes the Hessian for the atoms in ``$alist`` only, and prints them
    under "Hessian Limited to Following Atoms" with their coordinates. Q-Chem also *reorders*
    the molecule internally for such a job, so the index it prints beside each atom refers to
    the permuted molecule and means nothing outside it. Matching on coordinates instead maps
    them onto whichever geometry is passed in -- which is how the Hessian ends up expressed in
    the atom order the input used, rather than the one Q-Chem happened to work in.

    The returned order follows the block, so it lines up with the Hessian's own row blocks.

    Args:
        lines: The whole file.
        positions: Geometry from the same file, in Bohr, to match against.
    """
    hits = scan.find_anchors(lines, ANCHORS["partial_hessian"])
    if not hits:
        return None

    indices: list[int] = []
    for line in lines[hits[-1] + 1 :]:
        tokens = line.split()
        if len(tokens) != 5 or not scan.is_int_row(tokens[:2]) or not scan.is_float_row(tokens[2:]):
            if indices:
                break
            continue

        wanted = np.array([scan.to_float(t) for t in tokens[2:]]) / BOHR_ANG
        # The block prints five decimals against the geometry's ten, so match on distance.
        distances = np.linalg.norm(positions - wanted, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] > 1e-3:
            raise ParseError(f"partial-Hessian atom at {wanted * BOHR_ANG} matches no atom in the geometry")
        indices.append(nearest)

    if not indices:
        raise ParseError("found the partial-Hessian header but no atom rows under it")

    if len(set(indices)) != len(indices):
        raise ParseError(f"partial-Hessian atoms matched the same geometry atom twice: {indices}")

    return np.array(indices)


def read_hessian(lines: list[str], natoms: int, masses: np.ndarray) -> np.ndarray:
    """Cartesian Hessian in Hartree/Bohr^2, with Q-Chem's mass weighting removed.

    Args:
        lines: The whole file.
        natoms: How many atoms the Hessian covers -- every atom for a full Hessian, only the
            listed ones for a partial one. The block itself carries no dimension information.
        masses: In amu, as reported by the program, one per covered atom.
    """
    hits = scan.find_anchors(lines, ANCHORS["hessian"])
    if not hits:
        raise MissingBlock("Hessian", hint=HESSIAN_HINT)

    mass_weighted = scan.parse_bare_block(lines, hits[0], 3 * natoms)

    weights = np.repeat(np.sqrt(masses), 3)
    return scan.symmetrize(mass_weighted * np.outer(weights, weights))


def read_output(path: Path | str) -> QCOutput:
    """Parse a Q-Chem ``.out`` into a :class:`QCOutput`.

    A partial-Hessian job makes Q-Chem permute the molecule internally, but that permutation
    is an implementation detail of the program and not something the caller chose -- the same
    input geometry goes in either way. So the geometry comes back in the order it was given,
    taken from before the permutation, and the Hessian is expressed against *that* order.
    """
    path = Path(path)
    lines = scan.read_lines(path)
    check_converged(lines)

    # Everything printed after this marker is in Q-Chem's permuted order; the geometry is
    # taken from before it so the caller gets their own atom order back.
    reordering = scan.find_anchors(lines, ANCHORS["reordering"])
    numbers, positions = read_geometry(lines, before=reordering[0] if reordering else None)
    masses = read_masses(lines)
    indices = read_hessian_indices(lines, positions)
    covered = len(numbers) if indices is None else len(indices)

    hessian = None
    if scan.find_anchors(lines, ANCHORS["hessian"]):
        if masses is None:
            raise MissingBlock(
                "atomic masses",
                source=path,
                hint="They are needed to undo Q-Chem's mass weighting of the Hessian.",
            )
        if len(masses) != covered:
            raise ParseError(f"parsed {len(masses)} masses for {covered} atoms in the Hessian")
        hessian = read_hessian(lines, covered, masses)

    warn_imaginary_frequencies(count_imaginary(lines), path)

    return QCOutput(
        numbers=numbers,
        positions=positions,
        energy=read_energy(lines),
        hessian=hessian,
        masses=masses,
        hessian_indices=indices,
        program="qchem",
        version=read_version(lines),
        source=path,
    )

"""Reader for VASP ``OUTCAR`` files.

Verified against VASP 6.4.2. VASP differs from the molecular codes in three ways that matter:

* It is **periodic**, so the lattice is part of the answer rather than an afterthought. JEDI
  needs it -- ``bmatrix`` measures distances with the minimum-image convention.
* It works in **eV and Angstrom**, not atomic units, so this reader converts more than the
  others do.
* Its ``SECOND DERIVATIVES`` block is the *negative* of the Hessian. VASP prints the
  derivative of the force, and the printed diagonal is negative where a Hessian's is positive.
  Getting this wrong is not subtle -- the frequencies come out imaginary -- but it is silent
  if you never check them, which is what the round-trip test is for.

There is no file extension to go on: the file is simply called ``OUTCAR``. Detection is by
the ``vasp.`` banner on the first line, with the filename as a fallback.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from strainjedi.constants import BOHR_ANG, HARTREE_EV, HESSIAN_AU_TO_ASE
from strainjedi.io import scan
from strainjedi.io.elements import symbol_to_number
from strainjedi.io.report import warn_imaginary_frequencies
from strainjedi.io.types import MissingBlock, ParseError, QCOutput

MAGIC = (b" vasp.",)
"""The banner on OUTCAR's first line. Kept with its leading space so it cannot match prose."""

FILENAMES = ("OUTCAR",)
"""VASP output has no extension, so the registry matches on the name as well."""

ANCHORS = {
    "version": [re.compile(r"vasp\.(\d+)\.(\d+)\.(\d+)")],
    "geometry": ["POSITION", "TOTAL-FORCE (eV/Angst)"],
    "cell": ["direct lattice vectors"],
    # energy(sigma->0) is the value extrapolated to zero smearing, i.e. the 0 K electronic
    # energy, which is what a strain analysis wants. TOTEN still carries the -TS term.
    "energy": ["energy(sigma->0) ="],
    "energy_fallback": ["free  energy   TOTEN"],
    "hessian": ["SECOND DERIVATIVES (NOT SYMMETRIZED)"],
    "species": [re.compile(r"VRHFIN\s*=\s*([A-Za-z]+)")],
    "ions_per_type": ["ions per type"],
    "masses": ["Mass of Ions in am"],
    "frequencies": [re.compile(r"\d+\s+f(?:/i)?\s*=")],
    # IBRION=5/6 builds the Hessian by displacing every atom in turn, and each displacement
    # gets its own POSITION block and its own energy. Everything after this marker therefore
    # describes a displaced structure, not the one the Hessian belongs to.
    "displacements": ["Finite differences"],
}

HESSIAN_HINT = "Run a finite-difference frequency job (IBRION=5 or 6)."

NEAR_ZERO_CM = 10.0
"""Below this, an "imaginary" frequency is numerical noise rather than a real mode.

A periodic calculation has three translations whose frequencies should be exactly zero and
come out as tiny imaginary values instead -- 0.11, 0.16 and 0.38 cm^-1 in the 6.4.2 sample.
Counting those as imaginary would condemn every well-converged structure.
"""

_ROW_LABEL = re.compile(r"(\d+)([XYZ])\Z")
"""A Hessian row label such as ``10Z``: atom number, then the Cartesian direction."""

_CM = re.compile(r"(-?[\d.]+)\s*cm-1")


def read_version(lines: list[str]) -> tuple[int, ...]:
    for line in lines[:5]:
        match = ANCHORS["version"][0].search(line)
        if match:
            return tuple(int(g) for g in match.groups())
    return ()


def read_species(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Atomic numbers and masses per atom, from the POTCAR summary VASP echoes.

    VASP lists each species once and then says how many ions it has, so both arrays are
    rebuilt by expanding those counts.
    """
    symbols = [match.group(1) for line in lines if (match := ANCHORS["species"][0].search(line))]

    counts_hits = scan.find_anchors(lines, ANCHORS["ions_per_type"])
    if not symbols or not counts_hits:
        raise MissingBlock("species listing", hint="Expected 'VRHFIN' and 'ions per type' lines.")
    counts = [int(t) for t in lines[counts_hits[-1]].split("=")[1].split()]

    mass_hits = scan.find_anchors(lines, ANCHORS["masses"])
    if not mass_hits:
        raise MissingBlock("ion masses", hint="Expected a 'Mass of Ions in am' block.")
    masses_per_type = [scan.to_float(t) for t in lines[mass_hits[-1] + 1].split("=")[1].split()]

    # VRHFIN appears once per POTCAR; duplicates would mean a malformed header.
    symbols = symbols[: len(counts)]
    if not len(symbols) == len(counts) == len(masses_per_type):
        raise ParseError(f"{len(symbols)} species, {len(counts)} counts, {len(masses_per_type)} masses")

    numbers = np.concatenate([[symbol_to_number(s)] * n for s, n in zip(symbols, counts)])
    masses = np.concatenate([[m] * n for m, n in zip(masses_per_type, counts)])
    return numbers, masses


def read_cell(lines: list[str]) -> np.ndarray:
    """Lattice vectors in Bohr from the last ``direct lattice vectors`` block.

    Each row holds the direct vector followed by the reciprocal one; only the first three
    columns are the lattice.
    """
    hits = scan.find_anchors(lines, ANCHORS["cell"])
    if not hits:
        raise MissingBlock("lattice vectors", hint="Expected a 'direct lattice vectors' block.")

    rows = []
    for line in lines[hits[-1] + 1 :]:
        tokens = line.split()
        if len(tokens) != 6 or not scan.is_float_row(tokens):
            break
        rows.append([scan.to_float(t) for t in tokens[:3]])
        if len(rows) == 3:
            break

    if len(rows) != 3:
        raise ParseError(f"expected 3 lattice vectors, parsed {len(rows)}")

    return np.array(rows) / BOHR_ANG


def read_geometry(lines: list[str], before: int | None = None) -> np.ndarray:
    """Positions in Bohr from the last ``POSITION / TOTAL-FORCE`` block.

    Args:
        lines: The whole file.
        before: Ignore blocks printed after this line, so a finite-difference run reports the
            structure the Hessian belongs to rather than the last atom it nudged.
    """
    hits = [i for i, line in enumerate(lines) if all(a in line for a in ANCHORS["geometry"])]
    if before is not None:
        hits = [i for i in hits if i < before] or hits
    if not hits:
        raise MissingBlock("geometry", hint="Expected a 'POSITION ... TOTAL-FORCE' block.")

    positions = []
    for line in lines[hits[-1] + 1 :]:
        tokens = line.split()
        # Six columns: three coordinates then three force components.
        if len(tokens) != 6 or not scan.is_float_row(tokens):
            if positions:
                break
            continue
        positions.append([scan.to_float(t) for t in tokens[:3]])

    if not positions:
        raise ParseError("found a POSITION header but no coordinate rows under it")

    return np.array(positions) / BOHR_ANG


def read_energy(lines: list[str], before: int | None = None) -> float | None:
    """Last total energy in Hartree, extrapolated to zero smearing.

    Args:
        lines: The whole file.
        before: Ignore energies printed after this line. A finite-difference frequency run
            prints one per displaced structure, and none of those is the reference energy.
    """
    for key in ("energy", "energy_fallback"):
        hits = scan.find_anchors(lines, ANCHORS[key])
        if before is not None:
            hits = [i for i in hits if i < before] or hits
        for i in reversed(hits):
            tokens = [t for t in lines[i].replace("=", " ").split() if scan.is_float_row([t])]
            if tokens:
                return scan.to_float(tokens[-1]) / HARTREE_EV
    return None


def read_frequencies(lines: list[str]) -> np.ndarray | None:
    """Harmonic frequencies in cm^-1, negative for the modes VASP flags as imaginary."""
    hits = scan.find_anchors(lines, ANCHORS["frequencies"])
    if not hits:
        return None

    frequencies = []
    for i in hits:
        match = _CM.search(lines[i])
        if match:
            value = scan.to_float(match.group(1))
            frequencies.append(-value if "f/i=" in lines[i] else value)
    return np.array(frequencies)


def count_imaginary(lines: list[str]) -> int | None:
    """Genuinely imaginary modes, ignoring the near-zero ones a periodic cell always has."""
    frequencies = read_frequencies(lines)
    if frequencies is None:
        return None
    return int(np.sum(frequencies < -NEAR_ZERO_CM))


def read_hessian(lines: list[str], natoms: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Hessian in Hartree/Bohr^2 and the atoms it covers.

    The block is one row per Cartesian degree of freedom, labelled ``<atom><X|Y|Z>``, with
    every column on the same line -- no chunking to stitch back together. Those labels also
    say which atoms moved, so a run that froze some atoms with selective dynamics yields a
    partial Hessian without any extra bookkeeping.

    The printed values are negated on the way out: VASP reports the derivative of the force.
    """
    hits = scan.find_anchors(lines, ANCHORS["hessian"])
    if not hits:
        raise MissingBlock("Hessian", hint=HESSIAN_HINT)

    labels: list[str] = []
    rows: list[list[float]] = []
    for line in lines[hits[-1] + 1 :]:
        tokens = line.split()
        if len(tokens) < 2 or not _ROW_LABEL.match(tokens[0]) or not scan.is_float_row(tokens[1:]):
            # The dashed rule and the column header sit between the anchor and the data.
            if rows:
                break
            continue
        labels.append(tokens[0])
        rows.append([scan.to_float(t) for t in tokens[1:]])

    if not rows:
        raise ParseError("found the SECOND DERIVATIVES header but no data rows under it")

    matrix = np.array(rows)
    if matrix.shape[0] != matrix.shape[1]:
        raise ParseError(f"second-derivative block is {matrix.shape}, expected a square matrix")

    hessian = scan.symmetrize(-matrix) / HESSIAN_AU_TO_ASE

    covered = sorted({int(_ROW_LABEL.match(label).group(1)) - 1 for label in labels})
    indices = None if covered == list(range(natoms)) else np.array(covered)
    return hessian, indices


def read_output(path: Path | str) -> QCOutput:
    """Parse a VASP ``OUTCAR`` into a :class:`QCOutput`."""
    path = Path(path)
    lines = scan.read_lines(path)

    numbers, masses = read_species(lines)

    # A finite-difference run describes displaced structures from here on; the reference
    # geometry and energy are the last ones printed before it starts.
    displacements = scan.find_anchors(lines, ANCHORS["displacements"])
    cutoff = displacements[0] if displacements else None

    positions = read_geometry(lines, before=cutoff)

    hessian, indices = (None, None)
    if scan.find_anchors(lines, ANCHORS["hessian"]):
        hessian, indices = read_hessian(lines, len(numbers))

    warn_imaginary_frequencies(count_imaginary(lines), path)

    return QCOutput(
        numbers=numbers,
        positions=positions,
        energy=read_energy(lines, before=cutoff),
        hessian=hessian,
        masses=masses if indices is None else masses[indices],
        hessian_indices=indices,
        cell=read_cell(lines),
        pbc=np.ones(3, dtype=bool),
        program="vasp",
        version=read_version(lines),
        source=path,
    )

"""Reader for ORCA output and Hessian files.

Verified against ORCA 5.0.0. ORCA splits what JEDI needs across two files: the ``.out`` holds
the optimised geometry and the energy, while the ``.hess`` holds the Hessian -- and, in its
``$atoms`` block, the geometry and the masses as well. So a frequency job's ``.hess`` alone is
enough for geometry plus Hessian, and :func:`read_output` handles both file kinds.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from strainjedi.constants import BOHR_ANG
from strainjedi.io import scan
from strainjedi.io.elements import symbol_to_number
from strainjedi.io.report import warn_imaginary_frequencies
from strainjedi.io.types import MissingBlock, ParseError, QCOutput

MAGIC = (b"* O   R   C   A *", b"$orca_hessian_file")
"""The second entry identifies a ``.hess``, which carries no ORCA banner of its own."""

ANCHORS = {
    "version": [re.compile(r"Program Version\s+(\d+)\.(\d+)\.(\d+)")],
    "geometry": ["CARTESIAN COORDINATES (ANGSTROEM)"],
    "energy": ["FINAL SINGLE POINT ENERGY"],
    "hessian": [re.compile(r"^\s*\$hessian\b")],
    "atoms": [re.compile(r"^\s*\$atoms\b")],
    "frequencies": [re.compile(r"^\s*\$vibrational_frequencies\b")],
}

_HESS_FILE = re.compile(r"^\s*\$orca_hessian_file")
_NOT_CONVERGED = "Wavefunction not fully converged"

HESSIAN_HINT = "Point at the .hess file written by a frequency job (ORCA does not put the Hessian in the .out)."


def read_version(lines: list[str]) -> tuple[int, ...]:
    hits = scan.find_anchors(lines, ANCHORS["version"])
    if not hits:
        return ()
    match = ANCHORS["version"][0].search(lines[hits[0]])
    return tuple(int(g) for g in match.groups()) if match else ()


def read_energy(lines: list[str]) -> float | None:
    """Last converged ``FINAL SINGLE POINT ENERGY``, in Hartree.

    A geometry optimisation prints one per step, so the last is the converged one -- but ORCA
    also flags steps whose wavefunction did not converge on the same line, and those are
    skipped rather than trusted.
    """
    energy = None
    for i in scan.find_anchors(lines, ANCHORS["energy"]):
        if _NOT_CONVERGED in lines[i]:
            continue
        energy = scan.to_float(lines[i].split()[-1])
    return energy


def read_geometry(lines: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Last Cartesian geometry from a ``.out``, as (numbers, positions in Bohr)."""
    hits = scan.find_anchors(lines, ANCHORS["geometry"])
    if not hits:
        raise MissingBlock("geometry", hint="Expected a 'CARTESIAN COORDINATES (ANGSTROEM)' block.")

    numbers: list[int] = []
    positions: list[list[float]] = []

    for line in lines[hits[-1] + 1 :]:
        tokens = line.split()
        # The block opens with a rule of dashes and closes on a blank line.
        if not tokens or set(line.strip()) == {"-"}:
            if numbers:
                break
            continue
        if len(tokens) != 4 or not scan.is_float_row(tokens[1:]):
            break
        numbers.append(symbol_to_number(tokens[0]))
        positions.append([scan.to_float(t) for t in tokens[1:]])

    if not numbers:
        raise ParseError("found a geometry header but no atom rows under it")

    return np.array(numbers), np.array(positions) / BOHR_ANG


def read_frequencies(lines: list[str]) -> np.ndarray | None:
    """Harmonic frequencies in cm^-1 from a ``.hess``, or None if the block is absent.

    Translations and rotations are printed as exact zeros and imaginary modes as negatives.
    """
    hits = scan.find_anchors(lines, ANCHORS["frequencies"])
    if not hits:
        return None

    start = hits[0]
    count = int(lines[start + 1].split()[0])
    return np.array([scan.to_float(lines[start + 2 + i].split()[1]) for i in range(count)])


def count_imaginary(lines: list[str]) -> int | None:
    """How many imaginary frequencies the ``.hess`` reports, or None if it lists none at all."""
    frequencies = read_frequencies(lines)
    return None if frequencies is None else int((frequencies < 0).sum())


def read_atoms_block(lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``$atoms`` from a ``.hess``, as (numbers, positions in Bohr, masses in amu).

    Unlike the ``.out`` geometry, these coordinates are already in Bohr.
    """
    hits = scan.find_anchors(lines, ANCHORS["atoms"])
    if not hits:
        raise MissingBlock("$atoms block", hint="Expected it in the .hess file.")

    start = hits[0]
    count = int(lines[start + 1].split()[0])

    numbers: list[int] = []
    masses: list[float] = []
    positions: list[list[float]] = []

    for line in lines[start + 2 :]:
        tokens = line.split()
        if len(tokens) != 5 or not scan.is_float_row(tokens[1:]):
            break
        numbers.append(symbol_to_number(tokens[0]))
        masses.append(scan.to_float(tokens[1]))
        positions.append([scan.to_float(t) for t in tokens[2:]])
        if len(numbers) == count:
            break

    if len(numbers) != count:
        raise ParseError(f"$atoms declares {count} atoms but {len(numbers)} rows parsed")

    return np.array(numbers), np.array(positions), np.array(masses)


def read_hessian(lines: list[str]) -> np.ndarray:
    """Cartesian Hessian from a ``.hess``, in Hartree/Bohr^2.

    ORCA prints the block with both a column-index header and a row index, so the number of
    columns per chunk is read from the file rather than assumed -- 5 and 6 parse identically.
    The printed matrix is very slightly asymmetric (~4e-5), so it is symmetrised.
    """
    hits = scan.find_anchors(lines, ANCHORS["hessian"])
    if not hits:
        raise MissingBlock("Hessian", hint=HESSIAN_HINT)

    start = hits[0]
    n = int(lines[start + 1].split()[0])
    matrix = scan.parse_indexed_block(lines, start + 1, one_based=False, n=n, stop_prefix="$")
    return scan.symmetrize(matrix)


def read_output(path: Path | str) -> QCOutput:
    """Parse an ORCA ``.out`` or ``.hess`` into a :class:`QCOutput`."""
    path = Path(path)
    lines = scan.read_lines(path)
    is_hess = any(_HESS_FILE.match(line) for line in lines[:5])

    if is_hess:
        numbers, positions, masses = read_atoms_block(lines)
        warn_imaginary_frequencies(count_imaginary(lines), path)
        return QCOutput(
            numbers=numbers,
            positions=positions,
            masses=masses,
            hessian=read_hessian(lines),
            program="orca",
            source=path,
        )

    numbers, positions = read_geometry(lines)
    return QCOutput(
        numbers=numbers,
        positions=positions,
        energy=read_energy(lines),
        program="orca",
        version=read_version(lines),
        source=path,
    )

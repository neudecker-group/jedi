"""Reader for Gaussian output.

Verified against Gaussian 16 RevC.01. The Hessian can come from either of two files and the
formatted checkpoint is much the better source: ``.fchk`` states a value count and then prints
a packed lower triangle with no headers, row indices or block structure to drift between
revisions, whereas the ``.log`` block needs ``iop(7/33=1)`` and carries all three. So
:func:`read_hessian` prefers a sibling ``.fchk`` when one exists.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from strainjedi.constants import BOHR_ANG
from strainjedi.io import scan
from strainjedi.io.types import MissingBlock, ParseError, QCOutput

MAGIC = (b"Entering Gaussian System", b"Gaussian, Inc.")

ANCHORS = {
    "version": [re.compile(r"Gaussian\s+(\d+):")],
    # Standard orientation is what the force constants are expressed in; input orientation is
    # the fallback for jobs run with nosymm.
    "geometry": ["Standard orientation:", "Input orientation:", "Z-Matrix orientation:"],
    "energy": ["SCF Done:", "Energy=", "E(CORR)=", "EUMP2 ="],
    "hessian": ["Force constants in Cartesian coordinates:"],
    "masses": [re.compile(r"^\s*Atom\s+\d+\s+has atomic number\s+\d+\s+and mass\s")],
}

FCHK_FIELDS = {
    "numbers": "Atomic numbers",
    "positions": "Current cartesian coordinates",
    "masses": "Real atomic weights",
    "energy": "Total Energy",
    "hessian": "Cartesian Force Constants",
}

_TRANSLATION_VECTOR = -2
"""Gaussian encodes lattice vectors as pseudo-atoms with this atomic number."""

HESSIAN_HINT = "Add iop(7/33=1) to the route section, or run formchk to get a .fchk."


def read_version(lines: list[str]) -> tuple[int, ...]:
    hits = scan.find_anchors(lines, ANCHORS["version"])
    if not hits:
        return ()
    match = ANCHORS["version"][0].search(lines[hits[0]])
    return (int(match.group(1)),) if match else ()


def read_energy(lines: list[str]) -> float | None:
    """Last reported total energy, in Hartree.

    Every anchor spelling puts the number immediately after an ``=``, so one rule covers SCF,
    MP2 and coupled-cluster jobs alike.
    """
    energy = None
    for i in scan.find_anchors(lines, ANCHORS["energy"]):
        _, _, tail = lines[i].partition("=")
        tokens = tail.split()
        if tokens and scan.is_float_row(tokens[:1]):
            energy = scan.to_float(tokens[0])
    return energy


def read_geometry(lines: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Last geometry from a ``.log``, as (numbers, positions in Bohr, cell in Bohr, pbc).

    Gaussian writes lattice vectors as extra rows carrying atomic number -2, which are split
    off into the cell here rather than being mistaken for atoms.
    """
    hits = scan.find_anchors(lines, ANCHORS["geometry"])
    if not hits:
        raise MissingBlock("geometry", hint="Expected a 'Standard orientation:' block.")

    numbers: list[int] = []
    positions: list[list[float]] = []
    cell = np.zeros((3, 3))
    pbc = np.zeros(3, dtype=bool)
    npbc = 0

    for line in lines[hits[-1] + 1 :]:
        tokens = line.split()
        # Header rows and the rules of dashes that bracket them are skipped until data starts.
        if len(tokens) not in (5, 6) or not scan.is_float_row(tokens[-3:]) or not scan.is_int_row(tokens[:2]):
            if numbers:
                break
            continue

        number = int(tokens[1])
        position = [scan.to_float(t) for t in tokens[-3:]]

        if number == _TRANSLATION_VECTOR:
            if npbc < 3:
                cell[npbc] = position
                pbc[npbc] = True
                npbc += 1
        else:
            numbers.append(max(number, 0))
            positions.append(position)

    if not numbers:
        raise ParseError("found a geometry header but no atom rows under it")

    return np.array(numbers), np.array(positions) / BOHR_ANG, cell / BOHR_ANG, pbc


def read_masses(lines: list[str], natoms: int) -> np.ndarray | None:
    """Masses in amu from the thermochemistry listing, or None if absent.

    Gaussian's Cartesian force constants are not mass-weighted, so these are not needed to
    read the Hessian -- but they are the masses the printed frequencies were computed with,
    so carrying them makes those frequencies reproducible.
    """
    hits = scan.find_anchors(lines, ANCHORS["masses"])
    if len(hits) < natoms:
        return None
    return np.array([scan.to_float(lines[i].split()[-1]) for i in hits[:natoms]])


def read_hessian_log(lines: list[str]) -> np.ndarray:
    """Cartesian Hessian from a ``.log``, in Hartree/Bohr^2.

    The block carries a column-index header and a row index, so block width is read from the
    file rather than assumed. Only the lower triangle is printed.
    """
    hits = scan.find_anchors(lines, ANCHORS["hessian"])
    if not hits:
        raise MissingBlock("Hessian", hint=HESSIAN_HINT)

    return scan.mirror_lower(scan.parse_indexed_block(lines, hits[-1], one_based=True))


def _fchk_field(lines: list[str], label: str) -> tuple[int, int | None]:
    """Locate a formatted-checkpoint field, returning (line index, declared count).

    The count is None for scalar fields, whose value sits on the label line itself.
    """
    for i, line in enumerate(lines):
        if line.startswith(label):
            _, _, tail = line.partition("N=")
            return i, int(tail) if tail.strip() else None
    raise MissingBlock(f"'{label}' field", hint="Regenerate the .fchk with formchk.")


def read_fchk(path: Path | str) -> QCOutput:
    """Parse a Gaussian formatted checkpoint file.

    Coordinates in a ``.fchk`` are already in Bohr, and the Hessian is a packed lower triangle
    in Hartree/Bohr^2, so nothing here needs a unit conversion.
    """
    path = Path(path)
    lines = scan.read_lines(path)

    start, count = _fchk_field(lines, FCHK_FIELDS["numbers"])
    numbers = scan.parse_packed(lines, start, count).astype(int)

    start, count = _fchk_field(lines, FCHK_FIELDS["positions"])
    positions = scan.parse_packed(lines, start, count).reshape(-1, 3)

    energy = None
    try:
        start, _ = _fchk_field(lines, FCHK_FIELDS["energy"])
    except MissingBlock:
        pass
    else:
        energy = scan.to_float(lines[start].split()[-1])

    masses = None
    try:
        start, count = _fchk_field(lines, FCHK_FIELDS["masses"])
    except MissingBlock:
        pass
    else:
        masses = scan.parse_packed(lines, start, count)

    hessian = None
    try:
        start, count = _fchk_field(lines, FCHK_FIELDS["hessian"])
    except MissingBlock:
        pass
    else:
        hessian = scan.unpack_lower_triangle(scan.parse_packed(lines, start, count), 3 * len(numbers))

    return QCOutput(
        numbers=numbers,
        positions=positions,
        energy=energy,
        hessian=hessian,
        masses=masses,
        program="gaussian",
        source=path,
    )


def read_output(path: Path | str) -> QCOutput:
    """Parse a Gaussian ``.log`` or ``.fchk`` into a :class:`QCOutput`.

    For a ``.log``, a sibling ``.fchk`` is used for the Hessian when one exists, because it is
    the more version-stable source. The ``.log`` remains the source of geometry and energy.
    """
    path = Path(path)
    if path.suffix.lower() in (".fchk", ".fch"):
        return read_fchk(path)

    lines = scan.read_lines(path)
    numbers, positions, cell, pbc = read_geometry(lines)

    hessian = None
    fchk = next((path.with_suffix(s) for s in (".fchk", ".fch") if path.with_suffix(s).exists()), None)
    if fchk is not None:
        hessian = read_fchk(fchk).hessian
    if hessian is None and scan.find_anchors(lines, ANCHORS["hessian"]):
        hessian = read_hessian_log(lines)

    return QCOutput(
        numbers=numbers,
        positions=positions,
        energy=read_energy(lines),
        hessian=hessian,
        masses=read_masses(lines, len(numbers)),
        cell=cell,
        pbc=pbc,
        program="gaussian",
        version=read_version(lines),
        source=path,
    )

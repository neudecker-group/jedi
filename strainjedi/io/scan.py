"""Primitives for scraping blocked numeric data out of quantum-chemistry output files.

These files are semi-structured logs, not a language. Vendors change decoration -- banners,
blank lines, block widths -- freely between releases, but almost never the numeric payload.
Every helper here therefore locates data by *anchor plus shape*, never by a computed line
offset: a block that gains a header line, or goes from five columns to six, must keep parsing.

No ASE import, ever. See :mod:`strainjedi.io.readers`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from strainjedi.io.types import ParseError

Pattern = str | re.Pattern
"""An anchor. A plain string is matched as a substring, which keeps anchor tables readable."""

_INT = re.compile(r"[+-]?\d+\Z")
_FLOAT = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[DdEe][+-]?\d+)?\Z")

_MAX_BLANK = 2
"""How many consecutive blank lines a block may contain before we call it finished."""


def read_lines(path: Path | str) -> list[str]:
    """Read a file as a list of lines with newlines stripped.

    Note the newlines are *gone*: helpers here must therefore treat a blank line as ``""``
    and never as ``"\\n"``. Getting that backwards is what silently broke the previous
    Q-Chem parser.
    """
    return Path(path).read_text(errors="replace").splitlines()


def to_float(token: str) -> float:
    """Parse a float, accepting Fortran's ``D`` exponent (``0.843133D+00``)."""
    return float(token.replace("D", "E").replace("d", "e"))


def is_int_row(tokens: Sequence[str]) -> bool:
    """True if every token is a bare integer, i.e. this is a column-index header."""
    return bool(tokens) and all(_INT.match(t) for t in tokens)


def is_float_row(tokens: Sequence[str]) -> bool:
    """True if every token is a float, i.e. this is an unlabelled row of data."""
    return bool(tokens) and all(_FLOAT.match(t) for t in tokens)


def is_indexed_row(tokens: Sequence[str]) -> bool:
    """True if this is a row index followed by at least one value."""
    return len(tokens) > 1 and bool(_INT.match(tokens[0])) and is_float_row(tokens[1:])


def _matches(line: str, pattern: Pattern) -> bool:
    if isinstance(pattern, str):
        return pattern in line
    return pattern.search(line) is not None


def find_anchors(lines: Sequence[str], patterns: Sequence[Pattern]) -> list[int]:
    """Indices of every line matching an anchor, in file order.

    ``patterns`` is priority-ordered and the first pattern with any hit wins; later patterns
    are then ignored. That is what lets a reader list a renamed ORCA 6 block header ahead of
    the ORCA 5 spelling without ever branching on a version number -- and what lets an
    unrecognised future version still parse, as long as one known spelling survives.

    Returns an empty list if nothing matched; callers raise
    :class:`~strainjedi.io.types.MissingBlock` with a hint appropriate to the block.
    """
    for pattern in patterns:
        hits = [i for i, line in enumerate(lines) if _matches(line, pattern)]
        if hits:
            return hits
    return []


def parse_indexed_block(
    lines: Sequence[str],
    start: int,
    *,
    one_based: bool,
    n: int | None = None,
    stop_prefix: str | None = None,
) -> np.ndarray:
    """Parse a self-describing blocked matrix into a dense array.

    Handles the shape ORCA ``.hess`` and Gaussian ``.log`` share: repeating groups of one
    column-index header line followed by rows of ``<row index> <value>...``. Because both the
    row index and the column indices are read from the file, the number of columns per block
    is irrelevant -- 5 and 6 parse identically -- and so is any change in how the block is
    indented or separated.

    Lower-triangular blocks (Gaussian) are handled by the same code: a short row simply
    supplies fewer columns. Mirroring is *not* applied here; see :func:`mirror_lower`.

    Args:
        lines: The whole file.
        start: Index of the anchor line. Parsing begins on the line after it.
        one_based: Whether the program numbers rows and columns from 1 (Gaussian) or 0 (ORCA).
        n: Expected dimension, when the file states it. Inferred from the largest index seen
            if omitted.
        stop_prefix: Stop when a stripped line starts with this (ORCA's ``$``).

    Returns:
        An (n, n) array. Positions never written to stay zero.
    """
    entries: list[tuple[int, list[int], list[float]]] = []
    cols: list[int] | None = None
    blanks = 0

    for line in lines[start + 1 :]:
        stripped = line.strip()

        if not stripped:
            blanks += 1
            if blanks > _MAX_BLANK:
                break
            continue

        if stop_prefix is not None and stripped.startswith(stop_prefix):
            break

        tokens = stripped.split()

        if is_int_row(tokens):
            cols = [int(t) for t in tokens]
            blanks = 0
            continue

        if cols is not None and is_indexed_row(tokens):
            values = [to_float(t) for t in tokens[1:]]
            entries.append((int(tokens[0]), cols[: len(values)], values))
            blanks = 0
            continue

        break

    if not entries:
        raise ParseError(f"found the block header at line {start + 1} but no data rows under it")

    offset = 1 if one_based else 0
    if n is None:
        n = max(max(row for row, _, _ in entries), max(max(cs) for _, cs, _ in entries)) + 1 - offset

    matrix = np.zeros((n, n))
    for row, block_cols, values in entries:
        r = row - offset
        c = [col - offset for col in block_cols]
        if not (0 <= r < n) or not all(0 <= col < n for col in c):
            raise ParseError(f"index out of range for a {n}x{n} matrix: row {row}, columns {block_cols}")
        matrix[r, c] = values

    return matrix


def parse_bare_block(lines: Sequence[str], start: int, nrows: int) -> np.ndarray:
    """Parse a blocked matrix printed as bare numbers, with no row or column labels.

    This is Q-Chem's mass-weighted Hessian: column-chunks of a fixed width stacked
    vertically, separated by blank lines, with nothing in the file saying how wide a chunk is
    or where the matrix ends. So ``nrows`` has to come from the geometry -- that coupling is
    real and is why it is an argument rather than something inferred.

    The chunk width *is* read from the data, and reading stops as soon as the expected number
    of rows has been collected. That matters: Q-Chem prints the projected mass-weighted
    Hessian shortly afterwards in exactly the same format, and a parser that ran until it hit
    a non-numeric line would silently swallow part of it.
    """
    rows: list[list[float]] = []
    expected: int | None = None
    blanks = 0

    for line in lines[start + 1 :]:
        stripped = line.strip()

        if not stripped:
            blanks += 1
            if blanks > _MAX_BLANK and rows:
                break
            continue

        tokens = stripped.split()
        if not is_float_row(tokens):
            break

        rows.append([to_float(t) for t in tokens])
        blanks = 0

        if expected is None:
            width = len(rows[0])
            expected = nrows * math.ceil(nrows / width)
        if len(rows) == expected:
            break

    if expected is None:
        raise ParseError(f"found the block header at line {start + 1} but no data rows under it")
    if len(rows) != expected:
        raise ParseError(f"expected {expected} rows for a {nrows}x{nrows} matrix, collected {len(rows)}")

    chunks = [np.array(rows[k : k + nrows]) for k in range(0, expected, nrows)]
    matrix = np.hstack(chunks)

    if matrix.shape != (nrows, nrows):
        raise ParseError(f"assembled a {matrix.shape} matrix, expected {(nrows, nrows)}")

    return matrix


def parse_packed(lines: Sequence[str], start: int, count: int) -> np.ndarray:
    """Read ``count`` floats laid out free-form across the lines after ``start``.

    This is the Gaussian formatted-checkpoint layout: a declared count, then that many values
    with no indices, headers or block structure to drift between revisions.
    """
    values: list[float] = []

    for line in lines[start + 1 :]:
        tokens = line.split()
        if not tokens or not is_float_row(tokens):
            break
        values.extend(to_float(t) for t in tokens)
        if len(values) >= count:
            break

    if len(values) < count:
        raise ParseError(f"expected {count} values after line {start + 1}, found {len(values)}")

    return np.array(values[:count])


def unpack_lower_triangle(flat: np.ndarray, n: int) -> np.ndarray:
    """Expand a row-major packed lower triangle into a full symmetric matrix."""
    expected = n * (n + 1) // 2
    if flat.size != expected:
        raise ParseError(f"a packed {n}x{n} lower triangle needs {expected} values, got {flat.size}")

    matrix = np.zeros((n, n))
    matrix[np.tril_indices(n)] = flat
    return mirror_lower(matrix)


def mirror_lower(matrix: np.ndarray) -> np.ndarray:
    """Reflect a lower triangle into the upper one, leaving the diagonal alone."""
    return matrix + matrix.T - np.diag(np.diag(matrix))


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Average a matrix with its transpose.

    Programs that print a full Hessian print it slightly asymmetrically -- ORCA's ``.hess``
    is off by ~4e-5 -- and the downstream analysis assumes symmetry.
    """
    return 0.5 * (matrix + matrix.T)

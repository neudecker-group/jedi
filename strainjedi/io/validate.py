"""Physics checks on a parsed calculation, kept out of the parsers.

Reading a file and judging whether its contents are fit for a JEDI analysis are different
jobs, so this is where the judging lives. The readers do warn about imaginary frequencies as
they parse (see :mod:`strainjedi.io.report`); this module is for callers who want the number
rather than the warning.

No ASE import; the per-program knowledge lives in the reader anchor tables, so this module
stays program-agnostic.
"""

from __future__ import annotations

from pathlib import Path

from strainjedi.io import scan
from strainjedi.io.registry import reader_for


def imaginary_frequencies(path: Path | str, *, program: str | None = None) -> int | None:
    """How many imaginary frequencies the program itself reported.

    Args:
        path: Output file from a frequency calculation.
        program: Override detection. Rarely needed.

    Returns:
        The count, or None if the file contains no vibrational analysis to judge. Zero and
        None mean different things: zero is a clean minimum, None is "this file cannot say".
    """
    reader = reader_for(path, program)
    return reader.count_imaginary(scan.read_lines(path))

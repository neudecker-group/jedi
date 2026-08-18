"""Which reader handles which file.

Detection is by magic bytes first and file extension only as a fallback, because extensions
lie: ``.out`` is used by ORCA and Q-Chem alike.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from strainjedi.io.readers import gaussian, orca, qchem
from strainjedi.io.types import ProgramNotDetected

READERS: dict[str, ModuleType] = {
    "orca": orca,
    "gaussian": gaussian,
    "qchem": qchem,
}

SUFFIXES: dict[str, str] = {
    ".hess": "orca",
    ".fchk": "gaussian",
    ".fch": "gaussian",
    ".log": "gaussian",
}
"""Only unambiguous extensions. ``.out`` is deliberately absent."""

PROBE_BYTES = 65536
"""How much of the head to search. Every supported program prints its banner well inside this,
and a formatted checkpoint has no banner at all, so reading more would not help."""


def detect_program(path: Path | str) -> str:
    """Name of the program that produced a file.

    Raises:
        ProgramNotDetected: If neither the magic bytes nor the extension identify it.
    """
    path = Path(path)
    head = path.read_bytes()[:PROBE_BYTES]

    for name, module in READERS.items():
        if any(magic in head for magic in module.MAGIC):
            return name

    suffix = SUFFIXES.get(path.suffix.lower())
    if suffix is not None:
        return suffix

    raise ProgramNotDetected(
        f"Could not tell which program wrote '{path}'. "
        f"Supported: {', '.join(sorted(READERS))}. Pass program=... to say explicitly."
    )


def reader_for(path: Path | str, program: str | None = None) -> ModuleType:
    """The reader module for a file, detecting the program when not told."""
    name = program.lower() if program is not None else detect_program(path)
    try:
        return READERS[name]
    except KeyError:
        raise ProgramNotDetected(f"No reader for {name!r}. Supported: {', '.join(sorted(READERS))}.") from None

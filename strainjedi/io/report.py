"""Warnings the readers raise about the physics of what they just read.

Separate from :mod:`strainjedi.io.types` because these are not errors: the file parsed fine
and the numbers are exactly what the program produced. What is wrong is the *calculation*,
and only the person running it can decide what to do about that.

No ASE import; the readers depend on this.
"""

from __future__ import annotations

import sys
from pathlib import Path

WIDTH = 78

IMAGINARY_FREQUENCY_WARNING = """\
{n} imaginary {frequency} reported in
  {source}

This structure is a saddle point, not a minimum. A JEDI analysis expands the energy
harmonically about a relaxed structure, so strain energies computed from this Hessian
do not mean what they appear to mean. Re-optimise to a true minimum first.\
"""


def banner(title: str, body: str, stream=None) -> None:
    """Print a boxed message, hard to miss in a wall of calculation output."""
    stream = stream if stream is not None else sys.stderr

    print("=" * WIDTH, file=stream)
    print(f"  {title}", file=stream)
    print("-" * WIDTH, file=stream)
    for line in body.splitlines():
        print(f"  {line}", file=stream)
    print("=" * WIDTH, file=stream)


def warn_imaginary_frequencies(count: int | None, source: Path | str, stream=None) -> None:
    """Warn loudly if a frequency calculation did not land on a minimum.

    Deliberately a warning rather than an exception. The parse succeeded, and whether a
    saddle point is a mistake or the point of the exercise is not the parser's call.

    Args:
        count: How many imaginary frequencies the program reported. ``None`` means the file
            had no vibrational analysis to judge, and ``0`` means a clean minimum; neither
            warns.
        source: Path to name in the message.
        stream: Where to write. Defaults to stderr.
    """
    if not count:
        return

    banner(
        f"WARNING: imaginary frequencies in '{Path(source).name}'",
        IMAGINARY_FREQUENCY_WARNING.format(
            n=count,
            frequency="frequency" if count == 1 else "frequencies",
            source=source,
        ),
        stream=stream,
    )

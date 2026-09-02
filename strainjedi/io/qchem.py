"""Deprecated Q-Chem helpers, kept so existing scripts keep working.

Superseded by :mod:`strainjedi.io` and :mod:`strainjedi.io.adapter`::

    from strainjedi.io import read_output
    from strainjedi.io.adapter import to_atoms, to_vibrations

    mol     = to_atoms(read_output("opt.out"))
    hessian = to_vibrations(read_output("freq.out"), mol)

The Q-Chem calculator class moved to :mod:`strainjedi.calculators.qchem`.

Reading a Q-Chem Hessian previously never worked -- the block was searched for without
``re.MULTILINE``, so it was never found, and the chunk assembly behind it was broken too. It
works now, which means this shim returns results where it used to call ``sys.exit(1)``.
"""

from __future__ import annotations

import warnings

from ase.atoms import Atoms
from ase.vibrations.data import VibrationsData

from strainjedi.io import read_output
from strainjedi.io.adapter import to_atoms, to_vibrations


def _deprecated(old: str, new: str) -> None:
    warnings.warn(f"strainjedi.io.qchem.{old} is deprecated; use {new}.", DeprecationWarning, stacklevel=3)


def read(filename: str) -> Atoms:
    """Deprecated. Use ``to_atoms(read_output(filename))``."""
    _deprecated("read", "strainjedi.io.adapter.to_atoms(strainjedi.io.read_output(...))")
    return to_atoms(read_output(filename))


def get_vibrations(label: str, atoms: Atoms | None = None, indices=None) -> VibrationsData:
    """Deprecated. Use ``to_vibrations(read_output(label + '.out'), atoms)``."""
    _deprecated("get_vibrations", "strainjedi.io.adapter.to_vibrations(strainjedi.io.read_output(...))")
    return to_vibrations(read_output(f"{label}.out"), atoms, indices)

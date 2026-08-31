"""Deprecated Gaussian helpers, kept so existing scripts keep working.

Superseded by :mod:`strainjedi.io` and :mod:`strainjedi.io.adapter`::

    from strainjedi.io import read_output
    from strainjedi.io.adapter import to_atoms, to_vibrations

    mol     = to_atoms(read_output("opt.log"))
    hessian = to_vibrations(read_output("freq.log"), mol)

The Gaussian calculator class and ``write_gaussian_in`` moved to
:mod:`strainjedi.calculators.gaussian`.

The new reader prefers a sibling ``.fchk`` for the Hessian when one exists, since its packed
lower triangle has no headers or block structure to change between revisions.
"""

from __future__ import annotations

import warnings

from ase.atoms import Atoms
from ase.vibrations.data import VibrationsData

from strainjedi.io import read_output
from strainjedi.io.adapter import to_atoms, to_vibrations


def _deprecated(old: str, new: str) -> None:
    warnings.warn(f"strainjedi.io.gaussian.{old} is deprecated; use {new}.", DeprecationWarning, stacklevel=3)


def read_gaussian_out(label: str, index: int = -1) -> Atoms:
    """Deprecated. Use ``to_atoms(read_output(label + '.log'))``.

    Only the final configuration is returned; ``index`` is accepted and ignored.
    """
    _deprecated("read_gaussian_out", "strainjedi.io.adapter.to_atoms(strainjedi.io.read_output(...))")
    return to_atoms(read_output(f"{label}.log"))


def get_vibrations(label: str, atoms: Atoms | None = None, indices=None) -> VibrationsData:
    """Deprecated. Use ``to_vibrations(read_output(label + '.log'), atoms)``."""
    _deprecated("get_vibrations", "strainjedi.io.adapter.to_vibrations(strainjedi.io.read_output(...))")
    return to_vibrations(read_output(f"{label}.log"), atoms, indices)

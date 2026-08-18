"""Chemical symbol to atomic number mapping.

A local copy rather than ``ase.data.chemical_symbols`` so that the readers stay free of ASE;
see :mod:`strainjedi.io.readers`. The list is stable physics and needs no maintenance.
"""

from __future__ import annotations

from strainjedi.io.types import ParseError

# fmt: off
SYMBOLS: tuple[str, ...] = (
    "X",
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)
# fmt: on
"""Index is the atomic number; index 0 is the dummy/ghost placeholder."""

_NUMBERS = {symbol.lower(): number for number, symbol in enumerate(SYMBOLS)}


def symbol_to_number(symbol: str) -> int:
    """Atomic number for a chemical symbol, case-insensitively.

    Trailing decoration some programs attach to ghost atoms (``C:``) is stripped.
    """
    key = symbol.strip().rstrip(":").lower()
    try:
        return _NUMBERS[key]
    except KeyError:
        raise ParseError(f"unknown chemical symbol {symbol!r}") from None

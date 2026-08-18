"""The parser layer must not depend on ASE.

This is the guard rail behind the goal of a JEDI core that works without ASE: the readers
return plain NumPy in atomic units, and :mod:`strainjedi.io.adapter` is the single module
allowed to know ASE exists. Without a test, that boundary erodes the first time someone wants
``ase.data.chemical_symbols``.

The check is static rather than a runtime ``sys.modules`` probe because ``strainjedi/__init__``
imports ``Jedi`` eagerly, so importing *any* submodule pulls in ASE through the parent package.
That masks a runtime probe but says nothing about what the parser layer itself needs -- which
is what actually decides whether it can be lifted out.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parent.parent / "strainjedi"

ASE_FREE_MODULES = [
    "constants.py",
    "io/__init__.py",
    "io/elements.py",
    "io/registry.py",
    "io/scan.py",
    "io/types.py",
    "io/readers/__init__.py",
    "io/readers/gaussian.py",
    "io/readers/orca.py",
    "io/readers/qchem.py",
]

ALLOWED_STRAINJEDI_IMPORTS = {
    "strainjedi",
    "strainjedi.constants",
    "strainjedi.io",
    "strainjedi.io.elements",
    "strainjedi.io.readers",
    "strainjedi.io.registry",
    "strainjedi.io.scan",
    "strainjedi.io.types",
}
"""The ASE-free subset. Notably excludes strainjedi.io.adapter, which is ASE by design."""


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)

    return names


@pytest.mark.parametrize("relative", ASE_FREE_MODULES)
def test_module_does_not_import_ase(relative):
    offenders = {name for name in _imported_modules(PACKAGE / relative) if name == "ase" or name.startswith("ase.")}

    assert not offenders, (
        f"{relative} imports {sorted(offenders)}. The parser layer must stay ASE-free; "
        f"put the conversion in strainjedi/io/adapter.py instead."
    )


@pytest.mark.parametrize("relative", ASE_FREE_MODULES)
def test_module_only_imports_ase_free_strainjedi_modules(relative):
    """An indirect ASE import counts too, e.g. reaching for the adapter from a reader."""
    offenders = {
        name
        for name in _imported_modules(PACKAGE / relative)
        if name.startswith("strainjedi") and name not in ALLOWED_STRAINJEDI_IMPORTS
    }

    assert not offenders, f"{relative} imports {sorted(offenders)}, which are outside the ASE-free subset."


def test_adapter_is_the_one_module_that_imports_ase():
    """Stated as a test so the exception stays deliberate rather than becoming a habit."""
    assert any(name == "ase" or name.startswith("ase.") for name in _imported_modules(PACKAGE / "io/adapter.py"))

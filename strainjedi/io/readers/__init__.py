"""One module per quantum-chemistry program, each turning an output file into a QCOutput.

**Nothing in this package may import ASE.** The parsers return plain NumPy arrays in atomic
units; :mod:`strainjedi.io.adapter` is the only place that knows ASE exists. There is a test
(``tests/test_io_ase_free.py``) that imports each reader in a fresh interpreter and asserts
``"ase" not in sys.modules``, so this is enforced rather than merely intended.

Each module exposes the same small surface, which is what :mod:`strainjedi.io.registry`
dispatches on:

``MAGIC``
    Byte strings identifying the program from the head of a file.
``ANCHORS``
    Block name -> priority-ordered list of patterns. Adding support for a new release of a
    program should mean adding a pattern here, not writing a subclass.
``read_output(path)``
    Parse whatever that one file contains into a
    :class:`~strainjedi.io.types.QCOutput`.
"""

from strainjedi.io.readers import gaussian, orca, qchem

__all__ = ["gaussian", "orca", "qchem"]

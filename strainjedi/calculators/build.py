"""Helper functions to construct ASE calculators for supported programs."""

from __future__ import annotations

import re
import shutil

from ase.calculators.orca import ORCA, OrcaProfile

_CHARGE_MULT = re.compile(r"^\*\s*xyz(?:file)?\s+(-?\d+)\s+(\d+)", re.IGNORECASE)
"""ORCA accepts '*xyz 0 1' and '* xyz 0 1' alike, and ASE's own writer emits the first."""

_TOP_LEVEL = ("%", "!", "*")
"""What can begin a top-level item, and therefore what ends the preceding ``%`` block.

Finding the end of a ``%`` block by looking for its ``end`` is harder than it appears: blocks
nest, so ``%geom / POTENTIALS / ... / end / end`` closes twice, and single-line blocks such as
``%maxcore 4000`` never close at all. Both shapes appear in real inputs. Since the block body
is copied out verbatim -- and is therefore balanced however it was written -- it is enough to
know where the block *stops*, and that is simply the next top-level line.
"""


def orca_input_to_ase(inpfile: str) -> tuple[str, str, int, int]:
    """Convert an ORCA input file into ASE's orcasimpleinput/orcablocks form.

    Args:
        inpfile (str): ORCA input file.

    Notes:
        - Requires the "ENGRAD" keyword so the gradient is written and ASE can parse it.
        - ORCA 6 and newer need a one-line adjustment in ASE for its parser to work, cf.
          https://gitlab.com/ase/ase/-/issues/1513

    Returns:
        tuple[str, str, int, int]: simpleinput and orcablocks strings for ASE's ORCA
        calculator, plus charge and multiplicity from the input file.
    """
    with open(inpfile) as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    simple_lines: list[str] = []
    block_lines: list[str] = []
    in_block = False
    charge = None
    multiplicity = None

    for line in lines:
        if line.startswith(_TOP_LEVEL):
            in_block = False

        match = _CHARGE_MULT.match(line)
        if match:
            charge, multiplicity = int(match.group(1)), int(match.group(2))
            continue

        if line.startswith("%"):
            block_lines.append(line)
            in_block = True
            continue

        if in_block:
            # Body of a % block, including any nested sub-block, copied through verbatim.
            block_lines.append(line)
            continue

        if line.startswith("!"):
            simple_lines.append(line.lstrip("!").strip())

    if charge is None or multiplicity is None:
        raise ValueError(f"Charge and multiplicity not found in '{inpfile}' (expected a '*xyz charge mult' line).")

    return " ".join(simple_lines), "\n".join(block_lines), charge, multiplicity


def build_calc(inputfile: str | None = None, prog: str = "ORCA") -> ORCA:
    """Generate an ASE calculator from inputfile and program declaration.

    Args:
        inputfile (str | None, opt): Inputfile of QC program. None if not needed. Default: None.
        prog (str, opt): Program to initialize ASE calculator for. Default: ORCA.

    Returns:
        ORCA: A configured ASE calculator.
    """

    if prog.lower() != "orca":
        raise NotImplementedError(f"Cannot build calculator for {prog}.")

    ## Get ORCA executable. If None found, raise error
    orca_path = shutil.which("orca")

    if orca_path is None:
        raise RuntimeError("ORCA executable not found in PATH. Please load orca module or update PATH.")

    orca_profile = OrcaProfile(command=orca_path)

    # Init calculator and assign to mols
    sinp, blcks, chrg, mul = orca_input_to_ase(f"{inputfile}")

    print(f"Charge and Mult. from file: {chrg} {mul}")
    print(sinp)
    print(blcks)

    return ORCA(
        profile=orca_profile,
        charge=chrg,
        mult=mul,
        orcasimpleinput=sinp,
        orcablocks=blcks,
    )

"""Helper functions to construct ASE calculators for supported programs."""

import shutil

from ase.calculators.orca import ORCA, OrcaProfile
from strainjedi.IO.parser import orca_input_to_ase


def build_calc(inputfile: str | None = None, prog: str = "ORCA") -> None:
    """Generate an ASE calculator from inputfile and program declaration.

    Args:
        inputfile (str | None, opt): Inputfile of QC program. None if not needed. Default: None.
        prog (str, opt): Program to initialize ASE calculator for. Default: ORCA.

    Returns:
        ...
    """

    if prog.lower() == "orca":  # ORCA calculator
        ## Get ORCA executable. If None found, raise error
        orca_path = shutil.which("orca")

        if orca_path is None:
            raise RuntimeError(
                "ORCA executable not found in PATH. Please load orca module or update PATH."
            )

        orca_profile = OrcaProfile(command=orca_path)

        # Init calculator and assign to mols
        sinp, blcks, chrg, mul = orca_input_to_ase(f"{inputfile}")

        print(f"Charge and Mult. from file: {chrg} {mul}")
        print(sinp)
        print(blcks)

        calc = ORCA(
            profile=orca_profile,
            charge=chrg,
            mult=mul,
            orcasimpleinput=sinp,
            orcablocks=blcks,
        )

    else:
        raise NotImplementedError(f"Cannot build calculator for {prog}.")

    return calc

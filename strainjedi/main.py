"""Script for CL execution of Jedi strain analysis."""

from __future__ import annotations

import numpy as np
from ase import io
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Hartree, kcal, mol
from ase.vibrations import Vibrations
from ase.vibrations.data import VibrationsData

from strainjedi.calculators.build import build_calc
from strainjedi.cli import jedi_parser, read_energies
from strainjedi.constants import HESSIAN_AU_TO_ASE
from strainjedi.io import read_hessian
from strainjedi.jedi import Jedi


def main() -> None:
    # Parse command-line arguments
    args = jedi_parser()

    # Load structures as ASE objects
    ati = io.read(args.xyzi)
    atf = io.read(args.xyzf)

    nati, natf = len(ati), len(atf)

    # Check atom count in input files.
    if nati != natf:
        raise ValueError(f"Atom count mismatch: {nati} (initial) != {natf} (final)")

    # Parse init. and final state energy from file
    if args.energies:
        # Energies still in Hartree; convert to kcal/mol for JEDI analysis
        e_ = read_energies(args.energies)
        energies_kcal = e_ * Hartree * mol / kcal

        # Jedi reads the energies back off the structures via get_potential_energy(), so they
        # have to be attached to the Atoms rather than only passed as epot.
        ati.calc = SinglePointCalculator(ati, energy=e_[1] * Hartree)
        atf.calc = SinglePointCalculator(atf, energy=e_[2] * Hartree)

        print(f"Init. {args.xyzi} ({nati} atoms), Erel: 0.0 kcal/mol")
        print(f"Final {args.xyzf}: E: {energies_kcal[0]:.1f} kcal/mol")
    else:
        energies_kcal = None

    # Parse Hessian and initialize ASE VibDat object. read_hessian returns atomic units
    # (Hartree/Bohr^2); VibrationsData is defined in eV/Angstrom^2.
    if args.hessi:
        print(f"Reading Hessian from: {args.hessi}")
        # read_hessian warns by itself if the structure is a saddle point.
        h2d = read_hessian(args.hessi)

        h4d = VibrationsData.from_2d(ati, h2d * HESSIAN_AU_TO_ASE)
    else:
        h4d = None

    # Handle cases where either H or E was not provided via CL:
    # -> Use orca input file to build ASE calc. and follow conventional jedi usage as in docs.
    if (energies_kcal is None) or (h4d is None):
        print(
            f"Either Hessian or energies not provided via CL. "
            f"Using {args.oinp} to generate ASE calculator and determine internally."
        )

        if not args.oinp:  # Requires input file
            raise ValueError("Need a valid input file.")

        # Build calculator to get E and/or H
        calculator = build_calc(inputfile=args.oinp, prog="ORCA")

        # Set calc. for i and f
        print("Set calculator")
        ati.calc = calculator
        atf.calc = calculator

    # Handle Hessian
    if h4d is None:
        print(f"No Hessian found. Trying to compute this with {args.oinp} input via ASE calc. numerically.")

        vib = Vibrations(ati, name="jvibcalc")
        vib.run()

        # Set hessian tensor for use in jedi
        h4d = vib.get_vibrations()

    # Handle Energies
    if energies_kcal is None:
        print(f"No energies found. Trying to compute with {args.oinp}")

        e_i = ati.get_potential_energy()  # Energy in eV
        e_f = atf.get_potential_energy()

        de = e_f - e_i
        energies_kcal = np.array([x * mol / kcal for x in [de, e_i, e_f]])

        print(f"Init. {args.xyzi} ({nati} atoms), Erel: 0.0 kcal/mol")
        print(f"Final {args.xyzf}: E: {energies_kcal[0]:.1f} kcal/mol (from ASE)")

        # Freeze the computed energies onto the structures. Jedi calls get_potential_energy()
        # again in run(), and a live file-IO calculator is neither needed nor wanted there.
        ati.calc = SinglePointCalculator(ati, energy=e_i)
        atf.calc = SinglePointCalculator(atf, energy=e_f)

    jedi = Jedi(
        ati,
        atf,
        h4d,
        epot=energies_kcal,
    )
    jedi.run()


if __name__ == "__main__":
    main()

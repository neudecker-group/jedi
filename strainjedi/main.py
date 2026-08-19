"""Script for CL execution of Jedi strain analysis."""

from __future__ import annotations

from strainjedi.io.parser import jedi_parser, read_energies, parse_orca_hess
from strainjedi.io.calculator import build_calc

from ase import io
from ase.units import Hartree
from ase.vibrations import Vibrations
from ase.vibrations.data import VibrationsData

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
        # Energies still in Hartree; convert to eV for JEDI analysis
        e_ = read_energies(args.energies)
        energies_eV = e_ * Hartree
        de = energies_eV[1] - energies_eV[0]

        print(f"Init. {args.xyzi} ({nati} atoms), Erel: 0.0 eV")
        print(f"Final {args.xyzf}: E: {de:.1f} eV")
    else:
        energies_eV = None

    # Parse Hessian from ORCA output and initialize ASE VibDat object
    if args.hessi:
        print(f"Reading ORCA Hessian from: {args.hessi}")
        h2d = parse_orca_hess(args.hessi)

        h4d = VibrationsData(atoms=ati, hessian=h2d.reshape(nati, 3, nati, 3))
    else:
        h4d = None

    # Handle cases where either H or E was not provided via CL:
    # -> Use orca input file to build ASE calc. and follow conventional jedi usage as in docs.
    if (energies_eV is None) or (h4d is None):
        print(
            f"Either Hessian or energies not provided via CL. Using {args.oinp} to generate ASE calculator and determine internally."
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
    if energies_eV is None:
        print(f"No energies found. Trying to compute with {args.oinp}")

        e_i = ati.get_potential_energy()  # Energy in eV
        e_f = atf.get_potential_energy()

        de = e_f - e_i

        print(f"Init. {args.xyzi} ({nati} atoms), Erel: 0.0 eV")
        print(f"Final {args.xyzf}: E: {de:.1f} eV (from ASE)")

    # Delete calculators to avoid calling it in jedi.run
    ati.calc = None
    atf.calc = None

    jedi = Jedi(
        ati,
        atf,
        h4d,
        epot=energies_eV,
    )
    jedi.run()


if __name__ == "__main__":
    main()

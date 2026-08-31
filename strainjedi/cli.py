"""Argument parsing and plain-text input for the command-line jedi script."""

from __future__ import annotations

import argparse

import numpy as np
from ase.units import Hartree, kcal, mol


def read_energies(filename: str) -> np.ndarray:
    """Read two energies (initial and final) in Hartree from a plain text file.

    Args:
        filename (str): Input file with initial and final state energies (in Eh).

    Returns:
        ens (np.array): Vector with energy diff. (f - i), initial and final state energy (in Eh!).

    """

    with open(filename) as f:
        lines = [float(x.strip()) for x in f if x.strip()]

    if len(lines) != 2:
        raise ValueError(f"'{filename}' must contain initial, final energy in Eh.")

    e_i, e_f = lines[0], lines[1]
    de = e_f - e_i

    # Print a warning if initial and final state energy are interchanged.
    if e_i > e_f:
        print(f"Warning: E_i > E_f by {de * Hartree * mol / kcal:.2f} kcal/mol. You may want to change this...")

    return np.array([de, e_i, e_f])


def jedi_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JEDI strain analysis from quantum-chemistry outputs and XYZ structures.",
    )
    parser.add_argument(
        "--xyzi",
        required=True,
        help="XYZ file of the initial structure",
    )

    parser.add_argument(
        "--xyzf",
        required=True,
        help="XYZ file of the final structure",
    )

    parser.add_argument(
        "--hessi",
        help="Hessian file of the initial structure (ORCA .hess, Gaussian .log/.fchk or Q-Chem .out)",
    )

    parser.add_argument(
        "--energies",
        type=str,
        help="File containing initial, final energy (in Eh), one per line.",
    )

    parser.add_argument(
        "--oinp",
        type=str,
        help="Optional: ORCA input file to use for energy, num. Hessian calculation with ASE.",
    )

    args = parser.parse_args()

    return args

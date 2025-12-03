""" Argparser for command-line jedi script. """


from __future__ import annotations

from typing import Union

import argparse
import numpy as np
from ase.units import Hartree, kcal, mol


def read_energies(filename : str) -> np.array:
    """ Read two energies (initial and final) in Hartree from a plain text file.
    
    Args:
        filename (str): Input file with initial and final state energies (in Eh).
    
    Returns:
        ens (np.array): Vector with energy diff. (f - i), initial and final state energy (in Eh!). 
        
    """

    with open(filename) as f:
        lines = [float(x.strip()) for x in f.readlines() if x.strip()]

    if len(lines) != 2:
        raise ValueError(f"'{filename}' must contain initial, final energy in Eh.")
    
    e_i, e_f = lines[0], lines[1]
    de = e_f - e_i

    # Print a warning if initial and final state energy are interchanged.
    if e_i > e_f: 
        print(f"Warning: E_i > E_f by {de * Hartree * mol / kcal  :.2f} kcal/mol. You may want to change this...")
    
    return np.array([de, e_i, e_f])


def parse_orca_hess(filename : str) -> np.array:
    """ Parse ORCA .hess file.
    
    Args:
        filename (str): Input hessian file from ORCA.
    
    Returns:
        h (np.ndarray): Hessian matrix (dim: 3Nat x 3Nat).
    """

    with open(filename) as f:
        lines = f.readlines()

    try:
        start = next(i for i,l in enumerate(lines) if l.strip().lower() == "$hessian")
    except StopIteration:
        raise ValueError("No $hessian block found")

    # read line after $hessian: to get dimension of Hessian
    nrow = int(lines[start + 1].split()[0])
    H = np.zeros((nrow, nrow))

    col_indices = []
    i = start + 2

    # ---- parse block ----
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("$"):   # end of block
            break

        parts = line.split()

        # skip block label line
        if all(p.isdigit() for p in parts):
            col_indices = list(map(int, parts))
            i += 1
            continue

        # numeric row line
        if col_indices:
            row = int(parts[0])
            vals = np.array(parts[1:], float)
            H[row, col_indices[:len(vals)]] = vals

        i += 1

    return H

def orca_input_to_ase(inpfile : str) -> Union[str, str, int, int]: 
    """ Convert orca.inp content to ase orcasimpleinput and orcablocksblcks string. 
    
    Args:
        inpfile (str): ORCA input file. 

    Notes:
        - Requires "ENGRAD" keyword such that the gradient is printed and can be parsed with ASE.
        - ORCA version 6 and higher require a one-line adjustment in ASE for the parser to work, cf. https://gitlab.com/ase/ase/-/issues?sort=created_date&state=opened&search=orca&first_page_size=20&show=eyJpaWQiOiIxNTEzIiwiZnVsbF9wYXRoIjoiYXNlL2FzZSIsImlkIjoxNTAzODY4MTV9

    Returns:
        Union[str, str, int, int]: simpleinput, orcablock strings to be used with ASE's ORCA calculator, and charge, multiplicity from input file .  

    """

    with open(inpfile) as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith('#')]

    simple_lines = []
    block_lines = []
    block_active = False
    charge = None
    multiplicity = None

    for line in lines:

        # Detect *xyz charge mult
        if line.lower().startswith('* xyz'):
            parts = line.split()
            try:
                charge, multiplicity = map(int, parts[-2:])
            except ValueError:
                raise ValueError(f"Could not parse charge/multiplicity from line: {line}")

        # Handle blocks
        if line.startswith('%'):
            block_active = True

        if block_active:
            block_lines.append(line)

            if line.lower().endswith('end'):
                block_active = False

        else:
            if line.startswith('!'):
                simple_lines.append(line.lstrip('!').strip())

    # Safety check
    if charge is None or multiplicity is None:
        raise ValueError("Charge and multiplicity not found in the file (expected '*xyz charge mult').")

    return " ".join(simple_lines), "\n".join(block_lines), charge, multiplicity


def jedi_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JEDI strain analysis from ORCA outputs and XYZ structures.",
    )
    parser.add_argument("--xyzi", required=True,
    help="XYZ file of the initial structure",
    )

    parser.add_argument("--xyzf", required=True,
    help="XYZ file of the final structure",
    )

    parser.add_argument("--hessi",
    help="ORCA Hessian file of the initial structure",
    )

    parser.add_argument("--energies",
    type = str,
    help="File containing initial, final energy (in Eh), one per line.",
    )

    parser.add_argument("--oinp",
    type = str,
    help="Optional: ORCA input file to use for energy, num. Hessian calculation with ASE.",
    )

    args = parser.parse_args()

    return args

import numpy as np


def read_internal_strain(outcar_path: str) -> np.ndarray:
    """
    Reads 'INTERNAL STRAIN TENSORS FROM STRAINED CELLS' from a VASP OUTCAR.

    Args:
        outcar_path : str
            Path to OUTCAR

    Returns:
        internal_strain : np.ndarray
            Shape (N_ions * 3, 6)
            Rows ordered as:
            ion1-x, ion1-y, ion1-z, ion2-x, ...
            strain components: XX, YY, ZZ, YZ, XZ, XY
    """

    with open(outcar_path, "r") as f:
        lines = f.readlines()

    strain_rows = []
    i = 0
    nlines = len(lines)

    in_strained_cells_section = False

    while i < nlines:
        line = lines[i]

        # Enter correct OUTCAR section
        if "INTERNAL STRAIN TENSORS FROM STRAINED CELLS" in line:
            in_strained_cells_section = True
            i += 1
            continue

        # Leave section if next block starts
        if "INTERNAL STRAIN TENSORS FROM DISPLACED ATOMS" in line:
            in_strained_cells_section = False

        if in_strained_cells_section and "INTERNAL STRAIN TENSOR FOR ION" in line:
            # Skip header lines
            i += 3

            for _ in range(3):  # x, y, z
                parts = lines[i].split()
                values = [float(v) for v in parts[1:7]]
                strain_rows.append(values)
                i += 1

        else:
            i += 1

    internal_strain = np.array(strain_rows)
    vasp_to_voigt = [0, 1, 2, 4, 5, 3]  # different order in VASP than expected for voigt notation
    internal_strain_voigt = internal_strain[:, vasp_to_voigt]

    return internal_strain_voigt


def read_symmetrized_elastic_moduli(outcar_path):
    """
    Reads 'SYMMETRIZED ELASTIC MODULI' from a VASP OUTCAR.

    Args:
        outcar_path : str
            Path to OUTCAR

    Returns
        C_voigt : np.ndarray
            Shape (6, 6)
            Order: XX, YY, ZZ, YZ, ZX, XY
    """

    with open(outcar_path, "r") as f:
        lines = f.readlines()

    C = []
    i = 0
    nlines = len(lines)

    while i < nlines:
        line = lines[i]

        if "SYMMETRIZED ELASTIC MODULI" in line:
            # skip header lines
            i += 3

            # read 6 rows (XX, YY, ZZ, XY, YZ, ZX)
            for _ in range(6):
                parts = lines[i].split()
                # parts[0] = label (XX, YY, ...)
                values = [float(v) for v in parts[1:7]]
                C.append(values)
                i += 1

            break

        i += 1

    C = np.array(C) * 0.1  # kbar to GPa

    vasp_to_voigt = [0, 1, 2, 4, 5, 3]  # different order in VASP than expected for voigt notation
    C_voigt = C[np.ix_(vasp_to_voigt, vasp_to_voigt)]

    return C_voigt

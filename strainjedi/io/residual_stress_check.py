import json
import numpy as np
from ase import Atoms
from matscipy.elasticity import full_3x3_to_Voigt_6_strain


def _extract_cell_and_stress(atoms: Atoms):
    cell = np.array(atoms.cell.array, dtype=float)
    stress = atoms.get_stress()

    if stress is None:
        raise ValueError(f"No stress found in {atoms}")

    stress = np.array(stress, dtype=float)

    return cell, stress


def _compute_strain_voigt(cell_opt, cell_def):
    F = cell_def @ np.linalg.inv(cell_opt)
    strain = 0.5 * (F + F.T) - np.eye(3)

    return full_3x3_to_Voigt_6_strain(strain)


def residual_stress_check(
    optimized: Atoms,
    *deformed: Atoms,
    printout: bool = False,
    save: bool = False,
    out_file: str = "results.json",
    tol: float = 1e-8):

    if len(deformed) == 0:
        raise ValueError("At least one deformed structure required")

    # optimized structure
    cell_opt, stress_opt = _extract_cell_and_stress(optimized)

    results = []

    for idx, atoms in enumerate(deformed):
        cell_def, stress_def = _extract_cell_and_stress(atoms)

        strain_v = _compute_strain_voigt(cell_opt, cell_def)

        # relevant strain component
        relevant_idx = np.where(np.abs(strain_v) > tol)[0]

        if len(relevant_idx) == 0:
            raise ValueError(f"No relevant strain component found in deformed_{idx}")

        ratios = {}

        for i in relevant_idx:
            sigma_opt = stress_opt[i]
            sigma_def = stress_def[i]
            strain_i = strain_v[i]

            # only for relevant residual stress
            if abs(sigma_opt) <= 1e-6:
                continue

            if abs(sigma_def) < 1e-12:
                ratio_percent = np.nan
            else:
                ratio_percent = abs(sigma_opt / sigma_def) * 100

            ratios[int(i)] = {
                "strain": float(strain_i),
                "stress_opt": float(sigma_opt),
                "stress_def": float(sigma_def),
                "ratio": float(ratio_percent)
            }

        result = {
            "deformed_index": idx,
            "relevant_components": relevant_idx.tolist(),
            "ratios": ratios
        }

        results.append(result)

        if printout:
            print(f"\nDeformed structure {idx}")
            print("Relevant voigt components:", relevant_idx)

            for i in relevant_idx:
                r = ratios[int(i)]
                print(
                    f"  component {i}: "
                    f"strain={r['strain']:.2e}, "
                    f"|ratio|={r['ratio']:.2f}%, "
                    f"stress_def={r['stress_def']:.3e}, "
                    f"stress_opt={r['stress_opt']:.3e}"
                )

    output = {"results": results}

    if save:
        with open(out_file, "w") as f:
            json.dump(output, f, indent=2)

    return output

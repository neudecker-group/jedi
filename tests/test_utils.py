import numpy as np
import pytest
from ase import Atoms
from ase.vibrations.data import VibrationsData

from strainjedi.utils import validate_hessian


@pytest.mark.parametrize(
    "symbols,permutation,expected_ok",
    [
        pytest.param(["He"], [0], True, id="single_atom"),
        pytest.param(["H", "H"], [0, 1], True, id="identity"),
        pytest.param(["H", "H"], [1, 0], False, id="swap_two_atoms"),
        pytest.param(["C", "H", "H"], [1, 0, 2], False, id="partial_permutation"),
        pytest.param(["C", "H", "H"], [2, 1, 0], False, id="reverse_order"),
        pytest.param(["H", "H", "H"], [2, 0, 1], False, id="identical_species"),
        pytest.param(
            ["C", "H", "H", "H", "H"],
            [0, 3, 2, 1, 4],
            False,
            id="methane_hydrogen_permutation",
        ),
    ],
)
def test_validate_hessian_permutation(symbols, permutation, expected_ok):
    atoms0, _, modes, hessian = build_scenario(symbols, permutation=permutation)

    result, ok = validate_hessian(modes, atoms0)

    assert ok is expected_ok

    if expected_ok:
        expected = hessian
    else:
        expected = permute_hessian(hessian, np.argsort(permutation))

    assert np.allclose(result, expected)


@pytest.mark.parametrize(
    "offset,expected_ok",
    [
        pytest.param(0.0, True, id="exact_match"),
        pytest.param(1e-6, True, id="within_tolerance"),
        pytest.param(5e-6, True, id="near_tolerance_limit"),
        pytest.param(3e-5, False, id="outside_tolerance"),
        pytest.param(1e-4, False, id="well_outside_tolerance"),
    ],
)
def test_position_tolerance(offset, expected_ok):
    atoms0, _, modes, _ = build_scenario(
        ["N", "N"],
        permutation=None,
        offset=offset,
    )

    _, ok = validate_hessian(modes, atoms0)
    assert ok is expected_ok


@pytest.mark.parametrize(
    "symbols,permutation",
    [
        pytest.param(["H", "H"], [1, 0]),
        pytest.param(["N", "N", "N"], [2, 0, 1]),
        pytest.param(["C", "H", "H", "H"], [3, 1, 0, 2]),
    ],
)
def test_hessian_recovered_exactly(symbols, permutation):
    atoms0, _, modes, hessian = build_scenario(symbols, permutation=permutation)

    result, ok = validate_hessian(modes, atoms0)

    assert ok is False
    assert np.allclose(result, permute_hessian(hessian, np.argsort(permutation)))


# --------------------
# Helper functions
# --------------------


def build_scenario(symbols, permutation=None, offset=None):
    """
    Build a full test scenario:
    - atoms0 (reference)
    - vib_atoms (possibly permuted / perturbed)
    - modes object
    - hessian (base or transformed)
    """

    n_atoms = len(symbols)
    positions = np.zeros((n_atoms, 3))
    positions[:, 0] = np.arange(n_atoms, dtype=float)

    atoms0 = Atoms(symbols=symbols, positions=positions)

    # vib geometry
    vib_positions = positions.copy()

    if permutation is None:
        vib_atoms = atoms0
    else:
        vib_atoms = Atoms(
            symbols=[symbols[i] for i in permutation],
            positions=vib_positions[permutation],
        )

    # Hessian
    hessian = make_hessian(n_atoms)

    # optional perturbation (tolerance test)
    if offset is not None:
        vib_positions = vib_positions.copy()
        vib_positions[1, 2] += offset
        vib_atoms = Atoms(symbols=symbols, positions=vib_positions)

        hessian = np.eye(3 * n_atoms)

    modes = make_modes(vib_atoms, hessian)

    return atoms0, vib_atoms, modes, hessian


def make_modes(atoms, hessian_2d):
    """Create VibrationsData from a 2D Hessian."""
    n_atoms = len(atoms)

    hessian_4d = np.zeros((n_atoms, 3, n_atoms, 3))
    for i in range(n_atoms):
        for j in range(n_atoms):
            hessian_4d[i, :, j, :] = hessian_2d[
                3 * i : 3 * (i + 1),
                3 * j : 3 * (j + 1),
            ]

    modes = VibrationsData(atoms=atoms, hessian=hessian_4d)
    modes._hessian2d = hessian_2d
    return modes


def make_hessian(n_atoms: int) -> np.ndarray:
    """Deterministic test Hessian."""
    return np.arange((3 * n_atoms) ** 2, dtype=float).reshape(3 * n_atoms, 3 * n_atoms)


def atom_permutation_indices(permutation):
    """Expand atom permutation into Cartesian Hessian indices."""
    return np.concatenate([np.arange(3 * atom, 3 * atom + 3) for atom in permutation])


def permute_hessian(hessian, permutation):
    """Apply atom-wise permutation to a Hessian."""
    idx = atom_permutation_indices(permutation)
    return hessian[np.ix_(idx, idx)]

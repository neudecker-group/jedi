"""Golden-file tests for the output-file readers.

The headline test is :func:`test_hessian_reproduces_the_printed_frequencies`. Every program
prints its own harmonic frequencies in the same file as the Hessian, so re-diagonalising what
we parsed and comparing against those numbers validates extraction, row and column ordering,
symmetrisation, mass weighting *and* unit conversion in a single assertion. Supporting a new
release of a program should be: drop the files in, add an anchor, watch this test.

The sample outputs are the ones shipped for the documentation tutorials, so there is one copy
of each in the repository rather than two.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from strainjedi.constants import BOHR_ANG, HARTREE_EV
from strainjedi.io import MissingBlock, detect_program, read_hessian, read_output, scan
from strainjedi.io.adapter import to_atoms, to_vibrations

SAMPLES = Path(__file__).resolve().parent.parent / "docs" / "tutorials" / "calculators"

pytestmark = pytest.mark.skipif(not SAMPLES.is_dir(), reason="tutorial sample outputs are not present")


def _orca_frequencies(lines: list[str]) -> np.ndarray:
    start = scan.find_anchors(lines, ["$vibrational_frequencies"])[0]
    count = int(lines[start + 1].split()[0])
    return np.array([scan.to_float(lines[start + 2 + i].split()[1]) for i in range(count)])


def _gaussian_frequencies(lines: list[str]) -> np.ndarray:
    return np.array([scan.to_float(t) for line in lines if "Frequencies --" in line for t in line.split()[2:]])


def _qchem_frequencies(lines: list[str]) -> np.ndarray:
    return np.array(
        [scan.to_float(t) for line in lines if line.strip().startswith("Frequency:") for t in line.split()[1:]]
    )


@dataclass(frozen=True)
class Case:
    program: str
    geometry: str
    frequency: str
    natoms: int
    version: tuple[int, ...]
    frequencies: Callable[[list[str]], np.ndarray]

    @property
    def geometry_path(self) -> Path:
        return SAMPLES / self.geometry

    @property
    def frequency_path(self) -> Path:
        return SAMPLES / self.frequency


CASES = [
    Case("orca", "orca/output/opt.out", "orca/output/freq.hess", 8, (5, 0, 0), _orca_frequencies),
    Case("gaussian", "gaussian/output/opt.log", "gaussian/output/freq.log", 29, (16,), _gaussian_frequencies),
    Case("qchem", "qchem/output/opt.out", "qchem/output/freq.out", 12, (6, 0, 0), _qchem_frequencies),
]

CASE_IDS = [case.program for case in CASES]


@pytest.fixture(params=CASES, ids=CASE_IDS)
def case(request) -> Case:
    return request.param


def test_program_is_detected_from_content(case):
    assert detect_program(case.geometry_path) == case.program
    assert detect_program(case.frequency_path) == case.program


def test_detection_is_not_fooled_by_a_program_naming_another():
    """ORCA output mentions Gaussian basis sets, so the magics have to be the full banners."""
    assert detect_program(SAMPLES / "orca/output/opt.out") == "orca"


def test_geometry_and_energy(case):
    out = read_output(case.geometry_path)

    assert out.natoms == case.natoms
    assert out.version == case.version
    assert out.energy is not None
    assert out.positions.shape == (case.natoms, 3)


def test_positions_are_in_bohr(case):
    """A shortest interatomic distance below 1.2 would mean Angstrom leaked through."""
    positions = read_output(case.geometry_path).positions
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    shortest = distances[~np.eye(len(positions), dtype=bool)].min()

    assert 1.2 < shortest < 10.0


def test_hessian_is_square_and_symmetric(case):
    hessian = read_hessian(case.frequency_path)

    assert hessian.shape == (3 * case.natoms, 3 * case.natoms)
    assert hessian == pytest.approx(hessian.T, abs=1e-12)


def test_hessian_is_in_atomic_units(case):
    """In Hartree/Bohr^2 the stiffest diagonal element is order 1; in eV/A^2 it is order 100."""
    assert 0.05 < np.abs(read_hessian(case.frequency_path)).max() < 5.0


def test_hessian_reproduces_the_printed_frequencies(case):
    """The acceptance test: our Hessian must give back the program's own frequencies."""
    printed = case.frequencies(scan.read_lines(case.frequency_path))
    printed = np.sort(printed[printed != 0.0])

    computed = to_vibrations(read_output(case.frequency_path)).get_frequencies().real
    # Drop the six translations and rotations, which the programs do not list.
    computed = np.sort(computed[np.argsort(np.abs(computed))][6:])

    assert computed == pytest.approx(printed, abs=1.0)


def test_asking_an_optimisation_for_a_hessian_names_the_missing_keyword(case):
    with pytest.raises(MissingBlock) as excinfo:
        read_hessian(case.geometry_path)

    assert excinfo.value.hint, "a MissingBlock for a Hessian should say how to make the program print one"


def test_adapter_converts_back_to_the_angstroms_in_the_file(case):
    """Guards the Bohr round trip the atomic-unit convention introduces."""
    out = read_output(case.geometry_path)

    assert to_atoms(out).positions == pytest.approx(out.positions * BOHR_ANG, abs=1e-12)


def test_adapter_attaches_the_energy_where_jedi_looks_for_it(case):
    """Jedi reads energies via get_potential_energy(), not from any argument."""
    out = read_output(case.geometry_path)

    assert to_atoms(out).get_potential_energy() == pytest.approx(out.energy * HARTREE_EV)


def test_qchem_uses_the_masses_it_reported_rather_than_defaults():
    """Q-Chem weights its Hessian with 1.00783 for H, where ASE's default is 1.008."""
    out = read_output(SAMPLES / "qchem/output/freq.out")

    assert out.masses is not None
    assert out.masses[out.numbers == 1] == pytest.approx(1.00783, abs=1e-5)


def test_orca_hessian_values_land_where_the_file_puts_them():
    """Anchors the indexing: these are read straight off the first rows of freq.hess.

    The diagonal survives symmetrisation untouched, while [0, 3] is the average of the two
    off-diagonal partners the file prints (-7.0593692379E-02 and -7.0593692269E-02).
    """
    hessian = read_hessian(SAMPLES / "orca/output/freq.hess")

    assert hessian[0, 0] == pytest.approx(5.3373611284e-01)
    assert hessian[1, 1] == pytest.approx(5.3371132349e-01)
    assert hessian[0, 3] == pytest.approx(-7.0593692324e-02, abs=1e-12)
    assert hessian[3, 0] == pytest.approx(hessian[0, 3])


def test_jedi_runs_on_parsed_data():
    """End to end: two parsed geometries plus a parsed Hessian drive a full analysis."""
    from strainjedi import Jedi

    relaxed = to_atoms(read_output(SAMPLES / "orca/output/opt.out"))
    strained = to_atoms(read_output(SAMPLES / "orca/output/dist.out"))
    hessian = to_vibrations(read_output(SAMPLES / "orca/output/freq.hess"), relaxed)

    jedi = Jedi(relaxed, strained, hessian)
    jedi.run(printout=False)

    assert jedi.deltaE > 0
    assert np.isfinite(jedi.E_RIMs).all()

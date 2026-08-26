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

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "docs" / "tutorials" / "calculators"
RESOURCES = ROOT / "tests" / "resources" / "io"
"""Two roots, and only one of them is always there.

The tutorial outputs came with the documentation. Everything under ``tests/resources/io`` is a
git submodule -- private, because VASP's OUTCAR embeds POTCAR headers that say they may not be
redistributed -- so a clone without ``git submodule update --init`` simply has no such
directory. Cases backed by it are filtered out below rather than failing.

Adding a program version means dropping its outputs into the submodule and adding one row to
CASES."""

SUBMODULE_HINT = "run 'git submodule update --init' to fetch the QC fixture files"

pytestmark = pytest.mark.skipif(not SAMPLES.is_dir(), reason="tutorial sample outputs are not present")


def _orca_frequencies(lines: list[str]) -> np.ndarray:
    from strainjedi.io.readers import orca

    return orca.read_frequencies(lines)


def _gaussian_frequencies(lines: list[str]) -> np.ndarray:
    return np.array([scan.to_float(t) for line in lines if "Frequencies --" in line for t in line.split()[2:]])


def _qchem_frequencies(lines: list[str]) -> np.ndarray:
    return np.array(
        [scan.to_float(t) for line in lines if line.strip().startswith("Frequency:") for t in line.split()[1:]]
    )


def _vasp_frequencies(lines: list[str]) -> np.ndarray:
    from strainjedi.io.readers import vasp

    return vasp.read_frequencies(lines)


def signed(frequencies: np.ndarray) -> np.ndarray:
    """Flatten ASE's complex frequencies to signed reals, negative meaning imaginary.

    Programs print imaginary modes as negative numbers; ASE returns them as complex. VASP is
    the only fixture with any -- three near-zero translations of the periodic cell.
    """
    return np.where(frequencies.imag > 0, -frequencies.imag, frequencies.real)


@dataclass(frozen=True)
class Case:
    program: str
    root: Path
    geometry: str
    frequency: str
    natoms: int
    version: tuple[int, ...]
    frequencies: Callable[[list[str]], np.ndarray]
    trivial_modes: int = 6
    """How many modes the program leaves out of its printed list.

    An isolated molecule has six -- three translations and three rotations -- and none of the
    molecular codes print them. A periodic cell has only the three translations, and VASP
    prints those too, so nothing is dropped there.
    """

    @property
    def geometry_path(self) -> Path:
        return self.root / self.geometry

    @property
    def frequency_path(self) -> Path:
        return self.root / self.frequency

    @property
    def label(self) -> str:
        """Test id. Includes the size because one program version can appear twice."""
        return f"{self.program}-{'.'.join(str(p) for p in self.version)}-{self.natoms}atoms"


CASES = [
    # Shipped with the documentation tutorials.
    Case("orca", SAMPLES, "orca/output/opt.out", "orca/output/freq.hess", 8, (5, 0, 0), _orca_frequencies),
    Case("gaussian", SAMPLES, "gaussian/output/opt.log", "gaussian/output/freq.log", 29, (16,), _gaussian_frequencies),
    Case("qchem", SAMPLES, "qchem/output/opt.out", "qchem/output/freq.out", 12, (6, 0, 0), _qchem_frequencies),
    # H2O2 fixtures, generated from tests/resources/io/inputs/. Four atoms, so 3N = 12 --
    # still two or three column-chunks in every program's Hessian block.
    Case("orca", RESOURCES, "orca/6.1/h2o2_opt.out", "orca/6.1/h2o2_freq.hess", 4, (6, 1, 1), _orca_frequencies),
    Case("qchem", RESOURCES, "qchem/6.0/h2o2_opt.out", "qchem/6.0/h2o2_freq.out", 4, (6, 0, 0), _qchem_frequencies),
    Case("qchem", RESOURCES, "qchem/6.4/h2o2_opt.out", "qchem/6.4/h2o2_freq.out", 4, (6, 4, 0), _qchem_frequencies),
    Case("qchem", RESOURCES, "qchem/7.0/h2o2_opt.out", "qchem/7.0/h2o2_freq.out", 4, (7, 0, 0), _qchem_frequencies),
    # The only periodic fixture, so the only one exercising cell and pbc for real.
    Case("vasp", RESOURCES, "vasp/opt/OUTCAR", "vasp/freq/OUTCAR", 12, (6, 4, 2), _vasp_frequencies, trivial_modes=0),
]

CASES = [case for case in CASES if case.geometry_path.is_file() and case.frequency_path.is_file()]
"""Whatever is actually on disk. The tutorial cases always are; the submodule ones may not be."""

CASE_IDS = [case.label for case in CASES]


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

    computed = signed(to_vibrations(read_output(case.frequency_path)).get_frequencies())
    # Drop the modes the program leaves out of its own list; see Case.trivial_modes.
    computed = np.sort(computed[np.argsort(np.abs(computed))][case.trivial_modes :])

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


H2O2_ORCA = RESOURCES / "orca/6.1"
H2O2_QCHEM = RESOURCES / "qchem/6.4"

pytest_h2o2 = pytest.mark.skipif(
    not (H2O2_ORCA.is_dir() and H2O2_QCHEM.is_dir()), reason="the H2O2 fixtures are not present"
)


@pytest_h2o2
def test_two_programs_agree_on_the_same_molecule():
    """The strongest check available: independent programs, formats and readers, one answer.

    ORCA and Q-Chem ran identical PBE-D3(BJ)/def2-SVP jobs on H2O2, so anything the two
    parsers disagree about beyond method noise is a parsing error in one of them.
    """
    orca = to_atoms(read_output(H2O2_ORCA / "h2o2_opt.out"))
    qchem = to_atoms(read_output(H2O2_QCHEM / "h2o2_opt.out"))

    assert orca.get_distance(0, 1) == pytest.approx(qchem.get_distance(0, 1), abs=1e-3)
    assert orca.get_dihedral(2, 0, 1, 3) == pytest.approx(qchem.get_dihedral(2, 0, 1, 3), abs=0.1)
    assert list(orca.numbers) == list(qchem.numbers)


@pytest_h2o2
def test_two_programs_agree_on_the_strain_distribution():
    """End to end through both parsers: same molecule, same distortion, same JEDI answer."""
    from strainjedi import Jedi

    percentages = {}
    for name, folder, freq in [("orca", H2O2_ORCA, "h2o2_freq.hess"), ("qchem", H2O2_QCHEM, "h2o2_freq.out")]:
        relaxed = to_atoms(read_output(folder / "h2o2_opt.out"))
        strained = to_atoms(read_output(folder / "h2o2_dist.out"))
        jedi = Jedi(relaxed, strained, to_vibrations(read_output(folder / freq), relaxed))
        jedi.run(printout=False)
        percentages[name] = np.asarray(jedi.proc_E_RIMs)

    assert percentages["orca"] == pytest.approx(percentages["qchem"], abs=0.5)
    # The distortion stretched the O-O bond and twisted the dihedral, nothing else, so the
    # O-O bond must dominate. Anything else on top means the coordinates got mixed up.
    assert percentages["orca"][0] > 80.0


@pytest_h2o2
def test_h2o2_optimisations_found_a_real_minimum():
    """The start geometry was twisted off the planar trans saddle point on purpose."""
    from strainjedi.io import imaginary_frequencies

    assert imaginary_frequencies(H2O2_ORCA / "h2o2_freq.hess") == 0
    assert imaginary_frequencies(H2O2_QCHEM / "h2o2_freq.out") == 0


@pytest_h2o2
def test_each_program_reports_its_own_masses():
    """ORCA uses average atomic masses, Q-Chem the dominant isotope. Neither is ASE's default."""
    orca_masses = read_output(H2O2_ORCA / "h2o2_freq.hess").masses
    qchem_masses = read_output(H2O2_QCHEM / "h2o2_freq.out").masses

    assert orca_masses[0] == pytest.approx(15.999, abs=1e-3)
    assert qchem_masses[0] == pytest.approx(15.99491, abs=1e-5)


VASP = RESOURCES / "vasp"

pytest_vasp = pytest.mark.skipif(not VASP.is_dir(), reason="the VASP fixtures are not present")


@pytest_vasp
def test_vasp_second_derivatives_are_negated():
    """VASP prints the derivative of the *force*, so its diagonal is negative.

    Taking the block at face value is not a small error -- it turns every mode imaginary --
    but it is invisible unless the frequencies are checked, which is what this pins down.
    """
    lines = scan.read_lines(VASP / "freq" / "OUTCAR")
    start = scan.find_anchors(lines, ["SECOND DERIVATIVES"])[0]
    printed_diagonal = scan.to_float(lines[start + 3].split()[1])

    hessian = read_hessian(VASP / "freq" / "OUTCAR")

    assert printed_diagonal < 0, "the fixture should show VASP's negative sign convention"
    assert hessian[0, 0] > 0, "a Hessian diagonal is a restoring force and must be positive"


@pytest_vasp
def test_vasp_is_periodic_and_carries_its_cell():
    """The only fixture that exercises cell and pbc, which JEDI needs for minimum-image."""
    out = read_output(VASP / "opt" / "OUTCAR")
    atoms = to_atoms(out)

    assert out.pbc.all()
    assert np.diag(atoms.cell) == pytest.approx([4.059527, 4.611590, 8.470795], abs=1e-5)
    assert atoms.pbc.all()


@pytest_vasp
def test_vasp_masses_come_from_pomass():
    """VASP states masses per species; they have to be expanded over 'ions per type'."""
    out = read_output(VASP / "freq" / "OUTCAR")

    assert out.masses == pytest.approx([1.0] * 4 + [12.01] * 4 + [14.0] * 4)
    assert list(out.numbers) == [1] * 4 + [6] * 4 + [7] * 4


@pytest_vasp
def test_vasp_near_zero_translations_are_not_counted_as_imaginary():
    """A periodic cell's three translations come out as tiny imaginary values, not zero.

    Counting them would mean every converged solid failed the minimum check.
    """
    from strainjedi.io import imaginary_frequencies
    from strainjedi.io.readers import vasp

    printed = vasp.read_frequencies(scan.read_lines(VASP / "freq" / "OUTCAR"))

    assert sum(f < 0 for f in printed) == 3, "the fixture should have three near-zero modes"
    assert all(abs(f) < vasp.NEAR_ZERO_CM for f in printed if f < 0)
    assert imaginary_frequencies(VASP / "freq" / "OUTCAR") == 0


@pytest_vasp
def test_vasp_finite_differences_do_not_leak_into_the_reference():
    """IBRION=5 displaces every atom in turn, printing a geometry and energy for each.

    The same trap as Q-Chem's semi-numerical Hessian, in a different program: taking the last
    of anything gives a displaced structure. Here the displacement is VASP's default POTIM of
    0.015 A, which is small enough to look plausible and wrong enough to matter.
    """
    reference = read_output(VASP / "freq" / "OUTCAR")
    optimised = read_output(VASP / "opt" / "OUTCAR")

    assert reference.positions == pytest.approx(optimised.positions, abs=1e-9)

    lines = scan.read_lines(VASP / "freq" / "OUTCAR")
    marker = scan.find_anchors(lines, ["Finite differences"])[0]
    blocks = [i for i, line in enumerate(lines) if "POSITION" in line and "TOTAL-FORCE" in line]
    assert sum(1 for i in blocks if i > marker) > 50, "the fixture should contain the displacements"


@pytest_vasp
def test_outcar_is_detected_without_an_extension():
    """VASP output is just called OUTCAR, so the banner has to carry the detection."""
    assert detect_program(VASP / "opt" / "OUTCAR") == "vasp"


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

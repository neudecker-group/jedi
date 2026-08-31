"""Regressions for two defects found on a Q-Chem 6.3 semi-numerical frequency job.

1. A semi-numerical Hessian (``IDERIV=1``) differentiates analytic gradients numerically, so
   the frequency job prints one energy per *displaced* geometry. Taking the last one reported
   a displaced point instead of the equilibrium structure.
2. The imaginary-frequency check had no implementation at all, so a saddle point would sail
   through silently.

The fixtures here are synthetic but mirror the real layouts exactly; the real 11 MB file is
asserted against too when it happens to be present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from strainjedi.io import read_output, scan
from strainjedi.io.report import warn_imaginary_frequencies
from strainjedi.io.validate import imaginary_frequencies

SEMI_NUMERICAL = Path(__file__).resolve().parent / "resources" / "io" / "qchem" / "6.4" / "h2o2_freq_semi.out"
"""A real IDERIV=1 run: H2O2 through Q-Chem 6.4, small enough to keep in the repository.

Generated from ``tests/resources/io/inputs/qchem/h2o2_freq_semi.in``. This is the committed
regression fixture for both defects; the 92-atom Q-Chem 6.3 file that originally exposed them
is 11 MB and lives outside the repository.
"""

REAL_QCHEM_63 = Path("/home/rawsita/input_1103456.out")
"""The original 11 MB file, asserted against only when it happens to be on this machine."""

needs_fixtures = pytest.mark.skipif(
    not SEMI_NUMERICAL.is_file(),
    reason="QC fixture submodule not checked out; run 'git submodule update --init'",
)

QCHEM_GEOMETRY = """\
 A Quantum Leap Into The Future Of Chemistry
 Q-Chem 6.3.1 for Intel X86 EM64T Linux

             Standard Nuclear Orientation (Angstroms)
    I     Atom           X                Y                Z
 ----------------------------------------------------------------
    1      O       0.0000000000     0.0000000000     0.0000000000
    2      H       0.0000000000     0.0000000000     0.9600000000
 ----------------------------------------------------------------
"""

SEMI_NUMERICAL_TAIL = """\
 SCF   energy = -76.12345678
 Total energy = -76.12345678
 Final energy is -76.123456780000

	******************************
	**  OPTIMIZATION CONVERGED  **
	******************************

 Total energy = -76.09999901
 Total energy = -76.09999902
 Total energy = -76.09999903
"""
"""A converged optimisation followed by displacement points, as IDERIV=1 produces.

The plain energy and the marker agree here, as they do in a real file -- Q-Chem simply prints
the marker to more decimals. What must not be picked up are the three trailing values.
"""

EFEI_TAIL = """\
 Gradient from external distort forces
 SCF   energy = -76.12345678
 Total energy = -76.12345678
 Gradient from external distort forces
 Final energy is -76.222222220000
"""
"""An EFEI job, where the marker carries the work done by the external force.

Here the marker and the plain energy genuinely differ, and it is the *plain* one JEDI needs:
the force-modified surface is not the one strain is measured on.
"""


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


# --------------------------------------------------------------------------------------
# Defect 1: energy must not come from a displaced geometry
# --------------------------------------------------------------------------------------


def test_converged_energy_wins_over_trailing_displaced_energies(tmp_path):
    """The three trailing energies are displacement points; none of them is the answer."""
    path = write(tmp_path, "semi_numerical.out", QCHEM_GEOMETRY + SEMI_NUMERICAL_TAIL)

    assert read_output(path).energy == pytest.approx(-76.12345678)


def test_efei_energy_ignores_the_force_augmented_final_value(tmp_path):
    """EFEI's reported final energy includes the external work term; JEDI needs the plain one."""
    path = write(tmp_path, "efei.out", QCHEM_GEOMETRY + EFEI_TAIL)

    assert read_output(path).energy == pytest.approx(-76.12345678)


def test_last_energy_still_wins_when_there_is_no_converged_optimisation(tmp_path):
    """A plain single point or analytic freq has no 'Final energy is'; last hit is correct."""
    text = QCHEM_GEOMETRY + " Total energy in the final basis set =     -76.09999901\n"
    path = write(tmp_path, "single_point.out", text)

    assert read_output(path).energy == pytest.approx(-76.09999901)


def test_optimisation_trajectory_takes_its_final_step(tmp_path):
    """Within one anchor the last hit wins, so an opt trajectory still ends on its result."""
    text = QCHEM_GEOMETRY + "".join(
        f" Total energy in the final basis set =     {e}\n" for e in ("-76.01", "-76.05", "-76.09")
    )
    path = write(tmp_path, "opt.out", text)

    assert read_output(path).energy == pytest.approx(-76.09)


@needs_fixtures
def test_semi_numerical_fixture_has_the_shape_that_caused_the_bug():
    """Guards the fixture itself: without trailing displaced energies it proves nothing.

    The analytic run of the same molecule prints 7 of these lines; IDERIV=1 prints 31, one per
    displacement, all of them after the converged optimisation.
    """
    lines = scan.read_lines(SEMI_NUMERICAL)

    assert sum("Final energy is" in line for line in lines) == 1
    assert sum("Total energy =" in line for line in lines) > 20


@needs_fixtures
def test_semi_numerical_run_reports_the_converged_energy():
    """The real regression: -151.27075334 is the last displaced point, not the answer.

    The tolerance is 1e-8 rather than tighter because Q-Chem prints ``Final energy is`` to
    twelve decimals and ``Total energy =`` to eight. Both are the same converged energy; the
    parser reads the latter, so it is that precision we can assert. The difference is 5e-9
    Hartree, which is 3e-6 kcal/mol -- far below anything a strain analysis resolves.
    """
    lines = scan.read_lines(SEMI_NUMERICAL)
    displaced = [scan.to_float(line.split()[-1]) for line in lines if "Total energy =" in line][-1]

    energy = read_output(SEMI_NUMERICAL).energy

    assert energy == pytest.approx(-151.270753405138, abs=1e-8)
    assert energy != pytest.approx(displaced, abs=1e-9)


EFEI = SEMI_NUMERICAL.with_name("h2o2_dist_efei.out")
PARTIAL = SEMI_NUMERICAL.with_name("h2o2_p_freq.out")
ORCA_EFEI = SEMI_NUMERICAL.parents[2] / "orca" / "6.1" / "h2o2_dist_efei.out"


@pytest.mark.skipif(not EFEI.is_file(), reason="the Q-Chem EFEI fixture is not present")
def test_qchem_efei_reports_the_plain_electronic_energy():
    """Q-Chem's 'Final energy is' carries the external work term; the SCF energy does not.

    The two are 92.8 kcal/mol apart here, which is the whole strain signal several times over.
    """
    lines = scan.read_lines(EFEI)
    augmented = [scan.to_float(line.split()[-1]) for line in lines if "Final energy is" in line][-1]

    energy = read_output(EFEI).energy

    assert energy == pytest.approx(-151.26257274, abs=1e-8)
    assert augmented == pytest.approx(-151.410532225073, abs=1e-8)


@pytest.mark.skipif(not ORCA_EFEI.is_file(), reason="the ORCA EFEI fixture is not present")
def test_orca_efei_needs_no_special_handling():
    """ORCA splits the two the other way round, so the ordinary anchor is already correct.

    Its SCF 'Total Energy' absorbs the external-force term while FINAL SINGLE POINT ENERGY
    stays on the plain surface -- the opposite of Q-Chem, and the reason this is asserted
    rather than assumed.
    """
    lines = scan.read_lines(ORCA_EFEI)
    scf = [scan.to_float(line.split()[3]) for line in lines if line.strip().startswith("Total Energy       :")]

    energy = read_output(ORCA_EFEI).energy

    assert energy == pytest.approx(-151.10102726, abs=1e-8)
    assert scf[-1] == pytest.approx(-151.87013286, abs=1e-8)


@pytest.mark.skipif(not PARTIAL.is_file(), reason="the partial-Hessian fixture is not present")
def test_partial_hessian_is_read_with_its_atom_indices():
    """PHESS gives a Hessian over the $alist atoms only, expressed in the input atom order.

    Q-Chem permutes the molecule internally (printing O,H,O,H for H2O2), but that is its own
    business: the same geometry went in, so the same order comes back out, and the indices
    point at the $alist atoms within it -- 1-indexed {2,3} in the input, so 1 and 2 here.
    """
    out = read_output(PARTIAL)

    assert out.is_partial_hessian
    assert out.hessian.shape == (6, 6)
    assert list(out.hessian_indices) == [1, 2]
    assert out.masses == pytest.approx([15.99491, 1.00783])
    assert list(out.numbers) == [8, 8, 1, 1]


@pytest.mark.skipif(not PARTIAL.is_file(), reason="the partial-Hessian fixture is not present")
def test_partial_hessian_matches_the_optimised_structure():
    """The whole point of undoing the permutation: these two files now line up atom for atom."""
    partial = read_output(PARTIAL)
    optimised = read_output(PARTIAL.with_name("h2o2_opt.out"))

    assert list(partial.numbers) == list(optimised.numbers)
    assert partial.positions == pytest.approx(optimised.positions, abs=1e-6)


@pytest.mark.skipif(not PARTIAL.is_file(), reason="the partial-Hessian fixture is not present")
def test_partial_hessian_reproduces_its_printed_frequency():
    """A two-atom fragment is linear, so 3N-5 leaves exactly one mode: the O-H stretch."""
    from strainjedi.io.adapter import to_vibrations

    printed = [
        scan.to_float(t) for l in scan.read_lines(PARTIAL) if l.strip().startswith("Frequency:") for t in l.split()[1:]
    ]
    frequencies = to_vibrations(read_output(PARTIAL)).get_frequencies().real
    computed = np.sort(frequencies[np.argsort(np.abs(frequencies))][5:])

    assert computed == pytest.approx(np.sort(printed), abs=1.0)


@pytest.mark.skipif(not PARTIAL.is_file(), reason="the partial-Hessian fixture is not present")
def test_partial_hessian_attaches_to_the_optimised_structure():
    """What a caller actually wants to write, and what used to be impossible."""
    from strainjedi.io.adapter import to_atoms, to_vibrations

    optimised = to_atoms(read_output(PARTIAL.with_name("h2o2_opt.out")))
    vibrations = to_vibrations(read_output(PARTIAL), optimised)

    # A partial Hessian stays at its own size; `indices` is what ties it to the 4-atom
    # structure, so the pairing is meaningful even though the matrix is only 6x6.
    assert vibrations.get_hessian_2d().shape == (6, 6)
    assert vibrations.get_hessian().shape == (2, 3, 2, 3)
    assert len(optimised) == 4


@pytest.mark.skipif(not PARTIAL.is_file(), reason="the partial-Hessian fixture is not present")
def test_a_genuinely_mismatched_structure_is_still_refused():
    """Undoing the permutation removes the common trap, but the guard still has to hold."""
    from ase.atoms import Atoms

    from strainjedi.io.adapter import to_vibrations
    from strainjedi.io.types import ParseError

    with pytest.raises(ParseError, match="cannot attach"):
        to_vibrations(read_output(PARTIAL), Atoms("HHHH", positions=np.zeros((4, 3))))


@needs_fixtures
def test_semi_numerical_hessian_matches_the_analytic_one():
    """Finite differences of analytic gradients must agree with the analytic Hessian."""
    analytic = read_output(SEMI_NUMERICAL.with_name("h2o2_freq.out")).hessian
    semi = read_output(SEMI_NUMERICAL).hessian

    assert semi == pytest.approx(analytic, abs=1e-4)
    # ...but not bit-identical: this really is Q-Chem's other code path, not a relabelling.
    assert not np.array_equal(semi, analytic)


@pytest.mark.skipif(not REAL_QCHEM_63.is_file(), reason="the Q-Chem 6.3 sample is not on this machine")
def test_real_qchem_63_reports_the_converged_energy():
    """-2816.60896914 was the last displaced SCF; -2816.60938756 is the converged value."""
    assert read_output(REAL_QCHEM_63).energy == pytest.approx(-2816.60938756, abs=1e-8)


# --------------------------------------------------------------------------------------
# Defect 2: imaginary frequencies must be detectable
# --------------------------------------------------------------------------------------


def test_qchem_imaginary_frequencies_are_counted(tmp_path):
    text = QCHEM_GEOMETRY + "   This Molecule has  3 Imaginary Frequencies\n"
    path = write(tmp_path, "saddle.out", text)

    assert imaginary_frequencies(path) == 3


def test_qchem_reports_zero_for_a_clean_minimum(tmp_path):
    text = QCHEM_GEOMETRY + "   This Molecule has  0 Imaginary Frequencies\n"
    path = write(tmp_path, "minimum.out", text)

    assert imaginary_frequencies(path) == 0


def test_none_when_the_file_has_no_vibrational_analysis(tmp_path):
    """None and zero mean different things: 'cannot say' versus 'clean minimum'."""
    path = write(tmp_path, "opt_only.out", QCHEM_GEOMETRY)

    assert imaginary_frequencies(path) is None


GAUSSIAN_HEAD = " Entering Gaussian System, Link 0=g16\n Gaussian 16:  ES64L-G16RevC.01\n"


def test_gaussian_imaginary_frequencies_are_counted(tmp_path):
    text = GAUSSIAN_HEAD + " ****    2 imaginary frequencies (negative Signs) ****\n Frequencies --  -40.3  100.0\n"
    path = write(tmp_path, "saddle.log", text)

    assert imaginary_frequencies(path) == 2


def test_gaussian_infers_zero_from_a_frequency_run_without_the_warning(tmp_path):
    """Gaussian prints the count only when non-zero, so absence means zero -- if it ran freq."""
    path = write(tmp_path, "minimum.log", GAUSSIAN_HEAD + " Frequencies --    100.0   200.0\n")

    assert imaginary_frequencies(path) == 0
    assert imaginary_frequencies(write(tmp_path, "opt.log", GAUSSIAN_HEAD)) is None


ORCA_HESS = """\
$orca_hessian_file

$atoms
2
 O     15.99900      0.000000000000    0.000000000000    0.000000000000
 H      1.00800      0.000000000000    0.000000000000    1.814137000000

$vibrational_frequencies
6
    0        0.000000
    1        0.000000
    2        0.000000
    3        0.000000
    4     {fourth}
    5     3700.000000

$end
"""


def test_orca_counts_negative_entries_in_the_frequency_block(tmp_path):
    saddle = write(tmp_path, "saddle.hess", ORCA_HESS.format(fourth="-123.456789"))
    minimum = write(tmp_path, "minimum.hess", ORCA_HESS.format(fourth="1595.000000"))

    assert imaginary_frequencies(saddle) == 1
    assert imaginary_frequencies(minimum) == 0


def test_a_saddle_point_warns_instead_of_refusing(tmp_path):
    """A saddle point is a fact about the calculation, not a parse failure.

    The old code called sys.exit(1) from inside the parser. Warning leaves the decision with
    whoever ran the job -- a transition state is a legitimate thing to have computed.
    """
    import io

    text = QCHEM_GEOMETRY + "   This Molecule has  3 Imaginary Frequencies\n"
    path = write(tmp_path, "saddle.out", text)

    stream = io.StringIO()
    warn_imaginary_frequencies(imaginary_frequencies(path), path, stream=stream)
    message = stream.getvalue()

    assert "3 imaginary frequencies" in message
    assert "saddle point" in message
    assert read_output(path) is not None, "the parse itself must still succeed"


def test_a_clean_minimum_says_nothing(tmp_path):
    import io

    path = write(tmp_path, "minimum.out", QCHEM_GEOMETRY + "   This Molecule has  0 Imaginary Frequencies\n")

    stream = io.StringIO()
    warn_imaginary_frequencies(imaginary_frequencies(path), path, stream=stream)

    assert stream.getvalue() == ""


@needs_fixtures
def test_semi_numerical_run_is_a_clean_minimum():
    """The committed fixture is a true minimum, so nothing should warn about it."""
    assert imaginary_frequencies(SEMI_NUMERICAL) == 0


@pytest.mark.skipif(not REAL_QCHEM_63.is_file(), reason="the Q-Chem 6.3 sample is not on this machine")
def test_real_qchem_63_is_a_saddle_point():
    """The sample reports -40.30, -14.27 and -12.36 cm^-1."""
    assert imaginary_frequencies(REAL_QCHEM_63) == 3

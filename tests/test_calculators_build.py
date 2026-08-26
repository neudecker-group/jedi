"""Tests for converting an ORCA input file into ASE's orcasimpleinput/orcablocks form.

The awkward part is finding where a ``%`` block ends. Blocks nest, so a mechanochemistry
``%geom`` closes twice, while ``%maxcore 4000`` never closes at all -- and an earlier version
that looked for the first ``end`` silently dropped the outer one, producing an unbalanced
block string that ORCA would reject.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strainjedi.calculators.build import orca_input_to_ase

FIXTURES = Path(__file__).resolve().parent / "resources" / "io" / "inputs" / "orca"

GEOMETRY = "*xyz 0 1\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\n*\n"


def convert(tmp_path: Path, text: str):
    path = tmp_path / "job.inp"
    path.write_text(text)
    return orca_input_to_ase(str(path))


def test_nested_block_keeps_both_end_tokens(tmp_path):
    """A mechanochemistry %geom block contains a POTENTIALS sub-block, so it closes twice."""
    text = "! PBE\n%geom\n  POTENTIALS\n    { C 0 1 8.0 }\n  end\nend\n" + GEOMETRY

    _, blocks, _, _ = convert(tmp_path, text)

    assert blocks.splitlines() == ["%geom", "POTENTIALS", "{ C 0 1 8.0 }", "end", "end"]


def test_single_line_blocks_do_not_swallow_the_rest_of_the_file(tmp_path):
    """%maxcore and %base carry no 'end'; treating them as open blocks eats everything after."""
    text = '! PBE\n%maxcore 4000\n%base "job"\n' + GEOMETRY

    simple, blocks, charge, mult = convert(tmp_path, text)

    assert blocks.splitlines() == ["%maxcore 4000", '%base "job"']
    assert simple == "PBE"
    assert (charge, mult) == (0, 1)


def test_a_keyword_line_after_a_block_is_not_absorbed_into_it(tmp_path):
    text = "! PBE\n%scf\n  maxiter 200\nend\n! TightSCF\n" + GEOMETRY

    simple, blocks, _, _ = convert(tmp_path, text)

    assert simple == "PBE TightSCF"
    assert blocks.splitlines() == ["%scf", "maxiter 200", "end"]


@pytest.mark.parametrize(
    ("line", "expected"),
    [("*xyz 0 1", (0, 1)), ("* xyz -1 2", (-1, 2)), ("*XYZ 2 1", (2, 1)), ("*xyzfile 0 3 in.xyz", (0, 3))],
)
def test_charge_and_multiplicity_spellings(tmp_path, line, expected):
    """ORCA and ASE's own writer disagree about the space after the asterisk."""
    _, _, charge, mult = convert(tmp_path, f"! PBE\n{line}\nO 0.0 0.0 0.0\n*\n")

    assert (charge, mult) == expected


def test_missing_charge_and_multiplicity_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="Charge and multiplicity"):
        convert(tmp_path, "! PBE\n%maxcore 4000\n")


def test_comments_are_ignored(tmp_path):
    simple, blocks, _, _ = convert(tmp_path, "# a comment\n! PBE\n# another\n" + GEOMETRY)

    assert simple == "PBE"
    assert blocks == ""


def _end_tokens(text: str) -> int:
    """Lines closing a block. In an ORCA input 'end' only ever appears inside one."""
    return sum(
        1
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#") and line.split()[-1].lower() == "end"
    )


@pytest.mark.skipif(not FIXTURES.is_dir(), reason="the ORCA fixture inputs are not present")
@pytest.mark.parametrize("name", ["h2o2_opt", "h2o2_freq", "h2o2_dist", "h2o2_dist_efei", "h2o2_p_freq"])
def test_the_committed_fixture_inputs_lose_no_block_content(name):
    """Every input we ask people to run must convert without dropping a block terminator.

    Counting 'end' both sides is the direct test of the bug: nesting used to lose the outer
    one, leaving ORCA with an unbalanced %geom or %freq.
    """
    source = (FIXTURES / f"{name}.inp").read_text()
    _, blocks, charge, mult = orca_input_to_ase(str(FIXTURES / f"{name}.inp"))

    assert (charge, mult) == (0, 1)
    assert _end_tokens(blocks) == _end_tokens(source)

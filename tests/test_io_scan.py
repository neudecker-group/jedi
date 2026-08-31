"""Unit tests for the block-scraping primitives.

These use synthetic blocks so each shape is tested in isolation, including the deliberate
tolerances: a block that gains a header line, changes width, or picks up blank lines must
still parse, because that is exactly what changes between releases of a QC program.
"""

from __future__ import annotations

import numpy as np
import pytest

from strainjedi.io import scan
from strainjedi.io.types import ParseError


def test_to_float_accepts_fortran_d_exponent():
    assert scan.to_float("0.843133D+00") == pytest.approx(0.843133)
    assert scan.to_float("-0.601884D-01") == pytest.approx(-0.0601884)
    assert scan.to_float("5.3373611284E-01") == pytest.approx(0.53373611284)


def tokens(line: str) -> list[str]:
    """Split a line the way the scanners do, so the cases below read like real file lines."""
    return line.split()


def test_row_predicates_distinguish_headers_from_data():
    assert scan.is_int_row(tokens("1 2 3 4 5"))
    assert not scan.is_int_row(tokens("1 0.5"))
    assert scan.is_float_row(tokens("0.038179 -0.010483"))
    assert scan.is_indexed_row(tokens("6 0.000000D+00 -0.660206D-01"))
    # A bare index with no values is not a data row.
    assert not scan.is_indexed_row(["6"])


def test_find_anchors_returns_first_matching_pattern_only():
    lines = ["old spelling", "noise", "new spelling", "old spelling"]

    # Priority order decides: the first pattern with any hit wins outright.
    assert scan.find_anchors(lines, ["new spelling", "old spelling"]) == [2]
    assert scan.find_anchors(lines, ["old spelling", "new spelling"]) == [0, 3]
    assert scan.find_anchors(lines, ["absent"]) == []


ZERO_BASED_BLOCK = """
$hessian
3
          0         1
    0   1.0   2.0
    1   2.0   4.0
    2   3.0   6.0
          2
    0   3.0
    1   6.0
    2   9.0
$end
""".splitlines()


def test_parse_indexed_block_reads_width_from_the_file():
    matrix = scan.parse_indexed_block(ZERO_BASED_BLOCK, 2, one_based=False, n=3, stop_prefix="$")

    assert matrix == pytest.approx(np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]))


def test_parse_indexed_block_infers_dimension_when_not_declared():
    """Gaussian never states the dimension, so it comes from the largest index seen."""
    lines = "  1  1.0\n  2  2.0  3.0".splitlines()
    matrix = scan.parse_indexed_block(["anchor", "      1      2"] + lines, 0, one_based=True)

    assert matrix.shape == (2, 2)


def test_parse_indexed_block_tolerates_a_blank_line_inside_the_block():
    lines = ["anchor", "   0   1", "  0  1.0  2.0", "", "  1  2.0  4.0"]

    assert scan.parse_indexed_block(lines, 0, one_based=False, n=2).shape == (2, 2)


def test_parse_indexed_block_rejects_an_out_of_range_index():
    with pytest.raises(ParseError, match="index out of range"):
        scan.parse_indexed_block(ZERO_BASED_BLOCK, 2, one_based=False, n=2, stop_prefix="$")


def test_parse_indexed_block_reports_an_empty_block():
    with pytest.raises(ParseError, match="no data rows"):
        scan.parse_indexed_block(["anchor", "not data at all"], 0, one_based=False)


def _bare_block(matrix: np.ndarray, width: int) -> list[str]:
    """Render a matrix the way Q-Chem does: column chunks, no labels, blank separators."""
    lines = ["Mass-Weighted Hessian Matrix:", ""]
    for start in range(0, matrix.shape[1], width):
        for row in matrix[:, start : start + width]:
            lines.append("   " + "  ".join(f"{v:.6f}" for v in row))
        lines.extend(["", ""])
    return lines


@pytest.mark.parametrize("width", [3, 5, 6])
def test_parse_bare_block_round_trips_at_any_chunk_width(width):
    original = np.arange(36, dtype=float).reshape(6, 6)

    assert scan.parse_bare_block(_bare_block(original, width), 0, 6) == pytest.approx(original)


def test_parse_bare_block_stops_before_a_following_identical_block():
    """Q-Chem prints the projected Hessian right after this one, in the same format."""
    first = np.arange(36, dtype=float).reshape(6, 6)
    second = np.full((6, 6), 99.0)
    lines = _bare_block(first, 6) + ["Translations and Rotations Projected Out"] + _bare_block(second, 6)

    assert scan.parse_bare_block(lines, 0, 6) == pytest.approx(first)


def test_parse_bare_block_reports_a_truncated_block():
    truncated = _bare_block(np.zeros((6, 6)), 6)[:5]

    with pytest.raises(ParseError, match="expected 6 rows"):
        scan.parse_bare_block(truncated, 0, 6)


def test_unpack_lower_triangle_mirrors_into_a_symmetric_matrix():
    matrix = scan.unpack_lower_triangle(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 3)

    assert matrix == pytest.approx(np.array([[1.0, 2.0, 4.0], [2.0, 3.0, 5.0], [4.0, 5.0, 6.0]]))
    assert matrix == pytest.approx(matrix.T)


def test_unpack_lower_triangle_rejects_a_wrong_value_count():
    with pytest.raises(ParseError, match="needs 6 values"):
        scan.unpack_lower_triangle(np.zeros(5), 3)


def test_parse_packed_reads_across_line_breaks():
    lines = ["Cartesian Force Constants   R   N=  5", "  1.0  2.0  3.0", "  4.0  5.0", "Next Field  I  N= 3"]

    assert scan.parse_packed(lines, 0, 5) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])


def test_parse_packed_reports_a_short_field():
    with pytest.raises(ParseError, match="expected 9 values"):
        scan.parse_packed(["header", "  1.0  2.0"], 0, 9)


def test_symmetrize_averages_and_mirror_lower_reflects():
    asymmetric = np.array([[1.0, 0.0], [2.0, 3.0]])

    assert scan.symmetrize(asymmetric) == pytest.approx(np.array([[1.0, 1.0], [1.0, 3.0]]))
    assert scan.mirror_lower(asymmetric) == pytest.approx(np.array([[1.0, 2.0], [2.0, 3.0]]))

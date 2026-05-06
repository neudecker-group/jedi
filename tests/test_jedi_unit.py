import copy

import matplotlib
import numpy as np

from strainjedi import rics
from strainjedi.jedi import Jedi

# Headless backend for testing
matplotlib.use("agg")


class TestGetRims:
    """
    Tests for get_rims() and get_common_rics() methods.
    """

    def test_rim_list_shape(self, deds_jedi_with_rims):
        rim_list = deds_jedi_with_rims._rim_list
        assert len(rim_list) == 4
        assert rim_list[0].shape[1] == 2  # bonds
        assert rim_list[2].shape[1] == 3  # angles
        assert rim_list[3].shape[1] == 4  # dihedrals

    def test_custom_bonds_none_by_default(self, deds_jedi_fresh):
        j = copy.deepcopy(deds_jedi_fresh)
        assert j._custom_bonds is None

    def test_rim_list_no_custombonds(self, deds_jedi_with_rims):
        assert deds_jedi_with_rims._rim_list[1].shape[0] == 0

    def test_rim_list_custombonds(self, deds_jedi_fresh):
        j = copy.deepcopy(deds_jedi_fresh)
        j.add_custom_bonds([[0, 5], [2, 10]])
        j._indices = np.arange(0, len(j._atoms0))

        j._rim_list = rics.intersect(
            rics.calculate(j._atoms0, j._indices, j._custom_bonds),
            rics.calculate(j._atomsF, j._indices, j._custom_bonds),
        )

        assert j._rim_list[1].shape == (2, 2)
        assert list(j._rim_list[1][0]) == [0, 5]


class TestGetHessian:
    """
    Tests for get_hessian() method.
    """

    def test_hessian_reference(self, deds_jedi_fresh, deds_ref):
        j = copy.deepcopy(deds_jedi_fresh)
        np.testing.assert_allclose(j._H, deds_ref["jediInternalHessian"], atol=1e-05)


class TestGetBMatrix:
    """
    Test for get_b_matrix() method.
    """

    def test_B_matrix_reference(self, deds_jedi_fresh, deds_rim_list, deds_ref):
        j = copy.deepcopy(deds_jedi_fresh)
        j._rim_list = deds_rim_list
        np.testing.assert_allclose(j._B, deds_ref["bmatrix"], atol=1e-05)


class TestGetDeltaQ:
    """
    Tests for get_delta_q() method.
    """

    def test_delta_q_identical_structure(self, deds_opt, deds_vibdata):
        j = Jedi(deds_opt, deds_opt, deds_vibdata)
        j._indices = np.arange(len(deds_opt))
        j._rim_list = rics.intersect(
            rics.calculate(j._atoms0, j._indices, j._custom_bonds),
            rics.calculate(j._atomsF, j._indices, j._custom_bonds),
        )
        np.testing.assert_allclose(j._delta_q, 0.0, atol=1e-05)

    def test_delta_q_deds_reference(self, deds_jedi_fresh, deds_rim_list, deds_ref):
        j = copy.deepcopy(deds_jedi_fresh)
        j._rim_list = deds_rim_list
        j._B = deds_ref["bmatrix"]
        j.get_delta_q()
        np.testing.assert_allclose(j._delta_q, deds_ref["delta_q"], atol=1e-05)


class TestGetEnergy:
    """
    Tests for get_energies() method.
    """

    def test_energy_difference(self, deds_jedi_fresh):
        j = copy.deepcopy(deds_jedi_fresh)
        j.get_energies()
        assert abs(j._energies[0] - (j._energies[1] - j._energies[2])) < 1e-6

    def test_energies_reference(self, deds_jedi_fresh, deds_ref):
        j = copy.deepcopy(deds_jedi_fresh)
        j.get_energies()
        np.testing.assert_allclose(np.array(j._energies), deds_ref["energies"], atol=1e-05)


class TestJediAnalysis:
    """
    Tests the jedi_analysis() function.
    """

    def test_energy_consistency(self, deds_analysis, deds_ref):
        assert abs(sum(deds_analysis["procERIMs"]) - 100.0) < 1e-3
        assert abs(deds_analysis["ERIMs_total"] - sum(deds_analysis["ERIMs"])) < 1e-05

    def test_e_rims_reference(self, deds_analysis, deds_ref):
        np.testing.assert_allclose(deds_analysis["ERIMs"], deds_ref["ERIMs"], atol=1e-05)

    def test_proc_e_rims_reference(self, deds_analysis, deds_ref):
        np.testing.assert_allclose(deds_analysis["procERIMs"], deds_ref["procERIMs"], atol=1e-05)

    def test_e_rims_linear_molecule(self, hcn_analysis, hcn_ref):
        """Tests B.ndim==1 edge case in jedi_analysis()"""
        np.testing.assert_allclose(hcn_analysis["ERIMs"], hcn_ref["ERIMs"], atol=1e-05)


class TestSetBondParams:
    """
    Tests the set_bond_params() function.
    """

    def test_default_bond_params(self, deds_jedi_fresh):
        j = copy.deepcopy(deds_jedi_fresh)
        assert j._covf == 1.3
        assert j._vdwf == 0.9

    def test_set_bond_params(self, deds_jedi_fresh):
        j = copy.deepcopy(deds_jedi_fresh)
        j.set_bond_params(covf=1.6, vdwf=0.75)
        assert j._covf == 1.6
        assert j._vdwf == 0.75


class TestVisualize:
    """
    Tests the visualize() method.
    """

    def test_visualization_data_keys(self, deds_jedi_fresh, tmp_path):
        """visualization_data should contain correct mode keys."""
        j = copy.deepcopy(deds_jedi_fresh)
        j.run(printout=False)
        j.visualize(output_dir=tmp_path, show=False)
        assert set(j.visualization_data.keys()) == {"bl", "ba", "da", "all"}

    def test_visualization_data_bond_structure(self, deds_jedi_fresh, tmp_path):
        """Each mode should have the expected bond_data structure."""
        j = copy.deepcopy(deds_jedi_fresh)
        j.run(printout=False)
        j.visualize(output_dir=tmp_path, single_mode="all", show=False)
        bond_data = j.visualization_data["all"]["bond_data"]
        assert "bonds" in bond_data
        assert "energies" in bond_data
        assert "custom_bonds" in bond_data
        assert "custom_energies" in bond_data
        assert "atoms" in bond_data

    def test_bonds_and_energies_same_length(self, deds_jedi_fresh, tmp_path):
        """Number of bonds and energies must match."""
        j = copy.deepcopy(deds_jedi_fresh)
        j.run(printout=False)
        j.visualize(output_dir=tmp_path, single_mode="all", show=False)
        bond_data = j.visualization_data["all"]["bond_data"]
        assert len(bond_data["bonds"]) == len(bond_data["energies"])
        assert len(bond_data["custom_bonds"]) == len(bond_data["custom_energies"])

    def test_color_data_structure(self, deds_jedi_fresh, tmp_path):
        """color_data should have the expected keys."""
        j = copy.deepcopy(deds_jedi_fresh)
        j.run(printout=False)
        j.visualize(output_dir=tmp_path, single_mode="all", show=False)
        color_data = j.visualization_data["all"]["color_data"]
        assert "atom_colors" in color_data
        assert "max_strain" in color_data
        assert "colormap" in color_data

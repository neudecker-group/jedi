import pytest

from strainjedi.jedi import Jedi, jedi_analysis
import numpy as np
import copy

class TestGetRims:
    """
    Tests for get_rims() and get_common_rims() methods.
    """
    def test_rim_list_shape(self, deds_jedi_with_rims):
        rim_list = deds_jedi_with_rims.rim_list
        assert len(rim_list) == 4
        assert rim_list[0].shape[1] == 2  # bonds
        assert rim_list[2].shape[1] == 3  # angles
        assert rim_list[3].shape[1] == 4  # dihedrals
    
    def test_custom_bonds_none_by_default(self, deds_jedi_fresh):
        j = copy.deepcopy(deds_jedi_fresh)
        assert j.custom_bonds is None

    def test_rim_list_no_custombonds(self, deds_jedi_with_rims):
        assert deds_jedi_with_rims.rim_list[1].shape[0] == 0

    def test_rim_list_custombonds(self, deds_jedi_fresh):
        j = copy.deepcopy(deds_jedi_fresh)
        j.add_custom_bonds([[0, 5],[2, 10]])
        j.indices = np.arange(0, len(j.atoms0))
        j.get_common_rims()
        assert j.rim_list[1].shape == (2, 2)
        assert list(j.rim_list[1][0]) == [0, 5]


class TestGetHessian:
    """
    Tests for get_hessian() method.
    """
    def test_hessian_reference(self, deds_jedi_fresh, deds_ref):
        j = copy.deepcopy(deds_jedi_fresh)
        j.get_hessian()
        np.testing.assert_allclose(j.H, deds_ref["jediInternalHessian"], atol=1e-05)


class TestGetBMatrix:
    """
    Test for get_b_matrix() method.
    """

    def test_B_matrix_reference(self, deds_jedi_fresh, deds_rim_list, deds_ref):
        j = copy.deepcopy(deds_jedi_fresh)
        j.rim_list = deds_rim_list
        j.get_b_matrix()
        np.testing.assert_allclose(j.B, deds_ref["bmatrix"], atol=1e-05)


class TestGetDeltaQ:
    """
    Tests for get_delta_q() method.
    """
    def test_delta_q_identical_structure(self, deds_opt, deds_vibdata):
        j = Jedi(deds_opt, deds_opt, deds_vibdata)
        j.indices= np.arange(len(deds_opt))
        j.get_common_rims()
        j.get_b_matrix()
        j.get_delta_q()
        np.testing.assert_allclose(j.delta_q, 0.0, atol=1e-05)

    def test_delta_q_deds_reference(self, deds_jedi_fresh, deds_rim_list, deds_ref):
        j = copy.deepcopy(deds_jedi_fresh)
        j.rim_list = deds_rim_list
        j.B = deds_ref["bmatrix"]
        j.get_delta_q()
        np.testing.assert_allclose(j.delta_q, deds_ref["delta_q"], atol=1e-05)


class TestGetEnergy:
    """
    Tests for get_energies() method.
    """
    def test_energy_difference(self, deds_jedi_fresh):
        j = copy.deepcopy(deds_jedi_fresh)
        j.get_energies()
        assert abs(j.energies[0] - (j.energies[1] - j.energies[2])) < 1e-6

    def test_energies_reference(self, deds_jedi_fresh, deds_ref):
        j = copy.deepcopy(deds_jedi_fresh)
        j.get_energies()
        np.testing.assert_allclose(np.array(j.energies), deds_ref["energies"], atol=1e-05)

class TestJediAnalysis:
    """
    Tests the jedi_analysis() function.
    """
    def test_energy_consistency(self, deds_analysis, deds_ref):
        assert abs(sum(deds_analysis["procERIMs"]) - 100.0) < 1e-3
        assert abs(deds_analysis["ERIMs_total"] - sum(deds_analysis["ERIMs"])) < 1e-05

    def test_e_rims_reference(self, deds_analysis, deds_ref):
        np.testing.assert_allclose(deds_analysis["ERIMs"], deds_ref["ERIMs"], atol = 1e-05)

    def test_proc_e_rims_reference(self, deds_analysis, deds_ref):
        np.testing.assert_allclose(deds_analysis["procERIMs"], deds_ref["procERIMs"], atol = 1e-05)

    def test_e_rims_linear_molecule(self, hcn_analysis, hcn_ref):
        """Tests B.ndim==1 edge case in jedi_analysis()"""
        np.testing.assert_allclose(hcn_analysis["ERIMs"], hcn_ref["ERIMs"], atol=1e-05)

    
class TestSetBondParams:
    """
    Tests the set_bond_params() function.
    """
    def test_default_bond_params(self, deds_jedi_fresh):
        j = copy.deepcopy(deds_jedi_fresh)
        assert j.covf == 1.3
        assert j.vdwf == 0.9

    def test_set_bond_params(self, deds_jedi_fresh):
        j = copy.deepcopy(deds_jedi_fresh)
        j.set_bond_params(covf=1.6, vdwf=0.75)
        assert j.covf == 1.6
        assert j.vdwf == 0.75

'''
class TestVMDGen:
    """
    TEMPORARY tests for the vmd_gen() function.
    Fragile as it compares strings.
    Should be replaced/removed once new default visualization is implemented!
    """
    def test_vmd_gen(self, deds_jedi_full_run, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)


        j = copy.deepcopy(deds_jedi_full_run)
        j.vmd_gen(label=tmp_path)

        from tests.resources import path_to_test_resources
        ref_dir = path_to_test_resources() / "diethyldisulfid"
        fl = ["bl.vmd", "ba.vmd", "da.vmd", "all.vmd"]

        for f in fl:
            gen = (tmp_path / f).read_text()
            ref = (ref_dir / f).read_text()

            gen_lines = gen.splitlines()
            ref_lines = ref.splitlines()

            gen_cmp = "\n".join(gen_lines[4:181] + gen_lines[182:])
            ref_cmp = "\n".join(ref_lines[4:181] + ref_lines[182:])
            assert gen_cmp == ref_cmp

'''

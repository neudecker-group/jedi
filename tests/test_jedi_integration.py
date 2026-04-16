import ase.io as io
import numpy as np
import pytest

print("NumPy config:", np.show_config())
from ase.vibrations.vibrations import VibrationsData

from strainjedi.jedi import Jedi
from tests.resources import path_to_test_resources


class TestJEDIHCN:
    def test_erims(self, hcn_jedi_full_run, hcn_ref):
        """Test if ERIMs are correct"""
        np.testing.assert_allclose(hcn_jedi_full_run.E_RIMs, hcn_ref["ERIMs"], atol=1e-05)

    def test_partial_erims(self, hcn_jedi_partial, hcn_ref):
        """Test if ERIMs are correct for a partial analysis"""
        np.testing.assert_allclose(hcn_jedi_partial.E_RIMs, hcn_ref["procERIMs"], atol=1e-05)


class TestJEDIdiethyldisulfid:
    """
    Diethyldisulfide: Relaxed and distorted with B3LYP/6-31G* in QChem optimized.
    Distorted with CoGEF on atom 4 and 6 (terminal carbon atoms) with set
    atom-distance of 7.35 Ångström (Stretching of 0.79 Å).
    """

    def test_erims(self, deds_jedi_full_run, deds_ref):
        """Test if ERIMs are correct"""
        np.testing.assert_allclose(deds_jedi_full_run.E_RIMs, deds_ref["ERIMs"], atol=1e-05)

    def test_proc_erims(self, deds_jedi_full_run, deds_ref):
        """Test if proc_ERIMs are correct"""
        np.testing.assert_allclose(deds_jedi_full_run.proc_E_RIMs, deds_ref["procERIMs"], atol=1e-05)

    def test_delta_q(self, deds_jedi_full_run, deds_ref):
        """Test if delta_q are correct"""
        np.testing.assert_allclose(deds_jedi_full_run.delta_q, deds_ref["delta_q"], atol=1e-05)

    def test_energies(self, deds_jedi_full_run, deds_ref):
        """Test if energies are correct"""
        np.testing.assert_allclose(np.array(deds_jedi_full_run.energies), deds_ref["energies"], atol=1e-05)

    def test_hessian(self, deds_jedi_full_run, deds_ref):
        """Test if hessian inside of JEDI is correct (JEDI-hessian has transformed units)"""
        np.testing.assert_allclose(np.array(deds_jedi_full_run.energies), deds_ref["energies"], atol=1e-05)

    def test_b_matrix(self, deds_jedi_full_run, deds_ref):
        """Test if bmatrix are correct"""
        np.testing.assert_allclose(deds_jedi_full_run.B, deds_ref["bmatrix"], atol=1e-05)


class TestJediWarnings:
    @pytest.mark.parametrize("warning", ["broken_bond", "no_dihedral"])
    def test_rim_change_warning(self, warning):
        mol = io.read(path_to_test_resources() / "h2o2/h2o2.json")
        mol2 = io.read(path_to_test_resources() / f"h2o2/{warning}.json")
        hessian = VibrationsData.read(path_to_test_resources() / "h2o2/h2o2_hessian.json")
        with pytest.warns(UserWarning):
            _analysis = Jedi(mol, mol2, hessian)

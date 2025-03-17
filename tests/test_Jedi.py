import pytest

import ase.io as io
from ase.vibrations.vibrations import VibrationsData
import numpy as np
from strainjedi.jedi import Jedi
from tests.resources import path_to_test_resources


class TestJEDIHCN:

    @classmethod
    def setup_class(cls):
        mol = io.read(path_to_test_resources() / "hcn/opt.json")
        mol2 = io.read(path_to_test_resources() / "hcn/dis.json")
        hessian = VibrationsData.read(path_to_test_resources() / "hcn/modes.json")
        cls.hcn = Jedi(mol, mol2, hessian)
        cls.hcn.run()
        phessian = VibrationsData.from_2d(mol,
                                          np.loadtxt(path_to_test_resources() / "hcn/parthess"),
                                          indices=[2, 3, 5, 8, 9, 11])
        cls.phcn = Jedi(mol, mol2, phessian)
        cls.phcn.partial_analysis(indices=[2, 3, 5, 8, 9, 11])

    def test_run(cls):
        """Test if ERIMs are correct"""
        test = cls.hcn.E_RIMs.round(5)

        compare = np.loadtxt(path_to_test_resources() / "hcn/ERIMs").round(5)

        assert np.array_equal(test, compare)

    def test_partial(cls):
        """Test if ERIMs are correct"""
        test = cls.phcn.E_RIMs.round(5)
        compare = np.loadtxt(path_to_test_resources() / "hcn/pERIMs").round(5)
        assert np.array_equal(test, compare)


#Diethyldisulfide: Relaxed and distorted with B3LYP/6-31G* in QChem optimized. Distorted with CoGEF on atom 4 and 6 (terminal carbon atoms) with set atom-distance of 7.35 Ångström (Stretching of 0.79 Å).
class TestJEDIdiethyldisulfid:

    @classmethod
    def setup_class(cls):
        mol = io.read(path_to_test_resources() / "diethyldisulfid/opt.json")
        mol2 = io.read(path_to_test_resources() / "diethyldisulfid/dis.json")
        hessian = VibrationsData.read(path_to_test_resources() / "diethyldisulfid/modes.json")
        cls.diethyldisulfid = Jedi(mol, mol2, hessian)
        cls.diethyldisulfid.run()

    def test_run(cls):
        """Test if ERIMs are correct"""
        test = cls.diethyldisulfid.E_RIMs.round(5)
        compare = np.loadtxt(path_to_test_resources() / "diethyldisulfid/ERIMs").round(5)
        assert np.array_equal(test, compare)

    def test_run(cls):
        """Test if proc_ERIMs are correct"""
        test = cls.diethyldisulfid.proc_E_RIMs.round(5)
        compare = np.loadtxt(path_to_test_resources() / "diethyldisulfid/procERIMs").round(5)
        assert np.array_equal(test, compare)

    def test_run(cls):
        """Test if delta_q are correct"""
        test = cls.diethyldisulfid.delta_q.round(5)
        compare = np.loadtxt(path_to_test_resources() / "diethyldisulfid/delta_q").round(5)
        assert np.array_equal(test, compare)

    def test_run(cls):
        """Test if energies are correct"""
        test = cls.diethyldisulfid.energies.round(5)
        compare = np.loadtxt(path_to_test_resources() / "diethyldisulfid/energies").round(5)
        assert np.array_equal(test, compare)

    def test_run(cls):
        """Test if hessian inside of JEDI is correct (JEDI-hessian has transformed units)"""
        test = cls.diethyldisulfid.H.round(5)
        compare = np.loadtxt(path_to_test_resources() / "diethyldisulfid/jediInternalHessian").round(5)
        assert np.array_equal(test, compare)

    def test_run(cls):
        """Test if bmatrix are correct"""
        test = cls.diethyldisulfid.B.round(5)
        compare = np.loadtxt(path_to_test_resources() / "diethyldisulfid/bmatrix").round(5)
        assert np.array_equal(test, compare)

    def test_run(cls):
        """Test if indices are correct"""
        test = cls.diethyldisulfid.indices.round(5)
        compare = np.loadtxt(path_to_test_resources() / "diethyldisulfid/indices").round(5)
        assert np.array_equal(test, compare)
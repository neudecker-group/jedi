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



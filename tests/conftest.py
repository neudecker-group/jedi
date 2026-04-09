import pytest
from tests.resources import path_to_test_resources

from strainjedi.jedi import Jedi, jedi_analysis
from ase.io import read
from ase.vibrations.vibrations import VibrationsData
import numpy as np

"""Contains fixtures shared for all JEDI tests"""

########## ATOMS OBJECTS ############


# Diethyldisulfide
@pytest.fixture(scope="session")
def deds_opt():
    return read(path_to_test_resources() / "diethyldisulfid/opt.json")


@pytest.fixture(scope="session")
def deds_dist():
    return read(path_to_test_resources() / "diethyldisulfid/dis.json")


@pytest.fixture(scope="session")
def deds_vibdata():
    return VibrationsData.read(path_to_test_resources() / "diethyldisulfid/modes.json")


@pytest.fixture(scope="session")
def deds_rim_list():
    ref_file = np.load(path_to_test_resources() / "diethyldisulfid/rim_list.npz")
    return [
        ref_file["bonds"],
        ref_file["custom"],
        ref_file["angles"],
        ref_file["dihedrals"],
    ]


# HCN
@pytest.fixture(scope="session")
def hcn_opt():
    return read(path_to_test_resources() / "hcn/opt.json")


@pytest.fixture(scope="session")
def hcn_dist():
    return read(path_to_test_resources() / "hcn/dis.json")


@pytest.fixture(scope="session")
def hcn_vibdata():
    return VibrationsData.read(path_to_test_resources() / "hcn/modes.json")


@pytest.fixture(scope="session")
def hcn_rim_list():
    ref_file = np.load(path_to_test_resources() / "hcn/rim_list.npz")
    return [
        ref_file["bonds"],
        ref_file["custom"],
        ref_file["angles"],
        ref_file["dihedrals"],
    ]


########## REFERENCE DATA ###########


# Diethyldisulfide
@pytest.fixture(scope="session")
def deds_ref():
    deds_path = path_to_test_resources() / "diethyldisulfid"
    return {
        "ERIMs": np.loadtxt(deds_path / "ERIMs"),
        "procERIMs": np.loadtxt(deds_path / "procERIMs"),
        "delta_q": np.loadtxt(deds_path / "delta_q"),
        "energies": np.loadtxt(deds_path / "energies"),
        "jediInternalHessian": np.loadtxt(deds_path / "jediInternalHessian"),
        "bmatrix": np.loadtxt(deds_path / "bmatrix"),
    }


# HCN
@pytest.fixture(scope="session")
def hcn_ref():
    hcn_path = path_to_test_resources() / "hcn"
    return {
        "ERIMs": np.loadtxt(hcn_path / "ERIMs"),
        "procERIMs": np.loadtxt(hcn_path / "pERIMs"),
        "parthess": np.loadtxt(hcn_path / "parthess"),
        "bmatrix": np.loadtxt(hcn_path / "bmatrix"),
        "jediInternalHessian": np.loadtxt(hcn_path / "jediInternalHessian"),
        "energies": np.loadtxt(hcn_path / "energies"),
        "delta_q": np.loadtxt(hcn_path / "delta_q"),
    }


########## JEDI INSTANCES ###########
@pytest.fixture(scope="session")
def deds_jedi_fresh(deds_opt, deds_dist, deds_vibdata):
    """Fresh Jedi instance with only __init__."""
    return Jedi(deds_opt, deds_dist, deds_vibdata)


@pytest.fixture(scope="session")
def deds_jedi_with_rims(deds_opt, deds_dist, deds_vibdata):
    """Jedi instance with get_common_rims() executed"""
    j = Jedi(deds_opt, deds_dist, deds_vibdata)
    j.indices = np.arange(0, len(j.atoms0))
    j.get_common_rims()
    return j


@pytest.fixture(scope="session")
def deds_jedi_full_run(deds_opt, deds_dist, deds_vibdata):
    """Jedi instance after full run()"""
    j = Jedi(deds_opt, deds_dist, deds_vibdata)
    j.run(printout=False)
    return j


@pytest.fixture(scope="session")
def hcn_jedi_full_run(hcn_opt, hcn_dist, hcn_vibdata):
    """Jedi instance for HCN after full run()"""
    j = Jedi(hcn_opt, hcn_dist, hcn_vibdata)
    j.run(printout=False)
    return j


@pytest.fixture(scope="session")
def hcn_jedi_partial(hcn_opt, hcn_dist, hcn_ref):
    """Jedi instance for HCN after partial_analysis()."""
    parthess = VibrationsData.from_2d(
        hcn_opt, hcn_ref["parthess"], indices=[2, 3, 5, 8, 9, 11]
    )
    j = Jedi(hcn_opt, hcn_dist, parthess)
    j.partial_analysis(indices=[2, 3, 5, 8, 9, 11])
    return j


########### RESULTS OF JEDI_ANALYSIS() ############
@pytest.fixture(scope="session")
def deds_analysis(deds_opt, deds_rim_list, deds_ref):
    proc_E_RIMs, E_RIMs, E_RIMs_total, proc_geom_RIMs, delta_q = jedi_analysis(
        deds_opt,
        deds_rim_list,
        deds_ref["bmatrix"],
        deds_ref["jediInternalHessian"],
        deds_ref["delta_q"],
        deds_ref["energies"][0],
        printout=False,
    )
    return {
        "procERIMs": proc_E_RIMs,
        "ERIMs": E_RIMs,
        "ERIMs_total": E_RIMs_total,
        "proc_geom_RIMs": proc_geom_RIMs,
    }


@pytest.fixture(scope="session")
def hcn_analysis(hcn_opt, hcn_rim_list, hcn_ref):
    proc_E_RIMs, E_RIMs, E_RIMs_total, proc_geom_RIMs, delta_q = jedi_analysis(
        hcn_opt,
        hcn_rim_list,
        hcn_ref["bmatrix"],
        hcn_ref["jediInternalHessian"],
        hcn_ref["delta_q"],
        hcn_ref["energies"][0],
        printout=False,
    )
    return {
        "procERIMs": proc_E_RIMs,
        "ERIMs": E_RIMs,
        "ERIMs_total": E_RIMs_total,
        "proc_geom_RIMs": proc_geom_RIMs,
    }

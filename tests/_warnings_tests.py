import strainjedi.jedi as jedi
from ase.vibrations import VibrationsData
import ase.io
from tests.resources import path_to_test_resources

mol = ase.io.read(path_to_test_resources() / "h2o2/h2o2.json")
hessian = VibrationsData.read(path_to_test_resources() / "h2o2/h2o2_hessian.json")

mol2= ase.io.read(path_to_test_resources() / "h2o2/broken_bond.json")
jedianalysis = jedi.Jedi(mol, mol2, hessian)
jedianalysis.run()

mol3 = ase.io.read(path_to_test_resources() / "h2o2/no_dihedral.json")
jedianalysis = jedi.Jedi(mol, mol3, hessian)
jedianalysis.run()

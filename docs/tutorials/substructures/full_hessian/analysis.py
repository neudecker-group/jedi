import numpy as np
from ase.io.vasp import read_vasp_xml
from ase.vibrations import VibrationsData

from strainjedi import Jedi

mol = next(read_vasp_xml("output/opt.xml"))
mol2 = next(read_vasp_xml("output/dist.xml"))

hessian = VibrationsData.from_2d(mol, np.loadtxt("output/hessian"))

jedi = Jedi(mol, mol2, hessian)
jedi.run(indices=[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16])
jedi.visualize(show=True, show_indices=True)

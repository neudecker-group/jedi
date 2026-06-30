import numpy as np
from ase.constraints import FixAtoms
from ase.io.vasp import read_vasp_xml
from ase.vibrations import VibrationsData

from strainjedi import Jedi

ignored_atoms = FixAtoms(indices=[6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21])

mol = next(read_vasp_xml("output/opt.xml"))
mol.set_constraint(ignored_atoms)

mol2 = next(read_vasp_xml("output/dist.xml"))
mol.set_constraint(ignored_atoms)

interesting_atoms = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16]

hessian = VibrationsData.from_2d(mol, np.loadtxt("output/partial_hessian"), indices=interesting_atoms)

jedi = Jedi(mol, mol2, hessian)
jedi.partial_analysis(indices=interesting_atoms)
jedi.visualize(show=True, show_indices=True)

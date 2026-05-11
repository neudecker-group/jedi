from strainjedi.jedi import Jedi
from dipole import get_hbonds
import ase.io
from ase.vibrations.vibrations import VibrationsData
import numpy as np
from ase.visualize import view

mol=ase.io.read('opt.json')
modes=VibrationsData.read('modes.json')
mol_permutated=ase.io.read('opt_permutated.json')
modes_permutated=VibrationsData.read('modes_permutated.json')
mol2=ase.io.read('sp.json')
partmodes=VibrationsData.from_2d(mol,np.loadtxt('parthess'),indices=[2,3,5,8,9,11])

# view(mol)

jalldipole=Jedi(mol,mol2,modes)
jalldipole.add_custom_bonds(get_hbonds(mol))
jalldipole.run()

jspecialdipole=Jedi(mol,mol2,modes)
jspecialdipole.add_custom_bonds(get_hbonds(mol))
jspecialdipole.run(indices=[2,3,5,8,9,11])

jpartdipole=Jedi(mol,mol2,partmodes)
jpartdipole.add_custom_bonds(get_hbonds(mol))
jpartdipole.partial_analysis(indices=[2,3,5,8,9,11])

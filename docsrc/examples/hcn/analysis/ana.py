

from jedi.jedi import Jedi
from dipole import get_hbonds
import ase.io
from ase.vibrations.vibrations import VibrationsData
import os
import numpy as np
from ase.visualize import view
mol=ase.io.read('opt.json')
modes=VibrationsData.read('modes.json')

mol2=ase.io.read('sp.json')
jall=Jedi(mol,mol2,modes)
os.chdir('all')
jall.run()
jall.vmd_gen()

os.chdir('..')
jalldipole=Jedi(mol,mol2,modes)
jalldipole.add_custom_bonds(get_hbonds(mol))
os.chdir('alldipole')
jalldipole.run()
jalldipole.vmd_gen()

os.chdir('..')
modes=VibrationsData.from_2d(mol,np.loadtxt('parthess'),indices=[2,3,5,8,9,11])
jpart=Jedi(mol,mol2,modes)
os.chdir('part')
jpart.partial_analysis(indices=[2,3,5,8,9,11])
jpart.vmd_gen()

os.chdir('..')
jpartdipole=Jedi(mol,mol2,modes)
jpartdipole.add_custom_bonds(get_hbonds(mol))
os.chdir('partdipole')
jpartdipole.partial_analysis(indices=[2,3,5,8,9,11])
jpartdipole.vmd_gen()

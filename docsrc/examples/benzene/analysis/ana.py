from jedi.jedi import Jedi
from ase.vibrations.vibrations import VibrationsData
import ase.io
import numpy as np
import os

mol= ase.io.read('opt.json')
mol2= ase.io.read('6_6_6.json')
mol3= ase.io.read('8_8_8.json')
modes=VibrationsData.read('modes.json')

os.chdir('6_6_6')
j=Jedi(mol,mol2,modes)
j.run()
j.vmd_gen(modus='all',man_strain=0.655)


os.chdir('../../8_8_8')
j=Jedi(mol,mol3,modes)
j.run()
j.vmd_gen(modus='all',man_strain=0.655)
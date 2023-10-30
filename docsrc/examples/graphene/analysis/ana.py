from jedi.jedi import Jedi
from ase.vibrations.vibrations import VibrationsData
import ase.io
import numpy as np


mol= ase.io.read('opt.json')
mol2= ase.io.read('x-0_1.json')
modes=VibrationsData.read('modes.json')

j=Jedi(mol,mol2,modes)
j.run()
j.vmd_gen()
from jedi.jedi import Jedi
from ase.vibrations.vibrations import VibrationsData
import ase.io
import numpy as np
import os

mol= ase.io.read('opt.json')
mol2= ase.io.read('para-C-H.json')
modes=VibrationsData.from_2d(mol,np.loadtxt('hessian'))
partmodes=VibrationsData.from_2d(mol[0,1,2,3,4,5,12,13,14,15,16],np.loadtxt('p-hessian'))
j=Jedi(mol,mol2,modes)
os.chdir('all')
j.run()
j.vmd_gen()
os.chdir('../../partial')
jpart=Jedi(mol,mol2,partmodes)
jpart.partial_analysis(indices=[0,1,2,3,4,5,12,13,14,15,16])
jpart.vmd_gen()

os.chdir('../../special')
jspecial=Jedi(mol,mol2,modes)
jspecial.run(indices=[0,1,2,3,4,5,12,13,14,15,16])
jspecial.vmd_gen()
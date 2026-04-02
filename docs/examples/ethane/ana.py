import ase.io
from strainjedi.jedi import Jedi

from strainjedi.io.orca import get_vibrations

mol=ase.io.read('opt.json')
mol2=ase.io.read('force.json')
modes=get_vibrations('out/orcafreq',mol)#VibrationsData.from_2d(mol,np.loadtxt('hes'))
j=Jedi(mol,mol2,modes)


j.run()
j.vmd_gen()
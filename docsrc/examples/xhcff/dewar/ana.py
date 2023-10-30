from jedi.jedi import Jedi
import ase.io
from ase.vibrations.vibrations import VibrationsData


mol=ase.io.read('opt.json')
modes=VibrationsData.read('freq.json')
mol2=ase.io.read('force.json')
j=Jedi(mol,mol2,modes)

j.run()

j.vmd_gen()
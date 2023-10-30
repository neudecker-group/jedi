import ase.io
from ase.vibrations.vibrations import VibrationsData
from jedi.jedi import Jedi
from jedi.jedi import get_hbonds
from jedi.io.gaussian import get_vibrations,read_gaussian_out

file=open('output/opt.log')
mol=read_gaussian_out(file)
file2=open('output/dist.log')
mol2=read_gaussian_out(file2)
modes=get_vibrations('output/freq',mol)
j=Jedi(mol,mol2,modes)
j.add_custom_bonds(get_hbonds(mol2))

j.run()
j.vmd_gen()

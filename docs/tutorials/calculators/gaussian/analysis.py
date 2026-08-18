from strainjedi.io import read_output
from strainjedi.io.adapter import to_atoms, to_vibrations
from strainjedi.jedi import Jedi

mol = to_atoms(read_output("output/opt.log"))
mol2 = to_atoms(read_output("output/dist.log"))
hessian = to_vibrations(read_output("output/freq.log"), mol)

jedi = Jedi(mol, mol2, hessian)
jedi.run()
jedi.visualize(show=True)

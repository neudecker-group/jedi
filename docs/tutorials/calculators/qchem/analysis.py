from strainjedi.io import read_output
from strainjedi.io.adapter import to_atoms, to_vibrations
from strainjedi.jedi import Jedi

mol = to_atoms(read_output("output/opt.out"))
mol2 = to_atoms(read_output("output/dist.out"))
hessian = to_vibrations(read_output("output/freq.out"), mol)

jedi = Jedi(mol, mol2, hessian)
jedi.run()
jedi.visualize(show=True)

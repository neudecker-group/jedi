from strainjedi.jedi import Jedi
from strainjedi.io import read_output
from strainjedi.io.adapter import to_atoms, to_vibrations

mol = to_atoms(read_output("opt/OUTCAR"))
mol2 = to_atoms(read_output("dist/OUTCAR"))
hessian = to_vibrations(read_output("freq/OUTCAR"), mol)

jedi = Jedi(mol, mol2, hessian)
jedi.run()
jedi.visualize(show=True, show_indices=True)

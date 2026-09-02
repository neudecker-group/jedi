from strainjedi.jedi import Jedi
from strainjedi.io import read_output
from strainjedi.io.adapter import to_atoms, to_vibrations

mol = to_atoms(read_output("opt/OUTCAR"))
mol2 = to_atoms(read_output("dist/OUTCAR"))
substructure = [0,1,2,3,4,5,12,13,14,15,16]
part_hessian = to_vibrations(read_output("p-freq/OUTCAR"), mol, indices=substructure)

jedi = Jedi(mol, mol2, part_hessian)
jedi.partial_analysis(indices=substructure)
jedi.visualize(show=True, show_indices=True)

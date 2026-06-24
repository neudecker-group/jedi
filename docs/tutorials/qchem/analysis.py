from strainjedi.io.qchem import get_vibrations, read
from strainjedi.jedi import Jedi

mol = read("output/opt.out")
mol2 = read("output/dist.out")
hessian = get_vibrations("output/freq", mol)

jedi = Jedi(mol, mol2, hessian)
jedi.run()
jedi.visualize(show=True)

from strainjedi.io.gaussian import get_vibrations, read_gaussian_out
from strainjedi.jedi import Jedi

mol = read_gaussian_out("output/opt")
mol2 = read_gaussian_out("output/dist")
hessian = get_vibrations("output/freq", mol)

jedi = Jedi(mol, mol2, hessian)
jedi.run()
jedi.visualize(show=True)

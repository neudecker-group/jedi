import ase.io
from ase.io.jsonio import read_json

from strainjedi import Jedi
from strainjedi.utils import get_hbonds

mol = ase.io.read("opt.json")
mol2 = ase.io.read("sp.json")
vibdata = read_json("full_hessian.json")

jedi = Jedi(mol, mol2, vibdata)
jedi.add_custom_bonds(get_hbonds(mol, extra_hpartners=("C")))
jedi.run()
jedi.visualize(show=True, box=True, split_bonds=True, show_indices=True)

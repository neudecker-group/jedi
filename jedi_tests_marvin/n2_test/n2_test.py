from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations import Vibrations

n2 = Atoms('N2', [(0, 0, 0), (0, 0, 1.1)])

calc = EMT()
n2.set_calculator(calc)

BFGS(n2).run(fmax=0.01)

vib = Vibrations(n2)
vib.run()
hessian = vib.get_vibrations()

n21 = n2.copy()
n21.positions[1][2] = n2.positions[1][2]+0.1
n21.calc = EMT()
n21.get_potential_energy()

from strainjedi.jedi import Jedi

j = Jedi(n2, n21, hessian)
j.run()

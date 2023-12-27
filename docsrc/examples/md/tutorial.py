from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations import Vibrations
n2 = Atoms('N2', [(0, 0, 0), (0, 0, 1.1)],
            calculator=EMT())
BFGS(n2).run(fmax=0.01)

vib = Vibrations(n2)
vib.run()
modes = vib.get_vibrations()




from ase import units
from ase.io.trajectory import Trajectory

from ase.md.langevin import Langevin



T = 400  # Kelvin


atoms = n2.copy()

# Describe the interatomic interactions with the Effective Medium Theory
atoms.calc = EMT()

# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 5 fs, the temperature T and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(atoms, 5 * units.fs, T * units.kB, 0.002)


def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
          'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))


dyn.attach(printenergy, interval=50)

# We also want to save the positions of all atoms after every 100th time step.
traj = Trajectory('moldyn3.traj', 'w', atoms)
dyn.attach(traj.write, interval=4)

# Now run the dynamics
printenergy()
dyn.run(200)
import os

from jedi.jedi import Jedi
for i in range(1,51):
    j = Jedi(n2, Trajectory('moldyn3.traj')[i], modes)
    print(Trajectory('moldyn3.traj')[i].calc.get_potential_energy())
     
    j.run()

    j.vmd_gen(label=str(i), man_strain=0.3087887,modus='all')

=========================================
JEDI --- Judgement of Energy Distribution
=========================================

The JEDI analysis is a quantum chemical analysis tool for the distribution of strain in mechanically deformed systems.
It is implemented in the Python_ programming language, with support for Python 3.10 and upwards.
The code is made available under the Open Source `MIT License`_.

It is a goal to make this module useful for older Python versions as well, namely 3.8 and 3.9.
Contributions welcome!

JEDI provides a simple interface through the ``Jedi`` class. Here's a minimal example:

.. code-block:: python

   from ase import Atoms
   from ase.calculators.emt import EMT
   from ase.optimize import BFGS
   from ase.vibrations import Vibrations
   from strainjedi.jedi import Jedi
   
   # Create a structure
   atoms = Atoms('N2', [(0, 0, 0), (0, 0, 1.1)])
   
   # Set up calculator
   calc = EMT()
   atoms.set_calculator(calc)
   
   # Optimize geometry
   BFGS(atoms).run(fmax=0.01)
   
   # Calculate vibrational modes (Hessian)
   vib = Vibrations(atoms)
   vib.run()
   modes = vib.get_vibrations()
   
   # Create a distorted structure
   atoms_distorted = atoms.copy()
   atoms_distorted.positions[1][2] += 0.1
   atoms_distorted.set_calculator(EMT())
   atoms_distorted.get_potential_energy()
   
   # Run JEDI analysis
   jedi = Jedi(atoms, atoms_distorted, modes)
   jedi.run()

The analysis compares the relaxed structure with a mechanically deformed structure and distributes the strain energy across redundant internal coordinates (bonds, angles, dihedrals).

For more details, see the :ref:`full tutorial <tutorial>`.

.. _Python: https://www.python.org/
.. _MIT License: https://github.com/neudecker-group/jedi/blob/main/LICENSE

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   strainjedi/strainjedi

==================
Molecular Dynamics
==================

Performing a JEDI analysis on molecular dynamics simulations is a little more involved than the "classic" analysis we introduced over the previous pages.
To illustrate, we will --- like in the very first tutorial ---  use a nitrogen molecule and simulate it at 400 Kelvin.
For running the calculations, we will also use EMT for the same reasons: it is readily available and reasonably fast to run.

Create The Structure
====================

First, we will start by creating our nitrogen atom and perform a basic geometry optimisation.
Do note that this is to provide a self-contained example; as established previously, it is just as possible to use an :ref:`existing calculator <existing-calculators>`.

.. code-block:: python

   from ase import Atoms
   from ase.calculators.emt import EMT
   from ase.optimise import BFGS

   n2 = Atoms("N2", [(0, 0, 0), (0, 0, 1.1)])
   n2.calc = EMT()
   BFGS(n2).run(fmax=0.01)

Run The Simulation
==================

The ASE framework provides a very simple interface to run MD simulations and then store the entirety of that simulation.
Feel free to contribute examples for the calculator of your choice --- this makes JEDI more accessible to everyone.

.. code-block:: python

   from ase import units
   from ase.md.langevin import Langevin

   dist = n2.copy()
   dist.calc = EMT()

   temp = 400  # Kelvin
   # timestep of 5 fs, T = 400 K, friction coeff = 0.002 a.u.
   # save every 4th step in moldyn3.traj 
   dyn = Langevin(
       dist, timestep=5 * units.fs, temperature=temp * units.kB, friction=0.002, trajectory="moldyn3.traj", loginterval=4
   )

   # run the simulation for 200 timesteps.
   dyn.run(200)

This will store the results of the molecular dynamics simulation in ``moldyn3.traj``.
Now, we can read this ASE-specific format into JEDI.

Perform a JEDI Analysis
=======================

First, we will have to read in the trajectory file we created earlier.

.. code-block:: python

   from ase.io.trajectory import Trajectory

   traj = Trajectory("moldyn3.traj")

Then, also obtain the necessary inputs (the Hessian).

.. code-block:: python

   from ase.vibrations import Vibrations

   vib = Vibrations(n2)
   vib.run()
   hessian = vib.get_vibrations()

Finally, we can start a JEDI analysis for all time steps.
To avoid creating a lot (and we really mean a lot!) of visualisation windows, we will instead store them inside a ``moldyn_vis`` directory.
Also, let's tell the visualiser to only create figures pertaining to the bond length.

.. code-block:: python

   from strainjedi import Jedi

   for idx, atoms in enumerate(traj):
       print(f"performing JEDI analysis for step {idx}...", end=" ")
       j = Jedi(n2, atoms, hessian)
       j.run(printout=False)
       j.visualize(output_dir=f"moldyn_vis/{idx}", single_mode="bl")
       print("done!")

Putting it Together
===================

The full script, after some formatting work, should look like the following.
Please note, however, that *most* of this script is just boilerplate to get the example working in a self-contained manner.
The actual JEDI analysis is at the very end of this file.

.. code-block:: python
   :emphasize-lines: 39-44

   from ase import Atoms, units
   from ase.calculators.emt import EMT
   from ase.io.trajectory import Trajectory
   from ase.md.langevin import Langevin
   from ase.optimize import BFGS
   from ase.vibrations import Vibrations

   from strainjedi import Jedi

   # create an N2 molecule and optimise its geometry.
   n2 = Atoms("N2", [(0, 0, 0), (0, 0, 1.1)])
   n2.calc = EMT()
   BFGS(n2).run(fmax=0.01)

   dist = n2.copy()
   dist.calc = EMT()

   # timestep of 5 fs, T = 400 K, friction coeff = 0.002 a.u.
   # save every 4th step in moldyn3.traj
   dyn = Langevin(
       dist,
       timestep=5 * units.fs,
       temperature_K=400,
       friction=0.002,
       trajectory="moldyn3.traj",
       loginterval=4,
       fixcm=False,
   )

   # run the simulation for 200 timesteps.
   dyn.run(200)
   traj = Trajectory("moldyn3.traj")

   # obtain the Hessian
   vib = Vibrations(n2)
   vib.run()
   hessian = vib.get_vibrations()

   for idx, atoms in enumerate(traj):
       print(f"performing JEDI analysis for step {idx}...", end=" ")
       j = Jedi(n2, atoms, hessian)
       j.run(printout=False)
       j.visualize(output_dir=f"moldyn_vis/{idx}", single_mode="bl")
       print("done!")


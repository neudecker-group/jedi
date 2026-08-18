.. _existing-calculators:

============================
Reading Existing Calculators
============================

On the previous page, we created a nitrogen molecule programmatically by hand, entirely within our script.
In practise, however, JEDI is often used with geometries and Hessians obtained from external quantum-chemical programs.
Hence, on this page, we will cover a few calculators and how to read their output into JEDI.

Before We Start
===============

JEDI currently provides helpers to read in Gaussian, ORCA, and Q-Chem output files.
If the calculator of your choice is missing, please let us know!

You will, of course, need the output of both geometry optimisations and the frequency analysis of the relaxed structure.
Refer to your calculator's manual on how to obtain those.
Two of the three need to be asked for the Hessian explicitly:

* **Gaussian** --- add ``iop(7/33=1)`` to the route section, or run ``formchk`` to produce a ``.fchk``.
* **Q-Chem** --- add ``vibman_print 7`` to the ``$rem`` section.
* **ORCA** --- writes the Hessian to a ``.hess`` file automatically.

If you forget, JEDI will tell you which keyword is missing rather than failing obscurely.

For the below examples, we will assume your output files are all in an ``output/`` directory next to your script.

One Reader For Every Program
============================

A single pair of functions covers all three programs.
:func:`strainjedi.io.read_output` works out which program wrote a file by looking inside it, so you do not have to say::

   from strainjedi.io import read_output
   from strainjedi.io.adapter import to_atoms, to_vibrations

Note that you pass the **path of the file that actually holds the data**, extension included.
Which file that is differs by program, because ORCA keeps its Hessian somewhere else entirely:

=============  ====================================  ===================================
Program        Geometry and energy                   Hessian
=============  ====================================  ===================================
ORCA           ``opt.out``                           ``freq.hess``
Gaussian       ``opt.log``                           ``freq.fchk``, else ``freq.log``
Q-Chem         ``opt.out``                           ``freq.out``
=============  ====================================  ===================================

Reading The Structures
======================

First, we will read in the structures --- for the purposes of these examples, we store the relaxed structure as ``opt``, and the distorted structure as ``dist``, plus the respective file name extension.

.. tabs::

   .. code-tab:: python Gaussian

      mol = to_atoms(read_output("output/opt.log"))
      mol2 = to_atoms(read_output("output/dist.log"))

   .. code-tab:: python ORCA

      mol = to_atoms(read_output("output/opt.out"))
      mol2 = to_atoms(read_output("output/dist.out"))

   .. code-tab:: python Q-Chem

      mol = to_atoms(read_output("output/opt.out"))
      mol2 = to_atoms(read_output("output/dist.out"))

Reading The Hessian
===================

As discussed earlier, JEDI also needs a Hessian to reason about which structural elements have higher significance when judging strain distribution.
The Hessian is the result of a frequency analysis, which we will save under ``freq``.

Pass ``mol`` along as well, so the Hessian is attached to the optimised structure rather than to whichever geometry happened to sit in the frequency file.

.. tabs::

   .. code-tab:: python Gaussian

      hessian = to_vibrations(read_output("output/freq.log"), mol)

   .. code-tab:: python ORCA

      hessian = to_vibrations(read_output("output/freq.hess"), mol)

   .. code-tab:: python Q-Chem

      hessian = to_vibrations(read_output("output/freq.out"), mol)

Putting it Together
===================

From here on, starting a JEDI analysis is just as straightforward as introduced: Construct the Jedi object, call ``run()`` and, optionally, visualise the result.

.. tabs::

   .. code-tab:: python Gaussian

      from strainjedi.jedi import Jedi
      from strainjedi.io import read_output
      from strainjedi.io.adapter import to_atoms, to_vibrations

      mol = to_atoms(read_output("output/opt.log"))
      mol2 = to_atoms(read_output("output/dist.log"))
      hessian = to_vibrations(read_output("output/freq.log"), mol)

      jedi = Jedi(mol, mol2, hessian)
      jedi.run()
      jedi.visualize(show=True)

   .. code-tab:: python ORCA

      from strainjedi.jedi import Jedi
      from strainjedi.io import read_output
      from strainjedi.io.adapter import to_atoms, to_vibrations

      mol = to_atoms(read_output("output/opt.out"))
      mol2 = to_atoms(read_output("output/dist.out"))
      hessian = to_vibrations(read_output("output/freq.hess"), mol)

      jedi = Jedi(mol, mol2, hessian)
      jedi.run()
      jedi.visualize(show=True)

   .. code-tab:: python Q-Chem

      from strainjedi.jedi import Jedi
      from strainjedi.io import read_output
      from strainjedi.io.adapter import to_atoms, to_vibrations

      mol = to_atoms(read_output("output/opt.out"))
      mol2 = to_atoms(read_output("output/dist.out"))
      hessian = to_vibrations(read_output("output/freq.out"), mol)

      jedi = Jedi(mol, mol2, hessian)
      jedi.run()
      jedi.visualize(show=True)

Working Without ASE
===================

``read_output`` itself does not use ASE at all.
It returns a :class:`strainjedi.io.QCOutput`: plain NumPy arrays in **atomic units** --- positions in Bohr, energy in Hartree, Hessian in Hartree/Bohr². ::

   out = read_output("output/freq.hess")
   out.numbers      # (N,) atomic numbers
   out.positions    # (N, 3) Bohr
   out.hessian      # (3N, 3N) Hartree/Bohr², symmetric
   out.masses       # (N,) amu, as the program itself reported them
   out.energy       # Hartree, or None

``to_atoms`` and ``to_vibrations`` are the only step that converts to ASE's eV and Ångström.
If you want the numbers rather than the ASE objects, skip them.

.. _calculators-deprecated:

Older Scripts
=============

The previous per-program helpers still work, but emit a :class:`DeprecationWarning`:

=========================================================  ==================================================
Old                                                        New
=========================================================  ==================================================
``strainjedi.io.orca.read(f)``                             ``to_atoms(read_output(f))``
``strainjedi.io.gaussian.read_gaussian_out(label)``        ``to_atoms(read_output(label + ".log"))``
``strainjedi.io.qchem.read(f)``                            ``to_atoms(read_output(f))``
``strainjedi.io.<prog>.get_vibrations(label, mol)``        ``to_vibrations(read_output(path), mol)``
=========================================================  ==================================================

The calculator classes used to run these programs from ASE moved from ``strainjedi.io`` to
:mod:`strainjedi.calculators`.

Downloads
=========

To run these examples, feel free to download the archive below; in there, you will find three folders for each respective calculator, containing the relevant files.

:download:`Calculator examples <../_static/downloads/calculators.zip>`

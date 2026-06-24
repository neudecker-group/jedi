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

For the below examples, we will assume your output files are all in an ``output/`` directory next to your script.

Reading The Structures
======================

First, we will read in the structures --- for the purposes of these examples, we store the relaxed structure as ``opt``, and the distorted structure as ``dist``, plus the respective file name extension.

.. tabs::

   .. code-tab:: python Gaussian
      
      from strainjedi.io.gaussian import read_gaussian_out

      mol = read_gaussian_out("output/opt")
      mol2 = read_gaussian_out("output/dist")

   .. code-tab:: python ORCA

      from strainjedi.io.orca import read

      mol = read("output/opt.out")
      mol2 = read("output/dist.out")

   .. code-tab:: python Q-Chem

      from strainjedi.io.qchem import read

      mol = read("output/opt.out")
      mol2 = read("output/dist.out")

It is somewhat regrettable that the Gaussian helper is named differently.
In the future, we may provide a unified way for reading output files with automatic calculator detection.

Reading The Hessian
===================

As discussed earlier, JEDI also needs a Hessian to reason about which structural elements have higher significance when judging strain distribution.
The Hessian is the result of a frequency analysis, which we will save under ``freq`` (again, with the respective file name extension).

.. tabs::

   .. code-tab:: python Gaussian
      
      from strainjedi.io.gaussian import get_vibrations

      hessian = get_vibrations("output/freq")

   .. code-tab:: python ORCA

      from strainjedi.io.orca import get_vibrations
      
      hessian = get_vibrations("output/freq")


   .. code-tab:: python Q-Chem

      from strainjedi.io.qchem import get_vibrations

      hessian = get_vibrations("output/freq")

Putting it Together
===================

From here on, starting a JEDI analysis is just as straightforward as introduced: Construct the Jedi object, call ``run()`` and, optionally, visualise the result.

.. tabs::

   .. code-tab:: python Gaussian
      
      from strainjedi.jedi import Jedi
      from strainjedi.io.gaussian import read_gaussian_out, get_vibrations

      mol = read_gaussian_out("output/opt")
      mol2 = read_gaussian_out("output/dist")
      hessian = get_vibrations("output/freq", mol)

      jedi = Jedi(mol, mol2, hessian)
      jedi.run()
      jedi.visualize(show=True)

   .. code-tab:: python ORCA

      from strainjedi.jedi import Jedi
      from strainjedi.io.orca import read, get_vibrations
      
      mol = read("output/opt.out")
      mol2 = read("output/dist.out")
      hessian = get_vibrations("output/freq", mol)

      jedi = Jedi(mol, mol2, hessian)
      jedi.run()
      jedi.visualize(show=True)

   .. code-tab:: python Q-Chem

      from strainjedi.jedi import Jedi
      from strainjedi.io.qchem import read, get_vibrations

      mol = read("output/opt.out")
      mol2 = read("output/dist.out")
      hessian = get_vibrations("output/freq", mol)

      jedi = Jedi(mol, mol2, hessian)
      jedi.run()
      jedi.visualize(show=True)

Downloads
=========

To run these examples, feel free to download the archive below; in there, you will find three folders for each respective calculator, containing the relevant files.

:download:`Calculator examples <../_static/downloads/calculators.zip>`

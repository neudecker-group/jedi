=======================
Analysing Substructures
=======================

When analyzing large structures, but you're really only interested in small parts of the whole, it might be more useful to instead resort to a partial analysis.

We will use the biphenyl molecule to illustrate a partial analysis.
To start, there are two approaches to be considered:

#. You performed a frequency analysis on the entire structure, or
#. you did so only on parts of the molecule to save computational resources

Full Hessian
============

Assuming you performed a frequency analysis on the entire structure, but are only interested in parts thereof, you should pick this approach.

.. code-block:: python

   from strainjedi.jedi import Jedi
   from strainjedi.io import read_output
   from strainjedi.io.adapter import to_atoms, to_vibrations

   mol = to_atoms(read_output("opt/OUTCAR"))
   mol2 = to_atoms(read_output("dist/OUTCAR"))
   hessian = to_vibrations(read_output("freq/OUTCAR"), mol)

   jedi = Jedi(mol, mol2, hessian)
   jedi.run()
   jedi.visualize(show=True, show_indices=True)

Running this script opens our trusted visualization window as shown below.

.. image:: biphenyl_full.png
   :align: center
   :alt: JEDI visualization of the analysis of a biphenyl molecule. All the bonds are color-coded, indicating an analysis of the full structure.

As all bonds are colored, you can see that JEDI still analyzed the entire structure.
Let's fix that by providing the indices we are actually interested in, which is the benzene ring with the strained C-H bond.
From the image, it's clear that the atoms with the indices 6 to 11, and 17 to 21 do not (significantly) contribute to the strain, so we will ignore those.

Therefore will modify the ``run()`` call as following, keeping in mind to provide the indices we are *interested in* (as opposed to the ones we do not care about).

.. code-block:: python

   jedi.run(indices=[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16])

And execute the script again.

.. image:: biphenyl_partial.png
   :align: center
   :alt: JEDI visualization of the analysis of a biphenyl molecule. One benzene ring is colored black, indicating that those bonds were not considered during analysis.

The unanalyzed part of the structure is now indicated by black bonds.

.. _partial-hessian:

Partial Hessian
===============

JEDI also supports the result of a partial frequency analysis, though such an analysis may not be started by calling ``run()`` on your Jedi object.
Instead, we must explicitly call ``partial_analysis()``, with indices similar to how we performed an analysis with the full Hessian.
Furthermore, also the ``to_vibrations()`` function needs the atom indices to extract the partial hessian.

.. code-block:: python

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


Conclusion
==========

To summarise on which approach to pick: if you have a full frequency analysis, call ``run()`` as the "normal" entry point.
In case of a partial frequency analysis, use ``partial_analysis()``.
Either Jedi functions expect indices of the atoms you are interested in.

.. _setting constraints: https://docs.ase-lib.org/ase/constraints.html

.. _substr_download:

Downloads
=========

To follow along in these examples, feel free to download the archive listed below.
The calculations were performed using the VASP calculator.

:download:`Substructure examples <../_static/downloads/substructures.zip>`

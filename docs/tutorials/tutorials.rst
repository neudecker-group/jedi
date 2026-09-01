=========
Tutorials
=========

In this section, we will cover some examples on what JEDI is capable of and how it can be used.

For running the examples, you need to have the ``strainjedi`` package installed.
Everything else that JEDI needs, such as matplotlib and ASE, will be installed automatically as dependencies.
See :ref:`the previous page <install>` for installation instructions.

We assume some basic familiarity with Python, but will build gradually toward more advanced examples.
If you have suggestions for improvements or would like to contribute additional examples, do :ref:`feel free to contribute <contribute>`!

Input Data
==========

Before you can run a JEDI analysis, you need to provide three inputs:

#. The geometry and energy of the relaxed molecule, obtained from a standard quantum-chemical geometry optimization.
#. The geometry and energy of a mechanically deformed structure based on the relaxed geometry.
#. The Hessian matrix at the relaxed geometry, obtained from a standard frequency calculation.

Using these inputs, JEDI performs the required matrix transformations and calculations to determine the strain distribution.
In addition, JEDI computes the strain energy within the harmonic approximation.
To assess the validity of this approximation, the program compares the harmonic strain energy with the *ab initio* energy difference between the relaxed and deformed geometries.
The deviation between these values provides a measure of the accuracy of the harmonic approximation.

All input data must be generated using the ASE framework. Molecular geometries are provided as ASE ``Atoms`` objects, while the Hessian is supplied through the ``VibrationsData`` class.
Do note that ASE supports `a wide range of calculators <https://ase-lib.org/ase/calculators/calculators.html#supported-calculators>`_.
We will explore how to obtain a valid ``Atoms`` and ``VibrationsData`` object from some sample logfiles of selected calculators.

.. caution::

   JEDI requires a Hessian corresponding to a minimum on the potential energy surface.
   Hence, structures containing one or more imaginary frequences are not supported.
   Before running a JEDI analysis, ensure that the optimized geometry is free of imaginary frequencies.


.. toctree::
   :maxdepth: 1

   nitrogen
   calculators
   custom-bonds
   substructures
   periodic-boundary-conditions
   molecular-dynamics
   vmd
   json_io

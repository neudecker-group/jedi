========================================================================
Welcome to JEDI, the Judgement of Energy Distribution Analysis
========================================================================

The JEDI Analysis (Judgement of Energy Distribution Analysis) is a quantum chemical strain analysis tool for the distribution of strain
mechanically deformed molecules. JEDI is developed in Python. Currently, the main contributor to JEDI is the `Institute for physical and theoretical Chemistry 
<https://www.uni-bremen.de/institut-fuer-physikalische-und-theoretische-chemie>`_, 
respectively the `AG Neudecker <https://www.uni-bremen.de/ag-neudecker/>`_ of the University of Bremen (https://www.uni-bremen.de/ag-neudecker). 

Introduction and Overview
=========================
Based on the harmonic approximation, the JEDI Analysis calculates the strain energy for each bond, bending and torsion in a molecule, thus allowing the
identification of the mechanically most strained regions in a molecule as well as the rationalization of mechanochemical processes.


Usage
======

JEDI can be used in three different applications.

Mechanically deformed Molecules
-------------------------------
When a molecule is stretched, some internal modes store more energy than others. 
This leads to particularly large displacements of certain modes and to the preconditioning 
of selected bonds for rupture. Using the JEDI analysis the mechanochemical properties can be investigated.

Excited State 
-------------
Besides the description of mechanical deformation in the ground state, the JEDI
analysis can be used in the electronically excited state to quantify the energy gained by
relaxation on the excited state potential energy surface (PES). For this, the harmonic
approximation needs to be applicable on the excited state PES of interest. The physical
process that is described by the excited state JEDI analysis is fundamentally different
from the ground state variant. While in the ground state JEDI analysis the distribution of
stress energy in a mechanically deformed molecule is analyzed, i.e. energy is expended for
deformation, the excited state JEDI analysis quantifies the energy gained by the relaxation
of each internal mode upon relaxation on the excited state PES, i.e. energy becomes
available.


Adsorbed molecules
------------------
When a molecule adsorbs onto a surface, the geometry of the adsorbate changes as compared 
to the relaxed molecule in the gas phase. As JEDI allows force analysis for only a 
subset of atoms, it is possible to quantify the strain build up within an adsorbtion. 

Literature
=============


* T. Stauch, A. Dreuw, J. Chem. Phys. 140, 134107 (2014)

* T. Stauch, A. Dreuw, J. Chem. Phys. 143, 074118 (2015)

* T. Stauch, A. Dreuw, Acc. Chem. Res. 50, 1041-1048 (2017)

* T. Stauch, A. Dreuw, J. Chem. Phys. 7, 1298-1302 (2016)


Documentation
=============

.. toctree::
   :maxdepth: 1
   :caption: Set Up:

   setup



.. toctree::
   :maxdepth: 1
   :caption: User Guide

   preparation/preparation
   examples/tutorial
   examples/examples
    

.. toctree::
   :maxdepth: 1
   :caption: Modules

   jedi
   utils/utils




Indices and tables
==================

* :ref:`genindex`



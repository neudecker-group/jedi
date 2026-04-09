========================================================================
Welcome to JEDI, the Judgement of Energy Distribution Analysis
========================================================================

The JEDI (Judgement of Energy DIstribution) analysis is a quantum chemical analysis tool for the distribution of strain mechanically deformed systems. JEDI is implemented in Python. Currently, the main contributor to JEDI is the `AG Neudecker <https://www.uni-bremen.de/en/neudecker-group>`_ from the University of Bremen. 

Introduction and Overview
=========================
Based on the harmonic approximation, the JEDI analysis calculates the strain energy for each bond, bending and torsion in a molecule or an extended system, thus allowing the identification of the mechanically most strained regions in the system as well as the rationalization of mechanochemical processes.


Usage
======

JEDI can be used in various application scenarios, some of which are:

Mechanically Deformed Molecules
-------------------------------
When a molecule or a periodic system is deformed, e.g., due to mechanical stretching or hydrostatic compression, some internal coordinates store more energy than others. This leads to particularly large displacements of certain coordinates such as the stretching of bond lengths, and to the preconditioning of selected bonds for rupture. Using the JEDI analysis the mechanochemical properties of the system can be investigated.


Adsorbed Molecules
------------------
When a molecule adsorbs onto a surface, the geometry of the adsorbate changes as compared to the relaxed molecule in the gas phase. As JEDI allows strain analysis for only a subset of atoms, it is possible to quantify the strain due to adsorbtion. 


Dynamical Strain Analyses
-------------------------
During an Ab Initio Molecular Dynamics (AIMD) simulation, JEDI can quantify the potential energy part of the strain due to, e.g., stretching and compression. In this scenario, each time step is considered as a deformed geometry. Dynamical strain analyses using JEDI enable the creation of color-coded movies showing the propagation of strain in dynamical mechanochemical processes.


How to Cite JEDI
================

When using JEDI, please cite the following papers:

* Wang, H.; Benter, S.; Dononelli, W.; Neudecker, T.; JEDI: A versatile code for strain analysis of molecular and periodic systems under deformation, J. Chem. Phys. **2024**, 160, 152501. https://doi.org/10.1063/5.0199247

* Stauch, T.; Dreuw, A.; A quantitative quantum-chemical analysis tool for the distribution of mechanical force in molecules, J. Chem. Phys. **2014**, 140, 134107. https://doi.org/10.1063/1.4870334

Additional information on the theoretical background of the JEDI analysis can be found in the following papers:

* Stauch, T.; Dreuw, A.; On the use of different coordinate systems in mechanochemical force analyses, J. Chem. Phys. **2015**, 143, 074118. https://doi.org/10.1063/1.4928973

* Stauch, T.; Dreuw, A.; Predicting the Efficiency of Photoswitches Using Force Analysis, J. Phys. Chem. Lett. **2016**, 7, 1298-1302. https://doi.org/10.1021/acs.jpclett.6b00455

* Stauch, T.; Dreuw, A.; Quantum Chemical Strain Analysis For Mechanochemical Processes, Acc. Chem. Res. **2017**, 50, 1041-1048. https://doi.org/10.1021/acs.accounts.7b00038


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

.. toctree::
   :maxdepth: 1
   :caption: Hints

   extras/vmd
   extras/problems



Indices and tables
==================

* :ref:`genindex`



Judgement of Energy DIstribution
--------------------------------

JEDI is a quantum chemical strain analysis tool working with the Atomic Simulation Environment (ASE).



Requirements
------------

* Python_ 3.8 or later
* NumPy_ (base N-dimensional array package)
* ase_ 3.23 (functions to determine atomic structures' geometries and quantum chemical calculators)




Installation
------------

JEDI can be installed by running ``pip install strainjedi``. When using the ``git`` version, add ``~/jedi`` to your $PYTHONPATH environment variable. 



Tutorial
------------

A tutorial is available here: https://neudecker-group.github.io/jedi/



Citation
--------

When using JEDI, please cite the following papers:

Wang, H.; Benter, S.; Dononelli, W.; Neudecker, T.; JEDI: A versatile code for strain analysis of molecular and periodic systems under deformation, J. Chem. Phys. **2024**, 160, 152501. https://doi.org/10.1063/5.0199247

Stauch, T.; Dreuw, A.; A quantitative quantum-chemical analysis tool for the distribution of mechanical force in molecules, J. Chem. Phys. **2014**, 140, 134107. https://doi.org/10.1063/1.4870334

Additional information on the theoretical background of the JEDI analysis can be found in the following papers:

 T. Stauch, A. Dreuw, On the use of different coordinate systems in mechanochemical force analyses, J. Chem. Phys. **2015**, 143, 074118. https://doi.org/10.1063/1.4928973

 T. Stauch, A. Dreuw, Predicting the Efficiency of Photoswitches Using Force Analysis, J. Phys. Chem. Lett. **2016**, 7, 1298-1302. https://doi.org/10.1021/acs.jpclett.6b00455

 T. Stauch, A. Dreuw, Quantum Chemical Strain Analysis For Mechanochemical Processes, Acc. Chem. Res. **2017**, 50, 1041-1048. https://doi.org/10.1021/acs.accounts.7b00038



.. _Python: http://www.python.org/
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _ase: https://wiki.fysik.dtu.dk/ase/

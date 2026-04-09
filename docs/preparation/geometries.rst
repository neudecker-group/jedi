==========
Geometries
==========

The geometries and their energies need to be stored in an Atoms object.
This is possible using two different ways:

1. Conduct calculation in ASE and store the geometry is returned with the energy through ´´mol.get_potential_energy()´´
2. Read the output file with read funtions given by ASE ´´mol = ase.io.read("outputfile")´´
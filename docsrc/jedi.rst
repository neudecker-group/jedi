============
Jedi
============
noooooooooo
Jedi is written as a class which includes the structure's geometries and its hessian in the relaxed state. The analysis works automatically with its inbuilt funcions. It converts the Hessian in 
redundant internal coordinates and calculates the energy in every 
internal mode using the equation shown below.

.. math:: E_{RIMs} = \frac{1}{2} * {\Delta q}^T * H_q * \Delta q 



.. autofunction:: strainjedi.jedi.jedi_analysis

.. autofunction:: strainjedi.jedi.jedi_printout

.. autofunction:: strainjedi.jedi.jedi_printout_bonds

.. autoclass:: strainjedi.jedi.Jedi
   :members:   


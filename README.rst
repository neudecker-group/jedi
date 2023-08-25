Judgement of Energy Distribution
=============================

JEDI is a quantum chemical strain analysis tool working with the atomic simulation environment (ASE).

Webpage: 


Requirements
------------

* Python_ 3.8 or later
* NumPy_ (base N-dimensional array package)
* ase_ (functions to determine atomic structures' geometries and quantum chemical calculators)




Installation
------------

Add ``~/jedi`` to your $PYTHONPATH environment variable 




Example
-------

Geometry optimization of hydrogen molecule with NWChem:

>>> from jedi.jedi import Jedi
>>> from ase.vibrations.data import VibrationsData
>>> from ase.io import read
>>> Opt = read('opt.json')
>>> Strain = read('strain.json')
>>> Modes = VibrationsData.read('modes.json')
>>> Jedi = Jedi(Opt, Strain, Modes)
 ************************************************
 *                 JEDI ANALYSIS                *
 *       Judgement of Energy DIstribution       *
 ************************************************

                   Strain Energy (kcal/mol)  Deviation (%)
      Geometries     3.31333944                  -
 Red. Int. Modes      3.98913827                20.40

 RIM No.       RIM type                       indices        delta_q (au) Percentage    Energy (kcal/mol)
     1       bond                                C0 C1        0.0000000      0.0        0.0000000
     2       bond                                C0 C5        0.0000000      0.0        0.0000000
     3       bond                                C1 C2        0.0000000      0.0        0.0000000
     4       bond                                C1 H6        0.0000000      0.0        0.0000000
     5       bond                                C2 C3        0.0000000      0.0        0.0000000
     6       bond                                C2 H7        0.0000000      0.0        0.0000000
     7       bond                                C3 C4        0.0000000      0.0        0.0000000
     8       bond                                C3 H8        0.1889726    100.0        3.9891383
     9       bond                                C4 C5        0.0000000      0.0        0.0000000
    10       bond                                C4 H9        0.0000000      0.0        0.0000000
    11       bond                               C5 H10        0.0000000      0.0        0.0000000
    12       bond angle                       C0 C1 C2        0.0000000      0.0        0.0000000
    13       bond angle                       C0 C1 H6        0.0000000      0.0        0.0000000
    14       bond angle                       C0 C5 C4        0.0000000      0.0        0.0000000
    15       bond angle                      C0 C5 H10        0.0000000      0.0        0.0000000
    16       bond angle                       C1 C0 C5        0.0000000      0.0        0.0000000
    17       bond angle                       C1 C2 C3        0.0000000      0.0        0.0000000
    18       bond angle                       C1 C2 H7        0.0000000      0.0        0.0000000
    19       bond angle                       C2 C1 H6        0.0000000      0.0        0.0000000
    20       bond angle                       C2 C3 C4        0.0000000      0.0        0.0000000
    21       bond angle                       C2 C3 H8        0.0000000      0.0        0.0000000
    22       bond angle                       C3 C2 H7        0.0000000      0.0        0.0000000
    23       bond angle                       C3 C4 C5        0.0000000      0.0        0.0000000
    24       bond angle                       C3 C4 H9        0.0000000      0.0        0.0000000
    25       bond angle                       C4 C3 H8       -0.0000000     -0.0       -0.0000000
    26       bond angle                      C4 C5 H10        0.0000000      0.0        0.0000000

...


.. _Python: http://www.python.org/
.. _NumPy: http://docs.scipy.org/doc/numpy/reference/
.. _ase: https://wiki.fysik.dtu.dk/ase/

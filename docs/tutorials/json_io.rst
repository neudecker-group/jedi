.. _json_io:

===================
Saving JEDI results
===================

To save your JEDI results for later visualization or plotting, you can write your results into a JSON file.
Feel free to follow along with the tutorial using any of the last JEDI analysis examples.

.. code-block:: python

   j = Jedi(a, a2, modes)
   j.run()
   j.write("jedi_results.json")

After running your analysis, you can save the JEDI state as a JSON file with the ``write()`` method.
You can also specify the path and the JSON file name.

.. code-block:: python

   j = Jedi.read("jedi_results.json")
   E_RIMs = j.E_RIMs
   print(E_RIMs)
   j.visualize(show=True)

With ``Jedi.read()``, you can reinitiate the ``Jedi`` object using the saved analysis results.
Now you can access the relevant results and plot or visualize them.

.. hint::
   Internally, JEDI saves energies in Hartree, distances in Bohr and angles in radians.
   If you want to use other units for plotting, etc., feel free to convert them using the `ASE constants <https://docs.ase-lib.org/ase/units.html#module-ase.units>`_.

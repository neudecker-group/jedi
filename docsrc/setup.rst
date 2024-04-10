Installation
=============
-------
Python
-------
JEDI is a tool using the ASE api. JEDI requires at least Python version 3.8. In addition, the python packages Numpy and ASE have to be installed.

Pip install
-----------

JEDI is pip installable. You can get the latest stable version through:

.. code-block:: console

    pip install strainjedi



Building Guide
==============

If you want the developer's build with latest features you can get it by following the steps below.
JEDI is primarily developed on Linux systems but can also be used on Windows and Mac systems. 

-------------------------------------------------------
Update package sources, Check / install Git and Python:
-------------------------------------------------------

First the package sources should be updated:

.. code-block:: console

    sudo apt update


Python 
------


JEDI runs with ``Python 3.8``. Check if it is installed with

.. code-block:: console

    python --version 

If ``Python 2.7`` is running on your system or if ``Python`` is not installed, install with 

.. code-block:: console

    sudo apt install python3.8

Pip and Libraries
-----------------

The JEDI analysis uses the libraries ASE and NumPy. To run JEDI both these libraries need to be installed. 
You can use the package installer ``pip`` to do that. First, check if pip is available by running:

.. code-block:: console

    python -m pip --version

If pip is not already installed, then bootstrap it from the standard library:

.. code-block:: console

    python -m ensurepip --default-pip

After ensuring that pip is installed, ASE and NumPy can be installed via pip from PyPI using

.. code-block:: console

    pip install ase 

and 

.. code-block:: console

    pip install numpy


Git
---

`Git <https://git-scm.com/>`_ is used as version control system for JEDI. 
Thus it is also used to clone the `JEDI source code repository <https://github.com/neudecker-group/jedi>`_ 
from `Github <https://github.com/>`_ to a local system where JEDI should be installed. 


--------------------------
Clone the JEDI repository
--------------------------

Clone the JEDI repository to your local machine with ``git`` from GitHub: 

.. code-block:: console
    
    git clone https://github.com/neudecker-group/jedi.git

This clones the JEDI repository to a local folder ``jedi``. 

Add ~/jedi to your $PYTHONPATH environment variable (assuming ~/jedi is where your jedi folder is).

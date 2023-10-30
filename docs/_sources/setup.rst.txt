Installation
=============

---
Git
---

`Git <https://git-scm.com/>`_ is used as version control system for JEDI. 
Thus it is also used to clone the `JEDI source code repository <https://github.com/henrwang/jedi>`_ 
from `Github <https://github.com/>`_ to a local system where JEDI should be installed. 

------
Python
------

JEDI is a tool using the ASE api. JEDI requires at least Python version 3.7.


Building Guide
==============

JEDI is primarily developed on Linux systems but can also be used on Windows and Mac systems. 

-------------------------------------------------------
Update package sources, Check / install Git and Python:
-------------------------------------------------------

First the package sources should be updated:

.. code-block:: console

    sudo apt update

Git 
---

Usually ``git`` should be installed already, however check if git is really installed: 

.. code-block:: console

    git --version 

prints the installed ``git`` version. If ``git`` is not installed, install it with 

.. code-block:: console

    sudo apt install git 


Python 
------


JEDI runs with ``Python 3.7``. Check, if it is installed with

.. code-block:: console

    python --version 

If ``Python 2.7`` is running on your system or if ``Python`` is not installed, install with 

.. code-block:: console

    sudo apt install python3.7

pip and libraries
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


--------------------------
Clone the JEDI repository
--------------------------

Clone the JEDI repository to your local machine with ``git`` from GitHub: 

.. code-block:: console
    
    git clone https://github.com/henrwang/jedi.git

This clones the JEDI repository to a local folder ``jedi``. 

Add ~/jedi to your $PYTHONPATH environment variable (assuming ~/jedi is where your jedi folder is).
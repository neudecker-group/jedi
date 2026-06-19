============
Installation
============

System Requirements
===================

* Python_ 3.10 or newer

.. _Python: https://www.python.org/
.. _pip: https://pip.pypa.io/en/stable/
.. _PyPI: https://pypi.org/project/strainjedi/

.. _install:

Installation from PyPI
======================

The simplest way to install JEDI is to use pip_.

.. code-block:: bash

   $ pip install --upgrade strainjedi

The above command should work if you are using a virtual environment, Conda, or some other user-level Python installation.

If your Python/pip was provided by a system administrator or package manager, you might not have appropriate permissions to install JEDI.
Additionally, pip itself might complain about potentially breaking your system packages.
In that case, install with e.g.

.. code-block:: bash

   $ pip install --upgrade --user strainjedi

If that for `some reasons <https://xkcd.com/1987/>`_ still does not work, try the steps below as if you were working on a Python project.

Using JEDI in your projects
===========================

If you're intending to use JEDI in one of your python projects, add it to your dependencies in ``pyproject.toml``:

.. code-block:: toml

   [project]
   dependencies = [
       "strainjedi>=1.0.0",
   ]

Or, when you're using `uv <https://docs.astral.sh/uv/>`_ to manage your dependencies (you should):

.. code-block:: bash

   $ uv add strainjedi
   $ uv sync

Alternatively, consult the documentation of your favourite Python dependency manager.

Installing from source
======================

For the latest and greatest code updates, you can install JEDI from Git.

.. code-block:: bash

   $ pip install --upgrade git+https://github.com/neudecker-group/jedi@main

The ``--upgrade`` option ensures that pip always reinstalls, even if the version number has not changed.

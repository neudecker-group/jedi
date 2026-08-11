.. _contribute:

===========
Development
===========

This document outlines development guidelines: how to build the project, how to and where to contribute, and code style to name a few.
As with almost every contributing guideline, it's not strictly set in stone --- use your best judgement.

JEDI is developed on GitHub at https://github.com/neudecker-group/jedi --- it is also where bug reports, questions, feature requests, and whatever else are to be submitted.

Requirements
============

Small changes can be authored directly using the GitHub web interface, though we highly recommend a more capable editor or IDE for any extended work.

JEDI uses uv to manage the Python version, dependencies, and to build the project.
If you haven't already, `install uv <https://docs.astral.sh/uv/getting-started/installation/>`_ according to their instructions.
This approach enables us to use multiple Python versions in parallel, without ensuing `a total mess of Python installations <https://xkcd.com/1987/>`_.

You will also need to have ``git`` installed in order to submit your changes to version control.

Instructions
============

The usual set of steps to contribute to anything hosted on GitHub applies:
fork the repository, clone the fork, cut a new branch from ``main``, make and commit your changes, and submit a pull request **targetting the** ``develop`` **branch**.

Preparing Your Environment
--------------------------

In your local clone, install the required dependencies via uv by running the following command:

.. code-block::

   uv sync --all-groups

This will ensure that you can work within and build the entire project, including its documentation.
It will also isolate the dependencies into a venv, keeping your system Python installation clean.

Making Your Changes
-------------------

Python Code
~~~~~~~~~~~

Formatting
^^^^^^^^^^

The Python code should follow the `PEP8 Style Guide <https://peps.python.org/pep-0008/>`_ for Python code.
One minor change, however, is that we allow line lengths of up to 120 characters.

You can check your formatting by issuing the following command:

.. code-block::

   uv run ruff format --check

Or, if you want to apply all the changes:

.. code-block::

   uv run ruff format .

We've also installed the formatter as a GitHub workflow, so if you're unsure how to format it, said workflow will fix it for you.
Please refer to your IDE's (or editor's) documentation on how to integrate ``ruff`` as your Python formatter.

Ideally, you have also set up some sort of type checker --- whilst Python isn't exactly known for strict typing, it makes reading code and avoiding subtle bugs far easier.
Alternatively, from your console:

.. code-block::

   uv run ty check

We also configured this one as a GitHub workflow that checks your submission --- running ``ty`` from your console might result in stricter output than the workflow.
Here, we ask you to employ your best judgement when prioritizing (or ignoring) type checker errors.

Testing
^^^^^^^

After you made your changes, please ensure the existing tests are still passing:

.. code-block::

   uv run pytest

If necessary, add new tests for your additions.
They don't have to be perfect and/or cover every possible edge case, but by writing tests, you ensure you are thinking well about the solution.

Building
^^^^^^^^

In case you wish to test the built wheel, uv also provides a simple command for that:

.. code-block::

   uv build

This will build a source distribution tarball and a Python wheel into the ignored ``dist/`` directory.

Documentation
~~~~~~~~~~~~~

The documentation is written in reStructuredText (RST), with one sentence per line (so-called semantic line feeds).
While this approach might seem unusual in the beginning, its benefits become clear relatively quickly.
You soon realise when sentences have grown too long or are unusually inconsisten in length, which can hurt readability.
It also makes it easier to review, edit, and track changes in version control, as you can see exactly which sentences were changed without any noise from reflowing wrapped text.

In source form, it does read somewhat awkwardly, but it's not what the reader will see, so it is a small price to pay for the benefits it brings.

Preview
^^^^^^^

We use Sphinx plus some extensions to build the documentation --- as they're all defined inside ``pyproject.toml``, they should be installed by means of ``uv sync --all-groups`` already.
Then, to build the documentation:

.. code-block::

   uv run sphinx-build -a -b html docs docs/_builds/

This will rebuild all pages and generate HTML into ``docs/_build/``.
Finally, you can start a simple HTTP server to preview the changes:

.. code-block::

   uv run python -m http.server -d docs/_build/

Submitting Your Changes
-----------------------

After you are happy with your changes, commit them, push to your fork, and then create a pull request against the ``develop`` branch.

.. note::

   To improve discoverability of released code, the default branch is ``main``.
   GitHub will automatically pick the default branch as PR target, so make sure to change it.

A commit message consists of a header, a body, and a footer:

.. code-block::

   <header>
   <blank-line>
   <body>
   <blank-line>
   <footer>

The header is mandatory and should ideally be limited to about 50 characters.
GitHub will automatically break the header line once it exceeds 72 characters, so consider that the hard limit.
If your change is trivial enough, like fixing a typo or some grammar, the header is all you need.
It should be written in the imperative mood and be descriptive of what precisely was changed.

.. code-block:: diff

   -Update README.md
   +readme: fix broken link to website

If you find yourself struggling to concisely summarize your changes in the header, consider splitting them up into multiple commits or perhaps even multiple PRs if appropriate.

The body can be used for more sophisticated changes, e.g. to give rationale for the change.
Like the header, the body should be wrapped to 72 characters as well, but if you paste logs or other pre-formatted output, please do not wrap those lines.

.. tip::

   Most editors today are aware of Git's commit message formatting and should either wrap automatically when you reach those line limits, or at least tell you.
   In our experience, (Neo)vim and Visual Studio Code are both reasonably git-aware.

After You Opened Your Pull Request
----------------------------------

After you opened your pull request, make sure you check back in a timely manner, you may have received some feedback.
Once everything is in order and the changes are approved, your pull request will be merged into the ``develop`` branch until it's time to cut a new release.

We are not working off a strict release schedule --- when we think it's time, a new release will be issued.

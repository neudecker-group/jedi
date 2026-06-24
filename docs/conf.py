# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import datetime
from pathlib import Path
from urllib.parse import quote

from docutils import nodes
from docutils.parsers.rst.roles import set_classes
from sphinx.util.nodes import split_explicit_title

project = "JEDI"
author = "Neudecker Group"
copyright = f"{datetime.datetime.now(tz=datetime.timezone.utc).date().year}, {author}"
repository = "https://github.com/neudecker-group/jedi"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "sphinxcontrib.lightbox2",
    #    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,  # all public members
    "undoc-members": True,  # also render undocumented members
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
html_logo = f"{html_static_path[0]}/images/Logo_JEDI.png"
# html_theme = "sphinx_rtd_theme"
html_theme = "sphinx_book_theme"
html_theme_options = {
    "collapse_navigation": False,
    "repository_url": repository,
    "show_toc_level": 3,
    "use_repository_button": True,
}

# -- Further customisation ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/development/tutorials/extending_syntax.html
# https://docutils.sourceforge.io/docs/howto/rst-roles.html


def git_role(role, rawtext, text, lineno, inliner, options=None, content=None):
    """
    Provide a :git:`path/to/file` shorthand, rendering as a link to the file on GitHub.
    Also works as masked link: :git:`random text <path/to/file>`.

    """
    options = options or {}

    env = inliner.document.settings.env
    root = Path(env.srcdir).parent

    _has_title, title, target = split_explicit_title(text)
    title = title or target

    path = root / target
    if not path.exists():
        msg = f"Broken link: {rawtext}: {path}"
        err = inliner.reporter.error(msg, line=lineno)
        return [inliner.problematic(rawtext, rawtext, err)], [err]

    ref = f"{repository}/blob/main/{quote(target)}"

    set_classes(options)
    node = nodes.reference(rawtext, title, refuri=ref, **options)
    return [node], []


def setup(app):
    app.add_role("git", git_role)


def zip_dirs(dirs: list[str], out: str):
    import os
    import zipfile

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zipf:
        for dir in dirs:
            for root, _, files in os.walk(dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, start=os.path.dirname(dir))
                    zipf.write(full_path, arcname)


zip_dirs(["tutorials/gaussian", "tutorials/orca", "tutorials/qchem"], "_static/downloads/calculators.zip")

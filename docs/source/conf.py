# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import importlib
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..', 'aleatory')))
aleatory = importlib.import_module("aleatory")



# -- Project information -----------------------------------------------------

project = 'aleatory'
copyright = '2022, Dialid Santiago'
author = 'Dialid Santiago'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.mathjax",
              "sphinx.ext.viewcode",
              "myst_parser"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
# html_theme = 'renku'
# html_theme = 'press'
#html_theme = 'bizstyle'

# import sphinx_bernard_theme
# html_theme = 'sphinx_bernard_theme'
# html_theme_path = [sphinx_bernard_theme.get_html_theme_path()]

html_theme = 'furo'

# import hachibee_sphinx_theme
# html_theme = 'hachibee'
# html_theme_path = [hachibee_sphinx_theme.get_html_themes_path()]

# The name of the Pygments (syntax highlighting) style to use.
#pygments_style = "sphinx"
#pygments_style = "autumn"
#pygments_style = "paraiso-dark"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['css/custom.css',]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}
# conf.py

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

# -- General configuration ---------------------------------------------------
# You only need to define the extensions list ONCE.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.doctest',
    'myst_parser',
    #    'sphinx-autobuild',
    #    'sphinx-copybutton',
    #    'sphinx-gallery',
    #    'sphinx-design',
    #    'sphinx-sitemap',
    #    'sphinx-reredirects',
]

# Set the theme via the html_theme variable
html_theme = 'sphinx_rtd_theme'

# -- Project information -----------------------------------------------------
project = 'Observed Control'
copyright = '2025, Andrew Petruska'
author = 'Andrew Petruska'

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'alabaster'
html_static_path = ['_static']

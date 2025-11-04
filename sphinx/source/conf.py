# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from typing import Sequence
from lmpipe import MODULE
from importlib.metadata import metadata as load_metadata

metadata = load_metadata(MODULE)
print('Loaded metadata for', MODULE)
print('==================== Metadata ====================')
print(metadata)
print('==================================================')
"""
Metadata fields reference:
    see https://packaging.python.org/en/latest/specifications/core-metadata/
"""

_author: str = (
    metadata["Author"]
    or metadata["Author-email"].split(' <')[0]
    or "Unknown Author"
)

project: str = metadata["Name"]
"The documented project’s name."
author: str = _author
"The project’s author(s)."
copyright: str | Sequence[str] = f'2025, {_author}'
"A copyright statement."
# project_copyright is an alias for copyright
version: str = '.'.join(metadata["Version"].split('.', 3)[:2])
"The project’s version, without the patch level."
release: str = metadata["Version"]
"The project’s full version, including alpha/beta/rc tags."


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

needs_sphinx = load_metadata("sphinx")["Version"]

extensions: list[str] = [

    'sphinx.ext.napoleon',

    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx.ext.autosummary',

    'sphinx_wagtail_theme',

]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_wagtail_theme'
html_static_path = ['_static']


autodoc_member_order = 'bysource'  # keep the order in source code
autodoc_typehints = "signature"  # show type hints in signatures




python_maximum_signature_line_length: int = 88

# Napoleon configuration: preprocess types so that type names are wrapped with
# Sphinx roles (e.g. :class:`str`) which renders them as code/backticks in the
# generated documentation.
napoleon_preprocess_types = True
# Optional: make Napoleon render parameter and return types using Sphinx field
# lists (and rtype), which pairs well with preprocess_types.
napoleon_use_param = False
napoleon_use_rtype = False
typehints_use_rtype = False
typehints_document_rtype = False

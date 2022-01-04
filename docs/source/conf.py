"""Configure sphinx documentation builds."""
# -- Path setup --------------------------------------------------------------
import os
import sys

sys.path.append(os.path.abspath("../arpes"))
sys.path.append(os.path.abspath(".."))

import arpes
import arpes.config
import datetime
import sphinx_rtd_theme


# -- Project information -----------------------------------------------------
project = "arpes"
CURRENT_YEAR = datetime.datetime.now().year
copyright = f"2018-{CURRENT_YEAR}, Conrad Stansbury"
author = "Conrad Stansbury"

# The short X.Y version
version = ".".join(arpes.__version__.split(".")[:2])
# The full version, including alpha/beta/rc tags
release = arpes.__version__

# supress some output information for nbconvert, don't open tools
arpes.config.DOCS_BUILD = True


# -- Options for rst extension -----------------------------------------------

rst_file_suffix = ".rst"
rst_link_suffix = ""  # we will generate a webpage with docsify so leave this blank


def transform_rst_link(docname):
    """Make sure links to docstrings are rendered at the correct relative path."""
    return "api/rst/" + docname + rst_link_suffix


rst_link_transform = transform_rst_link

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.restbuilder",
    # "sphinxcontrib.katex",
    "sphinx_rtd_theme",
    "nbsphinx",
    "sphinx_copybutton",
]

apidoc_separate_modules = True

katex_version = "0.13.13"
katex_css_path = f"https://cdn.jsdelivr.net/npm/katex@{katex_version}/dist/katex.min.css"
katex_js_path = f"https://cdn.jsdelivr.net/npm/katex@{katex_version}/dist/katex.min.js"
katex_inline = [r"\(", r"\)"]
katex_display = [r"\[", r"\]"]
katex_prerender = False
katex_options = ""

# autodoc settings
def autodoc_skip_member(app, what, name, obj, skip, options):
    """Don't include parts of code which require optional dependencies for now."""
    # This is a noop for now
    return skip


def setup(app):
    """Add the autodoc skip member hook, and any other module level config."""
    app.connect("autodoc-skip-member", autodoc_skip_member)


autodoc_mock_imports = ["torch", "pytorch_lightning"]


# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# nbsphinx settings
nbsphinx_timeout = 600

if os.getenv("READTHEDOCS"):
    nbsphinx_execute = "never"
else:
    nbsphinx_execute = "always"


autosummary_generate = True


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

language = None

exclude_patterns = []

pygments_style = "sphinx"


# HTML Configuration
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["style.css"]
html_logo = "_static/PyARPES-Logo.svg"
# html_favicon = "_static/PyARPES-Logo.ico"
html_theme_options = {
    "analytics_id": "UA-55955707-2",
    "analytics_anonymize_ip": False,
    "logo_only": True,
    "display_version": False,
    "style_nav_header_background": "white",
}

# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "arpesdoc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}
latex_documents = [
    (master_doc, "arpes.tex", "arpes Documentation", "Conrad Stansbury", "manual"),
]

# -- Options for manual page output ------------------------------------------
man_pages = [(master_doc, "arpes", "arpes Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    (
        master_doc,
        "arpes",
        "arpes Documentation",
        author,
        "arpes",
        "One line description of project.",
        "Miscellaneous",
    ),
]

# -- Options for Epub output -------------------------------------------------
epub_title = project
epub_exclude_files = ["search.html"]

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

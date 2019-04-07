============================
Installation
============================

The latest version can always be installed from github using

::

  pip install git+https://github.com/kjappelbaum/structure_comp.git

All requirements are automatically installed by pip. If you want to install the development extras,
use

::

  pip install git+https://github.com/kjappelbaum/structure_comp.git#egg=project[testing, docs, pre-commit]


Known issues
------------

For the addition of missing hydrogens and the geometry relaxation (both in the :code:`Cleaner` class) we
rely on openbabel. We made the openbabel python package (:code:`pybel`) an optional requirement in the
:code:`setup.py` which you can install with
::

  pip install git+https://github.com/kjappelbaum/structure_comp.git#egg=project[openbabel]

Note, that it still requires that you already have openbabel installed on you machine. We found it most convenient
to use the anaconda package manager to do so. For this reason, we also provide an :code:`environment.yml` file
that easily allows to create a conda environment with all dependencies using

::

    conda env create --name <envname> -f=environment.yml

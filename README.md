# structure_comp 

[![Documentation Status](https://readthedocs.org/projects/structure-comp/badge/?version=latest)](https://structure-comp.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/kjappelbaum/structure_comp.svg?branch=master)](https://travis-ci.com/kjappelbaum/structure_comp)
[![Coverage Status](https://coveralls.io/repos/github/kjappelbaum/structure_comp/badge.svg?branch=master)](https://coveralls.io/github/kjappelbaum/structure_comp?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/ed971fb3b03f4b24bfd58600bd7a7254)](https://www.codacy.com/app/kjappelbaum/structure_comp?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kjappelbaum/structure_comp&amp;utm_campaign=Badge_Grade)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![HitCount](http://hits.dwyl.io/kjappelbaum/structure_comp.svg)](http://hits.dwyl.io/kjappelbaum/structure_comp)

*Warning* The code is still under heavy development! 

Small python package to efficiently remove duplicates from larger 
structure databases and get some statistics (and more).

## Acknowledgments

We build on prior work, check the [bibliography](https://structure-comp.readthedocs.io/en/latest/references.html)
for a listing of references.  

Besides the python standard library, we make use of the following open-source libraries:

*   [pandas](https://pandas.pydata.org/)
*   [pymatgen](http://pymatgen.org/index.html)
*   [ase](https://wiki.fysik.dtu.dk/ase/) 
*   [scipy](https://www.scipy.org/) 
*   [sklearn](https://scikit-learn.org/stable/index.html) 
*   [tqdm](https://pypi.org/project/tqdm/)
*   [numba](https://github.com/numba/numba)
*   [PyCifRW](https://pypi.org/project/PyCifRW/)
*   [matplotlib](https://matplotlib.org/)
*   [pytest](https://docs.pytest.org/en/latest/)
*   [sphinx](http://www.sphinx-doc.org/en/stable/) with the
    [napoleon](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/index.html) 
    and [bibtex](https://github.com/mcmtroffaes/sphinxcontrib-bibtex) extensions

Several of the libraries are supported by [NumFocus](https://numfocus.org/). If you love these tools and want to support
open code, open science and the development of these packages, you might consider a donation to NumFocus. 

If we use code-snippets from the internet, we acknowledge this in the docstring of the function. 

## Installation
More details can be found in the [documentation](https://structure-comp.readthedocs.io/en/latest/install.html).  

Developers install in editable mode

    pip -e . 
    
and 

    pip install pre-commit yapf prospector pylint
    pre-commit install
    
for pre-commit hooks. 

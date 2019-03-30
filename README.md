# Remove duplicates

[![Documentation Status](https://readthedocs.org/projects/structure-comp/badge/?version=latest)](https://structure-comp.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/kjappelbaum/structure_comp.svg?branch=master)](https://travis-ci.com/kjappelbaum/structure_comp)
[![Coverage Status](https://coveralls.io/repos/github/kjappelbaum/structure_comp/badge.svg?branch=master)](https://coveralls.io/github/kjappelbaum/structure_comp?branch=master)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/ed971fb3b03f4b24bfd58600bd7a7254)](https://www.codacy.com/app/kjappelbaum/structure_comp?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=kjappelbaum/structure_comp&amp;utm_campaign=Badge_Grade)

Small python package to efficiently remove duplicates from larger 
structure databases (and more). The main goal is to have something that is not $\mathcal{O}(N^2)$ 
and works without a lot of threshold optimization. 

The idea is to first map into a cheap scalar feature space, find composition duplicates
in this space and then compare the structure graphs of composition duplicates. 

Alternatively, one can also use the [Kabsch algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm) for comparison of the composition duplicates.

Moreover, a second module tries to implement comparatively cheap metrics for the assessment of database diversity. 
A third module allows to select samples for computational studies. 

## Acknowledgment
We build on prior work:

* A more advantaged, but less transferable, topology base comparison was published in 
    * Barthel, S.; Alexandrov, E. V.; Proserpio, D. M.; Smit, B. Distinguishing Metal–Organic Frameworks. Crystal Growth & Design 2018, 18 (3), 1738–1747. https://doi.org/10.1021/acs.cgd.7b01663.
    * Lee, Y.; Barthel, S. D.; Dłotko, P.; Moosavi, S. M.; Hess, K.; Smit, B. Quantifying Similarity of Pore-Geometry in Nanoporous Materials. Nature Communications 2017, 8, 15396. https://doi.org/10.1038/ncomms15396.
* The [pymatgen structure comparator](http://pymatgen.org/_modules/pymatgen/analysis/structure_matcher.html) was used e.g.  in Nazarian, D.; Camp, J. S.; Sholl, D. S. A Comprehensive Set of High-Quality Point Charges for Simulations of Metal–Organic Frameworks. Chemistry of Materials 2016, 28 (3), 785–793. https://doi.org/10.1021/acs.chemmater.5b03836.

We make use of the following open-source libraries:
* pandas
* pymatgen 
* ase 
* scipy 
* sklearn 

## ToDo's
* Clean-up main class
* Add unit-tests
* Add runscript
* Let users choose between structure graph and RMSD
* Publish on pypy (as soon as version == 1.0)
* Give option for caching
* use with `futures.ProcessPoolExecutor() as pool:` for multiprocessing in 
the graph comparison loop 
* Add class constructors from folder and AiiDA databases, i.e. add useful hash for CifData objects
* API design, especially for statistics module
* Organization of statistics module, probably more efficient to create structures and structure graphs only once, 
should probably save them to the object. 

## Installation
For developers install in editable mode

    pip -e . 
    
and 

    pip install pre-commit yapf prospector pylint
    pre-commit install
    
for pre-commit hooks. 

## Usage 




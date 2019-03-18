# Remove duplicates
Small python package to efficiently remove duplicates from larger 
structure databases. The main goal is to have something that is not $\mathcal{O}(N^2)$ 
and works without a lot of threshold optimization. 

## Acknowledgment
We build on prior work:

* Barthel, S.; Alexandrov, E. V.; Proserpio, D. M.; Smit, B. Distinguishing Metal–Organic Frameworks. Crystal Growth & Design 2018, 18 (3), 1738–1747. https://doi.org/10.1021/acs.cgd.7b01663.
* The [pymatgen structure comparator](http://pymatgen.org/_modules/pymatgen/analysis/structure_matcher.html) was used in Nazarian, D.; Camp, J. S.; Sholl, D. S. A Comprehensive Set of High-Quality Point Charges for Simulations of Metal–Organic Frameworks. Chemistry of Materials 2016, 28 (3), 785–793. https://doi.org/10.1021/acs.chemmater.5b03836.

We make heavy use of the following open-source libraries:
* pandas
* pymatgen 
* scipy 

## ToDo's
* Clean-up main class
* Add unit-tests
* Add runscript
* Publish on pypy
* Add class constructors from folder, AiiDA databases 

## Installation

## Usage 
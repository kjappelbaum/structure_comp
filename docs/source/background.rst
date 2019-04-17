============================
Background
============================


Introduction
-------------

In high-throughput studies one often uses databases such as the Core-COF :cite:`tong_exploring_2017` or Core-MOF :cite:`nazarian_large-scale_2017` database.
Over the course of our work, we noticed that these databases contain a non-negligible number of duplicates, hence we
wrote this tool to easily (and more or less efficiently) find and eliminate then.

Starting doing machine learning with these databases, we also noticed that we need tools for comparing them:
We wanted to quantify 'diversity' and 'distance' of and between databases.
Especially the :code:`DistExampleComparison` class was written due to the fact that ML
models are often good at interpolation but bad at extrapolation :cite:`meredig_can_2018` -- hence we needed tools to detect this.

For expensive simulations -- or more efficient ML training -- one also wants to have tools
for clever sampling. This is what the :code:`sampling` module tries to do.

Duplicate removal
-----------------

The approach in the main duplicate removal routines is the following:

1. get the the Niggli reduced cell to have the smallest possible, well defined structure
2. get some cheap scalar features that describe the composition and use them to filter out
   structures that probably have the same composition.
3. after we identified the structures with identical composition (which are usually not too many)
   we can run a more expensive structural comparisons using structure graphs or the Kabsch algorithm
   As the Kabsch algorithm is relatively inexpensive (but it needs a threshold) we also have an option,
   where one can use the Kabsch algorithm and then do a comparison based on the structure graphs.

Kabsch algorithm
.................

The Kabsch algorithm :cite:`kabsch_solution_1976,kabsch_discussion_1978` attempts to solve the problem of calculating the RMSD between to structures,
which is in general not well defined as it depends on the relative orientations of the structures w.r.t
each other. The algorithm calculates the optimal rotation matrix that mininmizes the RMSD between the structures and
it is often used in visualizations to e.g. align structures. The basic idea behind the algorithm is the following:

1. First we center the structures at the origin
2. Then, we calculate the covariance matrix

   .. math::

     \mathbf{H}_ij =  \sum_{k}^N \mathbf{P}_{ki} \mathbf{Q}_{kj}

   where :math:`P_{ki}` and :math:`Q_{kj}` are the point-clouds (position coordinates) of the two
   structures.
3. The optimal rotation matrix is then given by

   .. math::

     \mathbf{R} = \left(\mathbf{H}^\mathsf{T}\mathbf{H} \right)^{\frac{1}{2}} \mathbf{H}^{-1}

   which can be implemented using a SVD decomposition.


The implementation in this package is based on the rmsd package from Jimmy Charnley Kromann :cite:`kromann_calculate_2019`, we just added routines for
periodic cases.


Graph based
...........


Hashing
.......


Statistics
----------
The statistics module (:code:`comparators.py`) implements three different classes that allow to

* measure the structural diversity of one database (:code:`DistStatistic`)
* compare two databases (:code:`DistComparison`)
* compare one sample with a database (:code:`DistExample`)

The :code:`DistStatistic` class implements parallelized versions of random (with resampling)
RMSD and structure graph comparisons within a database whereas the :code:`DistComparison` class
also implements those but also several statistical tests like:

* maximum mean discrepancy
* Anderson-Darling
* Kolmogorov-Smirnov
* Mutual information

that work on a list of list or dataframe of features.

The main :code:`Statistics` class also implements further statistical metrics, such as
measures of central tendency like the trimean which are not that commonly used (unfortunately).

The :code:`DistExample` class clusters the database (e.g. based on same property space) and then
compares the sample to the :math:`k` samples closest to the centroids.

maximum mean discrepancy (MMD)
...............................

MMD basically uses the kernel trick.

.. warning::

    There are better implementations for MMD and especially the choice of the kernel width.
    In a future release, we might introduce shogon as optional dependency and use it if installed.


Sampling
---------
For all sampling, we standardize the features by default to avoid overly large effects by e.g. different units :cite:`tibshirani_elements_2017`.
In case you want to use different weights one different features you can multiply manually the columns of the dataframe
with weight factors and then turn the standardization off. 


Farthest point sampling
........................
The greedy farthest point sampling (FPS) :cite:`peyre_geodesic_2010` tries to find a good sampling of the point set :math:`S`
by selecting points according to

.. math::

  x_{k+1} = \text{argmax}_{x \in S} \min_{0\le i \le k} d(x_i, x)

where :math:`d(x_i, x)` is an appropriate distance metric, which in our case is by default Euclidean.
We initialize :math:`x_0` by choosing a random point from :math:`S`.

KNN based
.........

The :math:`k`-nearest neighbor based sample selection clusters the :math:`S` into :math:`k` cluster
and then selects the examples closest to the centroids. This is based on the rational that :math:`k`nn-clustering
tries to minimize the in-cluster variance :cite:`tibshirani_elements_2017?` (hence we sample from different clusters as we want a diverse set).


Cleaning
---------
A problem when attempting high-throughput studies with experimental structures, e.g from the Cambridge Structural Database,
is that structures :cite:`sturluson_role_2019`

* contain unbound water
* are disordered (e.g. methyl groups in two positions, aromatic carbon exist in several configurations in the :code:`.cif` file
* contain a lot of information that is not necessarily useful for the simulation and can cause problems when using the
  structure as an input file for simulation packages. Also, dropping unnecessary information can significantly
  reduce the filesize.

There has already been work done on this topic: The authors of the Core-MOF database described their approach
in the accompanying paper :cite:`chung_computation-ready_2014` and the group around David Fairen-Jimenez published small scripts that use Mercury
and a pre-defined list of solvents to remove unbound solvents :cite:`moghadam_development_2017`.

Unfortunately, to our knowledge, there exist no open-source tools at try to address all of
the three issues mentioned above.

.. warning::

    We are well aware of the problems of automatic structure sanitation tools :cite:`zarabadi-poor_comment_2019`.
    and also advise to use them with care and to report issues such that we can improve the tools.


Rewriting the :code:`cif` files
................................
For the first stage of rewriting the :code:`.cif` files, we use the `PyCifRW <https://pypi.org/project/PyCifRW/4.3/>`_ package :cite:`hester_validating_2006` which is the most robust
:code:`.cif` parser we are aware of. We keep only the lattice constants and the most important loops (fractional coordinates,
type and labels as well as the symmetry operations) whilst also using the atomic types as label as this is imperative for some simulation packages.

Furthermore, we remove all uncertainty indications and sanitize the filename (e.g. remove non-unicode and unsafe
characters such as parentheses).

Optionally, we also remove all disorder groups other than :code:`.` and :code:`1`. This assumes that the disorder
groups were properly labelled by the crystallographer.


Removing unbound solvent
........................
For removal of unbound solvent, we construct the structure graph and query for the molecular subgraphs (pymatgen internally
constructs a supercell to distinguish moleules from e.g. 2D layers).
If the composition of one of the molecular subgraphs is in our list of solvent molecules we delete the molecule
from the structure.


Removing disorder
.................

.. warning::

    Please note that this module is experimental and does not work in all cases.



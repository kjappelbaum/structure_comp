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

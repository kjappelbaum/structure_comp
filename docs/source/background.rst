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

Kabsch algorithm
.................

Graph based
...........

Hashing
.......


Statistics
----------


Sampling
---------
For all sampling, we standardize the features to avoid overly large effects by e.g. units.


Farthest point sampling
........................
The greedy farthest point sampling (FPS) :cite:`peyre_geodesic_2010` tries to find a good sampling of the point set :math:`S`
by selecting points according to

.. math::

  x_{k+1} = \argmax_{x \in S} \min_{0\le i \le k} d(x_i, x)

where :math:`d(x_i, x)` is an appropriate distance metric, which in our case is by default Euclidean.
We initialize $x_0$ by choosing a random point from :math:`S`.

KNN based
.........

The :math:`k`-nearest neighbor based sample selection clusters the :math:`S` into :math:`k` cluster
and then selects the examples closest to the centroids. This is based on the rational that :math:`k`nn-clustering
tries to minimize the in-cluster variance :cite:`tibshirani_elements_2017?` (hence we sample from different clusters as we want a diverse set).

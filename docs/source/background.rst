============================
Background
============================

In high-throughput studies one often uses databases such as the Core-COF:cite:`tong_exploring_2017` or Core-MOF:cite:`nazarian_large-scale_2017` database.
Over the course of our work, we noticed that these databases contain a non-negligible number of duplicates, hence we
wrote this tool to easily (and more or less efficiently) find and eliminate then.

Starting doing machine learning with these databases, we also noticed that we need tools for comparing them:
We wanted to quantify 'diversity' and 'distance' of and between databases.
Especially the :code:`DistExampleComparison` class was written due to the fact that ML
models are often good at interpolation but bad at extrapolation :cite:`meredig_can_2018` -- hence we needed tools to detect this.



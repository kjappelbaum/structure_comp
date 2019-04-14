=================
Quickstart Guide
=================

Removing Duplicates
-------------------


Getting Statistics
------------------



Sampling
--------
The sampler object works on dataframes, since this interfaces smoothly with featurization packages like
`matminer <https://github.com/hackingmaterials/matminer>`_.
So far, a greedy and a clustering-based farthest point
sampling have been implemented.

To start sampling you have to initialize a sampler object with dataframe, columns, the name of the identifier column
and the number of samples you want to have:

::

  from structure_comp.sampling import Sampler
  import pandas as pd
  zeolite_df = pd.read_csv('zeolite_pore_properties.csv')
  columns = ['ASA_m^2/g', 'Density', 'Largest_free_sphere',
       'Number_of_channels', 'Number_of_pockets', 'Pocket_surface_area_A^2']
  zeolite_sampler = Sampler(zeolite_df, columns=columns, k=9)

  # use the knn-based sampler
  zeolite_samples = zeolite_sampler.get_farthest_point_samples()

  # or use the greedy sampler
  zeolite_sampler.greedy_farthest_point_samples()


If you want to visualize the samples, you can call the :code:`inspect_sample` function on the sampler object:

::

    zeolite_sampler = inspect_sample()

If you work in a Jupyter Notebook, don't forget to call

::

    %matplotlib inline



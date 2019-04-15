=================
Quickstart Guide
=================

Removing Duplicates
-------------------
To get the duplicates in a directory with structures, you can run something like

::

    from structure_comp.remove_duplicates import RemoveDuplicates

    rd_rmsd_graph = RemoveDuplicates.from_folder(
        '/home/kevin/structure_source/csd_mofs_rewritten/', method='rmsd_graph')

    rd_rmsd_graph.run_filtering()

The filenames and the number of duplicates are saved as attributes of the :code:`RemoveDuplicates`
object.


Getting Statistics
------------------

Measuring the diversity of a dataset
`````````````````````````````````````

If you have properties -- great, use those! With you don't have any,
calculate some using some package like zeo++ or matminer.
If you really want to compare structures, you can use the :code:`DistStatistic` class. Using the
randomized RMSD is decently quick, constructing structure graphs can take some time and probably
does not lead to more insight:

::

    from structure_comp.comparators import DistStatistic
    core_cof_path = '/home/kevin/structure_source/Core_Cof/'

    # core cof statistics
    core_cof_statistics = DistStatistic.from_folder(core_cof_path, extension='cif')
    randomized_rmsd_cc = core_cof_statistics.randomized_rmsd()
    randomized_jaccard_cc = core_cof_statistics.randomized_graphs(iterations=100)

Then, it might be interesting to plot the resulting list as e.g. a violinplot to see whether
the distribution is uniform (which would be surprising) or which RMSDs are most common as well as
(what is probably most interesting) what is the width of the distribution. A example is shown in
the Figure below.


Comparing two property distributions
````````````````````````````````````

If you have two dataframes of properties and you want to find out if they come from the same
distribution the :code:`DistComparison` class is the one you might want to use.

Under the hood, it runs different statistical tests feature by feature and some also over the complete
dataset and then returns a dictionary with the test statistics.


Finding out if a structure is different from a distribution
````````````````````````````````````````````````````````````

In this case you have the following possibilities:

* you can do a property-based comparison
* you can do a structure based comparison, guided by properties
* you can do a random structure based comparison



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



Cleaning Structures
--------------------

Rewriting a :code:`.cif` file
``````````````````````````````
Most commonly we use the following function call to "clean" a :code:`.cif` file

::

    from structure_comp.cleaner import Cleaner

    cleaner_object = Cleaner.from_folder('/home/kevin/structure_source/csd_mofs/', '/home/kevin/structure_source/csd_mofs_rewritten')
    cleaner_object.rewrite_all_cifs()

You will find a new directory with structures that:

* have "safe" filenames
* have no experimental details in the :code:`cif` files
* are set to P1 symmetry
* have a :code:`_atom_site_label` column that is equal to :code:`_atom_site_type_symbol` which we found to work well
  with RASPA
* by default, we will also remove all disorder groups except :code:`.` and :code:`*`

If you input files have a :code:`_atom_site_charge` column, you wil also
find it in the output file.

.. note::

    You also have the option to symmetrization routines by setting
    :code:`clean_symmetry` to a float which is the tolerance for the symmetrization step.

Removing unbound solvent
````````````````````````
.. warning::

    Note that this routine is slow for large structures as it has to construct the
    structure graph.


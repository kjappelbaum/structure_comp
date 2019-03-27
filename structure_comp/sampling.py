#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Kevin M. Jablonka'
__copyright__ = 'MIT License'
__maintainer__ = 'Kevin M. Jablonka'
__email__ = 'kevin.jablonka@epfl.ch'
__version__ = '0.1.0'
__date__ = '27.03.19'
__status__ = 'First Draft, Testing'

from sklearn.cluster import KMeans
import pandas as pd

class Sampler():
    def __init__(self, dataframe, columns, name='name', k=10):
        self.dataframe = dataframe
        self.columns = columns
        self.name = name
        self.k = k

    def get_farthest_point_samples(self):
        """
        Gets the k farthest point samples on dataframe, returns the identifiers

        Args:
            dataframe (panadas DataFrame object): contains the features in the sampling space and the names
            columns (list): list of string with column names of property columns
            name (string): name of the column containing the identifiers
            k (int): number of samples

        Returns:
        """
        data = self.dataframe[self.columns]
        kmeans = KMeans(n_clusters=self.k).fit(data)
        cluster_centers = kmeans.cluster_centers_
        closest, _ = metrics.pairwise_distances_argmin_min(cluster_centers, data)
        return self.dataframe[self.name].iloc[closest]




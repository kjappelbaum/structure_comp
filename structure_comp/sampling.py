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
from sklearn import metrics
import pandas as pd
import numpy as np
from scipy.spatial import distance


class Sampler():
    def __init__(self, dataframe, columns, name='name', k=10):
        self.dataframe = dataframe
        self.columns = columns
        self.name = name
        self.k = k
        self.selection = []

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
        closest, _ = metrics.pairwise_distances_argmin_min(
            cluster_centers, data)

        selection = list(self.dataframe[self.name].iloc[closest].values)
        self.selection.append(selection)

        return selection

    def greedy_farthest_point_samples(self, metric='euclidean'):
        """

        Returns:

        """

        data = self.dataframe[self.columns]

        index = np.random.randint(0, len(data) - 1)

        greedy_data = []
        greedy_data.append(data[index])

        remaining = np.delete(data, index, 0)

        for _ in range(self.k - 1):
            dist = distance.cdist(remaining, greedy_data, metric)
            greedy_index = np.argmax(dist)
            greedy_data.append(remaining[greedy_index])
            remaining = np.delete(remaining, greedy_index, 0)

        greedy_indices = []

        for d in greedy_data:
            greedy_indices.append(np.where(data == d))

        return self.dataframe[self.name].iloc[greedy_indices]

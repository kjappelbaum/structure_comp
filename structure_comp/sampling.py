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
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.spatial import distance
from ase.visualize.plot import plot_atoms
import os
from ase.io import read, write
import matplotlib.pyplot as plt


class Sampler():
    def __init__(self,
                 dataframe: pd.DataFrame,
                 columns: list,
                 name: str = 'name',
                 k: int = 10):
        self.dataframe = dataframe
        self.columns = columns
        self.name = name
        self.k = k
        assert self.k < len(
            dataframe
        ), 'Sampling only possible if number of datapoints is greater than the number of requested samples'
        self.selection = []

    def get_farthest_point_samples(self) -> list:
        """
        Gets the k farthest point samples on dataframe, returns the identifiers

        Args:
            dataframe (panadas DataFrame object): contains the features in the sampling space and the names
            columns (list): list of string with column names of property columns
            name (string): name of the column containing the identifiers
            k (int): number of samples

        Returns:
            list with the sampled names 
        """
        self.selection = []
        data = StandardScaler().fit_transform(self.dataframe[self.columns])
        kmeans = KMeans(n_clusters=self.k).fit(data)
        cluster_centers = kmeans.cluster_centers_
        closest, _ = metrics.pairwise_distances_argmin_min(
            cluster_centers, data)

        selection = list(self.dataframe[self.name].iloc[closest].values)
        self.selection = selection

        return selection

    def greedy_farthest_point_samples(self, metric: str = 'euclidean') -> list:
        """

        Args:
            metric (string): metric to use for the distance, can be one from
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
            defaults to euclidean

        Returns:
            list with the sampled names

        """

        self.selection = []
        data = StandardScaler().fit_transform(self.dataframe[self.columns].values)

        index = np.random.randint(0, len(data) - 1)

        greedy_data = []
        greedy_data.append(data[index])

        remaining = np.delete(data, index, 0)

        for _ in range(self.k - 1):
            dist = distance.cdist(remaining, greedy_data, metric)
            greedy_index = np.argmax(np.argmax(np.min(dist, axis=0)))
            greedy_data.append(remaining[greedy_index])
            remaining = np.delete(remaining, greedy_index, 0)

        greedy_indices = []

        for d in greedy_data:
            greedy_indices.append(
                np.array(np.where(np.all(data == d, axis=1)))[0])

        greedy_indices = np.concatenate(greedy_indices).ravel()

        selection = list(self.dataframe[self.name][greedy_indices].values)
        self.selection = selection
        return selection

    def inspect_sample(self, path: str, mode: str = 'ase'):
        if mode == 'ase':
            if self.selection:
                for item in self.selection:
                    fig, axarr = plt.subplots(1, 1, figsize=(15, 15))
                    plt.title(item)
                    plot_atoms(read(os.path.join(path, item)), axarr)

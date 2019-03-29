#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Kevin M. Jablonka'
__copyright__ = 'MIT License'
__maintainer__ = 'Kevin M. Jablonka'
__email__ = 'kevin.jablonka@epfl.ch'
__version__ = '0.1.0'
__date__ = '27.03.19'
__status__ = 'First Draft, Testing'

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from ase.visualize.plot import plot_atoms
from scipy.spatial import distance
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class Sampler():
    def __init__(self,
                 dataframe: pd.DataFrame,
                 columns: list,
                 name: str = 'name',
                 k: int = 10):
        """
        Class for selecting samples from a collection of samples.
        Args:
            dataframe (pandas dataframe): dataframe with properties and names/identifiers
            columns (list): list of column names of the feature columns
            name (str): name of the identifier column
            k (int): number of the samples to select
        """
        self.dataframe = dataframe
        self.columns = columns
        self.name = name
        self.k = k
        assert self.k < len(
            dataframe
        ), 'Sampling only possible if number of datapoints is greater than the number of requested samples'
        self.selection = []

    def get_farthest_point_samples(self, standardize: bool = True) -> list:
        """
        Gets the k farthest point samples on dataframe, returns the identifiers

        Args:
            standardize (bool): Flag that indicates whether features are standardized prior to clustering (defaults to True)

        Returns:
            list with the sampled names 
        """
        self.selection = []

        if standardize:
            data = StandardScaler().fit_transform(
                self.dataframe[self.columns].values)
        else:
            data = self.dataframe[self.columns].values

        kmeans = KMeans(n_clusters=self.k).fit(data)
        cluster_centers = kmeans.cluster_centers_
        closest, _ = metrics.pairwise_distances_argmin_min(
            cluster_centers, data)

        selection = list(self.dataframe[self.name].iloc[closest].values)
        self.selection = selection

        return selection

    def greedy_farthest_point_samples(self,
                                      metric: str = 'euclidean',
                                      standardize: bool = True) -> list:
        """

        Args:
            metric (string): metric to use for the distance, can be one from
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
                defaults to euclidean
            standardize (bool): flag that indicates whether features are standardized prior to sampling

        Returns:
            list with the sampled names

        """

        self.selection = []

        if standardize:
            data = StandardScaler().fit_transform(
                self.dataframe[self.columns].values)
        else:
            data = self.dataframe[self.columns].values

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

    def inspect_sample(self, path: str = '', extension: str = '', mode: str = 'ase'):
        """
        Helps to quickly inspect the samples by plotting them (work great in e.g. jupyter notebooks,
        here you'll have to call %matplotlib inline).

        It assumes that the identifier the sampler returned are file-names, -stems or -paths.

        Args:
            path (str): path to the structure directory
            extension (str): extension (with the leading dot, e.g. '.cif')
            mode (str): visualization mode for the structures

        Returns:

        """
        if mode == 'ase':
            if self.selection:
                for item in self.selection:
                    fig, axarr = plt.subplots(1, 1, figsize=(15, 15))
                    plt.title(item)
                    plot_atoms(read(os.path.join(path, ''.join([item, extension]))), axarr)

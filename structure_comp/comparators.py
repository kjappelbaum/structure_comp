#!/usr/bin/python
# -*- coding: utf-8 -*-

# Get basic statistics describing the database
# Compare a structure to a database

from tqdm.autonotebook import tqdm
import logging
from pymatgen import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from .utils import get_structure_list, get_rmsd, closest_index, tanimoto_distance
import random
from scipy.spatial import distance
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

logger = logging.getLogger('RemoveDuplicates')
logger.setLevel(logging.DEBUG)


class Statistics():
    def __init__(self):
        pass

    @staticmethod
    def _randomized_graphs(structure_list_a: list,
                           structure_list_b: list,
                           iterations: int = 5000) -> list:
        """
        Randomly sample structures from the structure list and compare their Jaccard graph distance.

        Args:
            structure_list_a (list): list of paths to structures
            structure_list_b (list): list of paths to structures
            iterations (int): Number of comparisons (sampling works with replacement, i.e. the same pair might
            be sampled several times).

        Returns:
            list of length iterations of the Jaccard distances
        """
        diffs = []
        for _ in tqdm(range(iterations)):
            random_selection_1 = random.sample(structure_list_a, 1)[0]
            random_selection_2 = random.sample(structure_list_b, 1)[0]
            crystal_a = Structure.from_file(random_selection_1)
            crystal_b = Structure.from_file(random_selection_2)
            nn_strategy = JmolNN()
            sgraph_a = StructureGraph.with_local_env_strategy(
                crystal_a, nn_strategy)
            sgraph_b = StructureGraph.with_local_env_strategy(
                crystal_b, nn_strategy)
            diffs.append(sgraph_a.diff(sgraph_b, strict=False)['dist'])
        return diffs

    @staticmethod
    def _randomized_structure_property(structure_list_a: list,
                                       structure_list_b: list,
                                       property: str = 'density',
                                       iterations: int = 5000) -> list:
        """

        Args:
            structure_list_a (list): list of paths to structures
            structure_list_b (list): list of paths to structures
            property (str): property that is used for the structure comparisons, available options are
                density, num_sites, volume. Default is density.
            iterations (int): number of comparisons (sampling works with replacement, i.e. the same pair might
            be sampled several times).

        Returns:

        """
        diffs = []
        for _ in tqdm(range(iterations)):
            random_selection_1 = random.sample(structure_list_a, 1)[0]
            random_selection_2 = random.sample(structure_list_b, 1)[0]
            crystal_a = Structure.from_file(random_selection_1)
            crystal_b = Structure.from_file(random_selection_2)
            if property == 'density':
                diff = np.abs(crystal_a.density - crystal_b.density)
            elif property == 'num_sites':
                diff = np.abs(crystal_a.num_sites - crystal_b.num_sites)
            elif property == 'volume':
                diff = np.abs(crystal_a.volume - crystal_b.volume)
            diffs.append(diff)
        return diffs

    @staticmethod
    def _randomized_rmsd(structure_list_a: list,
                         structure_list_b: list,
                         iterations: float = 5000) -> list:
        """

        Args:
            structure_list_a (list): list of paths to structures
            structure_list_b (list): list of paths to structures
            iterations (int): number of comparisons (sampling works with replacement, i.e. the same pair might
                  be sampled several times).

        Returns:

        """
        rmsds = []
        for _ in tqdm(range(iterations)):
            random_selection_1 = random.sample(structure_list_a, 1)[0]
            random_selection_2 = random.sample(structure_list_b, 1)[0]
            a = get_rmsd(random_selection_1, random_selection_2)
            rmsds.append(a)

        return rmsds


class DistStatistic(Statistics):
    def __init__(self, structure_list):
        self.structure_list = structure_list

    @classmethod
    def from_folder(cls, folder: str, extension: str = '.cif'):
        """

        Args:
            folder (str): name of the folder which is used to create the structure list
            extension (str): extension of the structure files

        Returns:

        """
        sl = get_structure_list(folder, extension)
        return cls(sl)

    def randomized_graphs(self, iterations: int = 5000) -> list:
        """
        Returns iterations times the Jaccard distance between structure graph of two randomly chosen structures

        Args:
            iterations (int): number of comparisons (sampling works with replacement, i.e. the same pair might
                  be sampled several times).

        Returns:
            list of jaccard distances
        """
        jaccards = self._randomized_graphs(self.structure_list,
                                           self.structure_list, iterations)
        return jaccards

    def randomized_structure_property(self,
                                      property: str = 'density',
                                      iterations: int = 5000) -> list:
        """
        Returns iterations times the Euclidean distance between two randomly chosen structures
        Args:
            property (str): property that is used for the structure comparisons, available options are
                density, num_sites, volume. Default is density.
            iterations (int): number of comparisons (sampling works with replacement, i.e. the same pair might
                  be sampled several times).

        Returns:
            list of property distances
        """
        distances = self._randomized_structure_property(
            self.structure_list, self.structure_list, property, iterations)
        return distances

    def randomized_rmsd(self, iterations: int = 5000) -> list:
        """
        Returns iterations times the Kabsch RMSD between two randomly chosen structures
        Args:
            iterations (int): number of comparisons (sampling works with replacement, i.e. the same pair might
                  be sampled several times).

        Returns:
            list of Kabsch RMSDs
        """
        distances = self._randomized_rmsd(self.structure_list,
                                          self.structure_list, iterations)
        return distances


class DistComparison():
    def __init__(self, structure_list_1: list, structure_list_2: list):
        """

        Args:
            structure_list_1 (list):
            structure_list_2:
        """
        self.structure_list_1 = structure_list_1
        self.structure_list_2 = structure_list_2

    @classmethod
    def from_folders(cls, folder_1, folder_2, extension='.cif'):
        sl_1 = get_structure_list(folder_1, extension)
        sl_2 = get_structure_list(folder_2, extension)
        return cls(sl_1, sl_2)

    def randomized_graphs(self, iterations: int = 5000) -> list:
        """
        Returns iterations times the Jaccard distance between structure graph of two randomly chosen structures

        Args:
            iterations (int): number of comparisons (sampling works with replacement, i.e. the same pair might
                  be sampled several times).

        Returns:
            list of jaccard distances
        """
        jaccards = self._randomized_graphs(self.structure_list_1,
                                           self.structure_list_2, iterations)
        return jaccards

    def randomized_structure_property(self,
                                      property: str = 'density',
                                      iterations: int = 5000) -> list:
        """
        Returns iterations times the Euclidean distance between two randomly chosen structures
        Args:
            property (str): property that is used for the structure comparisons, available options are
                density, num_sites, volume. Default is density.
            iterations (int): number of comparisons (sampling works with replacement, i.e. the same pair might
                  be sampled several times).

        Returns:
            list of property distances
        """
        distances = self._randomized_structure_property(
            self.structure_list_1, self.structure_list_2, property, iterations)
        return distances

    def randomized_rmsd(self, iterations: int = 5000) -> list:
        """
        Returns iterations times the Kabsch RMSD between two randomly chosen structures
        Args:
            iterations (int): number of comparisons (sampling works with replacement, i.e. the same pair might
                  be sampled several times).

        Returns:
            list of Kabsch RMSDs
        """
        distances = self._randomized_rmsd(self.structure_list_1,
                                          self.structure_list_2, iterations)
        return distances


class DistExampleComparison():
    def __init__(self, structure_list, file):
        self.structure_list = structure_list
        self.file = file

    @classmethod
    def from_folder_and_file(class_object, folder, file, extension='.cif'):
        sl = get_structure_list(folder, extension)
        return class_object(sl, file)

    @staticmethod
    def property_based_distances_histogram(
            property_list: list) -> pd.DataFrame:
        ...

    def property_based_distances_clustered(
            self, property_list: list) -> pd.DataFrame:
        """
        Compares other structure to
            - lowest, highest, median, mean and random structure from structure list

        Returns the RMSD and the graph jaccard distance between other structure and the
        aforementioned structures.

        This can be useful in case there is a direct link between structure and property
        and we want to make sure that a structure is not to dissimilar from the set of structures
        in structure_list.
        Args:
            property_list:

        Returns:

        """

        median_index = closest_index(property_list, np.median(property_list))
        mean_index = closest_index(property_list, np.mean(property_list))
        random_index = np.random.randint(0, len(self.structure_list))
        q25_index = closest_index(property_list,
                                  np.quantile(property_list, 0.25))
        q75_index = closest_index(property_list,
                                  np.quantile(property_list, 0.75))

        lowest_structure = Structure.from_file(
            self.structure_list[np.argmin(property_list)])
        highest_structure = Structure.from_file(
            self.structure_list[np.argmax(property_list)])
        median_structure = Structure.from_file(
            self.structure_list[median_index])
        mean_structure = Structure.from_file(self.structure_list[mean_index])
        random_structure = Structure.from_file(
            self.structure_list[random_index])
        q25_structure = Structure.from_file(self.structure_list[q25_index])
        q75_structure = Structure.from_file(self.structure_list[q75_index])
        other_structure = Structure.from_file(self.file)

        nn_strategy = JmolNN()

        lowest_structure_graph = StructureGraph.with_local_env_strategy(
            lowest_structure, nn_strategy)
        highest_structure_graph = StructureGraph.with_local_env_strategy(
            highest_structure, nn_strategy)
        median_structure_graph = StructureGraph.with_local_env_strategy(
            median_structure, nn_strategy)
        mean_structure_graph = StructureGraph.with_local_env_strategy(
            mean_structure, nn_strategy)
        random_structure_graph = StructureGraph.with_local_env_strategy(
            random_structure, nn_strategy)
        q25_structure_graph = StructureGraph.with_local_env_strategy(
            q25_structure, nn_strategy)
        q75_structure_graph = StructureGraph.with_local_env_strategy(
            q75_structure, nn_strategy)
        other_structure_graph = StructureGraph.with_local_env_strategy(
            other_structure, nn_strategy)

        distances = {
            'mean_rmsd':
            get_rmsd(other_structure, mean_structure),
            'highest_rmsd':
            get_rmsd(other_structure, highest_structure),
            'lowest_rmsd':
            get_rmsd(other_structure, lowest_structure),
            'median_rmsd':
            get_rmsd(other_structure, median_structure),
            'random_rmsd':
            get_rmsd(other_structure, random_structure),
            'q25_rmsd':
            get_rmsd(other_structure, q25_structure),
            'q75_rmsd':
            get_rmsd(other_structure, q75_structure),
            'mean_jaccard':
            other_structure_graph.diff(mean_structure_graph,
                                       strict=False)['diff'],
            'highest_jaccard':
            other_structure_graph.diff(highest_structure_graph,
                                       strict=False)['diff'],
            'lowest_jaccard':
            other_structure_graph.diff(lowest_structure_graph,
                                       strict=False)['diff'],
            'median_jaccard':
            other_structure_graph.diff(median_structure_graph,
                                       strict=False)['diff'],
            'random_jaccard':
            other_structure_graph.diff(random_structure_graph,
                                       strict=False)['diff'],
            'q25_jaccard':
            other_structure_graph.diff(q25_structure_graph,
                                       strict=False)['diff'],
            'q75_jaccard':
            other_structure_graph.diff(q75_structure_graph,
                                       strict=False)['diff'],
        }

        return pd.DataFrame(distances)


def fingerprint_based_distances(fingerprint_list, other_fingerprint,
                                k=8) -> pd.DataFrame:
    """
    This comparator performes clustering in the fingerprint space and compares the distance
    of the other fingerprint to all clusters. This could be useful in a ML model to check
    if the model is performing interpolation (probably trustworthy) or extrapolation (random guessing might be involved).

    Returns a DataFrame with different distances between the clusters and the other fingerprint.

    :param fingerprint_list:
    :param other_fingerprint:
    :param k:
    :return:
    """
    if len(fingerprint_list) < k:
        logger.warning(
            'The number of points is smaller than the number of clusters, will reduce number of cluster'
        )
        k = len(fingerprint_list) - 1

    fingerprint_array = np.array(fingerprint_list)
    other_fingerprint_array = np.array(other_fingerprint)
    assert len(fingerprint_array.shape) == len(
        other_fingerprint_array.shape)  # they should have same dimensionality
    if len(fingerprint_array.shape) == 1:
        fingerprint_array = fingerprint_array.reshape(-1, 1)
        other_fingerprint_array = other_fingerprint_array.reshape(-1, 1)

    # Perform knn clustering in fingerprint space
    kmeans = KMeans(n_clusters=k, random_state=0).fit(fingerprint_array)

    cluster_centers = kmeans.cluster_centers_

    distances = []
    for cluster_center in cluster_centers:
        distance_dict = {
            'tanimoto':
            tanimoto_distance(cluster_center, other_fingerprint_array),
            'euclidean':
            distance.euclidean(cluster_center, other_fingerprint_array),
            'correlation':
            distance.correlation(cluster_center, other_fingerprint_array),
            'manhattan':
            distance.cityblock(cluster_center, other_fingerprint_array)
        }
        distances.append(distance_dict)

    return pd.DataFrame(distances)


# MMD test taken from https://github.com/paruby/ml-basics/blob/master/Statistical%20Hypothesis%20Testing.ipynb
def gaussian_kernel(z, length):
    z = z[:, :, None]
    pre_exp = ((z - z.T)**2).sum(axis=1)
    return np.exp(-(pre_exp / length))


def mmd(x, y, kernel, kernel_parameters):
    n = len(x)
    m = len(y)
    z = np.concatenate([x, y])
    k = kernel(z, kernel_parameters)

    kxx = k[0:n, 0:n]
    kxy = k[n:, 0:n]
    kyy = k[n:, n:]

    return (kxx.sum() / (n**2)) - (2 * kxy.sum() / (n * m)) + (kyy.sum() /
                                                               (m**2))


def mmd_test(x, y, kernel, kernel_parameters):
    mmd_array = mmd(x, y, kernel, kernel_parameters)
    n_samples = 100
    null_dist = mmd_null(x, y, kernel, kernel_parameters, n_samples)
    p_value = (null_dist[:, None] > mmd_array).sum() / float(n_samples)

    return mmd_array, p_value


def mmd_null(x, y, kernel, kernel_parameters, n_samples):
    s = [False for _ in range(n_samples)]
    z = np.concatenate([x, y])
    for i in range(n_samples):
        np.random.shuffle(z)
        s[i] = mmd(z[0:len(x)], z[len(x):], kernel, kernel_parameters)
    s = np.array(s)
    return s

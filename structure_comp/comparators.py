#!/usr/bin/python
# -*- coding: utf-8 -*-

# Get basic statistics describing the database
# Compare a structure to a database

from tqdm.autonotebook import tqdm
import logging
from pymatgen import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from .utils import get_structure_list, get_rmsd, closest_index, tanimoto_distance, get_number_bins
import random
from scipy.spatial import distance
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, ks_2samp
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import ndimage

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
                                       feature: str = 'density',
                                       iterations: int = 5000) -> list:
        """

        Args:
            structure_list_a (list): list of paths to structures
            structure_list_b (list): list of paths to structures
            feature (str): property that is used for the structure comparisons, available options are
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
            if feature == 'density':
                diff = np.abs(crystal_a.density - crystal_b.density)
            elif feature == 'num_sites':
                diff = np.abs(crystal_a.num_sites - crystal_b.num_sites)
            elif feature == 'volume':
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
    def from_folder(cls, folder: str, extension: str = 'cif'):
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
                                      feature: str = 'density',
                                      iterations: int = 5000) -> list:
        """
        Returns iterations times the Euclidean distance between two randomly chosen structures
        Args:
            feature (str): property that is used for the structure comparisons, available options are
                density, num_sites, volume. Default is density.
            iterations (int): number of comparisons (sampling works with replacement, i.e. the same pair might
                  be sampled several times).

        Returns:
            list of property distances
        """
        distances = self._randomized_structure_property(
            self.structure_list, self.structure_list, feature, iterations)
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
    """
    Comparator to compare the difference or similarity between two distributions. 
    The idea is here to save the test statistics to the object such that we can
    then implement some dunder methods to compare different Comparator objects and
    e.g. find out which distributions are most similar to each other. 
    """

    # ToDo: implement option to provide lists of lists of properties and then loop over the 'feature columns'
    # in the statistical tests

    def __init__(self,
                 structure_list_1: list = [],
                 structure_list_2: list = [],
                 property_list_1: list = [],
                 property_list_2: list = []):
        """

        Args:
            structure_list_1 (list):
            structure_list_2 (list):
        """
        self.structure_list_1 = structure_list_1
        self.structure_list_2 = structure_list_2
        self.property_list_1 = property_list_1
        self.property_list_2 = property_list_2
        self.qq_statistics = None
        self.rmsds = None
        self.jaccards = None
        self.random_structure_property = {}

    @classmethod
    def from_folders(cls, folder_1, folder_2, extension='cif'):
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
        self.jaccards = jaccards
        return jaccards

    def randomized_structure_property(self,
                                      feature: str = 'density',
                                      iterations: int = 5000) -> list:
        """
        Returns iterations times the Euclidean distance between two randomly chosen structures
        Args:
            feature (str): property that is used for the structure comparisons, available options are
                density, num_sites, volume. Default is density.
            iterations (int): number of comparisons (sampling works with replacement, i.e. the same pair might
                  be sampled several times).

        Returns:
            list of property distances
        """
        distances = self._randomized_structure_property(
            self.structure_list_1, self.structure_list_2, feature, iterations)
        self.random_structure_property[feature] = distances
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
        self.rmsds = distances
        return distances

    def qq_test(self, plot: bool = True) -> dict:
        """
        Performs a qq analysis and optionally a plot to compare two distributions.
        Works also for samples of unequal length by using interpolation to find the quantiles of the larger
        sample.
        
        As a measure of 'linearity' we perform a Huber linear regression and return the MSE and R^2 scores of the fit
        and pearson correlation coefficient with two-sided p-value

        Compare https://stackoverflow.com/questions/42658252/how-to-create-a-qq-plot-between-two-samples-of-different-size-in-python,
        from which I took the main outline of this function
        and https://www.itl.nist.gov/div898/handbook/eda/section3/eda33o.htm which contains background information. 

        Args:
            plot (bool): if true, it returns a qq plot with diagonal guideline as well as the Huber regression,
                use %matplotlib inline or %matplotlib notebook when using a jupyter notebook to show the plot inline 

        Returns:
            dictionary with the following statistics
            mse
            r2
            pearson_correlation_coefficient
            pearson_p_value


        """
        property_list_1 = self.property_list_1
        property_list_2 = self.property_list_2

        # Calculate quantiles
        property_list_1.sort()
        quantile_levels1 = np.arange(
            len(property_list_1), dtype=float) / len(property_list_1)

        property_list_2.sort()
        quantile_levels2 = np.arange(
            len(property_list_2), dtype=float) / len(property_list_2)

        # Use the smaller set of quantile levels to create the plot
        quantile_levels = quantile_levels2

        # We already have the set of quantiles for the smaller data set
        quantiles2 = property_list_1

        # We find the set of quantiles for the larger data set using linear interpolation
        quantiles1 = np.interp(quantile_levels, quantile_levels1,
                               property_list_2)

        maxval = max(property_list_1[-1], property_list_2[-1])
        minval = min(property_list_1[0], property_list_2[0])

        hr = HuberRegressor()
        hr.fit(quantiles1, quantiles2)
        predictions = hr.predict(quantiles1)
        mse = mean_squared_error(predictions, quantiles2)
        r2 = r2_score(predictions, quantiles2)
        pearson = pearsonr(quantiles1, quantiles2)

        if plot:
            plt.scatter(quantiles1, quantiles2)
            plt.plot([minval - minval * 0.1, maxval + maxval * 0.1],
                     [minval - minval * 0.1, maxval + maxval * 0.1], '--k')
            plt.plot(quantiles1, predictions, label='Huber Regression')
            plt.legend()

        results_dict = {
            'mse': mse,
            'r2': r2,
            'pearson_correlation_coefficient': pearson[0],
            'pearson_p_value': pearson[1],
        }

        self.qq_statistics = results_dict
        return results_dict

    @staticmethod
    def _mutual_information_2d(x, y, sigma_ratio=0.1, normalized=False):
        """
        Taken from https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
        Modified to automatically adjust bin width, if the distributions to not have the same width,
        both are binned to the optimal number of bins for the shorter distribution.

        Args:
            y:
            normalized:

        Returns:

        """
        EPS = np.finfo(float).eps
        width_x = max(x) - min(x)
        width_y = max(y) - min(y)
        stdev_x = np.std(x)
        stdev_y = np.std(y)

        if width_x < width_y:
            bin = get_number_bins(x)
        else:
            bin = get_number_bins(y)

        if stdev_x < stdev_y:
            sigma = sigma_ratio * stdev_x
        else:
            sigma = sigma_ratio * stdev_y

        bins = (bin, bin)

        # make joint histogram
        jh = np.histogram2d(x, y, bins=bins)[0]

        # smooth the jh with a gaussian filter of given sigma
        ndimage.gaussian_filter(jh, sigma=sigma, mode='constant', output=jh)

        # compute marginal histograms
        jh = jh + EPS
        sh = np.sum(jh)
        jh = jh / sh
        s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
        s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

        # Normalised Mutual Information of:
        # Studholme,  jhill & jhawkes (1998).
        # "A normalized entropy measure of 3-D medical image alignment".
        # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
        if normalized:
            mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2))) / np.sum(
                jh * np.log(jh))) - 1
        else:
            mi = (np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1)) - np.sum(
                s2 * np.log(s2)))

        return mi

    def properties_test_statistics(self):
        """
        Preforms a range of statistical tests of the property distributions and returns a dictionary with the statistics. 
        Returns:

        """

        # Mutual information, continuous form from https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429
        mi = DistComparison._mutual_information_2d(self.property_list_1,
                                                   self.property_list_2)

        # Kolmogorov-Smirnov
        ks = ks_2samp(self.property_list_1, self.property_list_2)

        # Anderson-Darlin


        result_dict = {
            'mutual_information': mi,
            'ks_statistic': ks[0],
            'ks_p_value': ks[1],
        }

        return result_dict

class DistExampleComparison():
    def __init__(self, structure_list, file):
        self.structure_list = structure_list
        self.file = file

    @classmethod
    def from_folder_and_file(class_object, folder, file, extension='cif'):
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

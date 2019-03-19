#!/usr/bin/python
# -*- coding: utf-8 -*-

# Get basic statistics describing the database
# Compare a structure to a database

from tqdm.autonotebook import tqdm
from pymatgen import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from rmsd import parse_periodic_case, rmsd
import random
from scipy.spatial import distance
from scipy import stats
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
logger = logging.getLogger('RemoveDuplicates')
logger.setLevel(logging.DEBUG)


def get_rmsd(structure_a: Structure, structure_b: Structure) -> float:
    p_atoms, P, q_atoms, Q = parse_periodic_case(structure_a, structure_b)
    result = rmsd(P, Q)
    return result


def randomized_rmsd(structure_list: list, iterations: float = 5000) -> list:
    rmsds = []

    for _ in tqdm(range(iterations)):
        random_selection = random.sample(structure_list, 2)
        a = get_rmsd(random_selection[0], random_selection[1])
        rmsds.append(a)

    return rmsds


def randomized_graphs(structure_list: list, iterations=5000) -> list:
    diffs = []
    for _ in tqdm(range(iterations)):
        random_selection = random.sample(structure_list, 2)
        crystal_a = Structure.from_file(random_selection[0])
        crystal_b = Structure.from_file(random_selection[1])
        nn_strategy = JmolNN()
        sgraph_a = StructureGraph.with_local_env_strategy(
            crystal_a, nn_strategy)
        sgraph_b = StructureGraph.with_local_env_strategy(
            crystal_b, nn_strategy)
        diffs.append(sgraph_a.diff(sgraph_b, strict=False))
    return diffs


def randomized_density():
    ...


def closest_index(array, target):
    return np.argmin(np.abs(array - target))


def property_based_distances(structure_list: list, property_list: list,
                             other_structure: str) -> pd.DataFrame:
    """
    Compares other structure to
        - lowest, highest, median, mean and random structure from structure list
    Returns the RMSD and the graph jaccard distance between other structure and the
    aforementioned structures.

    This can be useful in case there is a direct link between structure and property
    and we want to make sure that a structure is not to dissimilar from the set of structures
    in structure_list.

    :param structure_list:
    :param property_list:
    :param other_structure:
    :return:
    """

    median_index = closest_index(property_list, np.median(property_list))
    mean_index = closest_index(property_list, np.mean(property_list))
    random_index = np.random.randint(0, len(structure_list))
    q25_index = closest_index(property_list, np.quantile(property_list, 0.25))
    q75_index = closest_index(property_list, np.quantile(property_list, 0.75))

    lowest_structure = Structure.from_file(
        structure_list[np.argmin(property_list)])
    highest_structure = Structure.from_file(
        structure_list[np.argmax(property_list)])
    median_structure = Structure.from_file(structure_list[median_index])
    mean_structure = Structure.from_file(structure_list[mean_index])
    random_structure = Structure.from_file(structure_list[random_index])
    q25_structure = Structure.from_file(structure_list[q25_index])
    q75_structure = Structure.from_file(structure_list[q75_index])
    other_structure = Structure.from_file(other_structure)

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
        other_structure_graph.diff(mean_structure_graph, strict=False)['diff'],
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
        other_structure_graph.diff(q25_structure_graph, strict=False)['diff'],
        'q75_jaccard':
        other_structure_graph.diff(q75_structure_graph, strict=False)['diff'],
    }

    return pd.DataFrame(distances)


def kde_probability_observation(observations, other_observation) -> float:
    """
    Performs a KDE on the list of observation and the returns the
    log likelihood of the data under the kde model
    :param observations:
    :param other_observation:
    :return:
    """
    observations_array = np.array(observations)
    other_observation_array = np.array(other_observation)
    assert len(other_observation.shape) == len(observations_array.shape)

    if len(observations_array.shape) == 1:
        observations_array = observations_array.reshape(-1, 1)
        other_observation_array = other_observation_array.reshape(-1, 1)

    kd = KernelDensity(
        kernel='gaussian', bandwidth=0.75).fit(observations_array)
    return kd.score(other_observation_array)


def kl_divergence(array_1, array_2):
    """
    KL divergence could be used a measure of covariate shift.
    :param array_1:
    :param array_2:
    :return:
    """

    a = np.asarray(array_1, dtype=np.float)
    a /= a.sum()
    b = np.asarray(array_2, dtype=np.float)
    b /= b.sum()

    if len(a) > len(b):
        np.random.shuffle(a)
        a = a[:len(b)]
    elif len(b) > len(a):
        np.random.shuffle(b)
        b = b[:len(a)]

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


def tanimoto_distance(array_1, array_2):
    """
    Continous form of the Tanimoto distance measure.
    :param array_1:
    :param array_2:
    :return:
    """
    xy = np.dot(array_1, array_2)
    return xy / (np.abs(array_1) + np.abs(array_2) - xy)


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

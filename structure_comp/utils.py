#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Kevin M. Jablonka'
__copyright__ = 'MIT License'
__maintainer__ = 'Kevin M. Jablonka'
__email__ = 'kevin.jablonka@epfl.ch'
__version__ = '0.1.0'
__date__ = '25.03.19'
__status__ = 'First Draft, Testing'

from glob import glob
import os
import functools
from .rmsd import parse_periodic_case, rmsd
from pymatgen import Structure
import numpy as np
import concurrent.futures
from sklearn.neighbors import KernelDensity
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
import logging
logger = logging.getLogger()


def get_structure_list(directory: str, extension: str = 'cif') -> list:
    """
    :param directory: path to directory
    :param extension: fileextension
    :return:
    """
    logger.info('getting structure list')
    if extension:
        structure_list = glob(
            os.path.join(directory, ''.join(['*.', extension])))
    else:
        structure_list = glob(os.path.join(directory, '*'))
    return structure_list


@functools.lru_cache(maxsize=128, typed=False)
def get_rmsd(structure_a: Structure, structure_b: Structure) -> float:
    p_atoms, P, q_atoms, Q = parse_periodic_case(structure_a, structure_b)
    result = rmsd(P, Q)
    return result


def closest_index(array, target):
    return np.argmin(np.abs(array - target))


def kde_probability_observation(observations, other_observation) -> float:
    """
    Performs a KDE on the list of observation and the returns the
    log likelihood of the data under the kde model
    Args:
        observations:
        other_observation:

    Returns:

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
    Continuous form of the Tanimoto distance measure.
    Args:
        array_1:
        array_2:

    Returns:

    """
    xy = np.dot(array_1, array_2)
    return xy / (np.abs(array_1) + np.abs(array_2) - xy)


def get_hash(structure: Structure):
    """
    This gets hash for the Niggli reduced cell

    Args:
        structure: pymatgen structure object

    Returns:

    """
    crystal = structure.get_reduced_structure()
    nn_strategy = JmolNN()
    sgraph_a = StructureGraph.with_local_env_strategy(crystal, nn_strategy)
    graph_hash = str(hash(sgraph_a.graph))
    comp_hash = str(hash(str(crystal.composition)))
    density_hash = str(hash(crystal.density))
    return graph_hash + comp_hash + density_hash

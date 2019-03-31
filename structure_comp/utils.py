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
from scipy import stats
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
    _, P, _, Q = parse_periodic_case(structure_a, structure_b)
    result = rmsd(P, Q)
    return result


def closest_index(array, target):
    return np.argmin(np.abs(array - target))


def get_number_bins(array):
    """
    Get optimal number of bins according to the Freedman-Diaconis rule (https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule) 
    Args:
        array:

    Returns:
        number of bins
    """
    h = 2 * stats.iqr(array) * len(array)**(- 1.0 / 3.0)
    return int((max(array) - min(array)) / h)


def kl_divergence(array_1, array_2, bins=None):
    """
    KL divergence could be used a measure of covariate shift.
    """

    minimum =min([min(array_1), min(array_2)])
    maximum = max([max(array_1), max(array_2)])

    if bins is None:
        if len(array_1) < len(array_2):
            bins = get_number_bins(array_1)
        else:
            bins = get_number_bins(array_2)

    a = np.histogram(array_1, bins=bins, range=(minimum, maximum))[0]
    b = np.histogram(array_2, bins=bins, range=(minimum, maximum))[0]

    return stats.entropy(a, b)


def tanimoto_distance(array_1, array_2):
    """
    Continuous form of the Tanimoto distance measure.
    Args:
        array_1:
        array_2:

    Returns:

    """
    xy = np.dot(array_1, array_2)
    return xy / (np.sum(array_1**2) + np.sum(array_2**2) - xy)


def get_hash(structure: Structure, get_niggli=True):
    """
    This gets hash, using the structure graph as a part of the has

    Args:
        structure: pymatgen structure object
        get_niggli (bool):
    Returns:

    """
    if get_niggli:
        crystal = structure.get_reduced_structure()
    else:
        crystal = structure
    nn_strategy = JmolNN()
    sgraph_a = StructureGraph.with_local_env_strategy(crystal, nn_strategy)
    graph_hash = str(hash(sgraph_a.graph))
    comp_hash = str(hash(str(crystal.symbol_set)))
    density_hash = str(hash(crystal.density))
    return graph_hash + comp_hash + density_hash


def get_cheap_hash(structure: Structure, get_niggli=True):
    """
    This gets hash based on composition and density

    Args:
        structure: pymatgen structure object
        get_niggli (bool):
    Returns:

    """
    if get_niggli:
        crystal = structure.get_reduced_structure()
    else:
        crystal = structure
    comp_hash = str(hash(str(crystal.symbol_set)))
    density_hash = str(hash(crystal.density))
    return comp_hash + density_hash
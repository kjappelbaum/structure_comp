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
import re
import unicodedata
from collections import defaultdict  # thanks Raymond Hettinger!
import functools
from .rmsd import parse_periodic_case, kabsch_rmsd
from pymatgen import Structure
import numpy as np
from scipy import stats
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core import Molecule
import networkx as nx
import random
import logging
logger = logging.getLogger()


def get_structure_list(directory: str, extension: str = 'cif') -> list:
    """

    Args:
        directory:
        extension:

    Returns:

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
    result = kabsch_rmsd(P, Q)
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
    h = 2 * stats.iqr(array) * len(array)**(-1.0 / 3.0)
    return int((max(array) - min(array)) / h)


def kl_divergence(array_1, array_2, bins=None):
    """
    KL divergence could be used a measure of covariate shift.
    """

    minimum = min([min(array_1), min(array_2)])
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


def attempt_supercell_pymatgen(structure_1: Structure,
                               structure_2: Structure) -> Structure:
    lattice1 = np.array(structure_1.lattice.abc)
    lattice2 = np.array(structure_2.lattice.abc)

    one_larger_than_two = False

    if structure_1.volume > structure_2.volume:
        factors = lattice1 / lattice2
        one_larger_than_two = True
    else:
        factors = lattice2 / lattice1

    x = np.array(factors)
    x_int = x.astype(int)
    if np.all(np.isclose(x, x_int, 0.001)):
        x = x_int
        logger.debug('found supercell with scaling factors %s', x)
        if one_larger_than_two:
            structure_2 = structure_2 * x
        else:
            structure_1 = structure_1 * x

    return structure_1, structure_2


def slugify(value, allow_unicode=False):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.

    source: https://github.com/django/django/blob/master/django/utils/text.py
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode(
            'ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value).strip().lower()
    return re.sub(r'[-\s]+', '-', value)


def incremental_farthest_search(points, k):
    """
    Source: https://flothesof.github.io/farthest-neighbors.html
    Args:
        points:
        k:

    Returns:

    """
    remaining_points = points[:]
    solution_set = []
    solution_set.append(remaining_points.pop(\
                                             random.randint(0, len(remaining_points) - 1)))
    for _ in range(k - 1):
        distances = [
            np.linalg.norm(p - solution_set[0]) for p in remaining_points
        ]
        for i, p in enumerate(remaining_points):
            for _, s in enumerate(solution_set):
                distances[i] = min(distances[i], np.linalg.norm(p - s))
        solution_set.append(
            remaining_points.pop(distances.index(max(distances))))
    return solution_set


def get_symbol_list(structure: Structure) -> list:
    """
    Utility function to return symbol list even for structures with partial occupancy.
    Args:
        structure (pymatgen structure object):

    Returns:

    """
    return [s.species_string.split(':')[0] for s in structure.sites]


def get_symbol_indices(structure: Structure) -> list:
    """
    Utility function to get the symbols of a site even for a structure with partial occupancies.

    Args:
        structure:

    Returns:

    """
    symbol_list = get_symbol_list(structure)
    index_dict = defaultdict(list)
    for idx, item in enumerate(symbol_list):
        index_dict[item].append(idx)
    return dict(index_dict)


def get_subgraphs_as_molecules_all(sg, use_weights=False):
    """
    Adapatation of
    http://pymatgen.org/_modules/pymatgen/analysis/graphs.html#StructureGraph.get_subgraphs_as_molecules
    for our needs

    Args:
        sg: structure graph
        use_weights:

    Returns:
        list of molecules
    """

    # creating a supercell is an easy way to extract
    # molecules (and not, e.g., layers of a 2D crystal)
    # without adding extra logic

    supercell_sg = sg * (3, 3, 3)

    # make undirected to find connected subgraphs
    supercell_sg.graph = nx.Graph(supercell_sg.graph)

    # find subgraphs
    all_subgraphs = list(nx.connected_component_subgraphs(supercell_sg.graph))

    # discount subgraphs that lie across *supercell* boundaries
    # these will subgraphs representing crystals
    molecule_subgraphs = []
    for subgraph in all_subgraphs:
        intersects_boundary = any([
            d['to_jimage'] != (0, 0, 0)
            for u, v, d in subgraph.edges(data=True)
        ])
        if not intersects_boundary:
            molecule_subgraphs.append(subgraph)

    # add specie names to graph to be able to test for isomorphism
    for subgraph in molecule_subgraphs:
        for n in subgraph:
            subgraph.add_node(n, specie=str(supercell_sg.structure[n].specie))

    # now define how we test for isomorphism
    def node_match(n1, n2):
        return n1['specie'] == n2['specie']

    def edge_match(e1, e2):
        if use_weights:
            return e1['weight'] == e2['weight']
        else:
            return True

    # get Molecule objects for each subgraph
    molecules = []
    for subgraph in molecule_subgraphs:

        coords = [supercell_sg.structure[n].coords for n in subgraph.nodes()]
        species = [supercell_sg.structure[n].specie for n in subgraph.nodes()]

        molecule = Molecule(species, coords)

        molecules.append(molecule)

    return molecules

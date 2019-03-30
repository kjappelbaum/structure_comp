#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Kevin M. Jablonka'
__copyright__ = 'MIT License'
__maintainer__ = 'Kevin M. Jablonka'
__email__ = 'kevin.jablonka@epfl.ch'
__version__ = '0.1.0'
__date__ = '18.03.19'
__status__ = 'First Draft, Testing'

import os
from pathlib import Path
import numpy as np
import shutil
from pymatgen import Structure
import concurrent.futures
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from scipy.spatial.distance import pdist, squareform
import tempfile
import logging
from ase.visualize.plot import plot_atoms
from ase.io import read
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import pandas as pd
from .rmsd import parse_periodic_case, rmsd
from .utils import get_structure_list, get_hash
from collections import defaultdict

logger = logging.getLogger('RemoveDuplicates')
logger.setLevel(logging.DEBUG)


class RemoveDuplicates():
    """
    A RemoveDuplicates object operates on a collection of structure and allows
        - Removal of duplicates on the collection of structures using different methods, using the main
        function run_filtering()

        - Basic comparisons between different RemoveDuplicates objects (e.g. comparing which one contains more duplicates)
    """

    def __init__(self,
                 structure_list: list,
                 cached: bool = False,
                 method='standard'):

        self.structure_list = structure_list
        self.reduced_structure_dict = {}
        self.cached = cached
        self.pairs = None
        self.method = method
        self.similar_composition_tuples = []

    def __repr__(self):
        return f'RemoveDuplicates on {len(self.structure_list)!r} structures'

    @classmethod
    def from_folder(cls,
                    folder,
                    cached: bool = False,
                    extension='cif',
                    method='standard'):
        """

        Args:
            folder (str): path to folder that is used for construction of the RemoveDuplicates object
            reduced_structure_dir (str): name in which tempera
            extension:
            remove_reduced_structure_dir:
            method:

        Returns:

        """
        sl = get_structure_list(folder, extension)
        return cls(sl, cached, method)

    # Implement some logic in case someone wants to compare dbs
    def __len__(self):
        if self.pairs is not None:
            return len(self.pairs)
        else:
            return 0

    def __eq__(self, other):
        return set(self.pairs) == set(other.pairs)

    def __gt__(self, other):
        return len(self.pairs) > len(other.pairs)

    def __lt__(self, other):
        return len(self.pairs) < len(other.pairs)

    def __ge__(self, other):
        return len(self.pairs) >= len(other.pairs)

    def __le__(self, other):
        return len(self.pairs) <= len(other.pairs)

    def __iter__(self):
        return iter(self.pairs)

    def get_reduced_structure(self, structure):
        sname = Path(structure).name
        stem = Path(structure).stem
        crystal = Structure.from_file(structure)
        crystal = crystal.get_reduced_structure()
        if not self.cached:
            crystal.to(filename=os.path.join(self.tempdirpath, sname))
        else:
            self.reduced_structure_dict[stem] = crystal
        return stem

    def get_reduced_structures(self):
        """
        To make calculations cheaper, we first get Niggli cells.
        If caching is turned off, the structures are written to a temporary directory (useful for large
        databases), otherwise the reduced structures are stored in memory.
        """
        if not self.cached:
            self.tempdirpath = tempfile.mkdtemp()
            self.reduced_structure_dir = self.tempdirpath
        logger.info('creating reduced structures')
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for _ in tqdm(
                    executor.map(self.get_reduced_structure,
                                 self.structure_list),
                    total=len(self.structure_list)):
                logger.debug('reduced structure for {} created'.format(_))

    @staticmethod
    def get_scalar_features(structure: Structure):
        """
        Computes number of atoms and density for a pymatgen structure object.
        Args:
            structure:

        Returns:

        """
        number_atoms = structure.num_sites
        density = structure.density
        return number_atoms, density

    @staticmethod
    def get_scalar_features_from_file(structure_file):
        """
        Computes number of atoms and density for structure file.
        Args:
            structure_file:

        Returns:

        """
        structure = Structure.from_file(structure_file)
        number_atoms = structure.num_sites
        density = structure.density
        return number_atoms, density

    @staticmethod
    def get_scalar_df(reduced_structure_list: list):
        """

        Args:
            reduced_structure_list:

        Returns:

        """
        feature_list = []
        logger.info('creating scalar features')

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for structure, result in tqdm(
                    zip(
                        reduced_structure_list,
                        executor.map(
                            RemoveDuplicates.get_scalar_features_from_file,
                            reduced_structure_list)),
                    total=len(reduced_structure_list)):
                features = {
                    'name': structure,
                    'number_atoms': result[0],
                    'density': result[1]
                }
                feature_list.append(features)
            df = pd.DataFrame(feature_list)
        return df

    @staticmethod
    def get_scalar_df_cached(reduced_structure_dict: dict):
        """

        Args:
            reduced_structure_dict:

        Returns:

        """
        feature_list = []
        logger.info('creating scalar features')
        for structure in tqdm(reduced_structure_dict):
            crystal = reduced_structure_dict[structure]
            number_atoms, density = RemoveDuplicates.get_scalar_features(
                crystal)
            features = {
                'name': structure,
                'number_atoms': number_atoms,
                'density': density
            }
            feature_list.append(features)
        df = pd.DataFrame(feature_list)
        logger.debug('The df looks like'.format(df.head()))

        return df

    @staticmethod
    def get_scalar_distance_matrix(scalar_feature_df: pd.DataFrame,
                                   threshold: float = 0.5) -> list:
        """
        Get structures that probably have the same composition.

        Args:
            scalar_feature_df: pandas Dataframe object with the scalar features
            threshold: threshold: threshold for the Euclidean distance between structure features

        Returns:
            list of tuples which Euclidean distance is under threshold

        """
        distances = pdist(
            scalar_feature_df.drop(columns=['name']).values,
            metric='euclidean')
        dist_matrix = squareform(distances)
        i, j = np.where(dist_matrix + np.eye(len(dist_matrix)) < threshold)
        logger.debug('found {} and {} composition duplicates'.format(i, j))
        return list(set(map(tuple, map(sorted, list(zip(i,
                                                        j))))))  # super ugly

    @staticmethod
    def compare_rmsd(tupellist: list,
                     scalar_feature_df: pd.DataFrame,
                     threshold: float = 0.2) -> list:
        """

        Args:
            tupellist:
            scalar_feature_df:
            threshold:

        Returns:

        """
        logger.info('doing RMSD comparison')
        pairs = []
        for items in tqdm(tupellist):
            if items[0] != items[1]:
                p_atoms, P, q_atoms, Q = parse_periodic_case(
                    scalar_feature_df.iloc[items[0]]['name'],
                    scalar_feature_df.iloc[items[1]]['name'])
                result = rmsd(P, Q)
                if result < threshold:
                    pairs.append(items)
        return pairs

    def compare_graph_pair(self, items):
        nn_strategy = JmolNN()
        crystal_a = Structure.from_file(
            self.scalar_feature_matrix.iloc[items[0]]['name'])
        crystal_b = Structure.from_file(
            self.scalar_feature_matrix.iloc[items[1]]['name'])
        sgraph_a = StructureGraph.with_local_env_strategy(
            crystal_a, nn_strategy)
        sgraph_b = StructureGraph.with_local_env_strategy(
            crystal_b, nn_strategy)
        try:
            if sgraph_a == sgraph_b:
                logger.debug('Found duplicate')
                return items
        except ValueError:
            logger.debug('Structures were probably not different')
            return False

    def compare_graph_pair_cached(self, items):
        nn_strategy = JmolNN()
        crystal_a = self.reduced_structure_dict[self.scalar_feature_matrix.
                                                iloc[items[0]]['name']]
        crystal_b = self.reduced_structure_dict[self.scalar_feature_matrix.
                                                iloc[items[1]]['name']]
        sgraph_a = StructureGraph.with_local_env_strategy(
            crystal_a, nn_strategy)
        sgraph_b = StructureGraph.with_local_env_strategy(
            crystal_b, nn_strategy)
        try:
            if sgraph_a == sgraph_b:
                logger.debug('Found duplicate')
                return items
        except ValueError:
            logger.debug('Structures were probably not different')
            return False

    def compare_graphs(self, tupellist: list) -> list:
        """

        Args:
            tupellist:
            scalar_feature_df:

        Returns:

        """
        logger.info('constructing and comparing structure graphs')
        pairs = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            if not self.cached:
                for _, result in tqdm(
                        zip(tupellist,
                            executor.map(self.compare_graph_pair, tupellist)),
                        total=len(tupellist)):
                    logger.debug(result)
                    if result:
                        pairs.append(result)
            else:
                for _, result in tqdm(
                        zip(
                            tupellist,
                            executor.map(self.compare_graph_pair_cached,
                                         tupellist)),
                        total=len(tupellist)):
                    logger.debug(result)
                    if result:
                        pairs.append(result)
        return pairs

    def janitor(self):
        logger.debug('cleaning directory up')
        shutil.rmtree(self.tempdirpath)

    def get_graph_hash_dict(self, structure):
        crystal = Structure.from_file(structure)
        name = Path(structure).name
        hash = get_hash(crystal)
        self.hash_dict[hash].append(name)

    def get_graph_hash_dicts(self):
        self.hash_dict = defaultdict(list)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for structure in tqdm(
                    zip(
                        self.structure_list,
                        executor.map(self.get_graph_hash_dict,
                                     self.structure_list)),
                    total=len(self.structure_list)):
                logger.debug('getting hash for %s', structure)

        return self.hash_dict

    def run_filtering(self):
        """

        Returns:

        """
        logger.info('running filtering workflow')

        if self.method == 'standard':
            self.get_reduced_structures()

            if not self.cached:
                self.reduced_structure_list = get_structure_list(
                    self.reduced_structure_dir)
                logger.debug('we have %s reduced structures',
                             len(self.reduced_structure_list))

                self.scalar_feature_matrix = RemoveDuplicates.get_scalar_df(
                    self.reduced_structure_list)
            else:
                logger.debug('we have %s reduced structures',
                             len(self.reduced_structure_dict))
                self.scalar_feature_matrix = RemoveDuplicates.get_scalar_df_cached(
                    self.reduced_structure_dict)

            logger.debug('columns of dataframe are %s',
                         self.scalar_feature_matrix.columns)

            self.similar_composition_tuples = RemoveDuplicates.get_scalar_distance_matrix(
                self.scalar_feature_matrix)

            self.pairs = self.compare_graphs(self.similar_composition_tuples)

        elif self.method == 'hash':
            RemoveDuplicates.get_graph_hash_dict(self.structure_list)

    @staticmethod
    def get_rmsd_matrix():
        return NotImplementedError

    @staticmethod
    def get_jacard_graph_distance_matrix():
        return NotImplementedError

    @property
    def number_of_duplicates(self):
        try:
            if self.pairs:
                number_duplicates = len(self.pairs)
            else:
                number_duplicates = 0
        except AttributeError:
            number_duplicates = None
        return number_duplicates

    @property
    def duplicates(self):
        try:
            if self.pairs:
                duplicates = []
                for items in self.pairs:
                    name1 = Path(
                        self.scalar_feature_matrix.iloc[items[0]]['name']).name
                    name2 = Path(
                        self.scalar_feature_matrix.iloc[items[1]]['name']).name
                    duplicates.append((name1, name2))
            else:
                duplicates = 0
        except AttributeError:
            duplicates = None
        return duplicates

    def inspect_duplicates(self, mode: str = 'ase'):
        if mode == 'ase':
            if self.pairs:
                for items in self.pairs:
                    fig, axarr = plt.subplots(1, 2, figsize=(15, 5))
                    plot_atoms(
                        read(
                            self.scalar_feature_matrix.iloc[items[0]]['name']),
                        axarr[0])
                    plot_atoms(
                        read(
                            self.scalar_feature_matrix.iloc[items[1]]['name']),
                        axarr[1])
            else:
                logger.info('no duplicates to plot')

    def remove_duplicates(self):
        try:
            for items in self.pairs:
                os.remove(self.scalar_feature_matrix.iloc[items[0]]['name'])

            # Should we now also clean the pair list?
        except Exception:
            logger.error('Could not delete duplicates')

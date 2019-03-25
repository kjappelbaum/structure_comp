#!/usr/bin/python
# -*- coding: utf-8 -*-

# Efficient duplicate removal in large databases (not O(N^2)
# such as in https://pubs.acs.org/doi/pdf/10.1021/acs.chemmater.5b03836)
# Making it simpler, maybe cheaper and more general
# than https://pubs.acs.org/doi/pdf/10.1021/acs.cgd.7b01663
# Should work without a threshold

import os
from pathlib import Path
import numpy as np
import shutil
from glob import glob
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
from .utils import get_structure_list
from .comparators import get_hash
from collections import defaultdict

logger = logging.getLogger('RemoveDuplicates')
logger.setLevel(logging.DEBUG)


class RemoveDuplicates():
    """
    A RemoveDuplicates object operates on a collection of structure and allows
        - Removal of duplicates on the collection of structures using different methods
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

    @classmethod
    def from_folder(class_object,
                    folder,
                    cached: bool = False,
                    extension='.cif',
                    remove_reduced_structure_dir: bool = True,
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
        return class_object(sl, cached, remove_reduced_structure_dir, method)

    # Implement some logic in case someone wants to compare dbs
    def __len__(self):
        return len(self.pairs)

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

    def get_reduced_structures(self):
        """
        To make calculations cheaper, we first get Niggli cells.
        If caching is turned off, the structures are written to a temporary directory (useful for large
        databases), otherwise the reduced structures are stored in memory.
        """
        if not self.cached:
            self.tempdirpath = tempfile.mkdtemp()
        logger.info('creating reduced structures')
        for structure in tqdm(self.structure_list):
            sname = Path(structure).name
            stem = Path(structure).stem
            crystal = Structure.from_file(structure)
            crystal = crystal.get_reduced_structure()
            if not self.cached:
                crystal.to(filename=os.path.join(self.tempdirpath, sname))
            else:
                self.reduced_structure_dict[stem] = crystal

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

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for structure, number_atoms, density in tqdm(executor.map(
                    RemoveDuplicates.get_scalar_features_from_file,
                    reduced_structure_list), total=len(reduced_structure_list)):
                number_atoms, density = RemoveDuplicates.get_scalar_features(
                    structure)
                features = {
                    'name': structure,
                    'number_atoms': number_atoms,
                    'density': density
                }
                feature_list.append(features)

        return pd.DataFrame(feature_list)

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

        return pd.DataFrame(feature_list)

    @staticmethod
    def get_scalar_distance_matrix(scalar_feature_df: pd.DataFrame,
                                   threshold: float = 0.1) -> list:
        """
        Get structures that probably have the same composition.
        :param scalar_feature_df: pandas Dataframe object with the scalar features
        :param threshold: threshold for the euclidian distance between structure features
        :return: list of tuples which euclidian distance is under threshold
        """
        logger.debug('columns of dataframe are {}'.format(
            scalar_feature_df.columns))
        distances = pdist(
            scalar_feature_df.drop(columns=['name']).values,
            metric='euclidean')
        dist_matrix = squareform(distances)
        i, j = np.where(dist_matrix < threshold)
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

    @staticmethod
    def compare_graph_pair(items, scalar_feature_df):

        nn_strategy = JmolNN()
        crystal_a = Structure.from_file(
            scalar_feature_df.iloc[items[0]]['name'])
        crystal_b = Structure.from_file(
            scalar_feature_df.iloc[items[1]]['name'])
        sgraph_a = StructureGraph.with_local_env_strategy(
            crystal_a, nn_strategy)
        sgraph_b = StructureGraph.with_local_env_strategy(
            crystal_b, nn_strategy)
        try:
            if sgraph_a == sgraph_b:
                return items
        except ValueError:
            logger.debug('Structures were probably not different')
            return None

    def compare_graph_pair_cached(self, items, scalar_feature_df):

        nn_strategy = JmolNN()
        crystal_a = self.reduced_structure_dict[
            scalar_feature_df.iloc[items[0]]['name'])]
        crystal_b = self.reduced_structure_dict[
            scalar_feature_df.iloc[items[1]]['name'])]
        sgraph_a = StructureGraph.with_local_env_strategy(
            crystal_a, nn_strategy)
        sgraph_b = StructureGraph.with_local_env_strategy(
            crystal_b, nn_strategy)
        try:
            if sgraph_a == sgraph_b:
                return items
        except ValueError:
            logger.debug('Structures were probably not different')
            return None


    @staticmethod
    def compare_graphs(tupellist: list,
                       scalar_feature_df: pd.DataFrame) -> list:
        """

        Args:
            tupellist:
            scalar_feature_df:

        Returns:

        """
        logger.info('constructing and comparing structure graphs')
        pairs = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for items, result in tqdm(
                    executor.map(RemoveDuplicates.compare_graph_pair,
                                 tupellist, scalar_feature_df),
                    total=len(tupellist)):
                if result:
                    pairs.append(result)
        return pairs

    def janitor(self):
        logger.debug('cleaning directory up')
        shutil.rmtree(self.tempdirpath)

    @staticmethod
    def get_graph_hash_dict(structure_list: list):
        hash_dict = defaultdict(list)
        for structure in structure_list:
            crystal = Structure.from_file(structure)
            name = Path(structure).name
            hash = get_hash(crystal)
            hash_dict[hash].append(name)
        return hash_dict

    def run_filtering(self):
        """

        Returns:

        """
        logger.info('running filtering workflow')

        if self.method == 'standard':
            RemoveDuplicates.get_reduced_structures()

            if not self.cached:
                self.reduced_structure_list = get_structure_list(
                    self.reduced_structure_dir)

            self.scalar_feature_matrix = RemoveDuplicates.get_scalar_df(
                self.reduced_structure_list)

            logger.debug('columns of dataframe are {}'.format(
                self.scalar_feature_matrix.columns))

            self.similar_composition_tuples = RemoveDuplicates.get_scalar_distance_matrix(
                self.scalar_feature_matrix)

            self.pairs = RemoveDuplicates.compare_graphs(
                self.similar_composition_tuples, self.scalar_feature_matrix)

        elif self.method == 'graph_hash':
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

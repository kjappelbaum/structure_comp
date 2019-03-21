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
from concurrent import futures
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from scipy.spatial.distance import pdist, squareform
import logging
from ase.visualize.plot import plot_atoms
from ase.io import read
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import pandas as pd
from .rmsd import parse_periodic_case, rmsd

logger = logging.getLogger('RemoveDuplicates')
logger.setLevel(logging.DEBUG)


class RemoveDuplicates():
    def __init__(self,
                 structure_list: list,
                 reduced_structure_dir: str,
                 remove_reduced_structure_dir: bool = True):

        self.structure_list = structure_list
        self.reduced_structure_dir = reduced_structure_dir
        self.remove_rsd = remove_reduced_structure_dir
        self.pairs = None

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

    @staticmethod
    def get_reduced_structures(structure_list: list, new_dir: str):
        """
        To make feature calculation cheaper and to
        avoid issues with supercells.
        :param structure_list: list of paths to structure files (format that pymatgen can read)
        :param new_dir:  string with the name of the new directory for the reduced structures
        :return:
        """
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        logger.info('creating reduced structures')
        for structure in tqdm(structure_list):
            sname = Path(structure).name
            crystal = Structure.from_file(structure)
            crystal = crystal.get_reduced_structure()
            crystal.to(filename=os.path.join(new_dir, sname))

    @staticmethod
    def get_scalar_features(structure: Structure):
        """
        Get features for the first comparison matrix.
        :param structure: pymatgen structure object
        :return:
            number_atoms: float
            density: float
        """
        number_atoms = structure.num_sites
        density = structure.density
        return number_atoms, density

    @staticmethod
    def get_scalar_df(reduced_structure_list: list):
        """
        Collect scalar features in a dataframe
        :param reduced_structure_list: list of paths to reduced structures
        :return: pandas dataframe with scalar features (currently name, number of atoms and density)
        """
        feature_list = []
        logger.info('creating scalar features')
        for structure in tqdm(reduced_structure_list):
            crystal = Structure.from_file(structure)
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
        Potentially, we could run the RMSD code to detect same structures
        :param tupellist: list of tuples (indices of dataframe)
        :param scalar_feature_df: dataframe that contains the name column
        :param threshold: float for a RMSD structure distance threshold for consideration as duplicate
        :return:
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
    def compare_graphs(tupellist: list,
                       scalar_feature_df: pd.DataFrame) -> list:
        """
        Filter structures with same structure graph
        :param tupellist: list of tuples (indices of dataframe)
        :param scalar_feature_df: dataframe that contains the name column
        :return: list of tuples of structures that are identical
        """
        logger.info('constructing and comparing structure graphs')
        pairs = []
        for items in tqdm(tupellist):
            if items[0] != items[1]:
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
                        pairs.append(items)
                except ValueError:
                    logger.debug('Structures were probably not different')
        return pairs

    @staticmethod
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

    def janitor(self):
        logger.debug('cleaning directory up')
        shutil.rmtree(self.reduced_structure_dir)

    def run_filtering(self):
        """
        Runs the on the database.
        :return:
        """

        logger.info('running filtering workflow')

        RemoveDuplicates.get_reduced_structures(self.structure_list,
                                                self.reduced_structure_dir)

        self.reduced_structure_list = RemoveDuplicates.get_structure_list(
            self.reduced_structure_dir)

        self.scalar_feature_matrix = RemoveDuplicates.get_scalar_df(
            self.reduced_structure_list)

        logger.debug('columns of dataframe are {}'.format(
            self.scalar_feature_matrix.columns))

        self.similar_composition_tuples = RemoveDuplicates.get_scalar_distance_matrix(
            self.scalar_feature_matrix)

        self.pairs = RemoveDuplicates.compare_graphs(
            self.similar_composition_tuples, self.scalar_feature_matrix)

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

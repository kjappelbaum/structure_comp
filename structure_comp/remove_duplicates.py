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
from pymatgen.io.ase import AseAtomsAdaptor
from scipy.spatial.distance import pdist, squareform
import tempfile
import logging
from ase.visualize.plot import plot_atoms
from ase.io import read, write
from ase.build import niggli_reduce
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
import pandas as pd
from .rmsd import parse_periodic_case, kabsch_rmsd
from .utils import get_structure_list, get_hash, attempt_supercell_pymatgen
from collections import defaultdict

logger = logging.getLogger('RemoveDuplicates')
logger.setLevel(logging.DEBUG)


# ToDo: add XTalComp support
# ToDo: more useful error message when file cannot be read
# ToDo: run it in a contextmanager?

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
                 method='standard',
                 try_supercell=True):

        self.structure_list = structure_list
        self.reduced_structure_dict = None
        self.cached = cached
        self.pairs = None
        self.method = method
        self.similar_composition_tuples = []
        self.try_supercell = try_supercell

    def __repr__(self):
        return f'RemoveDuplicates on {len(self.structure_list)!r} structures'

    @classmethod
    def from_folder(cls,
                    folder,
                    cached: bool = False,
                    extension='cif',
                    method='standard',
                    try_supercell=True):
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
        return cls(sl, cached, method, try_supercell)

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
        if self.cached:
            self.reduced_structure_dict = {}
        try:
            # Cif reader in ASE seems more stable to me, especially for CSD data
            atoms = read(structure)
        except Exception:
            logger.error('Could not read structure %s', stem)
        else:
            niggli_reduce(atoms)
            if not self.cached:
                write(os.path.join(self.tempdirpath, sname), atoms)
            else:
                crystal = AseAtomsAdaptor.get_structure(atoms)
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
        symbol_hash = hash(structure.symbol_set)
        density = structure.density
        return symbol_hash, density

    @staticmethod
    def get_scalar_features_from_file(structure_file):
        """
        Computes number of atoms and density for structure file.
        Args:
            structure_file:

        Returns:

        """
        structure = Structure.from_file(structure_file)
        symbol_hash = hash(structure.symbol_set)
        density = structure.density
        return symbol_hash, density

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
                    'symbol_hash': result[0],
                    'density': result[1]
                }
                feature_list.append(features)
            df = pd.DataFrame(feature_list)
            logger.debug('the dataframe looks like %s', df.head())
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
            symbol_hash, density = RemoveDuplicates.get_scalar_features(
                crystal)
            features = {
                'name': structure,
                'symbol_hash': symbol_hash,
                'density': density
            }
            feature_list.append(features)
        df = pd.DataFrame(feature_list)
        logger.debug('the dataframe looks like %s', df.head())

        return df

    @staticmethod
    def get_scalar_distance_matrix(scalar_feature_df: pd.DataFrame,
                                   threshold: float = 0.001) -> list:
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
        np.fill_diagonal(dist_matrix, 1)
        i, j = np.where(dist_matrix < threshold)
        duplicates = list(set(map(tuple, map(sorted, list(zip(i,
                                                 j))))))

        logger.debug('found {} and {} composition duplicates'.format(i, j))
        return duplicates

    @staticmethod
    def compare_rmsd(tupellist: list,
                     scalar_feature_df: pd.DataFrame,
                     threshold: float = 0.05,
                     try_supercell: bool = True,
                     reduced_structure_dict=None) -> list:
        """

        Args:
            tupellist (list): list of indices of structures with identical compostion
            scalar_feature_df (pandas dataframe):
            threshold:
            try_supercell (bool): switch which control whether expansion to supercell is tested

        Returns:

        """
        logger.info('doing RMSD comparison')
        pairs = []
        for items in tqdm(tupellist):
            if reduced_structure_dict is not None:
                if items[0] != items[1]:
                    crystal_a = reduced_structure_dict[scalar_feature_df.iloc[
                        items[0]]['name']]
                    crystal_b = reduced_structure_dict[scalar_feature_df.iloc[
                        items[1]]['name']]

                    _, P, _, Q = parse_periodic_case(
                        crystal_a,
                        crystal_b,
                        try_supercell,
                        pymatgen=True,
                        get_reduced_structure=False)

                    logger.debug('Lengths are %s, %s', len(P), len(Q))
                    rmsd_result = kabsch_rmsd(P, Q, translate=True)
                    logger.debug('The Kabsch RMSD is %s', rmsd_result)
                    if rmsd_result < threshold:
                        pairs.append(items)
            else:
                if items[0] != items[1]:
                    _, P, _, Q = parse_periodic_case(
                        scalar_feature_df.iloc[items[0]]['name'],
                        scalar_feature_df.iloc[items[1]]['name'],
                        try_supercell,
                        get_reduced_structure=False)
                    logger.debug('Comparing %s and %s',
                                 scalar_feature_df.iloc[items[0]]['name'],
                                 scalar_feature_df.iloc[items[1]]['name'])
                    logger.debug('Lengths are %s, %s', len(P), len(Q))
                    rmsd_result = kabsch_rmsd(P, Q, translate=True)
                    logger.debug('The Kabsch RMSD is %s', rmsd_result)
                    if rmsd_result < threshold:
                        pairs.append(items)
        return pairs

    def compare_graph_pair(self, items):
        nn_strategy = JmolNN()
        crystal_a = Structure.from_file(
            self.scalar_feature_matrix.iloc[items[0]]['name'])
        crystal_b = Structure.from_file(
            self.scalar_feature_matrix.iloc[items[1]]['name'])
        if self.try_supercell:
            crystal_a, crystal_b = attempt_supercell_pymatgen(
                crystal_a, crystal_b)
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
        if self.try_supercell:
            crystal_a, crystal_b = attempt_supercell_pymatgen(
                crystal_a, crystal_b)
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
        graph_hash = get_hash(crystal)
        self.hash_dict[graph_hash].append(name)

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

        if self.method == 'standard':

            self.pairs = self.compare_graphs(self.similar_composition_tuples)

        elif self.method == 'rmsd':
            self.pairs = RemoveDuplicates.compare_rmsd(
                tupellist=self.similar_composition_tuples,
                scalar_feature_df=self.scalar_feature_matrix,
                try_supercell=self.try_supercell,
                reduced_structure_dict=self.reduced_structure_dict)

        elif self.method == 'rmsd_graph':
            self.rmsd_pairs = RemoveDuplicates.compare_rmsd(
                tupellist=self.similar_composition_tuples,
                scalar_feature_df=self.scalar_feature_matrix,
                try_supercell=self.try_supercell,
                reduced_structure_dict=self.reduced_structure_dict)

            self.pairs = self.compare_graphs(self.rmsd_pairs)

        elif self.method == 'hash':
            raise NotImplementedError
            # RemoveDuplicates.get_graph_hash_dict(self.structure_list)

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

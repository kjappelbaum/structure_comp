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
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JMolNN
from scipy.spatial.distance import pdist, squareform
import logging
from tqdm.autonotebook import tqdm
import pandas as pd


class RemoveDuplicates():
    def __init__(self, structure_list: list, reduced_structure_dir: str,
                 remove_reduced_structure_dir: bool):

        self.structure_list = structure_list
        self.reduced_structure_dir = reduced_structure_dir
        self.remove_rsd = remove_reduced_structure_dir

    @staticmethod
    def get_reduced_structures(structure_list, new_dir):
        """
        To make feature calculation cheaper and to
        avoid issues with supercells.
        """
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        for structure in structure_list:
            sname = Path(structure).name
            crystal = Structure.from_file(structure)
            crystal = crystal.get_reduced_structure()
            crystal.to(filename=os.path.join(new_dir, sname))

    @staticmethod
    def get_scalar_features(structure):
        """
        Get features for the first comparison matrix.
        """
        number_atoms = structure.num_sites
        density = structure.density
        return number_atoms, density

    @staticmethod
    def get_scalar_matrix(reduced_structure_list):
        feature_list = []
        for structure in tqdm(reduced_structure_list):
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
    def get_scalar_distance_matrix(scalar_feature_df, threshold=0.2):
        """
        Get structures that probably have the same composition.
        """
        distances = pdist(
            scalar_feature_df.drop(columns=['name']).values,
            metric='euclidean')
        dist_matrix = squareform(distances)
        i, j = np.where(dist_matrix < threshold)

        return list(zip(i, j))

    def compare_rmsd(self):
        """
        Potentially, we could run the RMSD code to detect same structures
        """
        return NotImplementedError

    @staticmethod
    def compare_graphs(tupellist, scalar_feature_df):
        """
        Filter structures with same structure graph
        """
        pairs = []
        for items in tqdm(tupellist):
            if items[0] != items[1]:
                nn_strategy = JMolNN()
                crystal_a = Structure.from_file(
                    scalar_feature_df.iloc[items[0]].name)
                crystal_b = Structure.from_file(
                    scalar_feature_df.iloc[items[1]].name)
                sgraph_a = StructureGraph.with_local_env_strategy(
                    crystal_a, nn_strategy)
                sgraph_b = StructureGraph.with_local_env_strategy(
                    crystal_b, nn_strategy)
                distance = sgraph_a.diff(sgraph_b)
                if distance == 0:
                    pairs.append(items)
        return pairs

    @staticmethod
    def get_structure_list(directory, extension='cif'):
        if extension:
            structure_list = glob(
                os.path.join(directory, ''.join([',.', extension])))
        else:
            structure_list = glob(os.path.join(directory, '*'))
        return structure_list

    def janitor(self):
        shutil.rmtree(self.reduced_structure_dir)

    def run_filtering(self):
        """
        Runs the on the database.
        :return:
        """
        RemoveDuplicates.get_reduced_structures(self.structure_list,
                                                self.reduced_structure_dir)

        self.reduced_structure_list = RemoveDuplicates.get_structure_list(
            self.reduced_structure_dir)

        self.scalar_feature_matrix = RemoveDuplicates.get_scalar_matrix(
            self.reduced_structure_list)

        self.similar_composition_tuples = RemoveDuplicates.get_scalar_distance_matrix(
            self.scalar_feature_matrix)

        self.pairs = RemoveDuplicates.compare_graphs(
            self.similar_composition_tuples, self.scalar_feature_matrix)


    def inspect_duplicates(self):
        return NotImplementedError

    def remove_duplicates(self):
        return NotImplementedError

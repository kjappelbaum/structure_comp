#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Kevin M. Jablonka'
__copyright__ = 'MIT License'
__maintainer__ = 'Kevin M. Jablonka'
__email__ = 'kevin.jablonka@epfl.ch'
__version__ = '0.1.0'
__date__ = '05.04.19'
__status__ = 'First Draft, Testing'

from pymatgen import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
import pandas as pd
import concurrent.futures
from tqdm.autonotebook import tqdm
from .utils import get_subgraphs_as_molecules_all, get_structure_list
import numpy as np
import logging
from pathlib import Path
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Checker():
    def __init__(self, structure_list):
        self.structure_list = structure_list

    @classmethod
    def from_folder(cls, folder: str):
        sl = get_structure_list(folder)
        return cls(sl)

    @staticmethod
    def check_clashing(s: Structure, threshold: float = 0.3) -> bool:
        """
        Takes a pymatgen structure object and checks if there are atoms that are too close (i.e. their distance
        is smaller than the threshold).

        Args:
            s (pymatgen structure object): structure to be checked
            threshold (float): used as a check for clashing atoms

        Returns:

        """
        dm = s.distance_matrix

        if np.min(dm + np.eye(len(dm))) < threshold:
            logger.debug('minimum is %s', np.min(dm + np.eye(len(dm))))
            return True
        else:
            return False

    @staticmethod
    def check_hydrogens(s: Structure,
                        neighbor_threshold: float = 2.0,
                        strictness: str = 'CH') -> bool:
        """
        checks if there are any hydrogens in a structure object.

        Args:
            s (pymatgen structure object): structure to be checked
            neighbor_threshold (float): threshold for distance that is still considered to be bonded
            strictness (str): available levels: 'tight': returns false if there is no H at all, 'medium'
                returns false if there are carbons but no hydrogens, 'CH' (default) checks if there are
                carbons with less or equal 2 non-hydrogen neighbors (e.g. the most common case for aromatic rings).
                If those have also no hydrogen bonded to them, it will return False

        Returns:

        """

        symbols = s.symbol_set

        return_val = True

        if strictness == 'tight':
            logger.debug('running H check with tight strictness')
            if not 'H' in symbols:
                return False
        elif strictness == 'medium':
            logger.debug('running H check with medium strictness')
            if 'C' in symbols:
                if not 'H' in symbols:
                    return False
        elif strictness == 'CH':
            logger.debug('running H check with CH strictness')
            if 'C' in symbols:
                c_sites = s.indices_from_symbol('C')
                for c_site in c_sites:
                    neighbors = s.get_neighbors(s[c_site], neighbor_threshold)
                    neighbors_symbol_list = [
                        neighbor_site[0].species_string
                        for neighbor_site in neighbors
                    ]
                    neighbors_no_h = [
                        neighbor_site for neighbor_site in neighbors
                        if neighbor_site[0].species_string != 'H'
                    ]
                    if len(neighbors_symbol_list) == 0:
                        return False
                    if len(neighbors_no_h) <= 2:
                        if len(neighbors) - len(neighbors_no_h) == 0:
                            return False
        return return_val

    @staticmethod
    def check_unbound(s: Structure,
                      whitelist: list = ['H'],
                      threshold: float = 2.5,
                      mode='naive') -> bool:
        """
        This uses the fact that unbound solvent is often in pores
        and more distant from all other atoms. So far this test focusses on water.

        Args:
            s (pymatgen structure object): structure to be checked
            whitelist (list): elements that are not considered in the check (they are basically removed from the
                structure)
            mode (str): checking mode. If 'naive' then a simple distance based check is used and a atom is detected
                as unbound it there is no other atom within the threshold distance.

        Returns:

        """

        crystal = s.copy()

        if whitelist:
            crystal.remove_species(whitelist)

        if mode == 'naive':
            for atom in crystal:
                neighbors = crystal.get_neighbors(atom, threshold)
                if len(neighbors) == 0:
                    return True
            return False

        if mode == 'graph':
            nn_strategy = JmolNN()
            sgraph = StructureGraph.with_local_env_strategy(
                crystal, nn_strategy)
            molecules = get_subgraphs_as_molecules_all(sgraph)

            if len(molecules) > 0:
                return True
            else:
                return False

    def run_flagging(self):
        """
        Runs for all structures of the structure list the flagging procedure
        Returns:

        """
        flag_list = []
        logger.debug('starting flagging')

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in tqdm(
                    executor.map(Checker.flag_potential_problems,
                                 self.structure_list),
                    total=len(self.structure_list)):
                flag_list.append(result)

        return pd.DataFrame(flag_list)

    @staticmethod
    def flag_potential_problems(structure_path: str,
                                clashing_threshold: float = 0.5,
                                bond_threshold: float = 2.0) -> dict:
        """
        Runs several naive checks on a structure to find out if there are potential
        problems.

        Args:
            structure_path (str): Path to structure
            name (str): name that will be used as value for the 'name' key of the output dictionary.
            clashing_threshold (float): used as a check for clashing atoms
            bond_threshold (float): threshold for what is still considered to be bonded

        Returns:

        """
        problem_dict = {}
        problem_dict['name'] = Path(structure_path).stem

        # In some cases, we might not be able to read the structure,
        # then we set all check results to np.nan and set cif_error = True
        # often, this is a good indication of major errors but in many other cases
        # if is also simply due to the fact that pymatgen can not read some symmetry string or so
        try:
            s = Structure.from_file(structure_path)
        except:
            problem_dict['cif_error'] = True
            problem_dict['clashing'] = np.nan
            problem_dict['unbound'] = np.nan
            problem_dict['hydrogens'] = np.nan
        else:
            # one potential problem is that we might have disorder/clashing atoms
            problem_dict['clashing'] = Checker.check_clashing(
                s, threshold=clashing_threshold)

            # one other potential problem is that there might be unbound solvent
            problem_dict['unbound'] = Checker.check_unbound(
                s, threshold=bond_threshold + 0.5)

            # one other problem is that there might be missing hydrogens
            # a naive check would be if there are hydrogens at all in the file
            problem_dict['hydrogens'] = Checker.check_hydrogens(
                s, bond_threshold)
            problem_dict['cif_problem'] = False
        return problem_dict

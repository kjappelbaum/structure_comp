#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Kevin M. Jablonka'
__copyright__ = 'MIT License'
__maintainer__ = 'Kevin M. Jablonka'
__email__ = 'kevin.jablonka@epfl.ch'
__version__ = '0.1.0'
__date__ = '05.04.19'
__status__ = 'First Draft, Testing'

from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from pymatgen import Structure

import CifFile
import tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage
import os
from .utils import slugify, closest_index


class Cleaner():
    def __init__(self, structure_list: list):
        self.structure_list = structure_list
        self.tempdirpath = tempfile.mkdtemp()
        pass

    @staticmethod
    def rewrite_cif(path: str, outdir: str) -> str:
        """
        Reads cif file and keeps only the relevant parts defined in RELEVANT_KEYS.
        Sometimes, it is good to loose information ...
        Args:
            path (str): Path to input file
            outdir (str): Path to output directory

        Returns:

        """
        RELEVANT_KEYS = [
            '_cell_volume', '_cell_angle_gamma', '_cell_angle_beta',
            '_cell_angle_alpha', '_cell_length_a', '_cell_length_b',
            '_cell_length_c', '_symmetry_space_group_name_hall',
            '_symmetry_space_group_name_h', '_symmetry_cell_setting',
            '_atom_site_label', '_atom_site_fract_x', '_atom_site_fract_y',
            '_atom_site_fract_z', '_atom_site_charge',
            '_symmetry_space_group_name_Hall', '_symmetry_equiv_pos_as_xyz',
            '_atom_site_type_symbol', '_space_group_crystal_system',
            '_space_group_symop_operation_xyz', '_space_group_name_Hall',
            '_space_group_crystal_system'
        ]
        cf = CifFile.ReadCif(path)
        image = cf[cf.keys()[0]]
        for key in image.keys():
            if key not in RELEVANT_KEYS:
                image.RemoveItem(key)

        name = slugify(Path(path).stem)
        outpath = os.path.join(outdir, '.'.join([name, 'cif']))
        with open(outpath, 'w') as f:
            f.write(cf.WriteOut() + '\n')

        return outpath

    @staticmethod
    def remove_unbound_solvent(crystal):
        nn_strategy = JmolNN()
        sgraph = StructureGraph.with_local_env_strategy(crystal, nn_strategy)
        sgraph.get_subgraphs_as_molecules()

    @staticmethod
    def remove_disorder(structure: Structure,
                        distance: float = 1.0, threshold_circle: float =0.1) -> Structure:
        """
        Merges sites within distance that are likely due to structural disorder.

        Inspired by pymatgen merge function code:
            - we assume that the site properties of the clustered species are all
              the same
            - we assume that we can replace the disorder with an averaged position

        Args:
            structure (pymatgen Structure object):
            distance (float): distance threshold for the merging operation

        Returns:
            structure object with merged sites
        """

        crystal = structure.copy()
        d = crystal.distance_matrix

        all_cords = crystal.frac_coords
        indices_to_dump = []
        for symbol in crystal.symbol_set:
            sub_matrix = d[crystal.indices_from_symbol(
                symbol), :][:, crystal.indices_from_symbol(symbol)]

            np.fill_diagonal(sub_matrix, 0)

            # perform hierarchical clustering, get flat array of indices
            clusters = fcluster(
                linkage(squareform((sub_matrix + sub_matrix.T) / 2)), distance,
                'distance')

            symbol_indices = crystal.indices_from_symbol(symbol)
            species_coord = [crystal[i].frac_coords for i in symbol_indices]
            species_prop = [crystal[i].properties for i in symbol_indices]

            print(symbol)
            # iterate over clusters
            print(clusters)
            for c in np.unique(clusters):
                inds = np.where(clusters == c)[0]
                print(inds)
                indices_to_dump.append([symbol_indices[i] for i in inds])
                coords = [species_coord[i] for i in inds]
                props = [species_prop[i] for i in inds]

                # now, average the coordinates, make at this point sure
                # that the case of len(coords) = 1 is handeled correctly
                print(coords)
                if len(coords) == 1:
                    average_coord = np.concatenate(coords).ravel().tolist()
                else:
                    # important insight
                    # if circular, we look at the element in the center and
                    # get the usual coordination number and average in this way
                    average_coord = np.mean(coords, axis=0).flatten()
                    distances = []
                    for coord in coords:
                        distances.append(np.linalg.norm(average_coord, coord))
                    if np.std(distances) < threshold_circle:
                        print('we found a circle')
                        # then figure out, what the central element is
                        # and what its neighbors are
                        # if one neighbor is metallic, then do not count it.
                        center_index = closest_index(all_cords, average_coord)
                        center_species = crystal[center_index].species
                        center_valence = center_species.group - 10

                    print(average_coord)

                print(average_coord)

                # assumptions:
                # - properties are the same
                # - averaged coordinates is a good approximation
                print('average coords are {}'.format(average_coord))
                crystal.append(
                    symbol,
                    average_coord,
                    validate_proximity=False,
                    properties=props[0])

        # Now remove the sites that we averaged.
        print('indices to dump {}'.format(indices_to_dump))
        indices_to_dump = list(set(np.concatenate(indices_to_dump).ravel().tolist()))
        print(indices_to_dump)
        crystal.remove_sites(indices_to_dump)

        return crystal

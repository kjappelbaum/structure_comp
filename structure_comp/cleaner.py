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
import re
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage
import os
from .utils import slugify, incremental_farthest_search, get_symbol_indices
from sklearn.cluster import DBSCAN


class Cleaner():
    def __init__(self, structure_list: list):
        self.structure_list = structure_list
        self.tempdirpath = tempfile.mkdtemp()
        pass

    @staticmethod
    def rewrite_cif(path: str, outdir: str,
                    remove_disorder: bool = True) -> str:
        """
        Reads cif file and keeps only the relevant parts defined in RELEVANT_KEYS.
        Sometimes, it is good to loose information ...
        Args:
            path (str): Path to input file
            outdir (str): Path to output directory
            remove_disorder (bool): If True (default), then disorder groups other than 1 and . are removed.

        Returns:

        """
        RELEVANT_KEYS = [
            '_cell_volume',
            '_cell_angle_gamma',
            '_cell_angle_beta',
            '_cell_angle_alpha',
            '_cell_length_a',
            '_cell_length_b',
            '_cell_length_c',
            '_symmetry_space_group_name_hall',
            '_symmetry_space_group_name_h',
            '_symmetry_cell_setting',
            '_atom_site_label',
            '_atom_site_fract_x',
            '_atom_site_fract_y',
            '_atom_site_fract_z',
            '_atom_site_charge',
            '_symmetry_space_group_name_Hall',
            '_symmetry_equiv_pos_as_xyz',
            '_atom_site_type_symbol',
            '_space_group_crystal_system',
            '_space_group_symop_operation_xyz',
            '_space_group_name_Hall',
            '_space_group_crystal_system',
            '_space_group_IT_number',
            '_space_group_name_H - M_alt',
            '_space_group_name_Hall'
        ]

        LOOP_KEYS = [
            '_atom_site_label', '_atom_site_fract_x', '_atom_site_fract_y',
            '_atom_site_fract_z', '_atom_site_charge', '_atom_site_occupancy'
        ]

        NUMERIC_LOOP_KEYS = ['_atom_site_fract_x', '_atom_site_fract_y',
            '_atom_site_fract_z', '_atom_site_charge', '_atom_site_occupancy'
        ]

        CELL_PROPERTIES = [
            '_cell_volume',
            '_cell_angle_gamma',
            '_cell_angle_beta',
            '_cell_angle_alpha',
            '_cell_length_a',
            '_cell_length_b',
            '_cell_length_c',
        ]

        cf = CifFile.ReadCif(path)
        image = cf[cf.keys()[0]]

        # First, make sure we have proper atom type labels.
        if '_atom_site_type_symbol' not in image.keys():
            # then loop over the label and strip all the floats
            type_symbols = []
            for label in image['_atom_site_label']:
                type_symbols.append(re.sub('[^a-zA-Z]+', '', label))
            image.AddItem('_atom_site_type_symbol', type_symbols)

        if remove_disorder and '_atom_site_disorder_group' in image.keys():
            indices_to_drop = []
            print('Removing disorder groups')
            for i, dg in enumerate(image['_atom_site_disorder_group']):
                if dg not in ('.', '1'):
                    indices_to_drop.append(i)

            if indices_to_drop:
                image['_atom_site_type_symbol'] = [
                    i for j, i in enumerate(image['_atom_site_type_symbol'])
                    if j not in indices_to_drop
                ]
                for key in LOOP_KEYS:
                    if key in image.keys():
                        loop_fixed = [
                            i for j, i in enumerate(image[key])
                            if j not in indices_to_drop
                        ]
                        image.RemoveItem(key)
                        image.AddItem(key, loop_fixed)
                        image.AddLoopName('_atom_site_type_symbol', key)

        for key in image.keys():
            if key not in RELEVANT_KEYS:
                image.RemoveItem(key)

        image['_atom_site_label'] = image['_atom_site_type_symbol']

        # remove uncertainty brackets
        for key in NUMERIC_LOOP_KEYS:
            if key in image.keys():
                image[key] = [
                    float(re.sub(r'\([^)]*\)', '', s)) for s in image[key]
                ]

        for property in CELL_PROPERTIES:
            image[property] = float(re.sub(r'\([^)]*\)', '', image[property]))

        # make filename that is safe
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
                        distance: float = 1.0) -> Structure:
        """
        Merges sites within distance that are likely due to structural disorder.

        2
            - we assume that the site properties of the clustered species are all
              the same
            - we assume that we can replace the disorder with an averaged position


        Due to the akward treatmeant of partial occupancies, we need to use a bit of overhead for pymatgen

        Args:
            structure (pymatgen Structure object):
            distance (float): distance threshold for the merging operation

        Returns:
            structure object with merged sites
        """

        crystal = structure.copy()
        d = crystal.distance_matrix

        indices_to_dump = []

        symbol_indices_dict = get_symbol_indices(crystal)
        symbol_set = symbol_indices_dict.keys()

        all_coords = crystal.frac_coords

        for symbol in symbol_set:
            sub_matrix = d[
                symbol_indices_dict[symbol], :][:, symbol_indices_dict[symbol]]

            np.fill_diagonal(sub_matrix, 0)

            # perform hierarchical clustering, get flat array of indices
            #clusters = fcluster(
            #    linkage(squareform(sub_matrix)), distance,
            #    'distance')

            clustering = DBSCAN(
                eps=distance,
                min_samples=2).fit(all_coords[symbol_indices_dict[symbol]])

            clusters = clustering.labels_
            #print(linkage(squareform(sub_matrix)))

            symbol_indices = symbol_indices_dict[symbol]
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

                # here, I assume that we did a good ob in finding equivalent atoms
                # they will probably get the same occupancy assigned in the cif file
                occupancy = crystal[inds[0]].as_dict()['species'][0]['occu']
                print(len(inds))
                sites_to_keep = len(inds) * occupancy
                print('sites to keep are {}'.format(sites_to_keep))
                print(coords)
                if len(coords) == 1:
                    average_coord = [np.concatenate(coords).ravel().tolist()]
                else:
                    # now as we now, how many sites we should keep we select the n farthest ones
                    average_coord = incremental_farthest_search(
                        coords, int(sites_to_keep))
                    print(average_coord)

                print(average_coord)

                print('average coords are {}'.format(average_coord))
                for coord in average_coord:
                    crystal.append(
                        symbol,
                        coord,
                        validate_proximity=False,
                        properties=props[0])

        # Now remove the sites that we averaged.
        print('indices to dump {}'.format(indices_to_dump))
        indices_to_dump = list(
            set(np.concatenate(indices_to_dump).ravel().tolist()))
        print(indices_to_dump)
        crystal.remove_sites(indices_to_dump)

        return crystal

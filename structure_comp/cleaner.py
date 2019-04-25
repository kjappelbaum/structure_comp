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
from pymatgen.symmetry import analyzer
from pymatgen import Structure
import CifFile
import tempfile
from pathlib import Path
import numpy as np
from ase.io import read, write
from ase.geometry import get_duplicate_atoms
import re
from tqdm.autonotebook import tqdm
from functools import partial
from collections import defaultdict
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import fcluster, linkage
import os
import concurrent.futures
from .utils import slugify, get_symbol_indices, get_structure_list, get_subgraphs_as_molecules_all, \
    get_duplicates_ktree, get_duplicates_dynamic_threshold
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Cleaner():
    def __init__(self, structure_list: list, outdir: str):
        self.structure_list = structure_list
        self.tempdirpath = tempfile.mkdtemp()
        self.outdir = outdir
        self.rewritten_paths = []

    def __repr__(self):
        return f'Cleaner with outdir {self.outdir}'

    def __len__(self):
        return len(self.structure_list)

    @classmethod
    def from_folder(cls, folder, outdir: str):
        """

        Args:
            folder (str): path to folder that is used for construction of the RemoveDuplicates object
            outdir (str): path to putput directory

        Returns:

        """
        sl = get_structure_list(folder)
        return cls(sl, outdir)

    @staticmethod
    def rewrite_cif(path: str,
                    outdir: str,
                    remove_disorder: bool = True,
                    remove_duplicates: bool = True,
                    p1: bool = False,
                    clean_symmetry: float = None) -> str:
        """
        Reads cif file and keeps only the relevant parts defined in RELEVANT_KEYS.
        Sometimes, it is good to loose information ...
        Args:
            path (str): Path to input file
            outdir (str): Path to output directory
            remove_disorder (bool): If True (default), then disorder groups other than 1 and . are removed.
            p1 (bool): If True, then we will set the symmetry to P1.
            clean_symmetry (float): uses spglib to symmetrize the structure with the specified tolerance, set to None
                if you do not want to use it

        Returns:
            outpath (str)
        """
        # ToDo: I want to have it pretty strict with pre-defined keys but maybe a regex can take care of captitiliaztion
        RELEVANT_KEYS = [
            '_cell_volume',
            '_cell_angle_gamma',
            '_cell_angle_beta',
            '_cell_angle_alpha',
            '_cell_length_a',
            '_cell_length_b',
            '_cell_length_c',
            '_atom_site_label',
            '_atom_site_fract_x',
            '_atom_site_fract_y',
            '_atom_site_fract_z',
            '_atom_site_charge',
            '_atom_site_type_symbol',
        ]

        RELEVANT_KEYS_NON_P1 = [
            '_symmetry_cell_setting',
            '_space_group_crystal_system',
            '_space_group_name_hall',
            '_space_group_crystal_system',
            '_space_group_it_number',
            '_symmetry_space_group_name_h-m',
            '_symmetry_int_tables_number',
            '_space_group_name_h-m_alt',
            '_symmetry_tnt_tables_number',
            '_symmetry_space_group_name_hall',
            '_symmetry_equiv_pos_as_xyz',
            '_space_group_symop_operation_xyz',
        ]

        LOOP_KEYS = [
            '_atom_site_label', '_atom_site_fract_x', '_atom_site_fract_y',
            '_atom_site_fract_z', '_atom_site_charge', '_atom_site_occupancy'
        ]

        NUMERIC_LOOP_KEYS = [
            '_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z',
            '_atom_site_charge', '_atom_site_occupancy'
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

        try:
            cf = CifFile.ReadCif(path)
            image = cf[cf.keys()[0]]
            if ('_atom_site_fract_x' not in image.keys()) or (
                    '_atom_site_fract_y' not in image.keys()) or (
                        '_atom_site_fract_z' not in image.keys()):
                raise ValueError
        except ValueError:
            logger.error(
                'the file %s seems to be invalid because we were unable to find '
                'the atomic positions will return input path but '
                'this file will likely cause errors and needs to be checked',
                path)
            return path
        except FileNotFoundError:
            logger.error('the file %s was not found', path)
            return path
        except Exception:
            logger.error(
                'the file %s seems to be invalid will return input path but '
                'this file will likely cause errors and needs to be checked',
                path)
            return path
        else:
            # First, make sure we have proper atom type labels.
            if ('_atom_site_type_symbol' not in image.keys()) and (
                    '_atom_site_symbol' not in image.keys()):
                # then loop over the label and strip all the floats
                type_symbols = []
                for label in image['_atom_site_label']:
                    type_symbols.append(re.sub('[^a-zA-Z]+', '', label))
                image.AddItem('_atom_site_type_symbol', type_symbols)
                image.AddLoopName('_atom_site_label', '_atom_site_type_symbol')

            if remove_disorder and '_atom_site_disorder_group' in image.keys():
                indices_to_drop = []
                logger.info('Removing disorder groups in %s', path)
                for i, dg in enumerate(image['_atom_site_disorder_group']):
                    if dg not in ('.', '1'):
                        indices_to_drop.append(i)

                if indices_to_drop:
                    image['_atom_site_type_symbol'] = [
                        i
                        for j, i in enumerate(image['_atom_site_type_symbol'])
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

            if not p1:
                RELEVANT_KEYS += RELEVANT_KEYS_NON_P1

            for key in image.keys():
                if key not in RELEVANT_KEYS:
                    image.RemoveItem(key)

            image['_atom_site_label'] = image['_atom_site_type_symbol']

            if p1:
                image.AddItem('_symmetry_space_group_name_H-M', 'P 1')
                image.ChangeItemOrder('_symmetry_space_group_name_h-m', -1)
                image.AddItem('_space_group_name_Hall', 'P 1')
                image.ChangeItemOrder('_space_group_name_Hall', -1)

            # remove uncertainty brackets
            for key in NUMERIC_LOOP_KEYS:
                if key in image.keys():
                    image[key] = [
                        float(re.sub(r'\([^)]*\)', '', s)) for s in image[key]
                    ]

            for prop in CELL_PROPERTIES:
                if prop in image.keys():
                    image[prop] = float(re.sub(r'\([^)]*\)', '', image[prop]))

            # make filename that is safe
            name = slugify(Path(path).stem)
            outpath = os.path.join(outdir, '.'.join([name, 'cif']))
            with open(outpath, 'w') as f:
                f.write(cf.WriteOut() + '\n')

            if clean_symmetry:
                crystal = Structure.from_file(outpath)
                spa = analyzer.SpacegroupAnalyzer(crystal, 0.1)
                crystal = spa.get_refined_structure()
                crystal.to(filename=outpath)

            if remove_duplicates:
                atoms = read(outpath)
                get_duplicate_atoms(atoms, 0.5, delete=True)
                write(outpath, atoms)

            return outpath

    @staticmethod
    def remove_unbound_solvent(structure: Structure) -> Structure:
        """
        Constructs a structure graph and removes unbound solvent molecules
        if they are in a hardcoded composition list.

        Args:
            structure (pymatgen structure object=:

        Returns:

        """
        crystal = structure.copy()
        molecules_solvent = ['H2 O1', 'H3 O1', 'C2 H6 O S', 'O1']
        nn_strategy = JmolNN()
        sgraph = StructureGraph.with_local_env_strategy(crystal, nn_strategy)
        molecules = get_subgraphs_as_molecules_all(sgraph)
        cart_coordinates = crystal.cart_coords
        indices = []
        for molecule in molecules:
            print(str(molecule.composition))
            if molecule.formula in molecules_solvent:
                for coord in [
                        site.as_dict()['xyz'] for site in molecule.sites
                ]:
                    if (coord[0] < crystal.lattice.a) and (
                            coord[1] < crystal.lattice.b) and (
                                coord[2] < crystal.lattice.c):
                        print(coord)
                        indices.append(
                            np.where(
                                np.prod(
                                    np.isclose(cart_coordinates - coord, 0),
                                    axis=1) == 1)[0][0])

        print(indices)
        crystal.remove_sites(indices)

        return crystal

    @staticmethod
    def openbabel(cifpath: str,
                  add_h: bool = True,
                  opt: bool = True,
                  ff: str = 'uff',
                  steps: int = 500,
                  overwrite: bool = True,
                  frozenats: list = []):
        """
        Use openbabel to do local optimization (molecular coordinates with forcefield) or addition of missing hydrogens.

        Args:
            cifpath (str): path to structure, currently hardcoded to be a cif file
            add_h (bool): If true (default) openbabel is used to add missing hydrogens
            opt (bool): If true (default), local optimization is performed
            ff (str): forcefield for the local optimization
            steps (int): number of steps for geometry optimization
            overwrite (bool): If true, input file is overwritten, if false e will add '_openbabel' to the ciffile path stem

        Returns:

        """

        import pybel  # we do not import by default, cause openbabel needs to be installed

        mol = next(pybel.readfile('cif', cifpath))

        if ff not in pybel.forcefields:
            logger.warning(
                'the forcefield you selected is not available, will default to uff'
            )
            ff = 'uff'
        if add_h:
            mol.addh()

        if opt:
            mol.localopt(forcefield=ff, steps=steps)
            # cehck https://github.com/hjkgrp/molSimplify/blob/ed0c63ea33f5ceb543aa6db5ab0f68ef568031a3/molSimplify/Scripts/cellbuilder_tools.py
            # metals = range(21, 31) + range(39, 49) + range(72, 81)
            # constr = pybel.OBFFConstraints()
            # indmtls = []
            # mtlsnums = []
            # for iiat, atom in enumerate(pybel.OBMolAtomIter(mol.OBMol)):
            #     if atom.GetAtomicNum() in metals:
            #         indmtls.append(iiat)
            #         mtlsnums.append(atom.GetAtomicNum())
            #         atom.SetAtomicNum(6)
            # for cat in frozenats:
            #     constr.AddAtomConstraint(cat + 1)
            #
            # forcefield = pybel.OBForceField.FindForceField(ff)
            # forcefield.Setup(mol.OBMol, constr)
            # ## force field optimize structure
            # forcefield.ConjugateGradients(steos)
            # forcefield.GetCoordinates(mol.OBMol)
            # # reset atomic number to metal
            # for i, iiat in enumerate(indmtls):
            #     mol.OBMol.GetAtomById(iiat).SetAtomicNum(mtlsnums[i])
            #

        if overwrite:
            outname = cifpath
        else:
            parent = Path(cifpath).parent
            stem = Path(cifpath).stem
            outname = os.path.join(parent,
                                   ''.join([stem, '_openbabel', '.cif']))

        output = pybel.Outputfile("cif", outname, overwrite=True)
        output.write(mol)
        output.close()

    @staticmethod
    def merge_clashing(s: Structure, tolerance_factor: float = 0.9):
        """
        Naive method for 'merging' clashing sites.
        In case it the two sites are not the same elements, priority will be given to the heavier one.

        Args:
            s:

        Returns:

        """
        crystal = s.copy()
        duplicates = get_duplicates_dynamic_threshold(s, tolerance_factor)
        logger.debug('found %s clashing sites', duplicates)
        deleted = []
        for duplicate in duplicates:
            if (not duplicate[0] in deleted) and (not duplicate[1] in deleted):
                element0 = crystal[duplicate[0]].specie.number
                element1 = crystal[duplicate[1]].specie.number
            if element0 == element1:
                deleted.append(duplicate[1])
            elif element0 > element1:
                deleted.append(duplicate[1])
            elif element0 < element1:
                deleted.append(duplicate[0])
        crystal.remove_sites(deleted)
        return crystal

    @staticmethod
    def remove_disorder(structure: Structure,
                        distance: float = 0.3) -> Structure:
        """
        Merges sites within distance that are likely due to structural disorder.

        Inspired by the pymatgen merge function
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

        indices_to_dump = []
        to_append = defaultdict(list)
        symbol_indices_dict = get_symbol_indices(crystal)
        symbol_set = symbol_indices_dict.keys()

        for symbol in symbol_set:
            sub_matrix = d[
                symbol_indices_dict[symbol], :][:, symbol_indices_dict[symbol]]
            # perform hierarchical clustering, get flat array of indices
            clusters = fcluster(
                linkage(squareform(sub_matrix)), distance, 'distance')

            symbol_indices = symbol_indices_dict[symbol]
            species_coord = [crystal[i].frac_coords for i in symbol_indices]

            for c in np.unique(clusters):
                inds = np.where(clusters == c)[0]
                indices_to_dump.append([symbol_indices[i] for i in inds])
                coords = [species_coord[i] for i in inds]

                if len(coords) == 1:
                    average_coord = np.concatenate(coords).ravel().tolist()
                else:
                    average_coord = np.mean(coords, axis=0).tolist()

                to_append[symbol].append(average_coord)

        indices_to_dump = list(
            set(np.concatenate(indices_to_dump).ravel().tolist()))

        crystal.remove_sites(indices_to_dump)
        to_append = dict(to_append)

        for symbol in to_append.keys():
            for coord in to_append[symbol]:
                crystal.append(
                    symbol,
                    coord,
                    validate_proximity=False,
                )

        return crystal

    def rewrite_all_cifs(self,
                         remove_disorder: bool = True,
                         p1: bool = False,
                         remove_duplicates: bool = True,
                         clean_symmetry: float = None):
        """
        Loops concurrently over all cifs in a folder and rewrites them.

        Args:
            remove_disorder (bool): If True (default), then disorder groups other than 1 and . are removed.
            p1 (bool): If True, sets symmetry to P1.
            clean_symmetry (float): uses spglib to symmetrize the structure with the specified tolerance, set to None
                if you do not want to use it

        Returns:

        """
        if not os.path.exists(self.outdir):
            logger.info('Will create directory %s', self.outdir)
            os.makedirs(self.outdir)

        partial_rewrite_cifs = partial(
            Cleaner.rewrite_cif,
            outdir=self.outdir,
            p1=p1,
            remove_duplicates=remove_duplicates,
            remove_disorder=remove_disorder,
            clean_symmetry=clean_symmetry)

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for outpath in tqdm(
                    executor.map(partial_rewrite_cifs, self.structure_list),
                    total=len(self.structure_list)):
                self.rewritten_paths.append(outpath)

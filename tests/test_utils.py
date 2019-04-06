#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Kevin M. Jablonka'
__copyright__ = 'MIT License'
__maintainer__ = 'Kevin M. Jablonka'
__email__ = 'kevin.jablonka@epfl.ch'
__version__ = '0.1.0'
__date__ = '30.03.19'
__status__ = 'First Draft, Testing'

import pytest
import numpy as np
import os
from pymatgen import Structure
from structure_comp.utils import get_hash, get_rmsd, kl_divergence, \
    tanimoto_distance, get_cheap_hash, attempt_supercell_pymatgen, get_symbol_list, get_symbol_indices

THIS_DIR = os.path.dirname(__file__)


def test_get_hash(get_all_structures):
    """
    For all structures make sure that the hash is only identical to the structure itself.
    """
    comp_matrix = np.zeros((len(get_all_structures), len(get_all_structures)))
    for i, structure_a in enumerate(get_all_structures):
        for j, structure_b in enumerate(get_all_structures):
            if i < j:
                hash_a = get_hash(structure_a)
                hash_b = get_hash(structure_b)
                if hash_a == hash_b:
                    comp_matrix[i][j] = 1
                else:
                    comp_matrix[i][j] = 0
    assert sum(comp_matrix) == sum(np.diag(comp_matrix))


def test_get_cheap_hash(get_all_structures):
    """
    For all structures make sure that the hash is only identical to the structure itself.
    """
    comp_matrix = np.zeros((len(get_all_structures), len(get_all_structures)))
    for i, structure_a in enumerate(get_all_structures):
        for j, structure_b in enumerate(get_all_structures):
            if i < j:
                hash_a = get_cheap_hash(structure_a)
                hash_b = get_cheap_hash(structure_b)
                if hash_a == hash_b:
                    comp_matrix[i][j] = 1
                else:
                    comp_matrix[i][j] = 0
    assert sum(comp_matrix) == sum(np.diag(comp_matrix))


def test_get_rmsd(get_all_structures):
    """
    For all structures make sure that the RMSD is null on the diagonal and not null on the off-diagonal elements.
    Make sure that result is symmetric.
    """
    comp_matrix = np.zeros((len(get_all_structures), len(get_all_structures)))
    for i, structure_a in enumerate(get_all_structures):
        for j, structure_b in enumerate(get_all_structures):
            comp_matrix[i][j] = get_rmsd(structure_a, structure_b)

    assert sum(np.diag(comp_matrix)) == 0
    assert np.allclose(comp_matrix, comp_matrix.T, atol=1e-8)


def test_kl_divergence(get_distributions):
    """
    Check that probabilty for observing the same distribution is one, regardless of test order and that for off-diagonal
    it is smaller than one. For speed reasons, we test the sum and not the single elements.
    """
    for i, dist_a in enumerate(get_distributions):
        for j, dist_b in enumerate(get_distributions):
            kl = kl_divergence(dist_a, dist_b)
            if i == j:
                assert pytest.approx(kl, 0.0001) == 0.0
            else:
                assert kl > 0


def test_tanimoto_distance(get_distributions):
    """
    Check that probabilty for observing the same distribution is one, regardless of test order and that for off-diagonal
    it is smaller than one. For speed reasons, we test the sum and not the single elements.
    """
    for i, dist_a in enumerate(get_distributions):
        for j, dist_b in enumerate(get_distributions):
            tanimototo = tanimoto_distance(dist_a, dist_b)
            if i == j:
                assert pytest.approx(tanimototo, 0.0001) == 1
            else:
                assert tanimototo < 1


def test_attempt_supercell_pymatgen():
    structure_1 = Structure.from_file(
        os.path.join(THIS_DIR, 'structures_supercells', 'UIO-66.cif'))
    structure_2 = Structure.from_file(
        os.path.join(THIS_DIR, 'structures_supercells', 'UIO-66_2_2_2.cif'))
    s1, s2 = attempt_supercell_pymatgen(structure_1, structure_2)
    assert pytest.approx(s1.lattice.abc, abs=0.01) == s2.lattice.abc


def test_get_symbol_list(get_cleaned_dmof_path):
    s = Structure.from_file(get_cleaned_dmof_path)
    symbol_list = get_symbol_list(s)
    true_symbols = [
        'Zn', 'Zn', 'Zn', 'Zn', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
        'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
        'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
        'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
        'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
        'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H',
        'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'H', 'C', 'C', 'C', 'C', 'C',
        'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
        'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
        'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
        'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
        'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'N', 'N', 'N',
        'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'N', 'O', 'O', 'O', 'O', 'O',
        'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
        'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'
    ]
    assert symbol_list == true_symbols


def test_get_symbol_indices(get_cleaned_dmof_path):
    s = Structure.from_file(get_cleaned_dmof_path)
    symbol_indices = get_symbol_indices(s)
    true_indices = {
        'Zn': [0, 1, 2, 3],
        'H': [
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
            39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
            56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
            73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
            90, 91
        ],
        'C': [
            92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
            107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
            120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,
            133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145,
            146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158,
            159, 160, 161, 162, 163
        ],
        'N': [164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175],
        'O': [
            176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188,
            189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201,
            202, 203, 204, 205, 206, 207
        ]
    }

    assert symbol_indices == true_indices
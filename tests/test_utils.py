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
from glob import glob
from structure_comp.utils import get_hash, get_rmsd

THIS_DIR = os.path.dirname(__file__)

@pytest.fixture
def get_all_structures():
    crystal_list = []
    structure_list = glob(os.path.join(THIS_DIR, 'structures', '.cif'))
    for structure in structure_list:
        crystal_list.append(Structure.from_file(structure))
    return crystal_list

def test_get_hash(get_all_structures):
    """
    For all structures make sure that the hash is only identical to the structure itself.
    """
    comp_matrix = np.zeros(len(get_all_structures), len(get_all_structures))
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

def test_rmsd(get_all_structures):
    """
    For all structures make sure that the RMSD is null on the diagonal and not null on the off-diagonal elements.
    Make sure that result is symmetric.
    """
    comp_matrix = np.zeros(len(get_all_structures), len(get_all_structures))
    for i, structure_a in enumerate(get_all_structures):
        for j, structure_b in enumerate(get_all_structures):
            comp_matrix[i][j] = get_rmsd(structure_a, structure_b)

    assert sum(np.diag(comp_matrix)) == 0
    assert np.allclose(comp_matrix, comp_matrix.T, atol=1e-8)
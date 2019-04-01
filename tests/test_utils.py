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
from structure_comp.utils import get_hash, get_rmsd, kl_divergence, tanimoto_distance, get_cheap_hash, attempt_supercell_pymatgen
from scipy import stats

THIS_DIR = os.path.dirname(__file__)


@pytest.fixture(scope='module')
def get_all_structures() -> list:
    crystal_list = []
    structure_list = glob(os.path.join(THIS_DIR, 'structures', '.cif'))
    for structure in structure_list:
        crystal_list.append(Structure.from_file(structure))
    return crystal_list


@pytest.fixture
def get_distributions():
    x = np.linspace(-5, 5, 100)

    t1 = stats.norm(0, 1)
    t2 = stats.expon(1.9)
    normal_dist = np.random.normal(0, 1, size=100)
    exponential_dist = np.random.exponential(1.9, size=100)
    t1_dist = t1.pdf(x)
    t2_dist = t2.pdf(x)

    return [t1_dist, t2_dist, normal_dist, exponential_dist]


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
    structure_1 = Structure.from_file(os.path.join(THIS_DIR, 'structures_supercells', 'UIO-66.cif'))
    structure_2 = Structure.from_file(os.path.join(THIS_DIR, 'structures_supercells', 'UIO-66_2_2_2.cif'))
    s1, s2 = attempt_supercell_pymatgen(structure_1, structure_2)
    assert pytest.approx(s1.lattice.abc, abs=0.01) == s2.lattice.abc

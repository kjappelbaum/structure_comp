#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Kevin M. Jablonka'
__copyright__ = 'MIT License'
__maintainer__ = 'Kevin M. Jablonka'
__email__ = 'kevin.jablonka@epfl.ch'
__version__ = '0.1.0'
__date__ = '28.03.19'
__status__ = 'First Draft, Testing'

import pytest
import os
from glob import glob
import numpy as np
from scipy import stats
from pymatgen import Structure

THIS_DIR = os.path.dirname(__file__)

@pytest.fixture(scope='module')
def get_disordered_dmof_path():
    return os.path.join(THIS_DIR, 'structures_w_disorder', '986883.cif')


@pytest.fixture(scope='module')
def get_cleaned_dmof_path():
    return os.path.join(THIS_DIR, 'structures_w_disorder',
                        '986883_cleaned.cif')


@pytest.fixture(scope='session')
def tmp_dir(tmpdir_factory):
    tmp_dir = tmpdir_factory.mktemp('test_data')
    return tmp_dir


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

@pytest.fixture(scope='module')
def get_ten_identical_files():
    return [os.path.join(THIS_DIR, 'structures', 'Cu-BTC.cif')] * 10


@pytest.fixture(scope='module')
def get_ten_identical_files_and_one_file():
    return [os.path.join(THIS_DIR, 'structures', 'Cu-BTC.cif')
            ] * 10 + os.path.join(THIS_DIR, 'structures', 'Cu-BTC.cif')

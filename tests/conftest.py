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
import pandas as pd
from scipy.stats import laplace, norm

THIS_DIR = os.path.dirname(__file__)


@pytest.fixture(scope='module')
def get_disordered_dmof_path():
    return os.path.join(THIS_DIR, 'structures_w_disorder', '986883.cif')


@pytest.fixture(scope='module')
def get_disordered_uiobipy_path():
    return os.path.join(THIS_DIR, 'structures_w_disorder', 'uio-bipy.cif')


@pytest.fixture(scope='module')
def get_cleaned_dmof_path():
    return os.path.join(THIS_DIR, 'structures_w_disorder',
                        '986883_cleaned.cif')


@pytest.fixture(scope='module')
def get_znbttbbdc_path():
    return os.path.join(THIS_DIR, 'structures_for_rewrite', 'znbttbbdc_a.cif')


@pytest.fixture(scope='module')
def get_cleaned_znbttbbdc_path():
    return os.path.join(THIS_DIR, 'structures_for_rewrite', 'znbttbbdc.cif')


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


@pytest.fixture(scope='module')
def get_two_numeric_property_dataframes():
    df0 = pd.read_csv(
        os.path.join(THIS_DIR, 'dataframes', 'cof_pore_properties.csv'))
    df1 = pd.read_csv(
        os.path.join(THIS_DIR, 'dataframes', 'mof_pore_properties.csv'))

    df0.drop(columns=['name', 'Unnamed: 0'], inplace=True)
    df1.drop(columns=['name', 'Unnamed: 0'], inplace=True)

    return df0, df1


@pytest.fixture(scope='module')
def get_two_distributions():
    n = 500
    mu = 0.0
    sigma = 1
    b = np.sqrt(0.5)

    x = norm.rvs(size=n) * np.sqrt(sigma) + mu
    y = laplace.rvs(size=n, loc=mu, scale=b)

    return x, y

@pytest.fixture(scope='module')
def get_uio_66_water_no_water():
    s = Structure.from_file(
        os.path.join(THIS_DIR, 'structures_w_water/UiO_66_water.cif'))
    s_no_water = Structure.from_file(
        os.path.join(THIS_DIR, 'structures_w_water/UiO-66_no_water.cif'))
    return s, s_no_water
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
from structure_comp.comparators import DistStatistic
import os
import numpy as np
THIS_DIR = os.path.dirname(__file__)


def test_randomized_rmsd(get_ten_identical_files):
    ds = DistStatistic(get_ten_identical_files)
    rmsds = ds.randomized_rmsd(iterations=100)
    assert pytest.approx(sum(rmsds), 0.001) == 0.0


def test_randomized_structure_property(get_ten_identical_files):
    ds = DistStatistic(get_ten_identical_files)
    densities = ds.randomized_structure_property(iterations=10)
    num_sites = ds.randomized_structure_property(
        feature='num_sites', iterations=10)
    volume = ds.randomized_structure_property(feature='volume', iterations=10)

    assert pytest.approx(np.std(densities), 0.001) == 0.0
    assert pytest.approx(np.std(num_sites), 0.001) == 0.0
    assert pytest.approx(np.std(volume), 0.001) == 0.0

    assert pytest.approx(sum(densities), 0.001) == 0.0
    assert pytest.approx(sum(num_sites), 0.001) == 0.0
    assert pytest.approx(sum(volume), 0.001) == 0.0


def test_randomized_graphs(get_ten_identical_files):
    ds = DistStatistic(get_ten_identical_files)
    jaccards = ds.randomized_graphs(iterations=3)
    assert pytest.approx(sum(jaccards), 0.001) == 0.0



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
from structure_comp.comparators import DistStatistic, DistComparison
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

def test_randomized_structure_property_dist_comparator(get_ten_identical_files):
    ds = DistComparison(get_ten_identical_files, get_ten_identical_files)
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
    jaccards = ds.randomized_graphs(iterations=2)
    assert pytest.approx(sum(jaccards), 0.001) == 0.0


def test_qq_test(get_two_numeric_property_dataframes):
    # test two dataframes of different length
    df0 = get_two_numeric_property_dataframes[0]
    df1 = get_two_numeric_property_dataframes[1]

    comparator = DistComparison(property_list_1=df0, property_list_2=df1)

    result_dict = comparator.qq_test(plot=False)

    assert len(result_dict) == len(df0.columns)

    # now make sure that it calculates something meaningful
    # for two times the same dataset, we should get a diagonal

    comparator = DistComparison(property_list_1=df0, property_list_2=df0)

    result_dict = comparator.qq_test(plot=False)

    for column in df0.columns.values:
        assert pytest.approx(result_dict[column]['mse'], 0.001) == 0.0
        assert pytest.approx(result_dict[column]['r2'], 0.001) == 1.0
        assert pytest.approx(
            result_dict[column]['pearson_correlation_coefficient'],
            0.001) == 1.0


def test_mmd_test(get_two_distributions):
    # make sure we get a p-value close to zero for different distributions and
    # a significant one if the distributions are the same

    normal_dist = get_two_distributions[0]
    laplace = get_two_distributions[1]
    statistic_0, pvalue_0 = DistComparison.mmd_test(
        normal_dist.reshape(-1, 1), laplace.reshape(-1, 1))
    statistic_1, pvalue_1 = DistComparison.mmd_test(
        normal_dist.reshape(-1, 1), normal_dist.reshape(-1, 1))

    assert pvalue_0 < 0.1
    assert pvalue_1 > 0.95


def test_property_statistics(get_two_numeric_property_dataframes):
    df0 = get_two_numeric_property_dataframes[0]
    df1 = get_two_numeric_property_dataframes[1]

    comparator = DistComparison(property_list_1=df0, property_list_2=df1)
    result_dict = comparator.properties_test()
    assert len(result_dict) == len(
        df0.columns) + 1  # it is one more because of global features

    comparator = DistComparison(property_list_1=df0, property_list_2=df0)
    result_dict = comparator.properties_test()
    assert len(result_dict) == len(
        df0.columns) + 1  # it is one more because of global

    # for column in list(df0.columns.values) + ['global']:
    #     sub_dict = result_dict[column]
    #     for _, value in sub_dict.iterdict():
    #         assert


def test_property_descriptive_statistics(get_two_numeric_property_dataframes):
    df0 = get_two_numeric_property_dataframes[0]
    df1 = get_two_numeric_property_dataframes[1]

    describer = DistStatistic(property_list=df0)

    result_dict = describer.properties_test_statistics()

    assert len(result_dict) == len(df0.columns)

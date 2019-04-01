#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Kevin M. Jablonka'
__copyright__ = 'MIT License'
__maintainer__ = 'Kevin M. Jablonka'
__email__ = 'kevin.jablonka@epfl.ch'
__version__ = '0.1.0'
__date__ = '28.03.19'
__status__ = 'First Draft, Testing'

import os
import pytest
from structure_comp.utils import get_structure_list
from structure_comp.remove_duplicates import RemoveDuplicates
THIS_DIR = os.path.dirname(__file__)


# compare all structures in structure folder, there should be now duplicates
@pytest.mark.slow
def test_structure_folder():
    structure_list = get_structure_list(os.path.join(THIS_DIR, 'structures'))
    rd_object_1 = RemoveDuplicates(
        structure_list, cached=True, method='standard')
    rd_object_2 = RemoveDuplicates(
        structure_list, cached=False, method='standard')

    rd_object_3 = RemoveDuplicates(structure_list, cached=True, method='rmsd')
    rd_object_4 = RemoveDuplicates(structure_list, cached=False, method='rmsd')

    rd_object_5 = RemoveDuplicates(structure_list, cached=True, method='rmsd_graph')
    rd_object_6 = RemoveDuplicates(structure_list, cached=False, method='rmsd_graph')

    rd_object_1.run_filtering()
    assert rd_object_1.number_of_duplicates == 0
    rd_object_2.run_filtering()
    assert rd_object_2.number_of_duplicates == 0

    rd_object_3.run_filtering()
    assert rd_object_3.number_of_duplicates == 0
    rd_object_4.run_filtering()
    assert rd_object_4.number_of_duplicates == 0

    rd_object_5.run_filtering()
    assert rd_object_5.number_of_duplicates == 0
    rd_object_6.run_filtering()
    assert rd_object_6.number_of_duplicates == 0

    assert not (rd_object_1.number_of_duplicates >
                rd_object_2.number_of_duplicates)
    assert not (rd_object_1.number_of_duplicates <
                rd_object_2.number_of_duplicates)


@pytest.mark.slow
def test_structure_folder_2():
    rd_object_1 = RemoveDuplicates.from_folder(
        os.path.join(THIS_DIR, 'structures'), cached=True, method='standard')
    rd_object_2 = RemoveDuplicates.from_folder(
        os.path.join(THIS_DIR, 'structures'), cached=False, method='standard')

    rd_object_3 = RemoveDuplicates.from_folder(
        os.path.join(THIS_DIR, 'structures'), cached=True, method='rmsd')
    rd_object_4 = RemoveDuplicates.from_folder(
        os.path.join(THIS_DIR, 'structures'), cached=False, method='rmsd')

    rd_object_1.run_filtering()
    assert rd_object_1.number_of_duplicates == 0
    rd_object_2.run_filtering()
    assert rd_object_2.number_of_duplicates == 0

    rd_object_3.run_filtering()
    assert rd_object_3.number_of_duplicates == 0
    rd_object_4.run_filtering()
    assert rd_object_4.number_of_duplicates == 0

    assert not (rd_object_1.number_of_duplicates >
                rd_object_2.number_of_duplicates)
    assert not (rd_object_1.number_of_duplicates <
                rd_object_2.number_of_duplicates)


@pytest.mark.slow
def test_structure_folder_3():
    structure_list = get_structure_list(os.path.join(
        THIS_DIR, 'structures')) + [
            os.path.join(THIS_DIR, 'structure_duplicates', 'UiO-66.cif')
        ]
    rd_object_1 = RemoveDuplicates(
        structure_list, cached=True, method='standard')
    rd_object_2 = RemoveDuplicates(
        structure_list, cached=False, method='standard')

    rd_object_3 = RemoveDuplicates(structure_list, cached=True, method='rmsd')
    rd_object_4 = RemoveDuplicates(structure_list, cached=False, method='rmsd')

    rd_object_1.run_filtering()
    assert rd_object_1.number_of_duplicates == 1
    rd_object_2.run_filtering()
    assert rd_object_2.number_of_duplicates == 1

    rd_object_3.run_filtering()
    assert rd_object_3.number_of_duplicates == 1
    rd_object_4.run_filtering()
    assert rd_object_4.number_of_duplicates == 1

    assert not (rd_object_1.number_of_duplicates >
                rd_object_2.number_of_duplicates)
    assert not (rd_object_1.number_of_duplicates <
                rd_object_2.number_of_duplicates)


# Check supercell handling
#@pytest.mark.skip(
#    reason="the graph construction for a huge unitcell is too slow")
def test_supercells():
    structure_list = get_structure_list(
        os.path.join(THIS_DIR, 'structures_supercells'))

    #rd_object_1 = RemoveDuplicates(
    #    structure_list, cached=True, method='rmsd')
    rd_object_2 = RemoveDuplicates(
        structure_list, cached=False, method='rmsd')

    #rd_object_1.run_filtering()
    #assert rd_object_1.number_of_duplicates == 1
    rd_object_2.run_filtering()
    assert rd_object_2.number_of_duplicates == 1



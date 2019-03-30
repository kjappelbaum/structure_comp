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
from structure_comp.utils import get_structure_list
from structure_comp.remove_duplicates import RemoveDuplicates
THIS_DIR = os.path.dirname(__file__)


# compare all structures in structure folder, there should be now duplicates
def test_structure_folder():
    structure_list = get_structure_list(os.path.join(THIS_DIR, 'structures'))
    rd_object_1 = RemoveDuplicates(structure_list, cached=True, method='standard')
    rd_object_2 = RemoveDuplicates(structure_list, cached=False, method='standard')

    #rd_object_1.run_filtering()
    rd_object_2.run_filtering()

# Make sure that structures are removed if they are supercells
def all_supercells_test():
    pass


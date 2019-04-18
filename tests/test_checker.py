#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Kevin M. Jablonka'
__copyright__ = 'MIT License'
__maintainer__ = 'Kevin M. Jablonka'
__email__ = 'kevin.jablonka@epfl.ch'
__version__ = '0.1.0'
__date__ = '18.04.19'
__status__ = 'First Draft, Testing'

import os
import pytest
from pymatgen import Structure
from structure_comp.checker import  Checker
THIS_DIR = os.path.dirname(__file__)

def test_check_clashing(get_disordered_dmof_path, get_disordered_uiobipy_path):
    s = Structure.from_file(get_disordered_dmof_path)
    s2 = Structure.from_file(get_disordered_uiobipy_path)
    assert Checker.check_clashing(s)
    assert Checker.check_clashing(s2)

    # it should certainly not flag methyl or amino groups in crowded structures
    s3 = Structure.from_file(os.path.join(THIS_DIR, 'clash_test', 'uio-66-ndc.cif'))
    assert Checker.check_clashing(s3) is False

def test_check_hydrogens():
    s0 = Structure.from_file(os.path.join(THIS_DIR, 'structures_no_h', 'znbttbbdc.cif'))
    assert Checker.check_hydrogens(s0) is False
    assert Checker.check_hydrogens(s0, strictness='tight') is False
    assert Checker.check_hydrogens(s0, strictness='medium') is False

@pytest.mark.slow
def test_check_unbound(get_uio_66_water_no_water):
    s, s_no_water = get_uio_66_water_no_water
    assert Checker.check_unbound(s)
    assert Checker.check_unbound(s_no_water) is False

    assert Checker.check_unbound(s, mode='graph')
    assert Checker.check_unbound(s_no_water, mode='graph') is False

def test_run_flagging(get_all_structures):
    checker_object = Checker.from_folder(os.path.join(THIS_DIR, 'structures'))
    flagging_result = checker_object.run_flagging()
    print(flagging_result)
    assert  0 == 1
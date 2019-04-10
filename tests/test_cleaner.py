#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'Kevin M. Jablonka'
__copyright__ = 'MIT License'
__maintainer__ = 'Kevin M. Jablonka'
__email__ = 'kevin.jablonka@epfl.ch'
__version__ = '0.1.0'
__date__ = '05.04.19'
__status__ = 'First Draft, Testing'

import pytest
import os
import filecmp
import shutil
from pymatgen import Structure
from ase.io import read
from structure_comp.cleaner import Cleaner
THIS_DIR = os.path.dirname(__file__)


def test_rewrite_cif(tmp_dir, get_disordered_dmof_path, get_cleaned_dmof_path,
                     get_znbttbbdc_path, get_cleaned_znbttbbdc_path):
    outfile = Cleaner.rewrite_cif(get_disordered_dmof_path, tmp_dir)
    assert filecmp.cmp(outfile, get_cleaned_dmof_path)

    outfile2 = Cleaner.rewrite_cif(get_znbttbbdc_path, tmp_dir)
    assert filecmp.cmp(outfile2, get_cleaned_znbttbbdc_path)


@pytest.mark.slow
def test_remove_solvent():
    s = Structure.from_file(
        os.path.join(THIS_DIR, 'structures_w_water/UiO_66_water.cif'))
    s_no_water = Structure.from_file(
        os.path.join(THIS_DIR, 'structures_w_water/UiO-66_no_water.cif'))
    s_cleaned = Cleaner.remove_unbound_solvent(s)
    assert s_cleaned == s_no_water


def test_remove_disorder():
    ...


@pytest.mark.slow
def test_openbabel(tmp_dir):
    # test structure in which composition shouldn't change
    test_path = os.path.join(tmp_dir, 'UiO-66_no_water.cif')
    shutil.copy(
        os.path.join(THIS_DIR, 'structures_w_water/UiO-66_no_water.cif'),
        test_path)
    Cleaner.openbabel(test_path)
    atoms = read(test_path)
    assert atoms.get_chemical_formula() == 'C192H96O120Zr24'

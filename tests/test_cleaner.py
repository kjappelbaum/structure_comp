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
from structure_comp.cleaner import Cleaner
THIS_DIR = os.path.dirname(__file__)


def test_rewrite_cif(tmp_dir, get_disordered_dmof_path, get_cleaned_dmof_path):
    outfile = Cleaner.rewrite_cif(get_disordered_dmof_path, tmp_dir)
    assert filecmp.cmp(outfile, get_cleaned_dmof_path)

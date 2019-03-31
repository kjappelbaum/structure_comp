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
from structure_comp.rmsd import attempt_supercell, parse_periodic_case, rmsd, kabsch_rmsd
from ase.io import read
from ase.build import niggli_reduce
import os
THIS_DIR = os.path.dirname(__file__)


@pytest.fixture(scope='module')
def get_supercell_paths():
    return [
        os.path.join(THIS_DIR, 'structures_supercells', 'UIO-66.cif'),
        os.path.join(THIS_DIR, 'structures_supercells', 'UIO-66_2_2_2.cif')
    ]


# Test we can successfully rebuild supercell
def test_attempt_supercell(get_supercell_paths):
    atoms1 = read(get_supercell_paths[0])
    atoms2 = read(get_supercell_paths[1])
    niggli_reduce(atoms1)
    niggli_reduce(atoms2)

    a_1, a_2 = attempt_supercell(atoms1, atoms2)
    assert pytest.approx(a_1.get_cell_lengths_and_angles(),
                         0.001) == a_2.get_cell_lengths_and_angles()


def test_rmsd(get_supercell_paths):
    _, P, _, Q = parse_periodic_case(get_supercell_paths[0],
                                     get_supercell_paths[1])
    result = kabsch_rmsd(P, Q)
    assert pytest.approx(result, abs=0.01) == 0.0

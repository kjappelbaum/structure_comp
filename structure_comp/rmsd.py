#!/usr/bin/python
# -*- coding: utf-8 -*-

# Extract and modification of the relevant RMSD functions of
# the RMSD package (https://github.com/charnley/rmsd), most of the code in this module is
# directly copied from aforementioned repository

import numpy as np
from ase.io import read
from ase.build import niggli_reduce
from pymatgen.io.ase import AseAtomsAdaptor
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def rmsd(V: np.array, W: np.array):
    """
    Calculate Root-mean-square deviation from two sets of vectors V and W.

    Parameters
    ----------
    V : array
        (N,D) matrix, where N is points and D is dimension.
    W : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    rmsd : float
        Root-mean-square deviation between the two vectors
    """

    return np.sqrt(np.mean((np.subtract(V, W))**2))


def kabsch_rmsd(P, Q, translate=False):
    """
    Rotate matrix P unto Q using Kabsch algorithm and calculate the RMSD.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    translate : bool
        Use centroids to translate vector P and Q unto each other.

    Returns
    -------
    rmsd : float
        root-mean squared deviation
    """
    if P.shape[0] > Q.shape[0]:
        Q_temp = np.zeros(P.shape)
        Q_temp[:Q.shape[0], :Q.shape[1]] = Q
        Q = Q_temp
    elif Q.shape[0] > P.shape[0]:
        P_temp = np.zeros(Q.shape)
        P_temp[:P.shape[0], :P.shape[1]] = P
        P = P_temp

    if translate:
        Q = Q - centroid(Q)
        P = P - centroid(P)

    P = kabsch_rotate(P, Q)
    return rmsd(P, Q)


def kabsch_rotate(P, Q):
    """
    Rotate matrix P unto matrix Q using Kabsch algorithm.

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    P : array
        (N,D) matrix, where N is points and D is dimension,
        rotated

    """

    U = kabsch(P, Q)

    # Rotate P
    P = np.dot(P, U)
    return P


def kabsch(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.

    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U

    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm

    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


def centroid(X):
    """
    Centroid is the mean position of all the points in all of the coordinate
    directions, from a vectorset X.

    https://en.wikipedia.org/wiki/Centroid

    C = sum(X)/len(X)

    Parameters
    ----------
    X : array
        (N,D) matrix, where N is points and D is dimension.

    Returns
    -------
    C : float
        centroid
    """
    C = X.mean(axis=0)
    return C


def hungarian(A, B):
    """
    Hungarian reordering.
    Assume A and B are coordinates for atoms of SAME type only
    """

    # should be kabasch here i think
    distances = cdist(A, B, 'euclidean')

    # Perform Hungarian analysis on distance matrix between atoms of 1st
    # structure and trial structure
    indices_a, indices_b = linear_sum_assignment(distances)

    return indices_b


def reorder_hungarian(p_atoms, q_atoms, p_coord, q_coord):
    """
    Re-orders the input atom list and xyz coordinates using the Hungarian
    method (using optimized column results)
    Parameters
    ----------
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_atoms : array
        (N,1) matrix, where N is points holding the atoms' names
    p_coord : array
        (N,D) matrix, where N is points and D is dimension
    q_coord : array
        (N,D) matrix, where N is points and D is dimension
    Returns
    -------
    view_reorder : array
             (N,1) matrix, reordered indexes of atom alignment based on the
             coordinates of the atoms
    """

    # Find unique atoms
    unique_atoms = np.unique(p_atoms)

    # generate full view from q shape to fill in atom view on the fly
    view_reorder = np.zeros(q_atoms.shape, dtype=int)
    view_reorder -= 1

    for atom in unique_atoms:
        p_atom_idx, = np.where(p_atoms == atom)
        q_atom_idx, = np.where(q_atoms == atom)

        A_coord = p_coord[p_atom_idx]
        B_coord = q_coord[q_atom_idx]

        view = hungarian(A_coord, B_coord)
        view_reorder[p_atom_idx] = q_atom_idx[view]

    return view_reorder

def parse_periodic_case(file_1,
                        file_2,
                        try_supercell: bool = True,
                        pymatgen: bool = False,
                        get_reduced_structure: bool = True):
    """
    Parser for periodic structures, handles two possible cases:
        (1) Structures are supercells (within tolerance), then one cell is multiplied by the scaling factors
        (2) Structures are not supercells of each other, then we rescale on cell to the volume of the other cell
        to make sure we have meaningful comparisons.

    Args:
        file_1 (str/pymatgen structure object): path to first file, in on format that ASE can parse, pymatgen structure
            object in case pymatgen=True
        file_2 (str/pymatgen structure object): path to second file, pymatgen structure object in case pymatgen=True
        try_supercell (bool): if true, we attempt to build a supercell, default: True
        pymatgen (bool): if true, then file_1 and file_2 take pymatgen structure objects
        get_reduced_structure (bool): if true (default) it gets the Niggli reduced cell.

    Returns:
        atomic symbols (list), cartesian positions (list) of structure 1,
        atomic symbols (list), cartesian positions (list) of structure 2

    """

    if pymatgen:
        atoms1 = AseAtomsAdaptor.get_atoms(file_1)
        atoms2 = AseAtomsAdaptor.get_atoms(file_2)
    else:
        atoms1 = read(file_1)
        atoms2 = read(file_2)

    if get_reduced_structure:
        niggli_reduce(atoms1)
        niggli_reduce(atoms2)

    if try_supercell:
        a1, a2 = attempt_supercell(atoms1, atoms2)
    else:
        a1, a2 = rescale_periodic_system(atoms1, atoms2)

    atomic_symbols_1 = a1.get_chemical_symbols()
    positions_1 = a1.get_positions()

    atomic_symbols_2 = a2.get_chemical_symbols()
    positions_2 = a2.get_positions()

    return atomic_symbols_1, positions_1, atomic_symbols_2, positions_2


def attempt_supercell(atoms1, atoms2):
    """
    Checks if the lattice vectors of one cell are integer multiples of the other cell.
    For this to be meaningful, the lattices should be Niggli reduced.

    To get the order of the check correct without to many checks, we use the volume.
    Args:
        atoms1 (ase atoms object):
        atoms2 (ase atoms object):

    Returns:

    """
    lattice1 = atoms1.get_cell_lengths_and_angles()[0:3]
    lattice2 = atoms2.get_cell_lengths_and_angles()[0:3]

    one_larger_than_two = False

    if atoms1.get_volume() > atoms2.get_volume():
        factors = lattice1 / lattice2
        one_larger_than_two = True
    else:
        factors = lattice2 / lattice1

    x = np.array(factors)
    x_int = x.astype(int)
    if np.all(np.isclose(x, x_int, 0.001)):
        x = x_int
        logger.debug('found supercell with scaling factors %s', x)
        if one_larger_than_two:
            atoms2 = atoms2 * x
        else:
            atoms1 = atoms1 * x

    return atoms1, atoms2


def rescale_periodic_system(atoms1, atoms2):
    """
    Scales two periodic systems to the same size.
    Not the most efficient implementation yet.

    For a first implementation, I assume that the number
    of atoms in both cells is the same. Later, I will
    create supercells to fix this.


    Parameters
    ----------
        atoms1: ASE atoms object
        atoms2: ASE atoms object

    Returns
    --------
        atoms1_copy: ASE atoms object
        atoms2: ASE atoms object
    """
    atoms1_copy = atoms1.copy()
    atoms1_copy.set_cell(atoms2.get_cell(), scale_atoms=True)

    return atoms1_copy, atoms2

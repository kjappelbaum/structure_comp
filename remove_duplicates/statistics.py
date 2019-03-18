#!/usr/bin/python
# -*- coding: utf-8 -*-

# Get basic statistics describing the database
# Compare a structure to a database

from tqdm.autonotebook import tqdm
from pymatgen import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN
from rmsd import parse_periodic_case, rmsd
import random


def get_rmsd_matrix(structure_a: Structure, structure_b: Structure) -> float:
    p_atoms, P, q_atoms, Q = parse_periodic_case(structure_a, structure_b)
    result = rmsd(P, Q)
    return result


def randomized_rmsd(structure_list: list, iterations: float = 5000) -> list:
    rmsds = []

    for _ in tqdm(range(iterations)):
        random_selection = random.sample(structure_list, 2)
        a = get_rmsd_matrix(random_selection[0], random_selection[1])
        rmsds.append(a)

    return rmsds


def randomized_graphs(structure_list, iterations=5000):
    diffs = []
    for _ in tqdm(range(iterations)):
        random_selection = random.sample(structure_list, 2)
        crystal_a = Structure.from_file(
            random_selection[0])
        crystal_b = Structure.from_file(
            random_selection[1])
        nn_strategy = JmolNN()
        sgraph_a = StructureGraph.with_local_env_strategy(
            crystal_a, nn_strategy)
        sgraph_b = StructureGraph.with_local_env_strategy(
            crystal_b, nn_strategy)
        diffs.append(sgraph_a.diff(sgraph_b, strict=False))
    return diffs 


def randomized_composition():
    ...

#!/usr/bin/env python
#_*_coding:utf-8_*_

import numpy as np
import os
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union, Tuple

def load_predicted_PDB(pdbfile):
    # Generate (diagonalized) C_alpha distance matrix from a pdbfile
    parser = PDBParser()
    structure = parser.get_structure(pdbfile.split('/')[-1].split('.')[0], pdbfile)
    # name = structure.header['name']
    # print(name)
    residues = [r for r in structure.get_residues()]

    # sequence from atom lines
    records = SeqIO.parse(pdbfile, 'pdb-atom')
    seqs = [str(r.seq) for r in records]

    distances = np.empty((len(residues), len(residues)))
    for x in range(len(residues)):
        for y in range(len(residues)):
            one = residues[x]["CA"].get_coord()
            two = residues[y]["CA"].get_coord()
            distances[x, y] = np.linalg.norm(one - two)

    return distances, seqs[0]

def pdb_process(pdb_feature):
    pdb_mean = np.mean(pdb_feature, axis=0, keepdims=True)
    pdb_feature = np.concatenate((pdb_feature,pdb_mean), axis=0)
    pdb_feature = 1 / (pdb_feature + 1)
    e_matrix = np.exp(pdb_feature - np.max(pdb_feature, axis=-1, keepdims=True))
    return e_matrix / e_matrix.sum(axis=-1, keepdims=True)

def get_pdb_feature(pdb_file_1, pdb_file_2, pdb_file_3, pdb_file_4, pdb_file_5):
    dis_1, seq_1 = load_predicted_PDB(pdb_file_1)
    dis_2, seq_2 = load_predicted_PDB(pdb_file_2)
    dis_3, seq_3 = load_predicted_PDB(pdb_file_3)
    dis_4, seq_4 = load_predicted_PDB(pdb_file_4)
    dis_5, seq_5 = load_predicted_PDB(pdb_file_5)

    dis_1 = np.expand_dims(dis_1, axis=0)
    dis_2 = np.expand_dims(dis_2, axis=0)
    dis_3 = np.expand_dims(dis_3, axis=0)
    dis_4 = np.expand_dims(dis_4, axis=0)
    dis_5 = np.expand_dims(dis_5, axis=0)

    dis = np.concatenate((dis_1, dis_2, dis_3, dis_4, dis_5), 0)

    pdb_feature = pdb_process(dis)

    return pdb_feature

if __name__ == '__main__':
    feature = get_pdb_feature("../examples/UNIPROT_E3XRD1_1.pdb",
                              "../examples/UNIPROT_E3XRD1_2.pdb",
                              "../examples/UNIPROT_E3XRD1_3.pdb",
                              "../examples/UNIPROT_E3XRD1_4.pdb",
                              "../examples/UNIPROT_E3XRD1_5.pdb")
    print(feature)
#!/usr/bin/env python
#_*_coding:utf-8_*_

import numpy as np
import torch
import esm
from features.msaprocessing import parse_a3m, get_msa_256
from Bio import SeqIO

model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


def get_msa_feature(msa_file, length=32, max_msa=256):

    a3m_file = msa_file
    with open(a3m_file, 'r') as f:
        a3m_str = f.read()
    # parse_a3m
    msa = parse_a3m(a3m_str)
    seq = msa.sequences[0]
    # print(seq)
    msa_matrix = get_msa_256(msa, length, max_msa)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    num = int(len(seq) / length) + 1

    # print(num)
    for i in range(num):
        # print(i)
        batch_labels, batch_strs, batch_tokens = batch_converter(msa_matrix[i])
        batch_tokens = batch_tokens.to(device)
        results = model(batch_tokens, repr_layers=[12], )
        msa_representations = results["representations"][12]
        msa_query_representation = msa_representations[:, 0, 1:, :]
        # print(msa_query_representation.shape)
        msa_query_representation = torch.squeeze(msa_query_representation, 0)
        # print(msa_query_representation.shape)
        if i == 0:
            msa = msa_query_representation.detach().cpu().numpy()
        else:
            msa = np.concatenate((msa, msa_query_representation.detach().cpu().numpy()), 0)


    return msa

if __name__ == '__main__':
    a3m_file = '../examples/UNIPROT_E3XRD1.a3m'
    msa_feature = get_msa_feature(a3m_file)
    print(msa_feature.shape)

# -*- coding: utf-8 -*-

import numpy as np
import os
from Bio import SeqIO
from typing import Any, Mapping, MutableMapping, Optional, Sequence, Union
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
import torch
import logging
import time
import tqdm


def pdb_process(pdb_feature):
    pdb_mean = np.mean(pdb_feature, axis=0, keepdims=True)
    pdb_feature = np.concatenate((pdb_feature,pdb_mean), axis=0)
    pdb_feature = 1 / (pdb_feature + 1)
    e_matrix = np.exp(pdb_feature - np.max(pdb_feature, axis=-1, keepdims=True))
    return e_matrix / e_matrix.sum(axis=-1, keepdims=True)


class ProteinDataset(Dataset):
    def __init__(self,
                 fasta_file=None,
                 label_file=None,
                 batch_size=20,
                 seq_len_batch=1000,
                 max_seq_len=1000,
                 pad_index=0,
                 msa_path=None,
                 pdb_path=None,
                 feature_path=None,
                 ):
        """

                :param vocab_path: AA_dict
                :param batch_size:
                :param seq_len_batch:
                    seq_len_batch = None, seq_len_batch = the longest sample length in the batch
                    seq_len_batch = 50, seq_len_batch = 50

                :param max_seq_len: Specify the maximum sample length, exceeding this length will be cut
                :param is_sample_shuffle:

         """

        self.PAD_IDX = pad_index
        # self.UNK_IDX = '[UNK]'

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.msa_path = msa_path
        self.pdb_path = pdb_path
        if isinstance(seq_len_batch, int) and seq_len_batch > max_seq_len:
            seq_len_batch = max_seq_len
        self.max_seq_len_batch = seq_len_batch

        self.seq = []
        self.name = []
        for seq_record in SeqIO.parse(fasta_file, "fasta"):
            self.seq.append(str(seq_record.seq))
            self.name.append(str(seq_record.id.replace("|", "_")))
        self.label = np.loadtxt(label_file)
        self.feature_path = feature_path


    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        seq = self.seq[index]
        name = self.name[index]
        label = self.label[index]
        if self.feature_path == None:
            other_feature = None
        else:
            other_feature = np.loadtxt(str(self.feature_path) + str(name) + '.txt')

        msa_feature = np.loadtxt(str(self.msa_path) + str(name) + '.txt')
        pdb_feature = np.load(str(self.pdb_path) + str(name) + '.npy')

        if len(seq) > self.max_seq_len_batch:
            msa_feature = msa_feature[0:self.max_seq_len_batch, :]
            pdb_feature = pdb_feature[:, 0:self.max_seq_len_batch, 0:self.max_seq_len_batch]
            pdb_feature = pdb_process(pdb_feature)
            pad_mask = [0] * self.max_seq_len_batch
        else:
            pad_length = self.max_seq_len_batch - len(seq)
            msa_feature = np.pad(msa_feature, ((0, pad_length), (0, 0)), "constant")
            pdb_feature = pdb_process(pdb_feature)
            pdb_feature = np.pad(pdb_feature, ((0, 0), (0, pad_length), (0, pad_length)), "constant")
            pad_mask = [0] * len(seq) + [1] * pad_length

        msa_feature = torch.tensor(msa_feature, dtype=torch.float)
        pdb_feature = torch.tensor(pdb_feature, dtype=torch.float)
        pad_mask = torch.tensor(pad_mask, dtype=torch.long)
        label = torch.tensor(int(label), dtype=torch.long)

        other_feature = torch.tensor(other_feature, dtype=torch.float)

        return seq, name, msa_feature, pdb_feature, pad_mask, other_feature, label


if __name__ == '__main__':
    data = ProteinDataset(fasta_file="../data/Uniprot_train.fasta",
                          label_file="../data/label_train.txt",
                          msa_path="../data/msa_feature/",
                          pdb_path="../data/pdb/",
                          feature_path="../data/other_feature/")
    dataloader = DataLoader(data, batch_size=2, shuffle=True, drop_last=True, num_workers=1)
    for seq, name, msa_feature, pdb_feature, pad_mask, other_feature, label in dataloader:
        print(pdb_feature)
        print(pdb_feature.sum(-2))


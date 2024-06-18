#!/usr/bin/env python
#_*_coding:utf-8_*_

import numpy as np
import torch
import os, datetime, argparse
from features.msaprocessing import parse_a3m
from features.msa_feature import get_msa_feature
from features.pdb_feature import get_pdb_feature
from features.other_feature import get_other_feature



def test(msa_file, pdb_file_1, pdb_file_2, pdb_file_3, pdb_file_4, pdb_file_5, pssm_file, model_file, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    with open(msa_file, 'r') as f:
        a3m_str = f.read()
    # parse_a3m
    msa = parse_a3m(a3m_str)
    sequence = msa.sequences[0]
    msa_feature = get_msa_feature(msa_file)
    pdb_feature = get_pdb_feature(pdb_file_1, pdb_file_2, pdb_file_3, pdb_file_4, pdb_file_5)
    other_feature = get_other_feature(sequence, pssm_file)

    max_seq_len =1000
    if len(sequence) > max_seq_len:
        msa_feature = msa_feature[0:max_seq_len, :]
        pdb_feature = pdb_feature[:, 0:max_seq_len, 0:max_seq_len]
        pad_mask = [0] * max_seq_len
    else:
        pad_length = max_seq_len - len(sequence)
        msa_feature = np.pad(msa_feature, ((0, pad_length), (0, 0)), "constant")
        pdb_feature = np.pad(pdb_feature, ((0, 0), (0, pad_length), (0, pad_length)), "constant")
        pad_mask = [0] * len(sequence) + [1] * pad_length

    msa_feature = torch.tensor(msa_feature, dtype=torch.float)
    pdb_feature = torch.tensor(pdb_feature, dtype=torch.float)
    pad_mask = torch.tensor(pad_mask, dtype=torch.long)
    other_feature = torch.tensor(other_feature, dtype=torch.float)

    model = torch.load(model_file)
    model.eval()
    with torch.no_grad():
        msa_feature = msa_feature.to(device).unsqueeze(0)
        pbd_feature = pdb_feature.to(device).unsqueeze(0)
        pad_mask = pad_mask.to(device).unsqueeze(0)
        other_feature = other_feature.to(device).unsqueeze(0)


        pred, attn = model(msa_feature, pbd_feature, pad_mask, other_feature)
        pred = pred.argmax(dim=1)

        pred = pred.cpu().numpy().flatten()
        if pred == 0:
            print("This sequence is predicted to be an ARG")
        elif pred == 1:
            print("This sequence is predicted to be a VF")
        elif pred == 3:
            print("This sequence is predicted to be neither an ARG nor a VF")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--msa_file", type=str, help="msa file")
    parser.add_argument("--pdb_file_1", type=str, help="first pdb file")
    parser.add_argument("--pdb_file_2", type=str, help="second pdb file")
    parser.add_argument("--pdb_file_3", type=str, help="third pdb file")
    parser.add_argument("--pdb_file_4", type=str, help="fourth pdb file")
    parser.add_argument("--pdb_file_5", type=str, help="fifth pdb file")
    parser.add_argument("--pssm_file", type=str, help="pssm file")
    parser.add_argument("--model_file", type=str, help="model file")
    args = parser.parse_args()

    test(msa_file=args.msa_file,
         pdb_file_1=args.pdb_file_1,
         pdb_file_2=args.pdb_file_2,
         pdb_file_3=args.pdb_file_3,
         pdb_file_4=args.pdb_file_4,
         pdb_file_5=args.pdb_file_5,
         pssm_file=args.pssm_file,
         model_file=args.model_file,)

    # test('../examples/UNIPROT_E3XRD1.a3m',
    #      "../examples/UNIPROT_E3XRD1_1.pdb",
    #      "../examples/UNIPROT_E3XRD1_2.pdb",
    #      "../examples/UNIPROT_E3XRD1_3.pdb",
    #      "../examples/UNIPROT_E3XRD1_4.pdb",
    #      "../examples/UNIPROT_E3XRD1_5.pdb",
    #      "../examples/UNIPROT_E3XRD1.pssm",
    #      "SEVA.pt"
    #      )

    # python prediction.py --msa_file ../examples/UNIPROT_E3XRD1.a3m --pdb_file_1 ../examples/UNIPROT_E3XRD1_1.pdb --pdb_file_2 ../examples/UNIPROT_E3XRD1_2.pdb --pdb_file_3 ../examples/UNIPROT_E3XRD1_3.pdb --pdb_file_4 ../examples/UNIPROT_E3XRD1_4.pdb --pdb_file_5 ../examples/UNIPROT_E3XRD1_5.pdb --pssm_file ../examples/UNIPROT_E3XRD1.pssm --model_file SEVA.pt
#!/usr/bin/env python
#_*_coding:utf-8_*_

import numpy as np
import torch
import os, datetime, argparse
from features.statistical_feature import get_statistical_feature
import time

import matplotlib.pyplot as plt

def test(fasta_file, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    for seq_record in SeqIO.parse(fasta_file, "fasta"):
            sequence = str(seq_record.seq)
    
    other_feature = get_statistical_feature(sequence, pssm_file)
    other_feature = torch.tensor(other_feature, dtype=torch.float)

    model = torch.load(model_file)
    model.eval()
    with torch.no_grad():
       
        other_feature = other_feature.to(device).unsqueeze(0)

        pred = model(other_feature)
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
    parser.add_argument("--fasta_file", type=str, help="pssm file")
    parser.add_argument("--model_file", type=str, help="model file")
    args = parser.parse_args()


    test('../examples/UNIPROT_test.fasta',
         "SEVA_fast.pt"
         )

    # python prediction.py --fasta_file  ../examples/UNIPROT_test.fasta --model_file SEVA_fast.pt

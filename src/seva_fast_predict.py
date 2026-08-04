#!/usr/bin/env python
# _*_coding:utf-8_*_

import numpy as np
import torch
import os, datetime, argparse
from features.statistical_feature import get_statistical_feature
import time

import matplotlib.pyplot as plt


def test(sequence, model_file):


    other_feature = get_statistical_feature(sequence)
    other_feature = torch.tensor(other_feature, dtype=torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        elif pred == 2:
            print("This sequence is predicted to be neither an ARG nor a VF")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=str, help="protein sequence")
    parser.add_argument("--model_file", type=str, help="model file")
    args = parser.parse_args()

    test(sequence=args.sequence,
         model_file=args.model_file, )

    # test('MFEIHPVKKVSVVIPVYNEQESLPELIRRTTAACESLGKEYEILLIDDGSSDNSAHMLVEASQAEGSHIVSILLNRNYGQHSAIMAGFSHVTGDLIITLDADLQNPPEEIPRLVAKADEGYDVVGTVRQNRQDSWFRKTASKMINRLIQRTTGKAMGDYGCMLRAYRRHIVDAMLHCHERSTFIPILANIFARRAIEIPVHHAEREFGESKYSFMHLINLMYDLVTCLTTTPLRMLSLLGSIIAIGGFSIAVLLVILRLTFGPQWAAEGVFMLFAVLFTFIGAQFIGMGLLGEYIGRIYTDVRARPRYFVQQVIRPSSKENE',
    #      "SEVA_fast.pt"
    #      )

    # python seva_fast_predict.py --sequence  'MFEIHPVKKVSVVIPVYNEQESLPELIRRTTAACESLGKEYEILLIDDGSSDNSAHMLVEASQAEGSHIVSILLNRNYGQHSAIMAGFSHVTGDLIITLDADLQNPPEEIPRLVAKADEGYDVVGTVRQNRQDSWFRKTASKMINRLIQRTTGKAMGDYGCMLRAYRRHIVDAMLHCHERSTFIPILANIFARRAIEIPVHHAEREFGESKYSFMHLINLMYDLVTCLTTTPLRMLSLLGSIIAIGGFSIAVLLVILRLTFGPQWAAEGVFMLFAVLFTFIGAQFIGMGLLGEYIGRIYTDVRARPRYFVQQVIRPSSKENE' --model_file SEVA_fast.pt

#!/usr/bin/env python
#_*_coding:utf-8_*_

import numpy as np
from features.AAC import AAC
from features.DPC import DPC
from features.DDE import DDE
from features.PAAC import PAAC
from features.QSOrder import QSOrder



def get_statistical_feature(sequence):
    AAC_feature = AAC(sequence)
    DPC_feature = DPC(sequence)
    DDE_feature = DDE(sequence)
    PAAC_feature = PAAC(sequence)
    QSOrder_feature = QSOrder(sequence)
  

    other_feature = np.hstack((AAC_feature,
                                DDE_feature,
                                DPC_feature,
                                PAAC_feature,
                                QSOrder_feature,
                                ))
    other_feature = other_feature.flatten().astype(np.float64)

    return other_feature

if __name__ == '__main__':
    sequence = "MFEIHPVKKVSVVIPVYNEQESLPELIRRTTAACESLGKEYEILLIDDGSSDNSAHMLVEASQAEGSHIVSILLNRNYGQHSAIMAGFSHVTGDLIITLDADLQNPPEEIPRLVAKADEGYDVVGTVRQNRQDSWFRKTASKMINRLIQRTTGKAMGDYGCMLRAYRRHIVDAMLHCHERSTFIPILANIFARRAIEIPVHHAEREFGESKYSFMHLINLMYDLVTCLTTTPLRMLSLLGSIIAIGGFSIAVLLVILRLTFGPQWAAEGVFMLFAVLFTFIGAQFIGMGLLGEYIGRIYTDVRARPRYFVQQVIRPSSKENE"
    statistical_feature = get_statistical_feature(sequence)
    print(statistical_feature)

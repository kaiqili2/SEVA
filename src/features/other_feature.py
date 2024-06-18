#!/usr/bin/env python
#_*_coding:utf-8_*_

import numpy as np
from features.AAC import AAC
from features.DPC import DPC
from features.DDE import DDE
from features.PAAC import PAAC
from features.QSOrder import QSOrder
from features.PSSMC import PSSMC
from features.AADPPSSM import AADPPSSM
from features.RPMPSSM import RPMPSSM


def get_other_feature(sequence, pssm_file):
    AAC_feature = AAC(sequence)
    DPC_feature = DPC(sequence)
    DDE_feature = DDE(sequence)
    PAAC_feature = PAAC(sequence)
    QSOrder_feature = QSOrder(sequence)
    PSSMC_feature = PSSMC(sequence,pssm_file)
    AADPPSSM_feature = AADPPSSM(sequence, pssm_file)
    RPMPSSM_feature = RPMPSSM(sequence, pssm_file)

    other_feature = np.hstack((AAC_feature,
                                DDE_feature,
                                DPC_feature,
                                PAAC_feature,
                                QSOrder_feature,
                                PSSMC_feature,
                                AADPPSSM_feature,
                                RPMPSSM_feature))
    other_feature = other_feature.flatten().astype(np.float64)

    return other_feature

if __name__ == '__main__':
    sequence = "MFEIHPVKKVSVVIPVYNEQESLPELIRRTTAACESLGKEYEILLIDDGSSDNSAHMLVEASQAEGSHIVSILLNRNYGQHSAIMAGFSHVTGDLIITLDADLQNPPEEIPRLVAKADEGYDVVGTVRQNRQDSWFRKTASKMINRLIQRTTGKAMGDYGCMLRAYRRHIVDAMLHCHERSTFIPILANIFARRAIEIPVHHAEREFGESKYSFMHLINLMYDLVTCLTTTPLRMLSLLGSIIAIGGFSIAVLLVILRLTFGPQWAAEGVFMLFAVLFTFIGAQFIGMGLLGEYIGRIYTDVRARPRYFVQQVIRPSSKENE"
    other_feature = get_other_feature(sequence,"../examples/UNIPROT_E3XRD1.pssm")
    print(other_feature)





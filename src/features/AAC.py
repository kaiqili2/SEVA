#!/usr/bin/env python
#_*_coding:utf-8_*_

import re
from collections import Counter

import numpy as np


def AAC(sequence):
	AA = 'ACDEFGHIKLMNPQRSTVWY'

	count = Counter(sequence)

	total_length = len(sequence)

	AAC_feature= []
	for key in AA:
		count[key] = count[key] / total_length
		AAC_feature.append(count[key])

	AAC_feature	= np.array(AAC_feature)

	return AAC_feature

if __name__ == '__main__':
	sequence = "MEIALALKAVILGIVEGLTEFLPISSTGHLILAGQLLDFNDEKGKIFEIVIQFGAILAVCWEFRARIGKVVRGLRDDPLSQRFAANVVIASVPAIVLAFIFGKWIKAHLFNPISVALAFIVGGVVILLAEWRDARRGTVSHPQGNALLEAAKAGAPRIESVDDLNWRDALKVGLAQCFALVPGTSRSGATIIGGMLFGLSRQVATEFSFFLAIPVIFGATVYELYKARALLNGDDLGIFAVGFVFAFLSAFLCVRWLLRFVATHDFKPFAWYRIAFGIVVLLTAYTGLVSWHA"
	AAC_feature= AAC(sequence)
	print(AAC_feature)

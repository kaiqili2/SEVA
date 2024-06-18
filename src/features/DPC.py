#!/usr/bin/env python
#_*_coding:utf-8_*_

import re
import numpy as np

def DPC(sequence):
	AA = 'ACDEFGHIKLMNPQRSTVWY'
	diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i



	tmpCode = [0] * 400
	for j in range(len(sequence) - 2 + 1):
		tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j + 1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[
				sequence[j + 1]]] + 1
	if sum(tmpCode) != 0:
		tmpCode = [i / sum(tmpCode) for i in tmpCode]

	DPC_feature = np.array(tmpCode)

	return DPC_feature

if __name__ == '__main__':
	sequence = "MEIALALKAVILGIVEGLTEFLPISSTGHLILAGQLLDFNDEKGKIFEIVIQFGAILAVCWEFRARIGKVVRGLRDDPLSQRFAANVVIASVPAIVLAFIFGKWIKAHLFNPISVALAFIVGGVVILLAEWRDARRGTVSHPQGNALLEAAKAGAPRIESVDDLNWRDALKVGLAQCFALVPGTSRSGATIIGGMLFGLSRQVATEFSFFLAIPVIFGATVYELYKARALLNGDDLGIFAVGFVFAFLSAFLCVRWLLRFVATHDFKPFAWYRIAFGIVVLLTAYTGLVSWHA"
	DPC_feature= DPC(sequence)
	print(DPC_feature.shape)
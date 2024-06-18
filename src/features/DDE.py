#!/usr/bin/env python
#_*_coding:utf-8_*_

import re
import math
import numpy as np

def DDE(sequence):
	AA =  'ACDEFGHIKLMNPQRSTVWY'

	myCodons = {
		'A': 4,
		'C': 2,
		'D': 2,
		'E': 2,
		'F': 2,
		'G': 4,
		'H': 2,
		'I': 3,
		'K': 2,
		'L': 6,
		'M': 1,
		'N': 2,
		'P': 4,
		'Q': 2,
		'R': 6,
		'S': 6,
		'T': 4,
		'V': 4,
		'W': 1,
		'Y': 2
	}

	diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]

	myTM = []
	for pair in diPeptides:
		myTM.append((myCodons[pair[0]] / 61) * (myCodons[pair[1]] / 61))

	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i



	tmpCode = [0] * 400
	for j in range(len(sequence) - 2 + 1):
		tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
	if sum(tmpCode) != 0:
		tmpCode = [i/sum(tmpCode) for i in tmpCode]

	myTV = []
	for j in range(len(myTM)):
		myTV.append(myTM[j] * (1-myTM[j]) / (len(sequence) - 1))

	for j in range(len(tmpCode)):
		tmpCode[j] = (tmpCode[j] - myTM[j]) / math.sqrt(myTV[j])

	DDE_feature = np.array(tmpCode)
	return DDE_feature

if __name__ == '__main__':
	sequence = "MEIALALKAVILGIVEGLTEFLPISSTGHLILAGQLLDFNDEKGKIFEIVIQFGAILAVCWEFRARIGKVVRGLRDDPLSQRFAANVVIASVPAIVLAFIFGKWIKAHLFNPISVALAFIVGGVVILLAEWRDARRGTVSHPQGNALLEAAKAGAPRIESVDDLNWRDALKVGLAQCFALVPGTSRSGATIIGGMLFGLSRQVATEFSFFLAIPVIFGATVYELYKARALLNGDDLGIFAVGFVFAFLSAFLCVRWLLRFVATHDFKPFAWYRIAFGIVVLLTAYTGLVSWHA"
	DDE_feature= DDE(sequence)
	print(DDE_feature)
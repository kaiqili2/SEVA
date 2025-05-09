#!/usr/bin/env python
#_*_coding:utf-8_*_

import re, sys, os, platform
import math
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import numpy as np

USAGE = """
USAGE:
	python PAAC.py input.fasta <lambda> <output>

	input.fasta:      the input protein sequence file in fasta format.
	lambda:           the lambda value, integer, defaule: 30
	output:           the encoding file, default: 'encodings.tsv'
"""

def Rvalue(aa1, aa2, AADict, Matrix):
	return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def PAAC(sequence, lambdaValue=30, w=0.05, **kw):


	dataFile = pPath + "/PAAC.txt"
	with open(dataFile) as f:
		records = f.readlines()
	AA = ''.join(records[0].rstrip().split()[1:])
	AADict = {}
	for i in range(len(AA)):
		AADict[AA[i]] = i
	AAProperty = []
	AAPropertyNames = []
	for i in range(1, len(records)):
		array = records[i].rstrip().split() if records[i].rstrip() != '' else None
		AAProperty.append([float(j) for j in array[1:]])
		AAPropertyNames.append(array[0])

	AAProperty1 = []
	for i in AAProperty:
		meanI = sum(i) / 20
		fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
		AAProperty1.append([(j-meanI)/fenmu for j in i])
	encodings = []



	code = []
	theta = []
	for n in range(1, lambdaValue + 1):
		theta.append(
				sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
				len(sequence) - n))
	myDict = {}
	for aa in AA:
		myDict[aa] = sequence.count(aa)
	code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
	code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
	encodings.append(code)

	PAAC_feature = np.array(encodings).flatten()
	return PAAC_feature

if __name__ == '__main__':
	sequence = "MEIALALKAVILGIVEGLTEFLPISSTGHLILAGQLLDFNDEKGKIFEIVIQFGAILAVCWEFRARIGKVVRGLRDDPLSQRFAANVVIASVPAIVLAFIFGKWIKAHLFNPISVALAFIVGGVVILLAEWRDARRGTVSHPQGNALLEAAKAGAPRIESVDDLNWRDALKVGLAQCFALVPGTSRSGATIIGGMLFGLSRQVATEFSFFLAIPVIFGATVYELYKARALLNGDDLGIFAVGFVFAFLSAFLCVRWLLRFVATHDFKPFAWYRIAFGIVVLLTAYTGLVSWHA"
	PAAC_feature = PAAC(sequence)
	print(PAAC_feature)
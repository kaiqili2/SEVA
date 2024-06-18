#!/usr/bin/env python
#_*_coding:utf-8_*_
import numpy as np
import sys, os
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)

from features.AADPPSSMTools import *
def AADPPSSM(sequence, file):

	AA = 'ARNDCQEGHILKMFPSTWYV'

	encodings = []
	# header = ['#']
	# for p in range(1, len(fastas[0][1]) + 1):
	# 	for aa in AA:
	# 		header.append('Pos.'+str(p) + '.' + aa)
	# encodings.append(header)


	length=len(sequence)
	code = []

	with open(file) as f:
		records = f.readlines()[3: 3+length]
	proteinSeq = ''
	pssmMatrix = []
	for line in records:
		array = line.strip().split()
		pssmMatrix.append(array[1:42])
		proteinSeq = proteinSeq + array[1]

	pos = proteinSeq.find(sequence)
	if pos == -1:
		print('Warning: could not find the peptide in proteins.\n\n')
		temlist = ["0"] * 420
		code = code + temlist
		encodings.append(code)
	else:
		pair = []
		aadppssmMatrix=aadp_pssm(np.array(pssmMatrix))
		pair=pair+list(aadppssmMatrix[0])
		pair = [str(pair[i]) for i in range(len(pair))]
		code = code + pair
	encodings.append(code)

	AADPPSSM_feature = np.array(encodings).flatten()


	return AADPPSSM_feature

if __name__ == '__main__':
	sequence = "MFEIHPVKKVSVVIPVYNEQESLPELIRRTTAACESLGKEYEILLIDDGSSDNSAHMLVEASQAEGSHIVSILLNRNYGQHSAIMAGFSHVTGDLIITLDADLQNPPEEIPRLVAKADEGYDVVGTVRQNRQDSWFRKTASKMINRLIQRTTGKAMGDYGCMLRAYRRHIVDAMLHCHERSTFIPILANIFARRAIEIPVHHAEREFGESKYSFMHLINLMYDLVTCLTTTPLRMLSLLGSIIAIGGFSIAVLLVILRLTFGPQWAAEGVFMLFAVLFTFIGAQFIGMGLLGEYIGRIYTDVRARPRYFVQQVIRPSSKENE"
	AADPPSSM_feature = AADPPSSM(sequence,"../examples/UNIPROT_E3XRD1.pssm")
	print(AADPPSSM_feature)
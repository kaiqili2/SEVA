#!/usr/bin/env python
#_*_coding:utf-8_*_

import sys, os
pPath = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(pPath)
import numpy as np


def PSSMC(sequence, file):
	# if checkFasta.checkFasta(fastas) == False:
	# 	print('Error: for "PSSM" encoding, the input fasta sequences should be with equal length. \n\n')
	# 	return 0

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
		pssmMatrix.append(array[2:22])
		proteinSeq = proteinSeq + array[1]

	pos = proteinSeq.find(sequence)
	if pos == -1:
		print('Warning: could not find the peptide in proteins.\n\n')
		temlist = ["0"] * 400
		code = code + temlist
		encodings.append(code)

	else:
		pair=[]
		for aa in AA:
			temlist = [0] * 20
			for p in range(pos, pos + len(sequence)):
				aa2=proteinSeq[p]
				aa2feature=pssmMatrix[p]
				if aa2==aa:
					temlist= [temlist[i]+float(aa2feature[i]) for i in range(min(len(temlist),len(aa2feature)))]
			temlist=[(temlist[i])/length for i in range(len(temlist))]
			pair=pair+temlist
		pair=[str(pair[i]) for i in range(len(pair))]
		code = code+pair
	encodings.append(code)

	PSSMC_feature = np.array(encodings).flatten()
	return PSSMC_feature


if __name__ == '__main__':
	sequence = "MFEIHPVKKVSVVIPVYNEQESLPELIRRTTAACESLGKEYEILLIDDGSSDNSAHMLVEASQAEGSHIVSILLNRNYGQHSAIMAGFSHVTGDLIITLDADLQNPPEEIPRLVAKADEGYDVVGTVRQNRQDSWFRKTASKMINRLIQRTTGKAMGDYGCMLRAYRRHIVDAMLHCHERSTFIPILANIFARRAIEIPVHHAEREFGESKYSFMHLINLMYDLVTCLTTTPLRMLSLLGSIIAIGGFSIAVLLVILRLTFGPQWAAEGVFMLFAVLFTFIGAQFIGMGLLGEYIGRIYTDVRARPRYFVQQVIRPSSKENE"
	PSSMC_feature = PSSMC(sequence,"../examples/UNIPROT_E3XRD1.pssm")
	print(PSSMC_feature)
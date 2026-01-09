# SEVA 
SEVA is a novel deep learning-based framework aggregating structural and evolutionary features for virulence factors and antibiotic resistance genes prediction. 

The latest version will be updated as soon as possible.

![Image text](https://github.com/kaiqili2/SEVA/blob/main/overview.png)

# System Requirements
SEVA is developed under Linux environment with:

python 3.8.18

numpy 1.24.4

torch 2.1.0

biopython 1.81

cuda 12.3

# Software Requirements
To run SEVA, you have to install the following software:

[BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) for PSSM feature generation.

[AlphaFold2](https://github.com/google-deepmind/alphafold) or [ColabFold](https://github.com/sokrypton/ColabFold) for protein structure feature generation.

[esm](https://github.com/facebookresearch/esm) for MSA embedding generation.

# Model Variants: Choose What Fits Your Project
We provide two prediction modes to balance speed and accuracy:
| model	| SEVA (Full Model)	| SEVA-Fast (Fast Mode) |
| ------ | ------ | ------ |
| Features |	Includes statistical features, PSSM, MSA search, and structural prediction.	| Omits intensive MSA/structural steps; uses statistical features only. |
| Best For	| Maximum accuracy on challenging, low-identity sequences.	| Rapid screening of large datasets in time-sensitive contexts. |
| Speed	| ~1000 seconds per sequence. |	~0.36 seconds per sequence. |
| Accuracy |	97.13% on the test dataset |91.23% on the test dataset. |


# Quick Start

# Run SEVA-Fast for rapid screening
The model checkpoint can be downloaded from [here](https://drive.google.com/file/d/1-hXc_dObTe8b8IfKeP3vyiAt5QGwwl58/view?usp=drive_link), and put them in /src folder.
```
cd src/
python seva_fast_predict.py --input your_sequences.fasta --model_file SEVA_fast.pt
```

# Run complete SEVA for detailed analysis
To run SEVA for VF and ARG prediction, it requires MSA file, PDB file, and PSSM file. These features files should pre-generated. All these files can be generated from the previously mentioned software. The model checkpoint can be downloaded from [here](https://drive.google.com/file/d/1-hXc_dObTe8b8IfKeP3vyiAt5QGwwl58/view?usp=drive_link), and put them in /src folder.

We provide an example to run SEVA_full prediction.
```
cd src/
python prediction.py --msa_file ../examples/UNIPROT_E3XRD1.a3m --pdb_file_1 ../examples/UNIPROT_E3XRD1_1.pdb --pdb_file_2 ../examples/UNIPROT_E3XRD1_2.pdb --pdb_file_3 ../examples/UNIPROT_E3XRD1_3.pdb --pdb_file_4 ../examples/UNIPROT_E3XRD1_4.pdb --pdb_file_5 ../examples/UNIPROT_E3XRD1_5.pdb --pssm_file ../examples/UNIPROT_E3XRD1.pssm --model_file SEVA.pt
```


# License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

# Contacts
If you have any questions or comments, please feel free to email: kaiqili2-c@my.cityu.edu.hk.

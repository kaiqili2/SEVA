# ARGVF
SEVA is a novel deep learning-based framework aggregating structural and evolutionary features for virulence factors and antibiotic resistance genes prediction.

![Image text](https://github.com/kaiqili2/SEVA/blob/main/framework.PNG)

# System Requirments
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


# Train SEVA
An easy way to train SEVA, the training dataset can be downloaded from [Google Drive](https://drive.google.com/file/d/1-hEIfi09xz-pLhVwCXhfmQNoBZUv8KFv/view?usp=drive_link), and unzip them to /data folder.
```
cd src/

python train_seva.py
```
The results of the first 10 epochs are shown below:
```
Epoch 0| Loss: 0.4799| Train accuracy: 0.8007| Validation accuracy: 0.8733

Epoch 1| Loss: 0.2602| Train accuracy: 0.9022| Validation accuracy: 0.8983

Epoch 2| Loss: 0.2097| Train accuracy: 0.9265| Validation accuracy: 0.9067

Epoch 3| Loss: 0.1868| Train accuracy: 0.9315| Validation accuracy: 0.9233

Epoch 4| Loss: 0.1459| Train accuracy: 0.9465| Validation accuracy: 0.9033

Epoch 5| Loss: 0.1230| Train accuracy: 0.9559| Validation accuracy: 0.9100

Epoch 6| Loss: 0.1191| Train accuracy: 0.9591| Validation accuracy: 0.9267

Epoch 7| Loss: 0.1037| Train accuracy: 0.9631| Validation accuracy: 0.9433

Epoch 8| Loss: 0.1059| Train accuracy: 0.9635| Validation accuracy: 0.8983

Epoch 9| Loss: 0.0949| Train accuracy: 0.9674| Validation accuracy: 0.9417

```

# Test SEVA
We can also test SEVA with the test dataset downloaded from [Google Drive](https://drive.google.com/file/d/1-hEIfi09xz-pLhVwCXhfmQNoBZUv8KFv/view?usp=drive_link) and unzip them to /data folder. The model_checkpoint in [here](https://drive.google.com/file/d/1-hXc_dObTe8b8IfKeP3vyiAt5QGwwl58/view?usp=drive_link), and put them in /src folder.
```
cd src/

python test_seva.py
```
The result is shown below:
```
The Test Dataset Accuracy: 0.9649

The Confusion Matrix:

[209   0   0]

[  0 195  14]

[  0   8 201]


```
# Run SEVA for VF and ARG prediction with your own data.
To run SEVA for VF and ARG prediction, it requires MSA file, PDB file, and PSSM file. All these files can be generated from the previously mentioned software. The model checkpoint can be downloaded from [here](https://drive.google.com/file/d/1-hXc_dObTe8b8IfKeP3vyiAt5QGwwl58/view?usp=drive_link), and put them in /src folder.

We provide an example to run SEVA prediction.
```
cd src/

python prediction.py --msa_file ../examples/UNIPROT_E3XRD1.a3m --pdb_file_1 ../examples/UNIPROT_E3XRD1_1.pdb --pdb_file_2 ../examples/UNIPROT_E3XRD1_2.pdb --pdb_file_3 ../examples/UNIPROT_E3XRD1_3.pdb --pdb_file_4 ../examples/UNIPROT_E3XRD1_4.pdb --pdb_file_5 ../examples/UNIPROT_E3XRD1_5.pdb --pssm_file ../examples/UNIPROT_E3XRD1.pssm --model_file SEVA.pt
```
# License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

# Contacts
If you have any questions or comments, please feel free to email: kaiqili2-c@my.cityu.edu.hk.

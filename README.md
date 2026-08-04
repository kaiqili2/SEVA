# SEVA 
SEVA is a deep learning framework for predicting virulence factors (VFs), antibiotic resistance genes (ARGs), and negative samples (NSs) from protein sequences.

The full SEVA model integrates complementary sequence, evolutionary, and structure-aware features, including statistical sequence descriptors, position-specific scoring matrices (PSSMs), multiple sequence alignment (MSA) embeddings, and protein structure-derived distance maps. SEVA-Fast is a lightweight version that uses statistical sequence features only for rapid preliminary screening.

![Image text](https://github.com/kaiqili2/SEVA/blob/main/Overview.png)

# Model Variants: Choose What Fits Your Project
We provide two prediction modes to balance speed and accuracy:
| model	| SEVA (Full Model)	| SEVA-Fast (Fast Mode) |
| ------ | ------ | ------ |
| Features |	Includes statistical features, PSSM, MSA search, and structural prediction.	| Omits intensive MSA/structural steps; uses statistical features only. |
| Best For	| Maximum accuracy.	| Rapid screening of large datasets. |
| Speed	| ~1070 seconds per sequence. |	~0.36 seconds per sequence. |
| Accuracy |	97.13% on the test dataset |91.23% on the test dataset. |

## Repository layout

```text
SEVA_3/
|-- data/                       # Training/test data 
|-- examples/                   # Example A3M, PDB, and PSSM files
|-- src/
|   |-- SEVA.pt                 # Full-model checkpoint
|   |-- SEVA_fast.pt            # SEVA-Fast checkpoint
|   |-- seva_predict.py         # Full-model single-protein prediction
|   |-- seva_fast_predict.py    # SEVA-Fast single-protein prediction
|   |-- train_seva.py           # Full-model training
|   |-- test_seva.py            # Full-model evaluation
|   `-- features/               # Feature-extraction modules
`-- README.md
```

## Requirements

SEVA was developed and tested on Linux with the following versions:

| Component | Version |
| --- | --- |
| Python | 3.8.18 |
| NumPy | 1.24.4 |
| PyTorch | 2.1.0 |
| Biopython | 1.81 |
| CUDA | 12.3 |

Install a Python environment first. Install the PyTorch build appropriate for your operating system and CUDA driver from the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the remaining packages:

```bash
conda create -n seva python=3.8.18 -y
conda activate seva

pip install numpy==1.24.4 biopython==1.81
pip install fair-esm einops torchvision matplotlib
```

The full model additionally requires the following external software to prepare its inputs:

- [BLAST+](https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/) to generate PSSM files.
- [AlphaFold2](https://github.com/google-deepmind/alphafold) or [ColabFold](https://github.com/sokrypton/ColabFold) to generate predicted protein structures.
- [ESM](https://github.com/facebookresearch/esm) for MSA embedding extraction (`fair-esm` provides the Python package used by this repository).

> The prediction scripts consume already generated A3M, PSSM, and PDB files. They do not automatically run BLAST, AlphaFold2, or ColabFold.

## Input requirements

Use uppercase, single-letter amino-acid sequences. For the current statistical feature implementation:

- sequences must be longer than 30 residues because PAAC uses `lambdaValue=30`;
- standard residues `ACDEFGHIKLMNPQRSTVWY` are expected;
- remove or handle non-standard symbols such as `X`, `B`, `Z`, `U`, `O`, and `*` before prediction.

For the full model, the first sequence in the A3M file is the query sequence. Its PSSM and all five PDB inputs must describe that same query protein. Inputs longer than 1000 residues are truncated to 1000 residues by `seva_predict.py`.

# Quick Start

# Run SEVA-Fast for rapid screening
SEVA-Fast only needs the amino-acid sequence and the included `SEVA_fast.pt` checkpoint. Run it from `src/`:

```bash
cd src

python seva_fast_predict.py \
  --sequence "MFEIHPVKKVSVVIPVYNEQESLPELIRRTTAACESLGKEYEILLIDDGSSDNSAHMLVEASQAEGSHIVSILLNRNYGQHSAIMAGFSHVTGDLIITLDADLQNPPEEIPRLVAKADEGYDVVGTVRQNRQDSWFRKTASKMINRLIQRTTGKAMGDYGCMLRAYRRHIVDAMLHCHERSTFIPILANIFARRAIEIPVHHAEREFGESKYSFMHLINLMYDLVTCLTTTPLRMLSLLGSIIAIGGFSIAVLLVILRLTFGPQWAAEGVFMLFAVLFTFIGAQFIGMGLLGEYIGRIYTDVRARPRYFVQQVIRPSSKENE" \
  --model_file SEVA_fast.pt
```
SEVA-Fast predicts whether each query protein is most consistent with an ARG, VF, or NS class using statistical sequence features.

# Run complete SEVA for detailed analysis
Full SEVA integrates statistical sequence features, PSSM features, MSA embeddings, and structure-derived distance maps.
The full model requires one A3M file, one PSSM file, and five structure files for a single query protein. Example inputs are provided in `examples/`.

We provide an example to run SEVA_full prediction.
```
cd src/
python seva_predict.py --msa_file ../examples/UNIPROT_E3XRD1.a3m --pdb_file_1 ../examples/UNIPROT_E3XRD1_1.pdb --pdb_file_2 ../examples/UNIPROT_E3XRD1_2.pdb --pdb_file_3 ../examples/UNIPROT_E3XRD1_3.pdb --pdb_file_4 ../examples/UNIPROT_E3XRD1_4.pdb --pdb_file_5 ../examples/UNIPROT_E3XRD1_5.pdb --pssm_file ../examples/UNIPROT_E3XRD1.pssm --model_file SEVA.pt
```

### Full-model parameters

| Parameter | Description |
| --- | --- |
| `--msa_file` | A3M multiple-sequence alignment; the first sequence is treated as the query. |
| `--pdb_file_1` ... `--pdb_file_5` | Five PDB structure predictions for the query protein. |
| `--pssm_file` | PSSM generated for the same query protein. |
| `--model_file` | Full-model checkpoint, normally `SEVA.pt`. |

# License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

# Contacts
If you have any questions or comments, please feel free to email: kaiqili2-c@my.cityu.edu.hk.

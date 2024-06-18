# -*- coding: utf-8 -*-


import numpy as np
import os, datetime, argparse
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split, TensorDataset
from dataloader import ProteinDataset


def test(loader, model_name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    model = torch.load(model_name)
    model.eval()

    pre_final = []
    label_final = []

    for seq, name, msa_feature, pbd_feature, pad_mask, other_feature, label in loader:
        with torch.no_grad():
            msa_feature = msa_feature.to(device)
            pbd_feature = pbd_feature.to(device)
            pad_mask = pad_mask.to(device)
            other_feature = other_feature.to(device)
            label = label.to(device)

            pred, attn = model(msa_feature, pbd_feature, pad_mask, other_feature)
            pred = pred.argmax(dim=1)

            pred = pred.cpu().numpy()
            label = label.cpu().numpy()

            pre_final.append(pred)
            label_final.append(label)

    pre_final = np.array(pre_final)
    label_final = np.array(label_final)

    mask = (label_final >= 0) & (label_final < 3)
    confusion_matrix = np.bincount(
        3 * label_final[mask].astype(int) +
        pre_final[mask], minlength=3 ** 2).reshape(3, 3)

    total_sum = np.sum(confusion_matrix)
    main_diagonal_sum = np.sum(np.diag(confusion_matrix))

    accuracy = main_diagonal_sum/total_sum

    return accuracy, confusion_matrix



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_file", type=str, default="../data/Uniprot_test.fasta", help="fasta file with all test data")
    parser.add_argument("--label_file", type=str, default="../data/label_test.txt", help="label of all test data")
    parser.add_argument("--msa_path", type=str, default="../data/msa_feature/", help="path of msa feature")
    parser.add_argument("--pdb_path", type=str, default="../data/pdb_feature/", help="path of pdb feature")
    parser.add_argument("--other_feature_path", type=str, default="../data/other_feature/", help="path of other feature path")
    parser.add_argument("--model_file", type=str, default="SEVA.pt", help="model file")
    args = parser.parse_args()

    test_data = ProteinDataset(fasta_file=args.fasta_file, label_file=args.label_file,
                               msa_path=args.msa_path, pdb_path=args.pdb_path,
                               feature_path=args.other_feature_path)

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
    accuracy, confusion_matrix = test(test_dataloader, args.model_file)
    print("The Test Dataset Accuracy: {:.4f}" .format(accuracy))
    print("The Confusion_Matrix:")
    print(confusion_matrix)


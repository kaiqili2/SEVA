# -*- coding: utf-8 -*-

import os, datetime, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, random_split, TensorDataset
import torch.optim as optim
import random
from collections import Counter
from model_seva import SEVA
from dataloader import ProteinDataset


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def train(fasta_file="../data/Uniprot_train.fasta",
          label_file="../data/label_train.txt",
          msa_path="../data/msa_feature/",
          pdb_path="../data/pdb_feature/",
          feature_path="../data/other_feature/",
          learning_rate=5e-5,
          batch_size=20,
          epoch_n=100,
          random_seed=2023,
          val_split=0.1,
          model_name="SEVA.pt",
          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          ):

    setup_seed(random_seed)

    data = ProteinDataset(fasta_file=fasta_file,
                          label_file=label_file,
                          msa_path=msa_path,
                          pdb_path=pdb_path,
                          feature_path=feature_path)

    train_data, val_data = random_split(data, [len(data) - int(len(data) * val_split), int(len(data) * val_split)])
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=10)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=10)

    # build model
    model = SEVA(in_channels=768,
                 emb_size=768,
                 depth=6,
                 num_heads=6,
                 dropout=0.2,
                 attention_dropout=0.1,
                 ffn_expansion=4,
                 n_classes=3,
                 other_feature_dim=2154
              ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    # train
    old_val_acc = 0
    for epoch in range(epoch_n):
        training_running_loss = 0.0
        train_acc = 0.0

        model.train()
        for i, (seq, name, msa_feature, pdb_feature, pad_mask, other_feature, label) in enumerate(train_dataloader):
            msa_feature = msa_feature.to(device)
            pdb_feature = pdb_feature.to(device)
            pad_mask = pad_mask.to(device)
            other_feature = other_feature.to(device)


            label = label.to(device)

            # forward + backprop + loss
            pred, attn = model(msa_feature, pdb_feature, pad_mask, other_feature)

            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()

            # update model params
            optimizer.step()

            training_running_loss += loss.detach().item()
            train_acc += (torch.argmax(pred, 1).flatten() == label).type(torch.float).mean().item()

        # val accuracy
        val_acc = evaluation(val_dataloader, model, device)

        if val_acc > old_val_acc:
            torch.save(model, model_name)
            old_val_acc = val_acc


        print("Epoch {}| Loss: {:.4f}| Train accuracy: {:.4f}| Validation accuracy: {:.4f}".format(epoch, training_running_loss / (i + 1), train_acc / (i + 1), val_acc))


    return model


def evaluation(loader, model, device):
    model.eval()
    correct = 0

    for seq, name, msa_feature, pbd_feature, pad_mask, other_feature, label in loader:
        with torch.no_grad():
            msa_feature = msa_feature.to(device)
            pbd_feature = pbd_feature.to(device)
            pad_mask = pad_mask.to(device)
            other_feature = other_feature.to(device)
            label = label.to(device)


            pred, attn = model(msa_feature, pbd_feature, pad_mask, other_feature)
            pred = pred.argmax(dim=1)

        correct += pred.eq(label).sum().item()

    total = len(loader.dataset)
    acc = correct / total

    return acc



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta_file", type=str, default="../data/Uniprot_train.fasta", help="fasta file with all test data")
    parser.add_argument("--label_file", type=str, default="../data/label_train.txt", help="label of all test data")
    parser.add_argument("--msa_path", type=str, default="../data/msa_feature/", help="path of msa feature")
    parser.add_argument("--pdb_path", type=str, default="../data/pdb_feature/", help="path of pdb feature")
    parser.add_argument("--other_feature_path", type=str, default="../data/other_feature/", help="path of other feature path")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="learning rate for model training")
    parser.add_argument("--batch_size", type=int, default=20, help="batch size for model training")
    parser.add_argument("--number_epoch", type=int, default=100, help="epochs of model training")
    parser.add_argument("--random_seed", type=int, default=2023, help="random seed for model training")
    parser.add_argument("--model_name", type=str, default="SEVA.pt", help="model name")
    args = parser.parse_args()

    train(fasta_file=args.fasta_file,
          label_file=args.label_file,
          msa_path=args.msa_path,
          pdb_path=args.pdb_path,
          feature_path=args.other_feature_path,
          learning_rate=args.learning_rate,
          batch_size=args.batch_size,
          epoch_n=args.number_epoch,
          random_seed=args.random_seed,
          val_split=0.1,
          model_name=args.model_name,
          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          )


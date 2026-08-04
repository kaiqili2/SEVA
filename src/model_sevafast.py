import torch
import torch.nn as nn
import math
from typing import Optional
from typing import Sequence, Tuple, List, Union
import torch.nn.functional as F
import itertools
import argparse
import esm
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce




class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int = 768, n_classes: int = 3, dropout: float = 0.2):
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes
        self.dropout = dropout


        self.classify = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )


    def forward(self, x):

        x = self.classify(x)

        return x



class SEVA(nn.Module):
    def __init__(self,
                 in_channels: int = 934,
                 n_classes: int = 3,
                 dropout: float = 0.2

                 ):
        super().__init__()



        self.classification_layer = ClassificationHead(in_channels, n_classes, dropout)

    def forward(self, x, feature=None):
        # x = [batch size, seq_len, emb_dim] dist_map = [batch size, dim, seq_len, seq_len] mask = [batch size, seq_len]

        x = self.classification_layer(x)


        return x



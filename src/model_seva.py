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


class SelfAttention(nn.Module):
    """Compute self-attention """

    def __init__(
            self,
            in_channels,
            embed_dim,
            num_heads,
            dropout=0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(in_channels, embed_dim)
        self.v_proj = nn.Linear(in_channels, embed_dim)
        self.q_proj = nn.Linear(in_channels, embed_dim)
        self.conv = nn.Conv2d(5, num_heads, 1, bias=False)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_module = nn.Dropout(dropout)


    def forward(
            self,
            x,
            dist_map=None,
            mask=None,

    ):
        batch_size, seq_len, embedding_dim = x.size()

        q = self.q_proj(x).view(batch_size, self.num_heads, seq_len, self.head_dim)
        k = self.k_proj(x).view(batch_size, self.num_heads, seq_len, self.head_dim)
        v = self.v_proj(x).view(batch_size, self.num_heads, seq_len, self.head_dim)

        attn_weights = torch.einsum('bhqd, bhkd -> bhqk', q, k)


        if mask is not None:
            mask = mask.bool()
            # mask [batch size, seq_len]
            padding_mask = mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
            attn_weights = attn_weights.masked_fill(
                padding_mask,
                -10000,
            )
        attn_weights = (attn_weights * self.scaling).softmax(-1)

        #print("att_weights_o", attn_weights)



        # if dist_map is not None:
        #     batch_size, dist_dim, seq_len, seq_len = dist_map.size()
        #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #
        #
        #     dist_map = self.conv(dist_map)
        #
        #
        #
        #     if mask is not None:
        #         mask = mask.bool()
        #         # mask [batch size, seq_len]
        #         padding_mask = mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
        #         dist_map = dist_map.masked_fill(
        #             padding_mask,
        #             -10000,
        #         )
        #
        #     dist_map = dist_map.softmax(-1)
        #     print("dis_map_s", dist_map)
        #
        #     attn_weights = attn_weights + dist_map
        #
        #
        #     if mask is not None:
        #         mask = mask.bool()
        #         # mask [batch size, seq_len]
        #         padding_mask = mask.view(batch_size, 1, 1, seq_len).expand(-1, self.num_heads, -1, -1)
        #         attn_weights = attn_weights.masked_fill(
        #             padding_mask,
        #             -10000,
        #         )
        #     attn_weights = attn_weights.softmax(-1)

        if dist_map is not None:

            #print("dis_map_s", dist_map)

            attn_weights = attn_weights + dist_map

            attn_weights = attn_weights/2.

        # print(attn_weights)
        # print(attn_weights.size())



        attn_probs = self.dropout_module(attn_weights)
        context = torch.einsum("bhal,bhlv->bhav", attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(context)


        return output, attn_probs


class ResidualBlock(nn.Module):
    def __init__(
            self,
            layer: nn.Module,
            embedding_dim: int,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.layer = layer
        self.dropout_module = nn.Dropout(
            dropout,
        )
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, x, *args, **kwargs):
        residual = x
        x = self.layer_norm(x)
        outputs = self.layer(x, *args, **kwargs)
        if isinstance(outputs, tuple):
            x, *out = outputs
        else:
            x = outputs
            out = None

        x = self.dropout_module(x)
        x = residual + x

        if out is not None:
            return (x,) + tuple(out)
        else:
            return x


class FeedForwardNetwork(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            expansion: int,
            activation_dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.expansion = expansion
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(
            activation_dropout,
        )
        self.fc1 = nn.Linear(embedding_dim, embedding_dim * expansion)
        self.fc2 = nn.Linear(embedding_dim * expansion, embedding_dim)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 in_channels: int = 768,
                 emb_size: int = 768,
                 num_heads: int = 8,
                 attention_dropout: float = 0.0,
                 drop_res: float = 0.1,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.1,
                 **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.drop_res = drop_res
        self.forward_expansion = forward_expansion
        self.forward_drop_p = forward_drop_p

        attention_layer = SelfAttention(in_channels, emb_size, num_heads, attention_dropout)
        ffn_layer = FeedForwardNetwork(emb_size, expansion=forward_expansion, activation_dropout=forward_drop_p)

        self.attention_layer = self.build_residual(attention_layer)
        self.ffn_layer = self.build_residual(ffn_layer)

    def build_residual(self, layer: nn.Module):
        return ResidualBlock(
            layer,
            self.emb_size,
            self.drop_res,
        )

    def forward(
            self,
            x,
            dist_map=None,
            mask=None,

    ):
        x, attn = self.attention_layer(x, dist_map, mask)
        x = self.ffn_layer(x)

        return x, attn


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 3, other_feature_dim: int = 0):
        super().__init__()

        self.emb_size = emb_size
        self.n_classes = n_classes
        self.other_feature_dim = other_feature_dim

        self.lm = nn.LayerNorm(emb_size)
        self.classify = nn.Sequential(
            nn.Linear(emb_size, 512),
            nn.Linear(512, 256),
            nn.Linear(256, n_classes)
        )
        self.classify_other = nn.Sequential(
            nn.Linear(emb_size + other_feature_dim,512),
            nn.Linear(512, 256),
            nn.Linear(256, n_classes)
        )


    def forward(self, x, mask, other_feature):

        if mask is not None:
            x = x.sum(1)/((1-mask).sum(1).unsqueeze(-1))
        else:
            x = x.mean(1)

        x = self.lm(x)

        if other_feature is not None:

            x = torch.cat((x, other_feature), dim=1)
            x = self.classify_other(x)
        else:
            x = self.classify(x)

        return x



class SEVA(nn.Module):
    def __init__(self,
                 in_channels: int = 768,
                 emb_size: int = 768,
                 depth: int = 6,
                 num_heads: int = 6,
                 dropout: float = 0.2,
                 attention_dropout: float = 0.1,
                 ffn_expansion: int = 4,
                 n_classes: int = 3,
                 other_feature_dim: int = 0,

                 ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    in_channels,
                    emb_size,
                    num_heads,
                    attention_dropout,
                    dropout,
                    ffn_expansion,
                    dropout,
                )
                for _ in range(depth)
            ]
        )

        self.classification_layer = ClassificationHead(emb_size, n_classes, other_feature_dim)

    def forward(self, x, dist_map=None, mask=None, other_feature=None):
        # x = [batch size, seq_len, emb_dim] dist_map = [batch size, dim, seq_len, seq_len] mask = [batch size, seq_len]
        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(x, dist_map, mask)

        x = self.classification_layer(x, mask, other_feature)


        return x, attn


if __name__ == '__main__':
    x = torch.randn((20,1000,768))
    dist_map = torch.randn((20,6,1000,1000))
    mask = torch.randint(0,2,[20,1000])
    other_feature = torch.randn((20,1000))
    #print(mask)
    model = SEVA(other_feature_dim=1000)
    y, attn = model(x,dist_map,mask,other_feature)
    print(model)
    print(y)



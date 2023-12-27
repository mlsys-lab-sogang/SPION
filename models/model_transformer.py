
import torch
import torch.nn as nn
import numpy as np
import math
import ctypes
from ctypes import *
from torch.utils.checkpoint import checkpoint

from models.attention import Attention

from config import Config


mask_task = ['lra-listops', 'lra-text', 'lra-news', 'lra-yelp', 'lra-text1',  'lra-news1', 'lra-yelp1', 'lra-text2',  'lra-news2', 'lra-yelp2', 'lra-retrieval']

class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config["embedding_dim"] == config["transformer_dim"]

        self.dim = config["embedding_dim"]

        self.word_embeddings = nn.Embedding(config["vocab_size"], config["embedding_dim"])
        torch.nn.init.normal_(self.word_embeddings.weight, std = 0.02)

        self.position_embeddings = nn.Embedding(config["max_seq_len"], config["embedding_dim"])
        torch.nn.init.normal_(self.position_embeddings.weight, std = 0.02)

        self.dropout = torch.nn.Dropout(p = config["dropout_prob"])

    def fixed_pos_emb(self, seq_len, device):
        position = torch.arange(0, seq_len, device = device)[:, np.newaxis]
        div_term = torch.exp(torch.arange(0, self.dim, 2, device = device) * -(math.log(10000.0) / self.dim))
        pos_embed = torch.stack([torch.sin(position * div_term), torch.cos(position * div_term)], -1).reshape(seq_len, -1)
        return pos_embed

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()

        X_token = self.word_embeddings(input_ids)

        position_ids = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)[None, :].repeat(batch_size, 1)
        X_pos = self.position_embeddings(position_ids)

        X = X_token + X_pos

        X = self.dropout(X)

        return X

    
class TransformerLayer(nn.Module):
    def __init__(self, config, inference=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(config["transformer_dim"])

        self.mha = Attention(config, inference)

        self.dropout1 = torch.nn.Dropout(p = config["dropout_prob"])
        self.norm2 = nn.LayerNorm(config["transformer_dim"])

        self.mlpblock = nn.Sequential(
                    nn.Linear(config["transformer_dim"], config["transformer_hidden_dim"]),
                    nn.GELU(),
                    torch.nn.Dropout(p = config["dropout_prob"]),
                    nn.Linear(config["transformer_hidden_dim"], config["transformer_dim"]),
                    torch.nn.Dropout(p = config["dropout_prob"])
        )

        self.inference = inference
        
    def forward(self, X, mask, mat=None):
        
        if self.inference:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        out, attn = self.mha(self.norm1(X), mask, mat)

        X = self.dropout1(out) + X

        if self.inference:
            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end)
            start.record()
            print(f"attn: {time} \n")

        X = self.mlpblock(self.norm2(X)) + X

        if self.inference:
            end.record()
            torch.cuda.synchronize()
            time = start.elapsed_time(end)
            print(f"ff: {time} \n")

        return X, attn
    
class Model(nn.Module):
    def __init__(self, config, inference=False):
        super().__init__()

        self.num_layers = config["num_layers"]
        self.tied_weights = config["tied_weights"]
        self.inference = inference

        self.embeddings = Embeddings(config)
        
        if self.tied_weights:
            self.transformer = TransformerLayer(config)
        else:
            for idx in range(self.num_layers):
                setattr(self, f"transformer_{idx}", TransformerLayer(config, inference))
                #setattr(self, f"transformer_{idx}", TransformerLayer(config, block_size=block_size))

        self.norm = nn.LayerNorm(config["transformer_dim"])

    
    def forward(self, input_ids, mask = None, mat_lst=[], is_attn=False):

        X = self.embeddings(input_ids)
        if mask is None:
            mask = torch.ones_like(input_ids)

        attn_lst = []
        attn_time = 0
        for idx in range(self.num_layers):
            torch.cuda.nvtx.range_push(f"transformer Layer{idx}")
            if len(mat_lst) == 0:
                X, attn = getattr(self, f"transformer_{idx}")(X, mask)
                if is_attn:
                    attn_lst.append(torch.mean(torch.mean(attn,dim=0),dim=0))
            else:
                X, attn = getattr(self, f"transformer_{idx}")(X, mask, mat_lst[0][idx])
                if self.inference:
                    attn_lst.append(attn)

        X = self.norm(X) * mask[:, :, None]

        return X, attn_lst

    
    
    

import torch
import torch.nn as nn
import math
import json
from torch.utils.checkpoint import checkpoint
import ctypes
from ctypes import *


mask_task = ['lra-listops', 'lra-text',  'lra-news', 'lra-yelp', 'lra-text1',  'lra-news1', 'lra-yelp1', 'lra-text2',  'lra-news2', 'lra-yelp2', 'lra-retrieval']

class SoftmaxAttention(nn.Module):
    def __init__(self, config,inference):
        super().__init__()
        self.drop_attn = torch.nn.Dropout(p = config["attention_dropout"])
        self.head_dim = config["head_dim"]
        self.inference = inference

    def forward(self, Q, K, V, mask):
        if self.inference:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        # input [batch_size, nb_heads, seq_len, dim_head]
        # print('Q', Q.abs().median()) # check scale
        # print('K', K.abs().median())
        dot = torch.matmul(Q, torch.transpose(K, -2, -1))
        dot = dot / math.sqrt(self.head_dim)
        dot = dot - 1e6 * (1 - mask[:, None, None, :])

        attn = nn.functional.softmax(dot, dim = -1)
        attn = self.drop_attn(attn)

        out = torch.matmul(attn, V)
        #print(X)
        # output [batch_size, nb_heads, seq_len, dim_head]
        return out, attn
    
class Attention(nn.Module):
    def __init__(self, config, inference=False):
        super().__init__()
        self.inference = inference

        self.dim = config["transformer_dim"] # input_dim
        self.head_dim = config["head_dim"]
        self.num_head = config["num_head"]
        self.seq_len = config["max_seq_len"]
        self.block_size = config["block_size"]
        self.num_blocks = int(self.seq_len/self.block_size)

        self.W_q = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_k = nn.Linear(self.dim, self.num_head * self.head_dim)
        self.W_v = nn.Linear(self.dim, self.num_head * self.head_dim)

        if not inference:
            if config['task'] == "lra-pathfinder32-curv_contour_length_14":
                self.attn = SoftmaxAttention(config)
            else:
                if config['task'] in mask_task:
                    attn_cpp = Attention._cuda_mask_compile_module()
                    attn_handle = attn_cpp.init(config["head_dim"], config["max_seq_len"], config["num_head"]*config["batch_size"])
                else:
                    attn_cpp = Attention._cuda_compile_module()
                    attn_handle = attn_cpp.init(config["head_dim"], config["max_seq_len"], config["num_head"]*config["batch_size"])
                    
                if config['task'] in mask_task:
                    from models.attention_cuda_mask import CUDAMaskAttention
                    self.attn = CUDAMaskAttention(config, attn_cpp, attn_handle)
                else:
                    from models.attention_cuda import CUDAAttention
                    self.attn = CUDAAttention(config, attn_cpp, attn_handle)

        #self.attn = SoftmaxAttention(config)
        
        self.ff = nn.Linear(self.num_head * self.head_dim, self.dim)
        

    def forward(self, X, mask, mat=None):
        """if self.inference:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()"""
        Q = self.split_heads(self.W_q(X))
        K = self.split_heads(self.W_k(X))
        V = self.split_heads(self.W_v(X))

        with torch.cuda.amp.autocast(enabled = False):
            if mat == None:
                attn_out, attn = self.attn(Q.float(), K.float(), V.float(), mask.float())
            else:
                attn_out, attn = self.block_attn(Q.float(), K.float(), V.float(), mask.float(), mat)
            torch.cuda.nvtx.range_pop()
        attn_out = self.combine_heads(attn_out)
        torch.cuda.nvtx.range_push(f"Feed Forward")
        out = self.ff(attn_out)

        """if self.inference:
            end.record()
            torch.cuda.synchronize()
            attn = (start.elapsed_time(end))"""

        return out, attn

    def reconstruct_for_blocksparse(self, config):

        emb_dim = config["head_dim"]
        num_heads = config["num_head"]
        seq_len = config["max_seq_len"]
        batch_size = config["batch_size"]
        block_size = config["block_size"]
        num_batches = batch_size * num_heads 

        if config['task'] in mask_task:
            block_attn_cpp = Attention._blockcuda_mask_compile_module()
            block_attn_handle = block_attn_cpp.init(config["head_dim"], config["max_seq_len"], int(config["num_head"]*config["batch_size"]), block_size)
        else:
            block_attn_cpp = Attention._blockcuda_compile_module()
            block_attn_handle = block_attn_cpp.init(config["head_dim"], config["max_seq_len"], int(config["num_head"]*config["batch_size"]), block_size)
        
        if config['task'] in mask_task:
            from models.attention_blockcuda_mask import CUDABlockMaskAttention
            self.block_attn = CUDABlockMaskAttention(config, block_attn_cpp, block_attn_handle)
        else:
            from models.attention_blockcuda import CUDABlockAttention
            self.block_attn = CUDABlockAttention(config, block_attn_cpp, block_attn_handle)

        model_dev = next(self.parameters()).device
        self.block_attn.to(model_dev)

    def combine_heads(self, X):
        X = X.transpose(1, 2)
        X = X.reshape(X.size(0), X.size(1), self.num_head * self.head_dim)
        return X

    def split_heads(self, X):
        X = X.reshape(X.size(0), X.size(1), self.num_head, self.head_dim)
        X = X.transpose(1, 2)
        return X

    @staticmethod
    def _cuda_compile_module():
        print("attn compile")
        attn_cpp = ctypes.CDLL('./models/attn.so')
        attn_cpp.init.argtypes = [c_int, c_int, c_int]
        attn_cpp.init.restype = ctypes.c_void_p
        attn_cpp.attn_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        attn_cpp.attn_backward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        attn_cpp.destroy.argtypes = [ctypes.c_void_p]
        return attn_cpp
    
    @staticmethod
    def _blockcuda_compile_module():
        print("block_attn compile")
        attn_cpp = ctypes.CDLL('./models/block_attn.so')
        attn_cpp.init.argtypes = [c_int, c_int, c_int, c_int]
        attn_cpp.init.restype = ctypes.c_void_p
        attn_cpp.attn_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                          ctypes.c_void_p, ctypes.c_void_p, c_int]
        attn_cpp.attn_backward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          c_int]
        attn_cpp.destroy.argtypes = [ctypes.c_void_p]
        return attn_cpp
    
    @staticmethod
    def _cuda_mask_compile_module():
        print("attn_mask compile")
        attn_cpp = ctypes.CDLL('./models/attn_mask.so')
        attn_cpp.init.argtypes = [c_int, c_int, c_int]
        attn_cpp.init.restype = ctypes.c_void_p
        attn_cpp.attn_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        attn_cpp.attn_backward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        attn_cpp.destroy.argtypes = [ctypes.c_void_p]
        return attn_cpp
    
    @staticmethod
    def _blockcuda_mask_compile_module():
        print("block_attn_mask compile")
        attn_cpp = ctypes.CDLL('./models/block_attn_mask.so')
        attn_cpp.init.argtypes = [c_int, c_int, c_int, c_int]
        attn_cpp.init.restype = ctypes.c_void_p
        attn_cpp.attn_forward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, c_int]
        attn_cpp.attn_backward.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, 
                                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                          c_int]
        attn_cpp.destroy.argtypes = [ctypes.c_void_p]
        return attn_cpp

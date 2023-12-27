
import torch
import torch.nn as nn
import ctypes
from ctypes import *

class CUDAAttention(nn.Module):
    def __init__(self, config, attn_cpp = None, attn_handle = None):
        super(CUDAAttention, self).__init__()
        emb_dim = config["head_dim"]
        num_heads = config["num_head"]
        seq_len = config["max_seq_len"]
        batch_size = config["batch_size"]
        num_batches = batch_size * num_heads

        class AttnFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, query, key, value, mask, layer_attn=None):
                tmp_hQuery = query.permute(0, 1, 3, 2).flatten()
                tmp_hValue = value.permute(0, 1, 3, 2).flatten()
                hAttn = torch.empty((batch_size,num_heads,seq_len,seq_len), dtype=torch.float32, device='cuda')
                hOut = torch.empty((batch_size,num_heads,seq_len,emb_dim), dtype=torch.float32, device='cuda')

                hQuery_p = tmp_hQuery.contiguous().data_ptr()
                hKey_p = key.contiguous().data_ptr()
                hValue_p = tmp_hValue.contiguous().data_ptr()
                hAttn_p = hAttn.contiguous().data_ptr()
                hOut_p = hOut.contiguous().data_ptr()

                attn_cpp.attn_forward(attn_handle, hQuery_p, hKey_p, hValue_p, hAttn_p, hOut_p)
                ctx.save_for_backward(query, key, value, hAttn)
                out = hOut.view(batch_size, num_heads, emb_dim, seq_len).permute(0, 1, 3, 2)
                #print(out)
                # output [batch_size, num_heads, seq_len, emb_dim]

                return out, hAttn.view(batch_size, num_heads, seq_len, seq_len).permute(0, 1, 3, 2)

            @staticmethod
            def backward(ctx, grad_output, grad_weights):

                query, key, value, attn_score = ctx.saved_tensors
                #[batch_size, num_heads, seq_len, emb_dim]
                
                hGradAttnScore = torch.zeros(num_batches*seq_len*seq_len, dtype=torch.float32, device='cuda')
                hGradAttnScale = torch.empty(num_batches*seq_len*seq_len, dtype=torch.float32, device='cuda')
                hGradAttn = torch.empty(num_batches*seq_len*seq_len, dtype=torch.float32, device='cuda')
                hGradQuery = torch.empty((batch_size,num_heads,seq_len,emb_dim), dtype=torch.float32, device='cuda')
                hGradKey = torch.empty((batch_size,num_heads,seq_len,emb_dim), dtype=torch.float32, device='cuda')
                hGradValue = torch.empty((batch_size,num_heads,seq_len,emb_dim), dtype=torch.float32, device='cuda')
                tmp_hGradOutput = grad_output.permute(0, 1, 3, 2)
                tmp_value = value.flatten()
                tmp_key = key.permute(0, 1, 3, 2).flatten()
                tmp_query = query.permute(0, 1, 3, 2).flatten()

                hQuery_p = tmp_query.contiguous().data_ptr()
                hKey_p = tmp_key.contiguous().data_ptr()
                hValue_p = tmp_value.contiguous().data_ptr()
                hAttnScore_p = attn_score.contiguous().data_ptr()
                hGradOutput_p = tmp_hGradOutput.contiguous().data_ptr()
                hGradAttnScore_p = hGradAttnScore.contiguous().data_ptr()
                hGradAttnScale_p = hGradAttnScale.contiguous().data_ptr()
                hGradAttn_p = hGradAttn.contiguous().data_ptr()
                hGradQuery_p = hGradQuery.contiguous().data_ptr()
                hGradKey_p = hGradKey.contiguous().data_ptr()
                hGradValue_p = hGradValue.contiguous().data_ptr()

                attn_cpp.attn_backward(attn_handle, hQuery_p, hKey_p, hValue_p, hAttnScore_p, hGradOutput_p, hGradAttnScore_p, 
                        hGradAttnScale_p, hGradAttn_p, hGradQuery_p, hGradKey_p, hGradValue_p)

                gradQuery = hGradQuery.view(batch_size, num_heads, emb_dim, seq_len).permute(0, 1, 3, 2)
                gradKey = hGradKey.view(batch_size, num_heads, emb_dim, seq_len).permute(0, 1, 3, 2)
                gradValue = hGradValue.view(batch_size, num_heads, emb_dim, seq_len).permute(0, 1, 3, 2)
    
                return gradQuery, gradKey, gradValue, None, None


        self.attn_func = AttnFunction

    def forward(self, query, key, value, mask):
        return self.attn_func.apply(query, key, value, mask)

    



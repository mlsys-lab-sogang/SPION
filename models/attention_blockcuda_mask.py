
import torch
import torch.nn as nn
import ctypes
from ctypes import *

class CUDABlockMaskAttention(nn.Module):
    def __init__(self, config, attn_cpp = None, attn_handle = None):
        super(CUDABlockMaskAttention, self).__init__()
        emb_dim = config["head_dim"]
        num_heads = config["num_head"]
        seq_len = config["max_seq_len"]
        batch_size = config["batch_size"]
        block_size = config["block_size"]
        num_batches = batch_size * num_heads
        num_blocks = int(seq_len/block_size)
        upsample = nn.Upsample(scale_factor=block_size, mode='nearest')

        class BlockAttnFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, query, key, value, mask, mat):
                query_tmp = query.flatten()
                key_tmp = key.permute(0, 1, 3, 2).flatten()
                value_tmp = value.flatten()

                sum_mat = torch.sum(mat,dim=1,dtype=torch.int32) * block_size 
                sum_mat = sum_mat.repeat(block_size, 1)
                sum_mat = sum_mat.transpose(1,0).flatten()

                Offsets = torch.zeros((seq_len+1), dtype=torch.int32, device='cuda')
                cum_sum_mat = torch.cumsum(sum_mat, dim=0)
                Offsets[1:] = torch.add(Offsets[:-1], cum_sum_mat)

                dense_matrix = upsample(mat.unsqueeze(0).unsqueeze(0))
                col_indices = dense_matrix[0][0].nonzero()[:, 1]
                Columns = col_indices.repeat(num_batches).to(torch.int32)

                nnz = Offsets[-1]

                hAttn = torch.empty(num_batches*nnz, dtype=torch.float32, device='cuda')
                hOut = torch.empty(seq_len*emb_dim*num_batches, dtype=torch.float32, device='cuda')
                mask = mask.repeat(1,num_heads)
                mask = 1e6 * (1 - mask[:, None, None, :])
                
                hQuery_p = query_tmp.contiguous().data_ptr()
                hKey_p = key_tmp.contiguous().data_ptr()
                hValue_p = value_tmp.contiguous().data_ptr()
                hMask_p = mask.contiguous().data_ptr()
                hAttn_p = hAttn.contiguous().data_ptr()
                hOut_p = hOut.contiguous().data_ptr()
                hOffsets_p = Offsets.contiguous().data_ptr()
                hColumns_p = Columns.contiguous().data_ptr()
                hSum_mat_p = sum_mat.contiguous().data_ptr()                    

                attn_cpp.attn_forward(attn_handle, hQuery_p, hKey_p, hValue_p, hMask_p, hAttn_p, hOut_p, hOffsets_p, hColumns_p, hSum_mat_p, nnz)

                ctx.save_for_backward(query, key, value, hAttn, Offsets, Columns, sum_mat)

                out = hOut.view(batch_size, num_heads, seq_len, emb_dim).contiguous()
                # out [batch_size, num_heads, seq_len, emb_dim]

                return out, 0

            @staticmethod
            def backward(ctx, grad_output, grad_weights):

                query, key, value, attn_score, hOffsets,hColumns,sum_mat = ctx.saved_tensors
                #[batch_size, num_heads, seq_len, emb_dim]
                
                query_tmp = query.flatten()
                key_tmp = key.flatten()
                tmp_value = value.permute(0, 1, 3, 2).flatten()

                nnz = hOffsets[-1]
                
                hGradAttnScore = torch.zeros((nnz*num_batches), dtype=torch.float32, device='cuda')
                hGradAttn = torch.empty((nnz*num_batches), dtype=torch.float32, device='cuda')
                hGradQuery = torch.empty((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
                hGradKey = torch.empty((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')
                hGradValue = torch.empty((seq_len*emb_dim*num_batches), dtype=torch.float32, device='cuda')

                hQuery_p = query_tmp.contiguous().data_ptr()
                hKey_p = key_tmp.contiguous().data_ptr()
                hValue_p = tmp_value.contiguous().data_ptr()
                hAttnScore_p = attn_score.contiguous().data_ptr()
                hGradOutput_p = grad_output.contiguous().data_ptr()
                hGradAttnScore_p = hGradAttnScore.contiguous().data_ptr()
                hGradAttn_p = hGradAttn.contiguous().data_ptr()
                hGradQuery_p = hGradQuery.contiguous().data_ptr()
                hGradKey_p = hGradKey.contiguous().data_ptr()
                hGradValue_p = hGradValue.contiguous().data_ptr()
                hOffsets_p = hOffsets.contiguous().data_ptr()
                hColumns_p = hColumns.contiguous().data_ptr()
                hSum_mat_p = sum_mat.contiguous().data_ptr()

                attn_cpp.attn_backward(attn_handle, hQuery_p, hKey_p, hValue_p, hAttnScore_p, hGradOutput_p, hGradAttnScore_p, 
                        hGradAttn_p, hGradQuery_p, hGradKey_p, hGradValue_p, hOffsets_p, hColumns_p, hSum_mat_p, nnz)

                gradQuery = hGradQuery.view(batch_size, num_heads, seq_len, emb_dim)
                gradKey = hGradKey.view(batch_size, num_heads, seq_len, emb_dim)
                gradValue = hGradValue.view(batch_size, num_heads, seq_len, emb_dim)

                return gradQuery, gradKey, gradValue, None, None, None, None

        self.attn_func = BlockAttnFunction

    def forward(self, query, key, value, mask, mat):
        return self.attn_func.apply(query, key, value, mask, mat)
    


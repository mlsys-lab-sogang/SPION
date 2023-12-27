#include <cuda_runtime_api.h> 
#include <cublas_v2.h>

#include <math.h>
#include <stdio.h>

//nvcc -Xcompiler -fPIC -shared -lcublas -o attn.so attn.cu

__global__ void max_sum_exp_kernel(float *d_values, int seq_len, float scale, int num_batches)
{
    int i = blockIdx.x * blockDim.x * seq_len + threadIdx.x;
    //printf("%d\n", i);
    //float scale = sqrtf(float(emb_dim));
    if (i >= seq_len*seq_len*num_batches)
        return;

    for (int k = 0; k < seq_len; k++){
        d_values[seq_len * k + i] = d_values[seq_len * k + i] * scale;
    }

    float max = -10.0;
    for (int k = 0; k < seq_len; k++){
        if (max < d_values[seq_len * k + i]){
            max = d_values[seq_len * k + i];
        }
    }

    float sum = 0.0f;
    for (int k = 0; k < seq_len; k++){
        sum += expf(d_values[seq_len * k + i] - max);
    }
    sum = 1/sum;
    for (int k = 0; k < seq_len; k++){
        d_values[seq_len * k + i] = expf(d_values[seq_len * k + i] - max) * sum;
    }
}

__global__ void max_sum_exp_kernel_large(float *d_values, int seq_len, float scale, int num_batches, int tmp)
{
    //int i = blockIdx.x * seq_len * seq_len + (threadIdx.x * tmp + idx);
    int bid = blockIdx.x * seq_len; //blockDim.x : 1024
    //printf("%d\n", i);
    for(int t = 0; t<tmp; t++){
        int i = blockIdx.x * seq_len * seq_len + (t * blockDim.x) + threadIdx.x;
        if (i >= seq_len*seq_len*num_batches)
            return;
        //printf("%d\n", bid);
        float max = -10.0;
        for (int k = 0; k < seq_len; k++){
            d_values[seq_len * k + i] = (d_values[seq_len * k + i] * scale);
            if (max < d_values[seq_len * k + i]){
                max = d_values[seq_len * k + i];
            }
        }

        float sum = 0.0f;
        for (int k = 0; k < seq_len; k++){
            sum += expf(d_values[seq_len * k + i] - max);
        }

        for (int k = 0; k < seq_len; k++){
            d_values[seq_len * k + i] = expf(d_values[seq_len * k + i] - max) / sum;
        }
    }
}

__global__ void softmax_scale_backward_kernel(float *dGradAttnScore, float *dAttnScore, float *dGradAttnScale,
                                            float *dGradAttn, int seq_len, int emb_dim, int num_batches, float scale)
{
    int i = blockIdx.x * blockDim.x * seq_len + threadIdx.x;

    if (i >= seq_len*seq_len*num_batches)
        return;
    
    for (int k = 0; k < seq_len; k++){
        dGradAttnScore[k * seq_len + i] = dGradAttnScore[k * seq_len + i] * dAttnScore[k * seq_len + i];
    }

    float sum = 0.0f;
    for (int k = 0; k < seq_len; k++){
        sum += dGradAttnScore[k * seq_len + i];

    }
    for (int k = 0; k < seq_len; k++){
        dGradAttn[k * seq_len + i] = (dGradAttnScore[k * seq_len + i] - (dAttnScore[k * seq_len + i] * sum)) * scale;
    }
}

__global__ void softmax_scale_backward_kernel_large(float *dGradAttnScore, float *dAttnScore, float *dGradAttnScale,
                                            float *dGradAttn, int seq_len, int num_batches, float scale, int tmp)
{
    //int i = blockIdx.x * seq_len * seq_len + (threadIdx.x * tmp + idx);
    for(int t = 0; t<tmp; t++){
        int i = blockIdx.x * seq_len * seq_len + (t * blockDim.x) + threadIdx.x;
        if (i >= seq_len*seq_len*num_batches)
            return;
        //printf("%d\n",i);
        for (int k = 0; k < seq_len; k++){
            dGradAttnScore[k * seq_len + i] = dGradAttnScore[k * seq_len + i] * dAttnScore[k * seq_len + i];
            //printf("%f * %f = %f\n",dGradAttnScore[k * seq_len + i],dAttnScore[k * seq_len + i],dGradAttnScoreScale[k * seq_len + i]);
        }

        float sum = 0.0f;
        for (int k = 0; k < seq_len; k++){
            sum += dGradAttnScore[k * seq_len + i];
        }
        //printf("%f\n",sum);
        for (int k = 0; k < seq_len; k++){
            dGradAttn[k * seq_len + i] = (dGradAttnScore[k * seq_len + i] - (dAttnScore[k * seq_len + i] * sum)) * scale;
        }
    }
}

void gemm_strided_batchedEx(cublasHandle_t handle, cublasOperation_t opA, const float *d_A, const float *d_B, float *d_C, const int M, const int N, const int K, const int batch_size) {

    int lda = M;
    int ldb = K;
    int ldc = M;

    int strideA = M*K;
    int strideB = K*N;
    int strideC = M*N;

    // Set the alpha and beta parameters for the gemm operation
    float alpha = 1.0f;
    float beta = 0.0f;

    // Perform the matrix multiplication using cublasGemmStridedBatchedEx
    cublasGemmStridedBatchedEx(handle,
                               opA, CUBLAS_OP_N,
                               M, N, K,
                               &alpha,
                               d_A, CUDA_R_32F, lda, strideA,
                               d_B, CUDA_R_32F, ldb, strideB,
                               &beta,
                               d_C, CUDA_R_32F, ldc, strideC,
                               batch_size,
                               CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
 
}
//CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT
//CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
struct Attention{
    cublasHandle_t handle;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    float scale = 0;
    int emb_dim, seq_len, num_batches, tmp_seq_len, tmp_num_batches, tmp;

    Attention(int hemb_dim, int hseq_len, int hnum_batches){
        cublasCreate(&handle);
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        emb_dim = hemb_dim;
        seq_len = hseq_len;
        num_batches = hnum_batches;
        tmp_seq_len = 1024;

        if(seq_len>tmp_seq_len){
            tmp = (int)(seq_len/tmp_seq_len);
            tmp_num_batches = num_batches * tmp;
            
        }
        else{
            tmp_seq_len = seq_len;
            tmp_num_batches = num_batches;
            tmp = 0;
        }

        scale = 1/sqrtf(float(emb_dim));
    }

    ~Attention(){
        cublasDestroy(handle);

    }

    void attn_forward(float *dQuery, float *dKey, float *dValue, float *dAttn, float *dOut)
    {
        // Host problem definition
        //cudaEventRecord(start);
        gemm_strided_batchedEx(handle, CUBLAS_OP_N, dQuery, dKey, dAttn, seq_len, seq_len, emb_dim, num_batches);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Forward gemm_strided_batchedEx 1 : %f\n",milliseconds);

        // cudaEventRecord(start);
        //scale_kernel<<<num_batches, seq_len*seq_len>>>(dAttn, seq_len, num_batches, scale);
        //cudaDeviceSynchronize();
        
        if(tmp == 0){
            max_sum_exp_kernel<<<num_batches, seq_len>>>(dAttn, seq_len, scale, num_batches);
        }
        else{
            max_sum_exp_kernel_large<<<num_batches, tmp_seq_len>>>(dAttn, seq_len, scale, num_batches, tmp);
        }
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Forward scale_softmax_function : %f\n",milliseconds);

        // cudaEventRecord(start);
        gemm_strided_batchedEx(handle, CUBLAS_OP_N, dAttn, dValue, dOut, seq_len, emb_dim, seq_len, num_batches);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Forward gemm_strided_batchedEx 2 : %f\n",milliseconds);
    }

    void attn_backward(float *dQuery, float *dKey, float *dValue, float *dAttnScore, float *dGradOutput, float *dGradAttnScore, 
                    float *dGradAttnScale, float *dGradAttn, float *dGradQuery, float *dGradKey, float *dGradValue)
    {
        //cudaEventRecord(start);
        gemm_strided_batchedEx(handle, CUBLAS_OP_N, dGradOutput, dValue, dGradAttnScore, seq_len, seq_len, emb_dim, num_batches);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Backward gemm_strided_batchedEx 1: %f\n",milliseconds);

        // cudaEventRecord(start);
        if(tmp == 0){
            softmax_scale_backward_kernel<<<num_batches, seq_len>>>(dGradAttnScore, dAttnScore, dGradAttnScale, dGradAttn, seq_len, emb_dim, num_batches, scale);
        }
        else{
            softmax_scale_backward_kernel_large<<<num_batches, tmp_seq_len>>>(dGradAttnScore, dAttnScore, dGradAttnScale, dGradAttn, seq_len, num_batches, scale, tmp);
        }
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Backward softmax_scale_backward_function 2: %f\n",milliseconds);

        // cudaEventRecord(start);
        gemm_strided_batchedEx(handle, CUBLAS_OP_N, dGradAttn, dKey, dGradQuery, seq_len, emb_dim, seq_len, num_batches);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Backward gemm_strided_batchedEx 2: %f\n",milliseconds);

        // cudaEventRecord(start);
        gemm_strided_batchedEx(handle, CUBLAS_OP_T, dGradAttn, dQuery, dGradKey, seq_len, emb_dim, seq_len, num_batches);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Backward gemm_strided_batchedEx 3: %f\n",milliseconds);

        // cudaEventRecord(start);
        gemm_strided_batchedEx(handle, CUBLAS_OP_T, dAttnScore, dGradOutput, dGradValue, seq_len, emb_dim, seq_len, num_batches);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Backward gemm_strided_batchedEx 4: %f\n",milliseconds);
    }
};

extern "C" Attention* init(int emb_dim, int seq_len, int num_batches) {
    return new Attention(emb_dim, seq_len, num_batches);
}

extern "C" void attn_forward(Attention* attn, float *hQuery, float *hKey, float *hValue, float *hAttn, float *hOut){
    attn->attn_forward(hQuery, hKey, hValue, hAttn, hOut);
}

extern "C" void attn_backward(Attention* attn, float *hQuery, float *hKey, float *hValue, float *hAttnScore, float *hGradOutput, float *hGradAttnScore, 
                    float *hGradAttnScale, float *hGradAttn, float *hGradQuery, float *hGradKey, float *hGradValue){
    attn->attn_backward(hQuery, hKey, hValue, hAttnScore, hGradOutput, hGradAttnScore, hGradAttnScale, hGradAttn, hGradQuery, hGradKey, hGradValue);
}

extern "C" void destroy(Attention* attn) {
    delete attn;
}

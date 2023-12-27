#include <cuda_runtime.h>
#include <cusparse.h>         // cusparseSpMM

#include <nvtx3/nvToolsExt.h>
#define IMPROVE 0

#define CUDA_BLOCKDIM 1024

#define WARP_SIZE 32



//nvcc -Xcompiler -fPIC -shared -lcusparse -o block_attn_mask.so block_attn_mask_warp.cu

//dAttn, dMask, dSum_mat, dOffsets, dColumns, nnz, seq_len, num_batches, scale, i

static __device__ __inline__ uint32_t __get_warpid(){
  uint32_t warpid;
  asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
  return warpid;}

static __device__ __inline__ uint32_t __get_laneid(){
  uint32_t laneid;
  asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;}


__global__ void scale_softmax_kernel_large(float *d_values, float *d_mask, int *dSum_mat, int *dOffsets, int *dColumns, int nnz, int seq_len, int num_batches, float scale, int tmp)
{
    // coalescing issue
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    if (i >= seq_len*num_batches)
        return;
    
    for(int t = 0; t < tmp; t++){
        //int tid = (threadIdx.x * tmp) + t; //0~4095
        int tid = (t * blockDim.x) + threadIdx.x; //0~4095
        int block_size = dSum_mat[tid];
        int block_idx = bid*nnz + dOffsets[tid];

        float max = -100.0;
        for (int k = 0; k < block_size; k++){
            d_values[block_idx + k] = d_values[block_idx + k] * scale - d_mask[blockIdx.x*seq_len + dColumns[block_idx + k]];
            if (max < d_values[block_idx + k]){
                max = d_values[block_idx + k];
            }
        }
        float sum = 0.0f;
        for (int k = 0; k < block_size; k++){
            sum += expf(d_values[block_idx + k] - max);

        }
        sum += (seq_len - block_size) * expf(-max);
        sum = 1/sum;
        for (int k = 0; k < block_size; k++){
            if(sum == 0){
                d_values[block_idx + k] = 0;
            }
            else{
                d_values[block_idx + k] = expf(d_values[block_idx + k] - max) * sum;
            }
        }
    }
}


__inline__ __device__
float warp_reduce_max(float value){
    for(int w = 1; w < WARP_SIZE; w = w << 1){
        float tmp = __shfl_xor_sync(0xffffffff, value, w);
        value = fmaxf(value, tmp);
    }
    return value;
}
__inline__ __device__
float warp_reduce_sum(float value){
    float ret = value;

    for(int w = 1; w < WARP_SIZE; w = w << 1){
        float tmp = __shfl_xor_sync(0xffffffff, ret, w);
        ret += tmp;
    }
    return ret;
}

__global__ void scale_softmax_kernel(float *d_values, float *d_mask, int *dSum_mat, int *dOffsets, int *dColumns, int nnz, int seq_len, int num_batches, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 0 ~ seq_len * num_batch * WARP_SIZE
    

    if(i > WARP_SIZE *seq_len * num_batches)
        return;
    
    int warp_id = i / WARP_SIZE; // 0 ~ seq_len * num_batch
    int lane = i % WARP_SIZE;
    int batch_id = warp_id / seq_len;
    int row_to_compute = warp_id % seq_len;//(t * blockDim.x) + threadIdx.x; //0~4095
    
    int block_size = dSum_mat[row_to_compute];
    int block_idx = batch_id *nnz + dOffsets[row_to_compute];


    float max1 = 0.0;
    int mask_tmp = batch_id*seq_len;
    float value_tmp;
    
    for (int k = block_idx + lane; k < block_idx + block_size; k+=WARP_SIZE){
        value_tmp = d_values[k] * scale - d_mask[mask_tmp + dColumns[k]];
        
        if (max1 < value_tmp){
            max1 = value_tmp;
        }
        d_values[k] = value_tmp;
        
    }

    float max = warp_reduce_max(max1);

    float sum = 0.0f;
    for (int k = block_idx + lane; k < block_idx + block_size; k+= WARP_SIZE){
        sum += expf(d_values[k] - max);
    }
    sum = warp_reduce_sum(sum);
    sum += (seq_len - block_size) * expf(-max);
    sum = 1/sum;

    if(sum == 0){
        for (int k = block_idx + lane; k < block_idx + block_size; k+= WARP_SIZE){
            d_values[k] = 0;
        }    
    }
    else{
        for (int k = block_idx + lane; k < block_idx + block_size; k+= WARP_SIZE){
            d_values[k] = expf(d_values[k] - max) * sum;
        } 
    }
    
}

__global__ void softmax_scale_backward_kernel(float *dGradAttnScore, float *dAttnScore,
                                            float *dGradAttn, int *dSum_mat, int *dOffsets, int nnz, int seq_len, int num_batches, float scale)
{
    // coalescing issue
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= WARP_SIZE * seq_len*num_batches)
        return;

    int warp_id = i / WARP_SIZE; // 0 ~ seq_len * num_batch
    int lane = i % WARP_SIZE;
    int batch_id = warp_id / seq_len;
    int row_to_compute = warp_id % seq_len;//(t * blockDim.x) + threadIdx.x; //0~4095
    
    int block_size = dSum_mat[row_to_compute];
    int block_idx = batch_id *nnz + dOffsets[row_to_compute];




    float grad_sum = 0.0f;
    for (int k = block_idx  + lane ; k < block_idx + block_size; k += WARP_SIZE){
        float val_tmp = dGradAttnScore[k] * dAttnScore[k];
        dGradAttnScore[k] = val_tmp;
        grad_sum += val_tmp;
    }
    grad_sum = warp_reduce_sum(grad_sum);
    
    for (int k = block_idx + lane; k < block_idx + block_size; k+= WARP_SIZE){
        dGradAttn[k] = (dGradAttnScore[k] - (dAttnScore[k] * grad_sum)) * scale;
    }
}

__global__ void softmax_scale_backward_kernel_large(float *dGradAttnScore, float *dAttnScore, 
                                            float *dGradAttn, int *dSum_mat, int *dOffsets, int nnz, int seq_len, int num_batches, float scale, int tmp)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int bid = blockIdx.x;

    if (i >= seq_len*num_batches)
        return;

    for(int t = 0; t < tmp; t++){
        //int tid = (threadIdx.x * tmp) + t; //0~4095
        int tid = (t * blockDim.x) + threadIdx.x; //0~4095
        int block_size = dSum_mat[tid];
        int block_idx = bid*nnz + dOffsets[tid];

        float grad_sum = 0.0f;
        for (int k = 0; k < block_size; k++){
            dGradAttnScore[block_idx + k] = dGradAttnScore[block_idx + k] * dAttnScore[block_idx + k];
            grad_sum += dGradAttnScore[block_idx + k];
        }
        
        // for (int k = 0; k < block_size; k++){
        //     grad_sum += dGradAttnScore[block_idx + k];
        // }
        
        for (int k = 0; k < block_size; k++){
            dGradAttn[block_idx + k] = (dGradAttnScore[block_idx + k] - (dAttnScore[block_idx + k] * grad_sum)) * scale;
        }
    }
}

struct SddmmOperation
{
    void *sddmm_dBuffer = NULL;
    cusparseHandle_t sddmm_handle = NULL;
    SddmmOperation()
    {
    }
    ~SddmmOperation()
    {
    }
    void prepare_resources(){

    }
    void sddmm(cusparseHandle_t handle, float *dQuery, float *dKey, float *dAttn, int *d_offsets, int *d_columns, int seq_len, int emb_dim, int nnz, int num_batches)
    {

        size_t bufferSize = 0;
        int lda = emb_dim;
        int ldb = seq_len;
        int input_size = seq_len * emb_dim;

        float alpha = 1.0f;
        float beta = 0.0f;

        if (sddmm_handle == NULL)
        {
        cusparseCreate(&sddmm_handle);
        }

        cusparseDnMatDescr_t matA, matB;
        cusparseSpMatDescr_t matC;
        nvtxRangePush("SDDMM prepare resources");

        // Create dense matrix A
        cusparseCreateDnMat(&matA, seq_len, emb_dim, lda, dQuery,
                            CUDA_R_32F, CUSPARSE_ORDER_ROW);
        cusparseDnMatSetStridedBatch(matA, num_batches, input_size);
        // Create dense matrix B
        cusparseCreateDnMat(&matB, emb_dim, seq_len, ldb, dKey,
                            CUDA_R_32F, CUSPARSE_ORDER_ROW);
        cusparseDnMatSetStridedBatch(matB, num_batches, input_size);
        // Create sparse matrix C in CSR format
        cusparseCreateCsr(&matC, seq_len, seq_len, nnz,
                          d_offsets, d_columns, dAttn,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCsrSetStridedBatch(matC, num_batches, 0, nnz);
        // allocate an external buffer if needed
        
        nvtxRangePush("SDDMM buffer size");
        cusparseSDDMM_bufferSize(
            sddmm_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize);

        nvtxRangePop();

        nvtxRangePop();
        nvtxRangePush("Malloc from sddmm");
        if (sddmm_dBuffer == NULL)
        {
            //printf("MALLOC from SDDMM %d %d %X\n", getpid(), gettid(), this);
            cudaMalloc(&sddmm_dBuffer, bufferSize);
        }
        nvtxRangePop();

        nvtxRangePush("SDDMM PREPROCESS");
        // execute preprocess (optional)
        cusparseSDDMM_preprocess(
            sddmm_handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SDDMM_ALG_DEFAULT, sddmm_dBuffer);
        nvtxRangePop();

        nvtxRangePush("SDDMM COMPUTE");
        cusparseSDDMM(sddmm_handle,
                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                      CUSPARSE_OPERATION_NON_TRANSPOSE,
                      &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                      CUSPARSE_SDDMM_ALG_DEFAULT, sddmm_dBuffer);
        // destroy matrix/vector descriptors
        nvtxRangePop();
        nvtxRangePush("nvtx DESTROY Resources");
        cusparseDestroyDnMat(matA);
        cusparseDestroyDnMat(matB);
        cusparseDestroySpMat(matC);
        // cudaFree(sddmm_dBuffer);
        nvtxRangePop();
    }
};
struct SpmmOperation
{
    void *spmm_dBuffer = NULL;
    cusparseHandle_t spmm_handle = NULL;
    SpmmOperation()
    {
    }
    ~SpmmOperation()
    {
    }
    void spmm(cusparseHandle_t handle, cusparseOperation_t opA, void *dBuffer, float *dA, float *dB, float *dC, int *d_offsets, int *d_columns, int seq_len, int emb_dim, int nnz, int num_batches)
    {
        // Host problem definition
        int ldb = emb_dim;
        int ldc = emb_dim;
        int output_size = seq_len * emb_dim;

        float alpha = 1.0f;
        float beta = 0.0f;
        // float milliseconds = 0;

        cusparseSpMatDescr_t matA;
        cusparseDnMatDescr_t matB, matC;

        size_t bufferSize = 0;

        if (spmm_handle == NULL)
        {
        cusparseCreate(&spmm_handle);
        }
        

        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // float milliseconds = 0;

        // cudaEventRecord(start);
        nvtxRangePush("SPMM resource prepare");
        cusparseCreateCsr(&matA, seq_len, seq_len, nnz,
                          d_offsets, d_columns, dA,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("cusparseCreateCsr A : %f\n",milliseconds);

        // cudaEventRecord(start);
        cusparseCsrSetStridedBatch(matA, num_batches, 0, nnz);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("cusparseCsrSetStridedBatch A : %f\n",milliseconds);

        // cudaEventRecord(start);
        cusparseCreateDnMat(&matB, seq_len, emb_dim, ldb, dB,
                            CUDA_R_32F, CUSPARSE_ORDER_ROW);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("cusparseCreateDnMat B : %f\n",milliseconds);

        // cudaEventRecord(start);
        cusparseDnMatSetStridedBatch(matB, num_batches, output_size);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("cusparseDnMatSetStridedBatch B : %f\n",milliseconds);

        // cudaEventRecord(start);
        cusparseCreateDnMat(&matC, seq_len, emb_dim, ldc, dC,
                            CUDA_R_32F, CUSPARSE_ORDER_ROW);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("cusparseCreateDnMat C : %f\n",milliseconds);

        // cudaEventRecord(start);
        cusparseDnMatSetStridedBatch(matC, num_batches, output_size);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("cusparseDnMatSetStridedBatch C : %f\n",milliseconds);

        // cudaEventRecord(start);
        nvtxRangePush("SPMM BUFFER SIZE");
        cusparseSpMM_bufferSize(spmm_handle,
                                opA, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                CUSPARSE_SPMM_CSR_ALG2, &bufferSize);
        nvtxRangePop();
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("cusparseSpMM_bufferSize dBuffer : %f\n",milliseconds);

        // cudaEventRecord(start);

        nvtxRangePop();
        nvtxRangePush("SPMM Malloc");
#if IMPROVE == 0
        if (dBuffer == NULL)
        {
            //printf("MALLOC from SPMM %d %d %X\n", getpid(), gettid(), this);
            cudaMalloc(&dBuffer, bufferSize);
        }
#else
        if (spmm_dBuffer == NULL)
        {
            //printf("MALLOC from SPMM %d %d %X\n", getpid(), gettid(), this);
            cudaMalloc(&spmm_dBuffer, bufferSize);
        }

#endif
        nvtxRangePop();
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("cudaMalloc dBuffer : %f\n",milliseconds);

        // nvtxRangePush("SpMM PREPROCESS");
        // // execute preprocess (optional)
        // cusparseSpMM_preprocess(
        //     spmm_handle,
        //     opA,
        //     CUSPARSE_OPERATION_NON_TRANSPOSE,
        //     &alpha, matA, matB, &beta, matC, CUDA_R_32F,
        //     CUSPARSE_SPMM_CSR_ALG2, spmm_dBuffer);
        // nvtxRangePop();

        // cudaEventRecord(start);
        nvtxRangePush("SPMM Compute");
#if IMPROVE == 0
        cusparseSpMM(spmm_handle,
                     opA, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                     CUSPARSE_SPMM_CSR_ALG2, dBuffer);
#else
        cusparseSpMM(spmm_handle,
                     opA, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                     CUSPARSE_SPMM_CSR_ALG2, spmm_dBuffer);
        
#endif
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("cusparseSpMM : %f\n",milliseconds);
        nvtxRangePop();
        nvtxRangePush("SPMM Resource Free");
        cusparseDestroySpMat(matA);
        cusparseDestroyDnMat(matB);
        cusparseDestroyDnMat(matC);
#if IMPROVE==0
        cudaFree(dBuffer);
#endif
        nvtxRangePop();
    }
};


struct Attention{

    cusparseHandle_t     handle = NULL;
    void* dBuffer    = NULL;
    cudaEvent_t start, stop;
    float milliseconds = 0;
    float scale = 0;
    int emb_dim, seq_len, num_batches, block_size, tmp_seq_len, tmp_num_batches, tmp, grid_size, blockdim_size;

    SddmmOperation *sd1, *sd2;
    SpmmOperation *sp1, *sp2, *sp3, *sp4;

    Attention(int hemb_dim, int hseq_len, int hnum_batches, int hblock_size)
    {
        scale = 1/sqrtf(float(hemb_dim));
        emb_dim = hemb_dim;
        seq_len = hseq_len;
        num_batches = hnum_batches;
        block_size = hblock_size;

        tmp_seq_len = 1024;

        if(seq_len>=tmp_seq_len){
            tmp = (int)(seq_len/tmp_seq_len);
            tmp_num_batches = num_batches * tmp;
                        
        }
        else{
            tmp_seq_len = seq_len;
            tmp_num_batches = num_batches;
            tmp = 0;
        }

        blockdim_size = CUDA_BLOCKDIM > seq_len ? seq_len : CUDA_BLOCKDIM;
        grid_size = (((seq_len * WARP_SIZE) + blockdim_size - 1) / blockdim_size) * num_batches;

        cusparseCreate(&handle);

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        sd1 = new SddmmOperation();
        sd2 = new SddmmOperation();
        sp1 = new SpmmOperation();
        sp2 = new SpmmOperation();
        sp3 = new SpmmOperation();
        sp4 = new SpmmOperation();
    }

    ~Attention(){
        delete sd1;
        delete sd2;
        delete sp1;
        delete sp2;
        delete sp3;
        delete sp4;
        cusparseDestroy(handle);
    }

    void attn_forward(float *dQuery, float *dKey, float *dValue, float *dMask, float *dAttn, float *dOut, int *dOffsets, int *dColumns, int *dSum_mat, int nnz)
    {
        //cudaEventRecord(start);
        sd1->sddmm(handle, dQuery, dKey, dAttn, dOffsets, dColumns, seq_len, emb_dim, nnz, num_batches);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("SDDMM : %f\n",milliseconds);

        // cudaEventRecord(start);
        //scale_softmax_kernel<<<num_batches, seq_len>>>(dAttnScore, dAttn, dMask, dSum_mat, dOffsets, dColumns, nnz, seq_len, num_batches, scale);
        // printf("tmp : %d\n",tmp);
        // printf("tmp_num_batches : %d\n",tmp_num_batches);
        // printf("tmp_seq_len : %d\n",tmp_seq_len);
        // printf("num_batches : %d\n",num_batches);
        // cudaEventRecord(start);
        
        // float* before;
        // float* host_dAttn, *host_before;
        // cudaMalloc(&before, sizeof(float) * nnz * num_batches);
        // host_dAttn = (float*)malloc(sizeof(float) * nnz * num_batches);
        // host_before = (float*)malloc(sizeof(float) * nnz * num_batches);
        // cudaMemcpy(before, dAttn, sizeof(float) * nnz * num_batches, cudaMemcpyDeviceToDevice);


        // scale_softmax_kernel_large<<<num_batches, tmp_seq_len>>>(before, dMask, dSum_mat, dOffsets, dColumns, nnz, seq_len, num_batches, scale, tmp);
        
        //cudaEventRecord(start);
        scale_softmax_kernel<<<grid_size, blockdim_size>>>(dAttn, dMask, dSum_mat, dOffsets, dColumns, nnz, seq_len, num_batches, scale);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("softmax : %f\n",milliseconds);



        // cudaDeviceSynchronize();

        // cudaMemcpy(host_dAttn, dAttn, sizeof(float) * nnz * num_batches, cudaMemcpyDeviceToHost);
        // cudaMemcpy(host_before, before, sizeof(float) * nnz * num_batches, cudaMemcpyDeviceToHost);

        // cudaDeviceSynchronize();
        // for (int i =0 ;i< nnz * num_batches; i++)
        //     {
        //         if (host_before[i] != host_dAttn[i])
        //             printf("AFTER KERNEL LAUNCH!!! %.20f\n", abs(host_before[i] - host_dAttn[i]));
        //     }
        // cudaFree(before);
        // free(host_before);
        // free(host_dAttn);

        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Forward scale_softmax_function : %f\n",milliseconds);

        //cudaEventRecord(start);
        sp1->spmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, dBuffer, dAttn, dValue, dOut, dOffsets, dColumns, seq_len, emb_dim, nnz, num_batches);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("SPMM : %f\n",milliseconds);
    }

    void attn_backward(float *dQuery, float *dKey, float *dValue, float *dAttnScore, float *dGradOutput, float *dGradAttnScore, 
                    float *dGradAttn, float *dGradQuery, float *dGradKey, float *dGradValue,
                    int *dOffsets, int *dColumns, int *dSum_mat, int nnz)
    {

        // cudaEventRecord(start);
        sd1->sddmm(handle, dGradOutput, dValue, dGradAttnScore, dOffsets, dColumns, seq_len, emb_dim, nnz, num_batches);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Backward SDDMM : %f\n",milliseconds);

        // cudaEventRecord(start);
        // softmax_scale_backward_kernel<<<num_batches, seq_len>>>(dGradAttnScore, dAttnScore, dGradAttnScoreScale, dGradAttnScale, dGradAttn, dSum_mat, dOffsets, nnz, seq_len, num_batches, scale);
        
        
        //softmax_scale_backward_kernel_large<<<num_batches, tmp_seq_len>>>(dGradAttnScore, dAttnScore, dGradAttn, dSum_mat, dOffsets, nnz, seq_len, num_batches, scale, tmp);
       
        softmax_scale_backward_kernel<<<grid_size, blockdim_size>>>(dGradAttnScore, dAttnScore, dGradAttn, dSum_mat, dOffsets, nnz, seq_len, num_batches, scale);
        
  
        
        
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Backward softmax_scale_backward_function : %f\n",milliseconds);

        // cudaEventRecord(start);
        sp2->spmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, dBuffer, dGradAttn, dKey, dGradQuery, dOffsets, dColumns, seq_len, emb_dim, nnz, num_batches);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Backward SPMM 1 : %f\n",milliseconds);

        // cudaEventRecord(start);
        sp3->spmm(handle, CUSPARSE_OPERATION_TRANSPOSE, dBuffer, dGradAttn, dQuery, dGradKey, dOffsets, dColumns, seq_len, emb_dim, nnz, num_batches);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Backward SPMM 2 : %f\n",milliseconds);

        // cudaEventRecord(start);
        sp4->spmm(handle, CUSPARSE_OPERATION_TRANSPOSE, dBuffer, dAttnScore, dGradOutput, dGradValue, dOffsets, dColumns, seq_len, emb_dim, nnz, num_batches);
        // cudaEventRecord(stop);
        // cudaEventSynchronize(stop);
        // milliseconds = 0;
        // cudaEventElapsedTime(&milliseconds, start, stop);
        // printf("Backward SPMM 3 : %f\n",milliseconds);

    }
};

extern "C" Attention *init(int emb_dim, int seq_len, int num_batches, int block_size)
{
    return new Attention(emb_dim, seq_len, num_batches, block_size);
}

extern "C" void attn_forward(Attention* attn, float *hQuery, float *hKey, float *hValue, float *hMask, float *hAttn, float *hOut, int *hOffsets, 
                    int *hColumns, int *hSum_mat, int nnz){
    attn->attn_forward(hQuery, hKey, hValue, hMask, hAttn, hOut, hOffsets, hColumns, hSum_mat, nnz);
}

extern "C" void attn_backward(Attention* attn, float *hQuery, float *hKey, float *hValue, float *hAttnScore, float *hGradOutput, float *hGradAttnScore, 
                    float *hGradAttn, float *hGradQuery, float *hGradKey, float *hGradValue,
                    int *hOffsets, int *hColumns, int *hSum_mat, int nnz){
    attn->attn_backward(hQuery, hKey, hValue, hAttnScore, hGradOutput, hGradAttnScore, hGradAttn, hGradQuery, hGradKey, hGradValue, 
                        hOffsets, hColumns, hSum_mat, nnz);
}

extern "C" void destroy(Attention* attn) {
    delete attn;
}

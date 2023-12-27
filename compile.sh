cd models
nvcc -Xcompiler -fPIC -shared -lcusparse -o block_attn.so block_attn.cu
nvcc -Xcompiler -fPIC -shared -lcusparse -o block_attn_mask.so block_attn_mask.cu
nvcc -Xcompiler -fPIC -shared -lcublas -o attn.so attn.cu
nvcc -Xcompiler -fPIC -shared -lcublas -o attn_mask.so attn_mask.cu
cd ..
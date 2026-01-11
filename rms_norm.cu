#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>


__global__ void square(float *d_inputVector, float *d_squaredVector, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (idx < size) {
        d_squaredVector[idx] = d_inputVector[idx] * d_inputVector[idx];
    }
}


__global__ void blockSum(float *d_inputVector, float *blockSumVector, int size) {
    __shared__ float shared[256];

    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tid;

    float val = (idx < size) ? d_inputVector[idx] : 0.0f;
    shared[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockSumVector[blockIdx.x] = shared[0];
    }
}


__global__ void compute_rms(float *sum, float *rms, int size) {
    const float epsilon = 1e-6f;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *rms = sqrtf((*sum) / size + epsilon);
    }
}


__global__ void rms_normalize(float *input, float *gamma, float *rms, float *output, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size) {
        output[idx] = (input[idx] / (*rms)) * gamma[idx];
    }
}


int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int size = 4096;

    float *h_inputVector = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; ++i) {
        h_inputVector[i] = i;
    }

    float *h_outputVector = (float *)malloc(size * sizeof(float));

    float *d_inputVector, *d_squaredVector, *d_outputVector;
    cudaMalloc(&d_inputVector, size * sizeof(float));
    cudaMalloc(&d_squaredVector, size * sizeof(float));
    cudaMalloc(&d_outputVector, size * sizeof(float));

    cudaMemcpy(d_inputVector, h_inputVector, size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    float *blockSumVector;
    cudaMalloc(&blockSumVector, blocksPerGrid * sizeof(float));


    float *h_gamma = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; ++i) {
        h_gamma[i] = 1.0f;
    }
    
    float *d_gamma;
    cudaMalloc((void **)&d_gamma, size * sizeof(float));

    cudaMemcpy(d_gamma, h_gamma, size * sizeof(float), cudaMemcpyHostToDevice);

    float *d_rms;
    cudaMalloc((void **)&d_rms, sizeof(float));

    // WARMUP
    for (int i = 0; i < 50; i++) {
        square<<<blocksPerGrid, threadsPerBlock>>>(d_inputVector, d_squaredVector, size);

        int currSizeTmp = size;
        while (currSizeTmp > 1) {
            int blocks = (currSizeTmp + threadsPerBlock - 1) / threadsPerBlock;
            blockSum<<<blocks, threadsPerBlock>>>(d_squaredVector, blockSumVector, currSizeTmp);
            currSizeTmp = blocks;
        }

        compute_rms<<<1, 1>>>(d_squaredVector, d_rms, size);
        rms_normalize<<<blocksPerGrid, threadsPerBlock>>>(d_inputVector, d_gamma, d_rms, d_outputVector, size);
    }
    cudaDeviceSynchronize();

    // Actual Benchmark
    cudaEventRecord(start);

    for (int iter = 0; iter < 1000; iter++) {

        square<<<blocksPerGrid, threadsPerBlock>>>(d_inputVector, d_squaredVector, size);

        int currSizeTmp = size;
        while (currSizeTmp > 1) {
            int blocks = (currSizeTmp + threadsPerBlock - 1) / threadsPerBlock;
            blockSum<<<blocks, threadsPerBlock>>>(d_squaredVector, blockSumVector, currSizeTmp);
            currSizeTmp = blocks;
        }

        compute_rms<<<1, 1>>>(d_squaredVector, d_rms, size);
        rms_normalize<<<blocksPerGrid, threadsPerBlock>>>(d_inputVector, d_gamma, d_rms, d_outputVector, size);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    printf("Avg RMSNorm time: %f us\n", (elapsed_ms * 1000.0f) / 1000);

    cudaMemcpy(h_outputVector, d_outputVector, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inputVector);
    cudaFree(d_squaredVector);
    cudaFree(d_outputVector);
    cudaFree(blockSumVector);
    cudaFree(d_gamma);
    cudaFree(d_rms);

    free(h_inputVector);
    free(h_gamma);
    free(h_outputVector);

    return 0;
}

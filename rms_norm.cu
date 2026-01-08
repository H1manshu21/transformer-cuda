#include <stdlib.h>

__global__ void square(float *d_inputVector, float *d_squaredVector, int size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < size) {
        d_squaredVector[idx] = d_inputVector[idx] * d_inputVector[idx];
    }
}

__global__ void blockSum(float *d_squaredVector, float *blockSum, int size) {
    __shared__ float shared[256];

    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    float val = 0.0f;
    
    if (idx < size) {
        val = d_squaredVector[idx];
    }

    shared[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (idx < stride) {
            shared[tid] += shared[tid + stride];
        }

        __syncthreads();
    }

    if (tid == 0) {
        blockSum[blockIdx.x] = shared[0]; 
    }
}

int main() {
    int size = 4096;

    // Allocate CPU Memory
    float *h_inputVector = (float *)malloc(size * sizeof(float));
    float *h_outputVector = (float *)malloc(size * sizeof(float));

    float *d_inputVector;
    float *d_squaredVector;
    float *d_outputVector;
    
    // Allocate GPU memory
    cudaMalloc((void **)&d_inputVector, size * sizeof(float));
    cudaMalloc((void **)&d_squaredVector, size * sizeof(float));
    cudaMalloc((void **)&d_outputVector, size * sizeof(float));

    // Assign value range from 1 to 4096
    for (int i = 0; i < size; ++i) {
        h_inputVector[i] = i + 1;
    }

    // Transfer data of h_inputVector to d_inputVector
    cudaMemcpy(d_inputVector, h_inputVector, size * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = size / threadsPerBlock;

    // Square each element
    square<<<blocksPerGrid, threadsPerBlock>>>(d_inputVector, d_squaredVector, size);

    float *blockSum;
    cudaMalloc((void **)&blockSum, blocksPerGrid * sizeof(float));

    // Store block sum
    blockSum<<<blocksPerGrid, threadsPerBlock>>>(d_squaredVector, blockSum, size);


    cudaMemcpy(d_outputVector, h_outputVector, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_inputVector);
    cudaFree(d_squaredVector);
    cudaFree(d_outputVector);

    free(h_inputVector);
    free(h_outputVector);
    
    return 0;
}

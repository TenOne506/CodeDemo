#include <cfloat>
#include <cmath>
#include <cstdlib>// For rand()
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 16

__global__ void softmaxKernel(float *input, float *output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        float max_val = input[row * cols];
        for (int i = 1; i < cols; ++i) {
            if (input[row * cols + i] > max_val) { max_val = input[row * cols + i]; }
        }

        float sum = 0.0f;
        for (int i = 0; i < cols; ++i) { sum += expf(input[row * cols + i] - max_val); }

        output[row * cols + col] = expf(input[row * cols + col] - max_val) / sum;
    }
}


__global__ void softmaxKernelOptimized(float* input, float* output, int rows, int cols) {
    extern __shared__ float shared_data[]; // 共享内存
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = threadIdx.x; // 线程在块内的列索引（行操作）

    if (row >= rows) return; // 超出范围返回

    // 计算索引
    float* row_data = &input[row * cols]; // 指向该行的起始地址
    float* shared_row = &shared_data[threadIdx.y * cols]; // 线程块共享行

    // **1. 计算最大值（归约计算）**
    float max_val = -INFINITY;
    for (int i = col; i < cols; i += blockDim.x) {
        max_val = fmaxf(max_val, row_data[i]);
    }

    // 线程块内归约最大值
    __shared__ float row_max;
    if (col == 0) row_max = -INFINITY;
    __syncthreads();

    atomicMax((int*)&row_max, __float_as_int(max_val)); // 原子操作更新最大值
    __syncthreads();

    // **2. 计算指数和**
    float sum = 0.0f;
    for (int i = col; i < cols; i += blockDim.x) {
        shared_row[i] = expf(row_data[i] - row_max); // 先存入共享内存
        sum += shared_row[i];
    }
    __syncthreads();

    // 线程块内求和归约
    __shared__ float row_sum;
    if (col == 0) row_sum = 0.0f;
    __syncthreads();

    atomicAdd(&row_sum, sum);
    __syncthreads();

    // **3. 计算 Softmax 并写回**
    for (int i = col; i < cols; i += blockDim.x) {
        output[row * cols + i] = shared_row[i] / row_sum;
    }
}


void softmaxCPU(float *input, float *output, int rows, int cols) {
    for (int row = 0; row < rows; ++row) {
        float max_val = input[row * cols];
        for (int col = 1; col < cols; ++col) {
            if (input[row * cols + col] > max_val) { max_val = input[row * cols + col]; }
        }

        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) { sum += expf(input[row * cols + col] - max_val); }

        for (int col = 0; col < cols; ++col) {
            output[row * cols + col] = expf(input[row * cols + col] - max_val) / sum;
        }
    }
}

void printMatrix(float *matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::fixed << std::setprecision(4) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}
__global__ void softmaxCUDA(float *input, float *output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;

    extern __shared__ float shared_mem[];
    float *temp = shared_mem;              // 临时归约数组
    float *max_val = &shared_mem[blockDim.x]; // +1
    float *sum_val = &shared_mem[blockDim.x + 1]; // +1

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // 阶段1: 计算最大值
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += stride) {
        local_max = fmaxf(local_max, input[row*cols + i]);
    }
    temp[tid] = local_max;
    __syncthreads();

    // 归约最大值
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            temp[tid] = fmaxf(temp[tid], temp[tid + offset]);
        }
        __syncthreads();
    }
    if (tid == 0) *max_val = temp[0];
    __syncthreads();

    // 阶段2: 计算指数和（直接写入output）
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += stride) {
        float exp_val = expf(input[row*cols + i] - *max_val);
        output[row*cols + i] = exp_val;  // 暂存指数结果
        local_sum += exp_val;
    }
    temp[tid] = local_sum;
    __syncthreads();

    // 归约求和
    for (int offset = blockDim.x/2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            temp[tid] += temp[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) *sum_val = temp[0];
    __syncthreads();

    // 阶段3: 归一化
    for (int i = tid; i < cols; i += stride) {
        output[row*cols + i] /= *sum_val;
    }
}

void testnative() {
    int rows = 1024;// 扩大行数
    int cols = 1024;// 扩大列数
    int size = rows * cols * sizeof(float);

    // 动态生成输入数据
    float *h_input = (float *) malloc(size);
    for (int i = 0; i < rows * cols; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;// 生成 0 到 10 之间的随机数
    }

    float *h_output = (float *) malloc(size);
    float *h_output_cpu = (float *) malloc(size);

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((cols + dimBlock.x - 1) / dimBlock.x, (rows + dimBlock.y - 1) / dimBlock.y);

    cudaEvent_t start, stop;

    //softmaxKernelOptimized<<<dimGrid, dimBlock, shared_mem_size>>>(d_input, d_output, rows, cols);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //softmaxKernel
    softmaxKernel<<<dimGrid, dimBlock>>>(d_input, d_output, rows, cols);

    cudaEventRecord(stop);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "GPU Time: " << milliseconds << " ms" << std::endl;

    softmaxCPU(h_input, h_output_cpu, rows, cols);

    // 验证 GPU 和 CPU 结果是否一致
    bool isCorrect = true;
    for (int i = 0; i < rows * cols; ++i) {
        if (fabs(h_output[i] - h_output_cpu[i]) > 1e-5) {
            isCorrect = false;
            break;
        }
    }
    std::cout << "Verification: " << (isCorrect ? "PASS" : "FAIL") << std::endl;

    // 释放内存
    free(h_input);
    free(h_output);
    free(h_output_cpu);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
void testopt() {
    int rows = 1024;
    int cols = 1024;
    int size = rows * cols * sizeof(float);
    
    float* h_input = (float*)malloc(size);
    for (int i = 0; i < rows * cols; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f;
    }
    
    float* h_output = (float*)malloc(size);
    float* h_output_cpu = (float*)malloc(size);
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int threadsPerBlock = 256;
    int sharedMemSize = (2 + cols) * sizeof(float);
    // 执行配置
    dim3 block(256);    // 每个block 256 threads
    dim3 grid(rows);    // 每个row一个block
    size_t shared_size = (block.x + 2) * sizeof(float);
    cudaEventRecord(start);
    softmaxCUDA<<<grid, block, shared_size>>>(d_input, d_output, rows, cols);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "GPU Time: " << milliseconds << " ms" << std::endl;
    
    softmaxCPU(h_input, h_output_cpu, rows, cols);

    // 验证 GPU 和 CPU 结果是否一致
    bool isCorrect = true;
    for (int i = 0; i < rows * cols; ++i) {
        if (fabs(h_output[i] - h_output_cpu[i]) > 1e-4) {
            isCorrect = false;
            break;
        }
    }
    std::cout << "Verification: " << (isCorrect ? "PASS" : "FAIL") << std::endl;

    free(h_input);
    free(h_output);
    free(h_output_cpu);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}
int main() {
    testnative();
    //testSoftmax();
    testopt();
    return 0;
}
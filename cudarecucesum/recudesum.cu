#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s)\n", \
        __FILE__, __LINE__, err, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// CPU验证函数
double cpu_sum(const float* data, int n) {
    double sum = .0;
    for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(data[i]); // 提升为双精度
    }
    return sum;
}

// GPU归约核函数（优化版）
template <int blockSize>
__global__ void reduce_sum_kernel(const float* input, float* output, int n) {
    extern __shared__ double sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    
    sdata[tid] = 0.0f;
    
    // 网格跨步数据加载
    while (i < n) {
        sdata[tid] += input[i];
        if (i + blockDim.x < n) {
            sdata[tid] += input[i + blockDim.x];
        }
        i += gridSize;
    }
    __syncthreads();

    // 共享内存归约（展开循环）
    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  sdata[tid] += sdata[tid + 64];  __syncthreads(); }
    
    // 最后warp归约
    if (tid < 32) {
        volatile double* vsdata = sdata;
        vsdata[tid] += vsdata[tid + 32];
        vsdata[tid] += vsdata[tid + 16];
        vsdata[tid] += vsdata[tid + 8];
        vsdata[tid] += vsdata[tid + 4];
        vsdata[tid] += vsdata[tid + 2];
        vsdata[tid] += vsdata[tid + 1];
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// GPU归约入口函数
float gpu_sum(const float* d_input, int n) {
    const int maxBlocks = 512;
    const int threads = 256;
    float *d_temp;
    
    CHECK_CUDA(cudaMalloc(&d_temp, maxBlocks * sizeof(float)));
    
    // 第一层归约
    dim3 block(threads);
    dim3 grid(min((n + threads*2-1) / (threads*2), maxBlocks));
    reduce_sum_kernel<threads><<<grid, block, threads*sizeof(float)>>>(d_input, d_temp, n);
    
    // 后续归约直到只剩一个元素
    int remain = grid.x;
    while (remain > 1) {
        int newGrid = min((remain + threads*2-1) / (threads*2), maxBlocks);
        reduce_sum_kernel<threads><<<newGrid, block, threads*sizeof(float)>>>(d_temp, d_temp, remain);
        remain = newGrid;
    }
    
    // 拷贝最终结果
    float result;
    CHECK_CUDA(cudaMemcpy(&result, d_temp, sizeof(float), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_temp));
    return result;
}

int main() {
    const int testSizes[] = {10000, 100000, 1000000, 10000000, 100000000};  // 修复点
    const int numTests = sizeof(testSizes)/sizeof(testSizes[0]);
    
    for (int t = 0; t < numTests; ++t) {
        const int n = testSizes[t];
        const size_t size = n * sizeof(float);
        
        // 生成随机数据
        float *h_data = (float*)malloc(size);
        for (int i = 0; i < n; ++i) {
            h_data[i] = (float)rand()/(RAND_MAX/100.0f);  // 0~100随机数
        }
        
        // CPU计算
        clock_t cpu_start = clock();
        float cpu_result = cpu_sum(h_data, n);
        double cpu_time = (double)(clock() - cpu_start)/CLOCKS_PER_SEC;
        
        // GPU计算
        float *d_data;
        CHECK_CUDA(cudaMalloc(&d_data, size));
        CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
        
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));
        CHECK_CUDA(cudaEventRecord(start));
        
        float gpu_result = gpu_sum(d_data, n);
        
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float gpu_time;
        CHECK_CUDA(cudaEventElapsedTime(&gpu_time, start, stop));
        
        // 验证结果
        float diff = fabs(cpu_result - gpu_result);
        float tolerance = cpu_result * 1e-6;
        bool correct = (diff <= tolerance);
        
        printf("N=%d:\n  CPU: %.4f (%.3fs)\n  GPU: %.4f (%.3fms)\n  Diff: %.6f (%s)\n\n",
               n, cpu_result, cpu_time, gpu_result, gpu_time, diff, 
               correct ? "PASS" : "FAIL");
        
        // 清理
        free(h_data);
        CHECK_CUDA(cudaFree(d_data));
    }
    
    return 0;
}
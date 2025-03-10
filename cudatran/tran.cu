#include <iostream>

#include <chrono> // 用于计时

// CUDA 核函数：矩阵转置
__global__ void transposeKernel(const float *A, float *B, int M, int N) {
    // 计算当前线程在输出矩阵 B 中的索引
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 如果索引在矩阵范围内，执行转置操作
    if (row < M && col < N) {
        B[col * M + row] = A[row * N + col];
    }
}

// 初始化矩阵 A
void initMatrix(float *A, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i * N + j; // 填充矩阵 A
        }
    }
}

// 打印矩阵
void printMatrix(const float *A, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << A[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}
#define TILE_WIDTH 32 
__global__ void transposeKernelOptimized(const float *A, float *B, int M, int N) {
    // 定义共享内存块
    __shared__ float tile[TILE_WIDTH][TILE_WIDTH + 1]; // 避免共享内存 bank 冲突

    // 计算当前线程在输入矩阵 A 中的索引
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

    // 将数据从全局内存加载到共享内存
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }

    // 同步线程，确保所有线程都完成了数据加载
    __syncthreads();

    // 计算当前线程在输出矩阵 B 中的索引
    int x_transposed = blockIdx.y * TILE_WIDTH + threadIdx.x;
    int y_transposed = blockIdx.x * TILE_WIDTH + threadIdx.y;

    // 将数据从共享内存写回全局内存（转置）
    if (x_transposed < M && y_transposed < N) {
        B[y_transposed * M + x_transposed] = tile[threadIdx.x][threadIdx.y];
    }
}

bool verifyTranspose(const float *A, const float *B, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (B[j * M + i] != A[i * N + j]) {
                std::cerr << "Verification failed at B[" << j << "][" << i << "]: "
                          << "expected " << A[i * N + j] << ", got " << B[j * M + i] << std::endl;
                return false;
            }
        }
    }
    return true;
}

int main() {
    // 定义矩阵大小
    int M = 1024; // 行数
    int N = 1024; // 列数

    // 分配主机内存
    float *h_A = (float *)malloc(M * N * sizeof(float));
    float *h_B = (float *)malloc(N * M * sizeof(float));

    // 初始化矩阵 A
    initMatrix(h_A, M, N);

    // 打印矩阵 A（可选，对于大矩阵可以注释掉）
    // std::cout << "Matrix A:" << std::endl;
    // printMatrix(h_A, M, N);

    // 分配设备内存
    float *d_A, *d_B;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * M * sizeof(float));

    // 将数据从主机拷贝到设备
    cudaMemcpy(d_A, h_A, M * N * sizeof(float), cudaMemcpyHostToDevice);

    
    // 定义线程块和网格大小
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH); // 每个线程块的大小
    dim3 gridSize((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH); // 网格大小

    // 创建 CUDA 事件用于计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start);

    // 调用 CUDA 核函数
    //transposeKernel

    transposeKernelOptimized<<<gridSize, blockSize>>>(d_A, d_B, M, N);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop); // 等待核函数执行完成

    // 计算执行时间
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // 将结果从设备拷贝回主机
    cudaMemcpy(h_B, d_B, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印转置后的矩阵 B（可选，对于大矩阵可以注释掉）
    // std::cout << "Matrix B (Transpose of A):" << std::endl;
    // printMatrix(h_B, N, M);
     // 验证转置结果
    if (verifyTranspose(h_A, h_B, M, N)) {
        std::cout << "Transpose verification succeeded!" << std::endl;
    } else {
        std::cerr << "Transpose verification failed!" << std::endl;
    }
    // 释放 CUDA 事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);

    // 释放主机内存
    free(h_A);
    free(h_B);

    return 0;
}
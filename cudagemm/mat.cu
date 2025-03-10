#include <iostream>

#define TILE_WIDTH 4

__global__ void matrixKernel(float *dA, float *dB, float *dC, int M, int K, int N) {
    __shared__ float sa[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sb[TILE_WIDTH][TILE_WIDTH];
    // 获取线程的唯一ID
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // 初始化输出值
    float sum = 0.0f;
    //分段共享读取
    // 计算每个线程块需要迭代的次数
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; t++) {
        if (row < M && t * TILE_WIDTH + threadIdx.x < K) {
            sa[threadIdx.y][threadIdx.x] = dA[row * K + t * TILE_WIDTH + threadIdx.x];
        } else {
            sa[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_WIDTH + threadIdx.y < K) {
            sb[threadIdx.y][threadIdx.x] = dB[(t * TILE_WIDTH + threadIdx.y) * N + col];
        } else {
            sb[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();
        // 计算部分积
        for (int k = 0; k < TILE_WIDTH; ++k) { sum += sa[threadIdx.y][k] * sb[k][threadIdx.x]; }
        __syncthreads();
    }

    // 将结果写回全局内存
    if (row < M && col < N) dC[row * N + col] = sum;
}
//#define TILE_WIDTH 32  // 增大分块大小，提升数据复用率

// __global__ void optimizedMatrixMulv1(float *dA, float *dB, float *dC, int M, int K, int N) {
//     // 双缓冲共享内存，隐藏全局内存延迟
//     __shared__ float sa[2][TILE_WIDTH][TILE_WIDTH];
//     __shared__ float sb[2][TILE_WIDTH][TILE_WIDTH];

//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int row = blockIdx.y * TILE_WIDTH + ty;// 计算全局行坐标
//     int col = blockIdx.x * TILE_WIDTH + tx;// 计算全局列坐标

//     float sum = 0.0f;
//     int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

//     // 双缓冲索引：异步加载下一个分块的同时计算当前分块
//     int loadBuffer = 0;
//     int computeBuffer = 0;

//     for (int t = 0; t < numTiles; ++t) {
//         // 异步预加载下一个分块到另一个缓冲区-----------------------------
//         int nextTile = t + 1;
//         if (nextTile < numTiles) {
//             // 使用向量化加载（float4），减少全局内存事务次数
//             int loadK = nextTile * TILE_WIDTH + tx;
//             if (row < M && loadK < K) {
//                 float4 val = reinterpret_cast<float4 *>(&dA[row * K + loadK])[0];
//                 sa[1 - loadBuffer][ty][tx] = val.x;
//                 if (tx + 1 < TILE_WIDTH && loadK + 1 < K) sa[1 - loadBuffer][ty][tx + 1] = val.y;
//                 if (tx + 2 < TILE_WIDTH && loadK + 2 < K) sa[1 - loadBuffer][ty][tx + 2] = val.z;
//                 if (tx + 3 < TILE_WIDTH && loadK + 3 < K) sa[1 - loadBuffer][ty][tx + 3] = val.w;
//             } else {
//                 sa[1 - loadBuffer][ty][tx] = 0.0f;
//             }

//             int loadK_B = nextTile * TILE_WIDTH + ty;
//             if (loadK_B < K && col < N) {
//                 float4 val = reinterpret_cast<float4 *>(&dB[loadK_B * N + col])[0];
//                 sb[1 - loadBuffer][ty][tx] = val.x;
//                 if (ty + 1 < TILE_WIDTH && loadK_B + 1 < K) sb[1 - loadBuffer][ty + 1][tx] = val.y;
//                 if (ty + 2 < TILE_WIDTH && loadK_B + 2 < K) sb[1 - loadBuffer][ty + 2][tx] = val.z;
//                 if (ty + 3 < TILE_WIDTH && loadK_B + 3 < K) sb[1 - loadBuffer][ty + 3][tx] = val.w;
//             } else {
//                 sb[1 - loadBuffer][ty][tx] = 0.0f;
//             }
//         }

//         __syncthreads();// 等待当前分块加载完成

//         // 计算当前分块的内积（循环展开 + 寄存器缓存）-------------------
//         //#pragma unroll// 强制循环展开，减少分支开销
//         for (int k = 0; k < TILE_WIDTH; ++k) {
//             float a = sa[computeBuffer][ty][k];
//             float b = sb[computeBuffer][k][tx];
//             sum += a * b;
//         }

//         // 切换缓冲区索引
//         computeBuffer = 1 - computeBuffer;
//         loadBuffer = 1 - loadBuffer;
//         __syncthreads();// 确保计算完成后再覆盖缓冲区
//     }

//     // 写入结果（条件判断外提，减少分支）
//     if (row < M && col < N) { dC[row * N + col] = sum; }
// }

__global__ void optimizedMatrixMulv1(float *dA, float *dB, float *dC, int M, int K, int N) {
    __shared__ float sa[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float sb[2][TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float sum = 0.0f;
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    int loadBuffer = 0;
    int computeBuffer = 1; // 先计算缓冲区 1

    for (int t = 0; t < numTiles; ++t) {
        // 预加载 tile
        int tileK = t * TILE_WIDTH + tx;
        if (row < M && tileK < K) {
            sa[loadBuffer][ty][tx] = dA[row * K + tileK];
        } else {
            sa[loadBuffer][ty][tx] = 0.0f;
        }

        int tileK_B = t * TILE_WIDTH + ty;
        if (tileK_B < K && col < N) {
            sb[loadBuffer][ty][tx] = dB[tileK_B * N + col];
        } else {
            sb[loadBuffer][ty][tx] = 0.0f;
        }

        __syncthreads(); // 保证数据加载完毕

        // 计算当前 tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += sa[computeBuffer][ty][k] * sb[computeBuffer][k][tx];
        }

        __syncthreads(); // 保证所有线程计算完毕

        // 切换缓冲区索引
        computeBuffer = loadBuffer;
        loadBuffer = 1 - loadBuffer;
    }

    // 写回结果
    if (row < M && col < N) {
        dC[row * N + col] = sum;
    }
}

__global__ void optimizedMatrixMulv2(float *dA, float *dB, float *dC, int M, int K, int N) {
    __shared__ float sa[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float sb[2][TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;// 全局行索引
    int col = blockIdx.x * TILE_WIDTH + tx;// 全局列索引

    float sum = 0.0f;
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    // 初始加载第一个分块到缓冲区0
    int loadBuffer = 0;
    int computeBuffer = 0;

    // 加载第一个分块到共享内存
    int loadK = 0 * TILE_WIDTH + tx;// t=0
    if (row < M && loadK < K) {
        sa[loadBuffer][ty][tx] = dA[row * K + loadK];
    } else {
        sa[loadBuffer][ty][tx] = 0.0f;
    }

    int loadK_B = 0 * TILE_WIDTH + ty;// t=0
    if (loadK_B < K && col < N) {
        sb[loadBuffer][ty][tx] = dB[loadK_B * N + col];
    } else {
        sb[loadBuffer][ty][tx] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < numTiles; ++t) {
        // 异步预加载下一个分块到另一个缓冲区
        int nextTile = t + 1;
        if (nextTile < numTiles) {
            int nextLoadK = nextTile * TILE_WIDTH + tx;
            if (row < M && nextLoadK < K) {
                sa[1 - loadBuffer][ty][tx] = dA[row * K + nextLoadK];
            } else {
                sa[1 - loadBuffer][ty][tx] = 0.0f;
            }

            int nextLoadK_B = nextTile * TILE_WIDTH + ty;
            if (nextLoadK_B < K && col < N) {
                sb[1 - loadBuffer][ty][tx] = dB[nextLoadK_B * N + col];
            } else {
                sb[1 - loadBuffer][ty][tx] = 0.0f;
            }
        }
        __syncthreads();

        // 计算当前分块的内积
        for (int k = 0; k < TILE_WIDTH; ++k) { sum += sa[computeBuffer][ty][k] * sb[computeBuffer][k][tx]; }

        // 切换缓冲区
        computeBuffer = 1 - computeBuffer;
        loadBuffer = 1 - loadBuffer;
        __syncthreads();
    }

    // 写入结果
    if (row < M && col < N) { dC[row * N + col] = sum; }
}
__global__ void optimizedMatrixMulv3(float *dA, float *dB, float *dC, int M, int K, int N) {
    __shared__ float sa[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float sb[2][TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;// 全局行索引
    int col = blockIdx.x * TILE_WIDTH + tx;// 全局列索引

    float sum = 0.0f;
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    // 初始化双缓冲索引
    int loadBuffer = 0;
    int computeBuffer = 0;

    // 初始加载第一个分块到缓冲区0
    for (int t = 0; t < numTiles; ++t) {
        int loadK = t * TILE_WIDTH + tx;
        if (row < M && loadK < K) {
            sa[loadBuffer][ty][tx] = dA[row * K + loadK];
        } else {
            sa[loadBuffer][ty][tx] = 0.0f;
        }

        int loadK_B = t * TILE_WIDTH + ty;
        if (loadK_B < K && col < N) {
            sb[loadBuffer][ty][tx] = dB[loadK_B * N + col];
        } else {
            sb[loadBuffer][ty][tx] = 0.0f;
        }
        __syncthreads();

        // 计算当前分块的内积
        for (int k = 0; k < TILE_WIDTH; ++k) { sum += sa[computeBuffer][ty][k] * sb[computeBuffer][k][tx]; }
        __syncthreads();

        // 切换缓冲区
        computeBuffer = 1 - computeBuffer;
        loadBuffer = 1 - loadBuffer;
    }

    // 写入结果
    if (row < M && col < N) { dC[row * N + col] = sum; }
}

__global__ void mymatmul(float* da,float* db,float* dc,int M,int K,int N){
    __shared__ float sa[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sb[TILE_WIDTH][TILE_WIDTH];

    int numTiles = (K+TILE_WIDTH-1)/TILE_WIDTH;
    int row = threadIdx.y+blockDim.y*blockIdx.y;
    int col = threadIdx.x+blockDim.x*blockIdx.x;

    float sum=0;
    for(int i=0;i<numTiles;++i){
        //读取a的共享内存
        if(row < M && i*TILE_WIDTH+threadIdx.x<K){
            sa[threadIdx.y][threadIdx.x] = da[row*K+i*TILE_WIDTH+threadIdx.x];
        }else{
            sa[threadIdx.y][threadIdx.x]=0.0f;
        }

        if(col<N && i*TILE_WIDTH+threadIdx.y<K){
            sb[threadIdx.y][threadIdx.x]=db[(i*TILE_WIDTH+threadIdx.y)*N+col];
        }else{
            sb[threadIdx.y][threadIdx.x]=0.0f;
        }
        __syncthreads();

        for(int j=0;j<TILE_WIDTH;++j){
            sum+=sa[threadIdx.y][j]*sb[j][threadIdx.x];
        }
        __syncthreads();
    }

    if(row<M && col<N){
        dc[row*N+col] = sum;
    }

}
int main() {
    // 测试数据
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int M = 1024;
    int K = 1024;
    int N = 1024;

    // 在主机上分配内存
    h_A = (float *) malloc(M * K * sizeof(float));
    h_B = (float *) malloc(K * N * sizeof(float));
    h_C = (float *) malloc(M * N * sizeof(float));

    // 初始化输入矩阵 A 和 B
    for (int i = 0; i < M * K; i++) { h_A[i] = rand() / (float) RAND_MAX; }
    for (int i = 0; i < K * N; i++) { h_B[i] = rand() / (float) RAND_MAX; }

    // 在设备上分配内存
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // 将输入矩阵复制到设备内存
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块和网格大小
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start);

    // 调用矩阵乘法核函数
    //matrixKernel
    //optimizedMatrixMulv2
    mymatmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

    // 记录结束时间
    cudaEventRecord(stop);

    // 等待核函数执行完成
    cudaEventSynchronize(stop);

    // 计算执行时间（毫秒）
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 输出执行时间
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // 检查 CUDA 错误
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "CUDA kernel execution failed!" << std::endl;
        return -1;
    }

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    bool allPassed = true;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            // 期望结果
            float expected = 0.0f;
            for (int k = 0; k < K; k++) { expected += h_A[i * K + k] * h_B[k * N + j]; }

            // 比较实际结果和期望结果
            if (std::abs(h_C[i * N + j] - expected) > 1e-4) {
                allPassed = false;
                std::cout << "Mismatch at (" << i << ", " << j << "): " << h_C[i * N + j] << " vs " << expected
                          << std::endl;
                break;
            }
        }
        if (!allPassed) break;
    }

    if (allPassed) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }

    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
// int main() {
//     // 定义测试矩阵维度 (M x K) * (K x N) = (M x N)
//     const int M = 2, K = 3, N = 2;

//     // 在主机端分配内存并初始化矩阵
//     float h_A[M][K] = {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}};  // 2x3 全1矩阵
//     float h_B[K][N] = {{2.0f, 2.0f}, {2.0f, 2.0f}, {2.0f, 2.0f}};// 3x2 全2矩阵
//     float h_C[M][N] = {{0.0f}};

//     // 在设备端分配内存
//     float *dA, *dB, *dC;
//     cudaMalloc(&dA, M * K * sizeof(float));
//     cudaMalloc(&dB, K * N * sizeof(float));
//     cudaMalloc(&dC, M * N * sizeof(float));

//     // 拷贝数据到设备端
//     cudaMemcpy(dA, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(dB, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

//     // 调用 CUDA 内核
//     dim3 block(TILE_WIDTH, TILE_WIDTH);
//     dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
//     optimizedMatrixMul<<<grid, block>>>(dA, dB, dC, M, K, N);

//     // 检查内核错误
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("Kernel error: %s\n", cudaGetErrorString(err));
//         return -1;
//     }

//     // 拷贝结果回主机端
//     cudaMemcpy(h_C, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

//     // 打印输入和结果
//     printf("Input Matrix A (2x3):\n");
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < K; ++j) printf("%.1f ", h_A[i][j]);
//         printf("\n");
//     }

//     printf("\nInput Matrix B (3x2):\n");
//     for (int i = 0; i < K; ++i) {
//         for (int j = 0; j < N; ++j) printf("%.1f ", h_B[i][j]);
//         printf("\n");
//     }

//     printf("\nOutput Matrix C (2x2):\n");
//     for (int i = 0; i < M; ++i) {
//         for (int j = 0; j < N; ++j) printf("%.1f ", h_C[i][j]);
//         printf("\n");
//     }

//     // 释放内存
//     cudaFree(dA);
//     cudaFree(dB);
//     cudaFree(dC);
//     return 0;
// }

#include <iostream>

#define TILE_WIDTH 32 

__global__ void matrixKernel(float *dA, float *dB, float *dC, int M, int K, int N){
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
  for(int t=0;t<numTiles;t++){
    if(row < M && t*TILE_WIDTH+threadIdx.x < K){
      sa[threadIdx.y][threadIdx.x]=dA[row*K+t*TILE_WIDTH+threadIdx.x];
    }else{
      sa[threadIdx.y][threadIdx.x]=0.0f;
    }

    if(col < N && t*TILE_WIDTH+threadIdx.y < K){
      sb[threadIdx.y][threadIdx.x]=dB[(t*TILE_WIDTH+threadIdx.y)*N+col];
    }else{
      sb[threadIdx.y][threadIdx.x]=0.0f;
    }
    __syncthreads(); 
    // 计算部分积
    for (int k = 0; k < TILE_WIDTH; ++k) {
      sum += sa[threadIdx.y][k] * sb[k][threadIdx.x];
    }
    __syncthreads(); 
	}

  // 将结果写回全局内存
  if (row < M && col < N)
    dC[row * N + col] = sum;
}

int main(int argc, char **argv) {
    // 测试数据
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int M = 4;
    int K = 4;
    int N = 4;

    // 在主机上分配内存
    h_A = (float *)malloc(M * K * sizeof(float));
    h_B = (float *)malloc(K * N * sizeof(float));
    h_C = (float *)malloc(M * N * sizeof(float));

    // 初始化输入矩阵 A 和 B
    for (int i = 0; i < M * K; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = rand() / (float)RAND_MAX;
    }

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

    // 调用矩阵乘法核函数
    matrixKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);

    // 将结果从设备内存复制回主机内存
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    bool allPassed = true;
    for (int i = 0; i < M * N; i++) {
        // 期望结果
        float expected = 0.0f;
        for (int j = 0; j < K; j++) {
            expected += h_A[i / N * K + j] * h_B[j * N + i % N];
        }

        // 比较实际结果和期望结果
        if (std::abs(h_C[i] - expected) > 1e-6) {
            allPassed = false;
            break;
        }
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
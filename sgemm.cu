#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

template <int _m, int _n, int _k = 1>
struct Layout {
  static constexpr int kM = _m;
  static constexpr int kN = _n;
  static constexpr int kK = _k;
};

void cpuSgemm(
    float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}

template <typename CTAShape, typename ThreadShape>
__global__ void mySgemmV1Aligned(
    float * __restrict__ devA, float * __restrict__ devB, float * __restrict__ devC,
    const int M, const int N, const int K) {
        const int bx = blockIdx.x;
        const int by = blockIdx.y;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        const int tid = ty * blockDim.x + tx;

        __shared__ float tile_a[CTAShape::kM][CTAShape::kK];
        __shared__ float tile_b[CTAShape::kK][CTAShape::kN];

        float regC[ThreadShape::kM][CTAShape::kN] = {0.f};

        const int load_a_smem_m = (tid / 2);
        const int load_a_smem_k = (tid % 2) << 2;
        const int load_b_smem_k = (tid / 32);
        const int load_b_smem_n = (tid % 32) << 2;

        const int load_a_gmem_m = by * CTAShape::kM + load_a_smem_m;
        const int load_b_gmem_n = bx * CTAShape::kN + load_b_smem_n;
        
        for( int bk = 0; bk < (K + CTAShape::kK - 1) / CTAShape::kK; bk++){
            int load_a_gmem_k = bk * CTAShape::kK + load_a_smem_k;
            int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
            FLOAT4(tile_a[load_a_smem_m][load_a_smem_k]) = FLOAT4(devA[load_a_gmem_addr]);
            int load_b_gmem_k = bk * CTAShape::kK + load_b_smem_k;
            int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
            FLOAT4(tile_b[load_b_smem_k][load_b_smem_n]) = FLOAT4(devB[load_b_gmem_addr]);
            __syncthreads();

            #pragma unroll
            for( int k = 0; k < CTAShape::kK; k++){
                #pragma unroll
                for(int m = 0; m < ThreadShape::kM; m++){
                    #pragma unroll 
                    for (int n = 0; n < ThreadShape::kN; n++){
                        int comp_c_smem_m = ty * ThreadShape::kM + m;
                        int comp_c_smem_n = tx * ThreadShape::kN + n;
                        regC[m][n] += tile_a[comp_c_smem_m][k] * tile_b[k][comp_c_smem_n];
                    }
                }
            }
            __syncthreads(); 
        }
        
        #pragma unroll
        for(int m = 0; m < ThreadShape::kM; m++){
            int store_c_gmem_m = by * CTAShape::kM + ty * ThreadShape::kM + m;
            #pragma unroll 
            for (int n = 0; n < ThreadShape::kN; n += 4){
                int store_c_gmem_n = bx * CTAShape::kN + tx * ThreadShape::kN + n;
                int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
                FLOAT4(devC[store_c_gmem_addr]) = FLOAT4(regC[m][n]);
            }
        }
}
/*
template<typename CTAShape, typename ThreadShape>
__global__ void mySgemmV2Aligned(
    float * __restrict__ devA, float * __restrict__ devB, float * __restrict__ devC,
    const int M, const int N, const int K) {

        const int bx = blockIdx.x;
        const int by = blockIdx.y;
        const int tx = threadIdx.x;
        const int ty = threadIdx.y;

        const int tid = ty * blockDim.x + tx;

        __shared__ float tileA[2][CTAShape::kM][CTAShape::kK];
        __shared__ float tileB[2][CTAShape::kK][CTAShape::kN];

        float resC[ThreadShape::kM][ThreadShape::kN] = {0.0};

        int load_a_smem_m = tid >> 1;
        int load_a_smem_k = (tid & 1) << 2;
        int load_b_smem_k = tid >> 5;
        int load_b_smem_n = (tid & 31) << 2;

        int load_a_gmem_m = by * CTAShape::kM + load_a_smem_m;
        int load_b_gmem_n = bx * CTAShape::kN + load_b_smem_n;

        {
            int load_a_gmem_k = load_a_smem_k;
            int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
            FLOAT4(tileA[0][load_a_smem_m][load_a_smem_k]) = FLOAT4(devA[load_a_gmem_addr]);
            int load_b_gmem_k = load_b_smem_k;
            int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
            FLOAT4(tileB[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(devB[load_b_gmem_addr]);
            
        }
        __syncthreads();

        for(int bk = 1; bk < (K + CTAShape::kK -1) / CTAShape::kK; bk++){
            int smem_sel = (bk - 1) & 1;
            int smem_sel_next = bk & 1;

            int load_a_gmem_k = bk * CTAShape::kK + load_a_smem_k;
            int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
            int load_b_gmem_k = bk * CTAShape::kK + load_b_smem_k;
            int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
            
            #pragma unroll
            for(int k = 0; k < CTAShape::kK; k++){
                #pragma unroll
                for(int m =0; m < ThreadShape::kM; m++){
                    #pragma unroll
                    for(int n =0; n < ThreadShape::kN; n++){
                        int comp_a_smem_m = ty * ThreadShape::kM + m;
                        int comp_b_smem_n = tx * ThreadShape::kN + n;
                        resC[m][n] += tileA[smem_sel][comp_a_smem_m][k] * tileB[smem_sel][k][comp_b_smem_n];
                    }
                }
            }
            FLOAT4(tileA[smem_sel_next][load_a_smem_m][load_a_smem_k]) = FLOAT4(devA[load_a_gmem_addr]);
            FLOAT4(tileB[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(devB[load_b_gmem_addr]);
            
            __syncthreads();
        }

        #pragma unroll
        for(int k = 0; k < CTAShape::kK; k++){
            #pragma unroll
            for(int m =0; m < ThreadShape::kM; m++){
                #pragma unroll
                for(int n =0; n < ThreadShape::kN; n++){
                    int comp_a_smem_m = ty * ThreadShape::kM + m;
                    int comp_b_smem_n = tx * ThreadShape::kN + n;
                    resC[m][n] += tileA[1][comp_a_smem_m][k] * tileB[1][k][comp_b_smem_n];
                }
            }
        }

        #pragma unroll
        for(int m =0; m < ThreadShape::kM; m++){
            int store_c_gmem_m = by * CTAShape::kM + ty * ThreadShape::kM + m;
            #pragma unroll
            for(int n =0; n < ThreadShape::kN; n+=4){
                int store_c_gmem_n = bx * CTAShape::kN + tx * ThreadShape::kN + n;
                int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
                FLOAT4(devC[store_c_gmem_addr]) = FLOAT4(resC[m][n]);
            }
        }
}
*/

template<typename CTAShape, typename ThreadShape>
__global__ void mySgemmV3Aligned(float* __restrict__ devA, 
                                float* __restrict__ devB, 
                                float* __restrict__ devC,
                                const int M, const int N, const int K){
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int tid = ty * blockDim.x + tx;
    // double buffering
    __shared__ float tileA[2][CTAShape::kM][CTAShape::kK];
    __shared__ float tileB[2][CTAShape::kK][CTAShape::kN];

    float regC[ThreadShape::kM][ThreadShape::kN] = {0.f};
    const int load_a_smem_m = (tid / 2); 
    const int load_a_smem_k = (tid % 2) << 2;
    const int load_b_smem_k = (tid / 32);
    const int load_b_smem_n = (tid % 32) << 2;
    // 0-th load Gmem-->Smem
    const int load_a_gmem_m = by * CTAShape::kM + load_a_smem_m;
    const int load_b_gmem_n = bx * CTAShape::kN + load_b_smem_n;
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        FLOAT4(tileA[0][load_a_smem_m][load_a_smem_k]) = FLOAT4(devA[load_a_gmem_addr]);
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        FLOAT4(tileB[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(devB[load_b_gmem_addr]);
    }
    int smem_sel, smem_sel_next;
    // i-th compute with (i+1)th load
    for(int bk = 1; bk < (CTAShape::kK + K -1) / CTAShape::kK; bk++){
        __syncthreads();
        smem_sel = (bk - 1) & 1;
        smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * CTAShape::kK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * CTAShape::kK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        // i-th compute
        #pragma unroll
        for( int k = 0; k < CTAShape::kK; k++){
            #pragma unroll
            for(int m = 0; m < ThreadShape::kM; m++){
                #pragma unroll 
                for (int n = 0; n < ThreadShape::kN; n++){
                    int comp_c_smem_m = ty * ThreadShape::kM + m;
                    int comp_c_smem_n = tx * ThreadShape::kN + n;
                    regC[m][n] += tileA[smem_sel][comp_c_smem_m][k] * tileB[smem_sel][k][comp_c_smem_n];
                }
            }
        }
        // (i+1)th load Gmem-->Smem
        FLOAT4(tileA[smem_sel_next][load_a_smem_m][load_a_smem_k]) = FLOAT4(devA[load_a_gmem_addr]);
        FLOAT4(tileB[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(devB[load_b_gmem_addr]);
    }
    __syncthreads();
    // the last compute
    #pragma unroll
    for( int k = 0; k < CTAShape::kK; k++){
        #pragma unroll
        for(int m = 0; m < ThreadShape::kM; m++){
            #pragma unroll 
            for (int n = 0; n < ThreadShape::kN; n++){
                int comp_c_smem_m = ty * ThreadShape::kM + m;
                int comp_c_smem_n = tx * ThreadShape::kN + n;
                regC[m][n] += tileA[smem_sel_next][comp_c_smem_m][k] * tileB[smem_sel_next][k][comp_c_smem_n];
            }
        }
    }
    // load Smem-->Gmem
    #pragma unroll
    for(int m = 0; m < ThreadShape::kM; m++){
        int store_c_gmem_m = by * CTAShape::kM + ty * ThreadShape::kM + m;
        #pragma unroll 
        for (int n = 0; n < ThreadShape::kN; n += 4){
            int store_c_gmem_n = bx * CTAShape::kN + tx * ThreadShape::kN + n;
            int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            FLOAT4(devC[store_c_gmem_addr]) = FLOAT4(regC[m][n]);
        }
    }
}

template<typename CTAShape, typename ThreadShape>
__global__ void mySgemmV4Aligned(float* __restrict__ devA, 
                                float* __restrict__ devB, 
                                float* __restrict__ devC,
                                const int M, const int N, const int K){
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * blockDim.x + tx;

    __shared__ float tileA[2][CTAShape::kM][CTAShape::kK];
    __shared__ float tileB[2][CTAShape::kK][CTAShape::kN];
    float regC[ThreadShape::kM][ThreadShape::kN]={0.f};

    const int load_a_smem_m = (tid / 2);
    const int load_a_smem_k = (tid & 1) * 4;
    const int load_b_smem_k = (tid / 32);
    const int load_b_smem_n = (tid & 31) * 4;

    const int load_a_gmem_m = by * CTAShape::kM + load_a_smem_m;
    const int load_b_gmem_n = bx * CTAShape::kN + load_b_smem_n;
    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        FLOAT4(tileA[0][load_a_smem_m][load_a_smem_k]) = FLOAT4(devA[load_a_gmem_addr]);
        FLOAT4(tileB[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(devB[load_b_gmem_addr]);
    }
    int smem_sel, smem_sel_next;
    for(int bk =1 ; bk < (CTAShape::kK + K - 1)/ CTAShape::kK; bk++) {
        __syncthreads();

        smem_sel = (bk - 1) & 1;
        smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * CTAShape::kK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * CTAShape::kK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;

        #pragma unroll
        for(int k = 0; k < CTAShape::kK; k++) {
            #pragma unroll
            for(int m = 0; m < ThreadShape::kM; m++){
                #pragma unroll
                for(int n = 0; n < ThreadShape::kN; n++){
                    int comp_c_smem_m = ty * ThreadShape::kM + m;
                    int comp_c_smem_n = tx * ThreadShape::kN + n;
                    regC[m][n] += tileA[smem_sel][comp_c_smem_m][k] * tileB[smem_sel][k][comp_c_smem_n];
                }
            }
        }

        FLOAT4(tileA[smem_sel_next][load_a_smem_m][load_a_smem_k]) = FLOAT4(devA[load_a_gmem_addr]);
        FLOAT4(tileB[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(devB[load_b_gmem_addr]);
    }
    __syncthreads();

    #pragma unroll
    for(int k = 0; k < CTAShape::kK; k++) {
        #pragma unroll
        for(int m = 0; m < ThreadShape::kM; m++){
            #pragma unroll
            for(int n = 0; n < ThreadShape::kN; n++){
                int comp_c_smem_m = ty * ThreadShape::kM + m;
                int comp_c_smem_n = tx * ThreadShape::kN + n;
                regC[m][n] += tileA[smem_sel_next][comp_c_smem_m][k] * tileB[smem_sel_next][k][comp_c_smem_n];
            }
        }
    }

    #pragma unroll
    for( int m = 0; m < ThreadShape::kM; m++){
        int store_c_gmem_m = by * CTAShape::kM + ty * ThreadShape::kM + m;
        #pragma unroll
        for(int n = 0; n < ThreadShape::kN; n+=4){
            int store_c_gmem_n = bx * CTAShape::kN + tx * ThreadShape::kN + n;
            int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
            FLOAT4(devC[store_c_gmem_addr]) = FLOAT4(regC[m][n]);
        }
    }

}

int main() {
  
    const int M = 512, N = 512, K = 512;

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    using CTAShape = Layout<128, 128, 8>;
    using ThreadShape = Layout<8, 8, 1>;
    
    dim3 blockDim(CTAShape::kN / ThreadShape::kN, CTAShape::kM / ThreadShape::kM);
    dim3 gridDim((N + CTAShape::kN - 1) / CTAShape::kN, (M + CTAShape::kM - 1) / CTAShape::kM);
    
    mySgemmV4Aligned<CTAShape, ThreadShape><<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    printf("max_error: %f\n", max_error);

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);
    

    return 0;
}

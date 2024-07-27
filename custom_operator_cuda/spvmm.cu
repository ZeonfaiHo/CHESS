#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int BLOCK_N = 32;
const int PARALLEL_K = 32;
const int BLOCK_K = 4096;

template <typename scalar_t>
__global__ void spvmm_kernel(
    scalar_t* __restrict__ output, 
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ W,
    const int N,
    const int K
) {

    __shared__ scalar_t shared_x[BLOCK_K];
    __shared__ scalar_t shared_output[PARALLEL_K][BLOCK_N];
    
    shared_output[threadIdx.y][threadIdx.x] = 0;

    if (blockIdx.x * blockDim.x >= N) {
        return;
    }

    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    int total_threads_in_block = blockDim.x * blockDim.y;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        
        int k1_upperbound = k0 + BLOCK_K > K ? K - k0 : BLOCK_K;

        for (int k1 = thread_id; k1 < k1_upperbound; k1 += total_threads_in_block) {
            shared_x[k1] = x[k0 + k1];
        }

        __syncthreads();

        if (n < N) {
            for (int k1 = threadIdx.y; k1 < k1_upperbound; k1 += PARALLEL_K) {
                if (shared_x[k1] != 0.0) {
                        shared_output[threadIdx.y][threadIdx.x] += shared_x[k1] * W[(k0 + k1) * N + n];
                }
            }
        }
        
        __syncthreads();
    }

    for (int boundary = PARALLEL_K / 2; boundary; boundary >>= 1) {
        if (threadIdx.y < boundary) {
            shared_output[threadIdx.y][threadIdx.x] += shared_output[threadIdx.y + boundary][threadIdx.x];
        }

        __syncthreads();
    }

    if (threadIdx.y == 0) {
        output[n] = shared_output[0][threadIdx.x];
    }
}


torch::Tensor spvmm_launch(torch::Tensor x, torch::Tensor W)
{
    CHECK_INPUT(x);
    CHECK_INPUT(W);
    TORCH_CHECK(x.size(0) == 1 && x.size(1) == W.size(0), "incorrect input shapes")
    TORCH_CHECK(x.scalar_type() == W.scalar_type(), "inconsistent input dtype")
    TORCH_CHECK(x.device() == W.device(), "inputs not on same device")

    torch::Tensor output = torch::empty({1, W.size(1)}, x.options());

    const int n_blocks = (W.size(1) + BLOCK_N - 1) / BLOCK_N;

    if (x.scalar_type() == torch::kBFloat16) {
        AT_DISPATCH_REDUCED_FLOATING_TYPES(
            x.scalar_type(),
            "spvmm_launch",
            ([&] {
                spvmm_kernel<<<n_blocks, dim3(BLOCK_N, PARALLEL_K)>>>(
                    output.data_ptr<scalar_t>(),
                    x.data_ptr<scalar_t>(),
                    W.data_ptr<scalar_t>(),
                    W.size(1),
                    x.size(1)
                );
            })
        );
    } else if (x.scalar_type() == torch::kFloat) {
        AT_DISPATCH_FLOATING_TYPES(
            x.scalar_type(),
            "spvmm_launch",
            ([&] {
                spvmm_kernel<<<n_blocks, dim3(BLOCK_N, PARALLEL_K)>>>(
                    output.data_ptr<scalar_t>(),
                    x.data_ptr<scalar_t>(),
                    W.data_ptr<scalar_t>(),
                    W.size(1),
                    x.size(1)
                );
            })
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("launch", &spvmm_launch, "launch SpVMM (CUDA)");
}
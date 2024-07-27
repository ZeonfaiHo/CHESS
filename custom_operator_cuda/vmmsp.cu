#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

const int BLOCK_N = 32;
const int PARALLEL_K = 1024;
const int BLOCK_K = 4096;

template <typename scalar_t>
__global__ void vmmsp_kernel(
    scalar_t* __restrict__ output, 
    const scalar_t* __restrict__ x, 
    const scalar_t* __restrict__ W,
    const scalar_t* __restrict__ mask,
    const int N,
    const int K
) {
    __shared__ scalar_t shared_x[BLOCK_K];
    __shared__ scalar_t shared_out[PARALLEL_K];

    int n_begin = blockIdx.x * BLOCK_N;
    int n_end = (blockIdx.x + 1) * BLOCK_N > N ? N : (blockIdx.x + 1) * BLOCK_N;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        int k1_upperbound = k0 + BLOCK_K > K ? K - k0 : BLOCK_K;
        for (int k1 = threadIdx.x; k1 < k1_upperbound; k1 += PARALLEL_K) {
            shared_x[k1] = x[k0 + k1];
        }
        for (int n = n_begin; n < n_end; n++) {
            scalar_t mask_scalar = mask[n];
            if (mask_scalar) {
                shared_out[threadIdx.x] = 0.0;
                for (int k1 = threadIdx.x; k1 < k1_upperbound; k1 += PARALLEL_K) {
                    shared_out[threadIdx.x] += shared_x[k1] * W[n * K + k0 + k1];
                }

                __syncthreads();

                for (int boundary = PARALLEL_K / 2; boundary; boundary >>= 1) {
                    if (threadIdx.x < boundary) {
                        shared_out[threadIdx.x] += shared_out[threadIdx.x + boundary];
                    }

                    __syncthreads();
                }

                output[n] += shared_out[0] * mask_scalar;
            } 
        }
    }
}


torch::Tensor vmmsp_launch(torch::Tensor x, torch::Tensor W, torch::Tensor mask)
{
    CHECK_INPUT(x);
    CHECK_INPUT(W);
    CHECK_INPUT(mask);
    TORCH_CHECK(x.size(0) == 1 && x.size(1) == W.size(1) && W.size(0) == mask.size(1), "incorrect input shapes")
    TORCH_CHECK(x.scalar_type() == W.scalar_type() && x.scalar_type() == mask.scalar_type(), "inconsistent input dtype")
    TORCH_CHECK(x.device() == W.device() && x.device() == mask.device(), "inputs not on same device")

    torch::Tensor output = torch::zeros({1, W.size(0)}, x.options());

    const int n_blocks = (W.size(0) + BLOCK_N - 1) / BLOCK_N;

    if (x.scalar_type() == torch::kBFloat16) {
        AT_DISPATCH_REDUCED_FLOATING_TYPES(
            x.scalar_type(),
            "vmmsp_launch",
            ([&] {
                vmmsp_kernel<<<n_blocks, PARALLEL_K>>>(
                    output.data_ptr<scalar_t>(),
                    x.data_ptr<scalar_t>(),
                    W.data_ptr<scalar_t>(),
                    mask.data_ptr<scalar_t>(),
                    W.size(0),
                    x.size(1)
                );
            })
        );
    } else if (x.scalar_type() == torch::kFloat) {
        AT_DISPATCH_FLOATING_TYPES(
            x.scalar_type(),
            "vmmsp_launch",
            ([&] {
                vmmsp_kernel<<<n_blocks, PARALLEL_K>>>(
                    output.data_ptr<scalar_t>(),
                    x.data_ptr<scalar_t>(),
                    W.data_ptr<scalar_t>(),
                    mask.data_ptr<scalar_t>(),
                    W.size(0),
                    x.size(1)
                );
            })
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("launch", &vmmsp_launch, "launch VMMSp (CUDA)");
}
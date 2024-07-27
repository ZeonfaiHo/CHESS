#include <torch/extension.h>
#include <omp.h>
#include <immintrin.h>

const int BLOCK_N = 64;

void spvmm_fp32(
    float* __restrict__ output, 
    const float* __restrict__ x, 
    const float* __restrict__ W,
    const int N,
    const int K
) {
    #pragma omp parallel for
    for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
        int n1_upperbound = n0 + BLOCK_N > N ? N - n0 : BLOCK_N;
        if (n1_upperbound == BLOCK_N) {
            for (int k = 0; k < K; k++) {
                if (x[k] != 0.0f) {
                    __m256 x_k_vec = _mm256_set1_ps(x[k]); // Broadcast x[k] to all elements of the vector

                    for (int n1 = 0; n1 < n1_upperbound; n1 += 8) {
                        // Load 8 elements from output and W
                        __m256 output_vec = _mm256_loadu_ps(&output[n0 + n1]);
                        __m256 W_vec = _mm256_loadu_ps(&W[k * N + n0 + n1]);

                        // Perform the multiplication and addition
                        __m256 result_vec = _mm256_fmadd_ps(x_k_vec, W_vec, output_vec);

                        // Store the result back to output
                        _mm256_storeu_ps(&output[n0 + n1], result_vec);
                    }
                }
            }
        }
    }
}


torch::Tensor spvmm_launch(torch::Tensor x, torch::Tensor W)
{
    TORCH_CHECK(x.is_contiguous() && W.is_contiguous(), 'inputs must be contiguous')
    TORCH_CHECK(x.dim() == 2 && W.dim() == 2 && x.size(0) == 1 && x.size(1) == W.size(0), "incorrect input shapes")
    TORCH_CHECK(x.scalar_type() == W.scalar_type(), "inconsistent input dtype")
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only support float32")
    TORCH_CHECK(x.device().type() == torch::kCPU && W.device().type() == torch::kCPU, "only support CPU")

    torch::Tensor output = torch::zeros({1, W.size(1)}, x.options());


    if (x.scalar_type() == torch::kFloat32) {
        spvmm_fp32(
            output.data_ptr<float>(),
            x.data_ptr<float>(),
            W.data_ptr<float>(),
            W.size(1),
            x.size(1)
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch", &spvmm_launch, "launch SpVMM (CPU)");
}
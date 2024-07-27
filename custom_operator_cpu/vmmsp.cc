#include <torch/extension.h>
#include <omp.h>
#include <immintrin.h>

const int BLOCK_N = 256;

void vmmsp_fp32(float* output, float* x, float* W, float* mask, int N, int K) {
    #pragma omp parallel for
    for (int n0 = 0; n0 < N; n0 += BLOCK_N) {
        int n1_upperbound = n0 + BLOCK_N > N ? N - n0 : BLOCK_N;
        for (int n1 = 0; n1 < n1_upperbound; n1++) {
            float mask_scalar = mask[n0 + n1];
            if (mask_scalar != 0.0) {
                int tmp = (n0 + n1) * K;
                __m256 accum = _mm256_setzero_ps(); 

                for (int k = 0; k < K; k += 8) {
                    __m256 w = _mm256_loadu_ps(&W[tmp + k]); 
                    __m256 x_vec = _mm256_loadu_ps(&x[k]); 
                    accum = _mm256_fmadd_ps(w, x_vec, accum);
                }

                __m256 hsum = _mm256_hadd_ps(accum, accum);
                hsum = _mm256_hadd_ps(hsum, hsum);
                __m128 hsum_high = _mm256_extractf128_ps(hsum, 1);
                __m128 hsum_low = _mm256_castps256_ps128(hsum);
                hsum_low = _mm_add_ps(hsum_low, hsum_high);
                float result = _mm_cvtss_f32(hsum_low);

                output[n0 + n1] = result * mask_scalar;
            }
        }
    }
}

torch::Tensor vmmsp_launch(torch::Tensor x, torch::Tensor W, torch::Tensor mask)
{
    TORCH_CHECK(x.is_contiguous() && W.is_contiguous() && mask.is_contiguous(), 'inputs must be contiguous')

    TORCH_CHECK(x.dim() == 2 && W.dim() == 2 && mask.dim() == 2 && x.size(0) == 1 && x.size(1) == W.size(1) && W.size(0) == mask.size(1), "incorrect input shapes")
    TORCH_CHECK(x.scalar_type() == W.scalar_type() && x.scalar_type() == mask.scalar_type(), "inconsistent input dtype")
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "only support float32")
    TORCH_CHECK(x.device().type() == torch::kCPU && W.device().type() == torch::kCPU && mask.device().type() == torch::kCPU, "only support CPU")


    torch::Tensor output = torch::zeros({1, W.size(0)}, x.options());

    const int n_blocks = (W.size(0) + BLOCK_N - 1) / BLOCK_N;

    if (x.scalar_type() == torch::kFloat32) {
        vmmsp_fp32(
            output.data_ptr<float>(),
            x.data_ptr<float>(),
            W.data_ptr<float>(),
            mask.data_ptr<float>(),
            W.size(0),
            x.size(1)
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("launch", &vmmsp_launch, "launch VMMSp (CPU)");
}
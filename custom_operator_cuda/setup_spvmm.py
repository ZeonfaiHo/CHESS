from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="spvmm_impl",
    ext_modules=[CUDAExtension(
            name="spvmm_impl", 
            sources=["spvmm.cu"],
            extra_compile_args={'nvcc': ['-O3']}
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
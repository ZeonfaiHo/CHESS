from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="vmmsp_impl",
    ext_modules=[CUDAExtension(
        name="vmmsp_impl", 
        sources=["vmmsp.cu"], 
        extra_compile_args={'nvcc': ['-O3']}
    )],
    cmdclass={"build_ext": BuildExtension}
)
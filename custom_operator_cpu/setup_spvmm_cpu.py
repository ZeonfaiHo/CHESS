from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="spvmm_impl_cpu",
    ext_modules=[CppExtension(
            name="spvmm_impl_cpu", 
            sources=["spvmm.cc"],
            extra_compile_args=['-Ofast', '-mavx2', '-mfma', '-fopenmp']
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name="vmmsp_impl_cpu",
    ext_modules=[CppExtension(
            name="vmmsp_impl_cpu", 
            sources=["vmmsp.cc"], 
                extra_compile_args=['-Ofast', '-mavx2', '-mfma', '-fopenmp']
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)
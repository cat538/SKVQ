from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


extra_compile_args = {
    "cxx": [
        "-g", 
        "-O3", 
        "-fopenmp", 
        "-lgomp", 
        "-std=c++17",
        # "-DENABLE_BF16"
    ],
    "nvcc": [
        "-O3", 
        "-std=c++17",
        # "-DENABLE_BF16",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--threads=32"
    ],
}

setup(
    name="skvq_quant",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="skvq_quant",
            sources=[
                "csrc/pybind.cc", 
                "csrc/skvq_quant.cu"
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    install_requires=["torch"],
)
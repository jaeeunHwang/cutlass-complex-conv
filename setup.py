from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent

cutlass_overlay = ROOT / "csrc" / "cutlass_ext"
cutlass_include = ROOT / "third_party" / "cutlass" / "include"

if not cutlass_include.exists():
    raise RuntimeError(
        "CUTLASS submodule not found.\n"
        "Run:\n"
        "  git submodule update --init --recursive"
    )

ext_modules = [
    CUDAExtension(
        name="complex_conv2d._C",
        sources=[
            str(ROOT / "csrc" / "bindings.cpp"),
            str(ROOT / "csrc" / "complex_conv2d.cu"),
        ],
        include_dirs=[
            str(cutlass_overlay),   # custom headers first
            str(cutlass_include),   # upstream CUTLASS second
        ],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-std=c++17",
            ],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
            ],
        },
    )
]

setup(
    name="cutlass-complex-conv",
    version="0.1.0",
    description="PyTorch ComplexConv2d backed by a unified CUTLASS implicit-CGEMM kernel",
    packages=find_packages(exclude=("tests", "benchmarks", "docs", "examples")),
    python_requires=">=3.9",
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
    },
    zip_safe=False,
)

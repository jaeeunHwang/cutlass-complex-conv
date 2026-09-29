# cutlass-complex-conv

A PyTorch CUDA extension for complex-valued 2D convolution built on top of NVIDIA CUTLASS.

This repository implements a unified implicit-CGEMM execution workflow for complex-valued convolution. Instead of decomposing a complex convolution into four independent real-valued `Conv2d` calls and reconstructing the result afterward, the CUDA kernel jointly processes the real and imaginary components inside a single implicit GEMM workflow.

The implementation is based on the method described in:

> **Restructuring the Implicit GEMM Workflow for Complex-Valued Convolution**  
> IEEE Access, 2026

***

## Performance

Benchmarks were measured on an **NVIDIA L4** with **PyTorch 2.11.0 + CUDA 12.8**.
All benchmark cases use batch size 1, a 3×3 kernel, stride 1, and padding 1.

| Cin → Cout, H×W | PyTorch Baseline (ms) | CUTLASS E2E (ms) | Fast Path (ms) | E2E Speedup | Fast-path Speedup |
|---|---:|---:|---:|---:|---:|
| 32 → 32, 32×32 | 0.3695 | 0.1240 | 0.0144 | 2.98× | 25.58× |
| 32 → 64, 56×56 | 0.4454 | 0.1229 | 0.0154 | 3.62× | 28.84× |
| 64 → 128, 28×28 | 0.4466 | 0.1245 | 0.0247 | 3.59× | 18.08× |
| 128 → 256, 14×14 | 0.4638 | 0.1230 | 0.0450 | 3.77× | 10.31× |
| 256 → 512, 7×7 | 0.4672 | 0.1227 | 0.0858 | 3.81× | 5.45× |

**Average end-to-end speedup: 3.55×**  
**Average planar fast-path speedup: 17.65×**

The PyTorch baseline uses four real-valued `Conv2d` operations followed by complex output reconstruction.

`CUTLASS E2E` includes the standard `torch.complex64` NCHW interface, including planar layout conversion and output reconstruction.

`Fast Path` measures `forward_planar()` with real and imaginary NHWC inputs already prepared, so it excludes complex-to-planar layout/materialization overhead and should be interpreted separately from end-to-end performance.

These repository benchmarks use a different environment and measurement setup from the paper experiments.

---

## Installation

### Requirements

- CUDA-enabled PyTorch
- NVIDIA CUDA Toolkit with `nvcc`
- A CUDA-capable NVIDIA GPU

Clone the repository with the CUTLASS submodule:

```bash
git clone --recurse-submodules https://github.com/jaeeunHwang/cutlass-complex-conv.git
cd cutlass-complex-conv
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Install the package:

```bash
pip install . --no-build-isolation
```

The CUDA extension is compiled during installation.

---

## Usage

```python
import torch

from complex_conv2d import ComplexConv2d

conv = ComplexConv2d(
    in_channels=32,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=False,
    device="cuda",
)

x = torch.randn(
    1,
    32,
    56,
    56,
    device="cuda",
    dtype=torch.complex64,
)

y = conv(x)
```

The standard interface accepts `torch.complex64` tensors in NCHW format.

If real and imaginary inputs are already stored separately in contiguous NHWC format, the planar fast path can be used directly:

```python
y_real, y_imag = conv.forward_planar(x_real, x_imag)
```

Correctness is validated against a PyTorch reference implementation based on four real-valued `Conv2d` operations. Tests are provided under `tests/`.

---

## Limitations

The current implementation is an inference-focused research prototype.

Supported:

- CUDA
- `torch.complex64`
- forward inference
- `bias=False`
- `groups=1`
- `dilation=1`
- `padding_mode="zeros"`

Not currently supported:

- autograd / backward
- complex bias
- grouped or depthwise convolution
- dilation other than 1
- CPU execution
- `complex128`
- FP16/BF16 complex paths

The CUDA kernel currently uses a fixed Tensor Core configuration and is not expected to be optimal for every convolution shape.

---

## Citation

If you use this implementation in academic work, please cite:

```bibtex
@article{hwang2026restructuring,
  title={Restructuring the Implicit GEMM Workflow for Complex-Valued Convolution},
  author={Hwang, Jaeeun and Jeong, Eunbi and Rhee, Jane and Yoon, Myung Kuk},
  journal={IEEE Access},
  year={2026},
  publisher={IEEE}
}
```

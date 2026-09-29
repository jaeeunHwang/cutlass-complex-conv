#include <torch/extension.h>

#include <cstdint>
#include <vector>

std::vector<torch::Tensor> complex_conv2d_cuda(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor w_real,
    torch::Tensor w_imag,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "forward",
      &complex_conv2d_cuda,
      "CUTLASS planar-complex Conv2d forward",
      pybind11::arg("x_real"),
      pybind11::arg("x_imag"),
      pybind11::arg("w_real"),
      pybind11::arg("w_imag"),
      pybind11::arg("stride_h"),
      pybind11::arg("stride_w"),
      pybind11::arg("pad_h"),
      pybind11::arg("pad_w"));
}

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/cutlass.h"
#include "cutlass/conv/conv2d_problem_size.h"

#include "cutlass/conv/kernel/default_planar_complex_conv2d_fprop.h"
#include "cutlass/conv/device/implicit_planar_cgemm_convolution.h"

#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"

#include <vector>

using ElementAccumulator = float;
using ElementComputeEpilogue = float;
using ElementInputA = float;
using ElementInputB = float;
using ElementOutput = float;

using LayoutInputA = cutlass::layout::TensorNHWC;
using LayoutInputB = cutlass::layout::TensorNHWC;
using LayoutOutput = cutlass::layout::TensorNHWC;

using MMAOp = cutlass::arch::OpClassTensorOp;
using SmArch = cutlass::arch::Sm80;

using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;

using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

constexpr int NumStages = 3;

static cutlass::conv::IteratorAlgorithm const IteratorAlgorithm =
    cutlass::conv::IteratorAlgorithm::kOptimized;

static cutlass::conv::StrideSupport const OutputStride =
    cutlass::conv::StrideSupport::kUnity;

using EpilogueOp =
    cutlass::epilogue::thread::LinearCombinationPlanarComplex<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementComputeEpilogue>;

using Conv2dFpropKernel =
    typename cutlass::conv::kernel::DefaultPlanarComplexConv2dFprop<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator,
        MMAOp,
        SmArch,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOp,
        SwizzleThreadBlock,
        NumStages,
        cutlass::arch::OpMultiplyAdd,
        IteratorAlgorithm,
        OutputStride>::Kernel;

using ImplicitPlanarCGemm =
    cutlass::conv::device::ImplicitPlanarCGemmConvolution<
        Conv2dFpropKernel>;

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.is_cuda(), #x " must be CUDA")

#define CHECK_FLOAT(x) \
  TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")

#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

std::vector<torch::Tensor> complex_conv2d_cuda(
    torch::Tensor x_real,
    torch::Tensor x_imag,
    torch::Tensor w_real,
    torch::Tensor w_imag,
    int64_t stride_h,
    int64_t stride_w,
    int64_t pad_h,
    int64_t pad_w) {

  CHECK_CUDA(x_real);
  CHECK_CUDA(x_imag);
  CHECK_CUDA(w_real);
  CHECK_CUDA(w_imag);

  CHECK_FLOAT(x_real);
  CHECK_FLOAT(x_imag);
  CHECK_FLOAT(w_real);
  CHECK_FLOAT(w_imag);

  CHECK_CONTIGUOUS(x_real);
  CHECK_CONTIGUOUS(x_imag);
  CHECK_CONTIGUOUS(w_real);
  CHECK_CONTIGUOUS(w_imag);

  TORCH_CHECK(x_real.dim() == 4, "x_real must be a 4D NHWC tensor");
  TORCH_CHECK(x_imag.dim() == 4, "x_imag must be a 4D NHWC tensor");
  TORCH_CHECK(w_real.dim() == 4, "w_real must be a 4D KRSC tensor");
  TORCH_CHECK(w_imag.dim() == 4, "w_imag must be a 4D KRSC tensor");

  TORCH_CHECK(
      x_real.sizes() == x_imag.sizes(),
      "x_real and x_imag must have identical shapes");

  TORCH_CHECK(
      w_real.sizes() == w_imag.sizes(),
      "w_real and w_imag must have identical shapes");

  TORCH_CHECK(
      x_real.device() == x_imag.device() &&
      x_real.device() == w_real.device() &&
      x_real.device() == w_imag.device(),
      "all inputs and weights must be on the same CUDA device");

  TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be positive");
  TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "padding must be non-negative");

  // Ensure allocations and the kernel launch use the input tensor's device.
  c10::cuda::CUDAGuard device_guard(x_real.device());

  // x: NHWC
  int N = x_real.size(0);
  int H = x_real.size(1);
  int W = x_real.size(2);
  int C = x_real.size(3);

  // w: KRSC
  int K = w_real.size(0);
  int R = w_real.size(1);
  int S = w_real.size(2);

  TORCH_CHECK(
      w_real.size(3) == C,
      "input channel count must match the weight channel count");

  TORCH_CHECK(R > 0 && S > 0, "kernel dimensions must be positive");

  int P =
      (H + 2 * pad_h - R) / stride_h + 1;

  int Q =
      (W + 2 * pad_w - S) / stride_w + 1;

  TORCH_CHECK(
      P > 0 && Q > 0,
      "invalid convolution configuration: output spatial size must be positive");

  auto y_real =
      torch::empty({N, P, Q, K}, x_real.options());

  auto y_imag =
      torch::empty({N, P, Q, K}, x_real.options());

  cutlass::Tensor4DCoord input_size(
      N, H, W, C);

  cutlass::Tensor4DCoord filter_size(
      K, R, S, C);

  cutlass::Tensor4DCoord padding(
      pad_h, pad_h,
      pad_w, pad_w);

  cutlass::MatrixCoord stride(
      stride_h, stride_w);

  cutlass::MatrixCoord dilation(1, 1);

  cutlass::Tensor4DCoord output_size(
      N, P, Q, K);

  int split_k_slices = 1;

  cutlass::conv::Conv2dProblemSize problem_size(
      input_size,
      filter_size,
      padding,
      stride,
      dilation,
      output_size,
      cutlass::conv::Mode::kCrossCorrelation,
      split_k_slices);

  auto layout_a =
      LayoutInputA::packed(input_size);

  auto layout_b =
      LayoutInputB::packed(filter_size);

  auto layout_c =
      LayoutOutput::packed(output_size);

  cutlass::TensorRef<ElementInputA, LayoutInputA>
      ref_a_real(
          x_real.data_ptr<float>(),
          layout_a);

  cutlass::TensorRef<ElementInputA, LayoutInputA>
      ref_a_imag(
          x_imag.data_ptr<float>(),
          layout_a);

  cutlass::TensorRef<ElementInputB, LayoutInputB>
      ref_b_real(
          w_real.data_ptr<float>(),
          layout_b);

  cutlass::TensorRef<ElementInputB, LayoutInputB>
      ref_b_imag(
          w_imag.data_ptr<float>(),
          layout_b);

  cutlass::TensorRef<ElementOutput, LayoutOutput>
      ref_d_real(
          y_real.data_ptr<float>(),
          layout_c);

  cutlass::TensorRef<ElementOutput, LayoutOutput>
      ref_d_imag(
          y_imag.data_ptr<float>(),
          layout_c);

  // beta = 0, so the epilogue does not need a separate source tensor C.
  // Alias C to D to avoid allocating and zero-filling two temporary tensors.
  cutlass::TensorRef<ElementOutput, LayoutOutput> ref_c_real = ref_d_real;
  cutlass::TensorRef<ElementOutput, LayoutOutput> ref_c_imag = ref_d_imag;

  EpilogueOp::Params epilogue_params(
      cutlass::complex<float>(1.0f, 0.0f),
      cutlass::complex<float>(0.0f, 0.0f));

  typename ImplicitPlanarCGemm::Arguments arguments{
      problem_size,

      ref_a_real,
      ref_a_imag,

      ref_b_real,
      ref_b_imag,

      ref_c_real,
      ref_c_imag,

      ref_d_real,
      ref_d_imag,

      epilogue_params
  };

  ImplicitPlanarCGemm op;

  auto status =
      op.can_implement(arguments);

  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS cannot implement this convolution configuration");

  size_t workspace_size =
      op.get_workspace_size(arguments);

  torch::Tensor workspace;

  void* workspace_ptr = nullptr;

  if (workspace_size > 0) {
    workspace = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        torch::TensorOptions()
            .dtype(torch::kUInt8)
            .device(x_real.device()));

    workspace_ptr = workspace.data_ptr();
  }

  cudaStream_t stream =
      at::cuda::getCurrentCUDAStream();

  status =
      op.initialize(
          arguments,
          workspace_ptr,
          stream);

  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS convolution initialization failed");

  status = op(stream);

  TORCH_CHECK(
      status == cutlass::Status::kSuccess,
      "CUTLASS kernel launch failed");

  return {y_real, y_imag};
}

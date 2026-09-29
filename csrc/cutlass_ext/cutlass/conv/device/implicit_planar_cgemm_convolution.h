#pragma once

#include <limits>

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/cuda_host_adapter.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ImplicitPlanarCGemmKernel_>
class ImplicitPlanarCGemmConvolution {
public:

  using UnderlyingKernel = GetUnderlyingKernel_t<ImplicitPlanarCGemmKernel_>;

  using ElementA = typename UnderlyingKernel::ElementA;
  using LayoutA = typename UnderlyingKernel::LayoutA;
  using ElementB = typename UnderlyingKernel::ElementB;
  using LayoutB = typename UnderlyingKernel::LayoutB;
  using ElementC = typename UnderlyingKernel::ElementC;
  using LayoutC = typename UnderlyingKernel::LayoutC;
  using ElementAccumulator = typename UnderlyingKernel::ElementAccumulator;
  using ElementCompute = typename UnderlyingKernel::ElementCompute;
  using OperatorClass = typename UnderlyingKernel::OperatorClass;
  using ArchTag = typename UnderlyingKernel::ArchTag;
  using ThreadblockShape = typename UnderlyingKernel::ThreadblockShape;
  using WarpShape = typename UnderlyingKernel::WarpShape;
  using InstructionShape = typename UnderlyingKernel::InstructionShape;
  using ThreadblockSwizzle = typename UnderlyingKernel::ThreadblockSwizzle;
  using EpilogueOutputOp = typename UnderlyingKernel::EpilogueOutputOp;
  static int const kStages = UnderlyingKernel::kStages;
  static int const kConvDim = UnderlyingKernel::kConvDim;
  using WarpMmaOperator = typename UnderlyingKernel::WarpMmaOperator;
  using ArchMmaOperator = typename UnderlyingKernel::ArchMmaOperator;
  using MathOperator = typename UnderlyingKernel::MathOperator; 

  static cutlass::conv::Operator const kConvolutionalOperator = UnderlyingKernel::kConvolutionalOperator;
  static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm = UnderlyingKernel::kIteratorAlgorithm;
  static cutlass::conv::StrideSupport const kStrideSupport = UnderlyingKernel::kStrideSupport;
  static cutlass::conv::GroupMode const kGroupMode = UnderlyingKernel::kGroupMode;

  static bool const kEnableCudaHostAdapter = CUTLASS_ENABLE_CUDA_HOST_ADAPTER;

  static int const kWarpCount = 
    (ThreadblockShape::kM / WarpShape::kM) * 
    (ThreadblockShape::kN / WarpShape::kN) *
    (ThreadblockShape::kK / WarpShape::kK);

  /// Argument structure
  using Arguments = typename UnderlyingKernel::Arguments;

private:

  /// Kernel parameters object
  typename UnderlyingKernel::Params params_;

public:

  /// Constructs a planar-complex implicit GEMM convolution operator.
  ImplicitPlanarCGemmConvolution() { }

  /// Determines whether the planar-complex implicit GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {
    // Dispatch validation to operand iterators.
    Status status = UnderlyingKernel::Mma::IteratorA::can_implement(args.problem_size);
    if (Status::kSuccess != status) {
      return status;
    }

    status = UnderlyingKernel::Mma::IteratorB::can_implement(args.problem_size);
    if (Status::kSuccess != status) {
      return status;
    }

    // Check that tensor sizes do not exceed the maximum supported size.
    if (kConvolutionalOperator == conv::Operator::kFprop) {
      if (args.problem_size.activation_size() * sizeof(ElementA) >=
              (1ull << 31) ||
          args.problem_size.filter_size() * sizeof(ElementB) >= (1ull << 31) ||
          args.problem_size.output_size() * sizeof(ElementC) >= (1ull << 31)) {
        return Status::kErrorInvalidProblem;
      }
    }
    else if (kConvolutionalOperator == conv::Operator::kDgrad ||
               kConvolutionalOperator == conv::Operator::kDeconv) {
      if (args.problem_size.activation_size() * sizeof(ElementC) >=
              (1ull << 31) ||
          args.problem_size.filter_size() * sizeof(ElementB) >= (1ull << 31) ||
          args.problem_size.output_size() * sizeof(ElementA) >= (1ull << 31)) {
        return Status::kErrorInvalidProblem;
      }
    }
    else if (kConvolutionalOperator == conv::Operator::kWgrad) {
      if (args.problem_size.activation_size() * sizeof(ElementB) >=
              (1ull << 31) ||
          args.problem_size.filter_size() * sizeof(ElementC) >= (1ull << 31) ||
          args.problem_size.output_size() * sizeof(ElementA) >= (1ull << 31)) {
        return Status::kErrorInvalidProblem;
      }
    }

    // Check grouped-convolution constraints.
    if (args.problem_size.groups != 1) {
      if (kGroupMode == conv::GroupMode::kNone) {
        return Status::kErrorInvalidProblem;
      } 

      // C and K must be divisible by the number of groups.
      if (args.problem_size.K % args.problem_size.groups ||
        args.problem_size.C % args.problem_size.groups) {
        return Status::kErrorInvalidProblem;
      }

      // Split-K is not supported for grouped convolution.
      if (args.problem_size.split_k_slices != 1) {
        return Status::kErrorInvalidProblem;
      }

      int k_per_group = args.problem_size.K / args.problem_size.groups;
      // k_per_group must be a multiple of ThreadblockShape::kN so one CTA computes one group.
      if (kGroupMode == conv::GroupMode::kSingleGroup && k_per_group % ThreadblockShape::kN) {
        return Status::kErrorInvalidProblem;
      }
      // ThreadblockShape::kN must be divisible by k_per_group so one CTA can compute multiple groups.
      if (kGroupMode == conv::GroupMode::kMultipleGroup && ThreadblockShape::kN % k_per_group) {
        return Status::kErrorInvalidProblem;
      }

      // The optimized iterator currently supports only SingleGroup mode.
      if (kIteratorAlgorithm == IteratorAlgorithm::kOptimized &&
        kGroupMode != conv::GroupMode::kSingleGroup) {
        return Status::kErrorInvalidProblem;
      }
    }

    static int const kAlignmentC = UnderlyingKernel::Epilogue::OutputTileIterator::kElementsPerAccess;
    if (kConvolutionalOperator == conv::Operator::kFprop) {
      if (args.problem_size.K % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    } else if (kConvolutionalOperator == conv::Operator::kDgrad || kConvolutionalOperator == conv::Operator::kDeconv) {
       if (args.problem_size.C % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    } else if (kConvolutionalOperator == conv::Operator::kWgrad) {
       if (args.problem_size.C % kAlignmentC)
        return Status::kErrorMisalignedOperand;
    }

    // Check unsupported cases for strided Dgrad/Deconv.
    if ((kConvolutionalOperator == conv::Operator::kDgrad || kConvolutionalOperator == conv::Operator::kDeconv) &&
      kStrideSupport == conv::StrideSupport::kStrided) {
      // split-k (serial or parallel) is not supported for strided dgrad / deconv
      if (args.problem_size.split_k_slices > 1 && (args.problem_size.stride().at(args.problem_size.stride().max_dim_index()) > 1)) {
        return Status::kErrorNotSupported;
      }

      // dilation > {1x1} is not supported for strided dgrad / deconv
      if (args.problem_size.dilation_h > 1 || args.problem_size.dilation_w > 1) {
        return Status::kErrorNotSupported;
      }
    }

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(
      threadblock_swizzle.get_tiled_shape(
        kConvolutionalOperator,
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices));

    if (!(grid.y <= std::numeric_limits<uint16_t>::max() &&
          grid.z <= std::numeric_limits<uint16_t>::max())) {

      return Status::kErrorInvalidProblem;
    }

    return Status::kSuccess;
  }

  /// Gets the required workspace size.
  static size_t get_workspace_size(Arguments const &args) {
  
    size_t workspace_bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        kConvolutionalOperator,
        args.problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices);

    if (args.split_k_mode == SplitKMode::kParallel) {

      // Split-K parallel: CTAs in k-dimension write the partial results in a temporary workspace.
      // The user needs to call a reduction operator to obtain the final output tensor
      workspace_bytes = 
        sizeof(ElementAccumulator) *
        size_t(cutlass::conv::implicit_gemm_tensor_c_size(kConvolutionalOperator, args.problem_size)) *
        size_t(grid_tiled_shape.k());
    }

    else if (args.split_k_mode == SplitKMode::kSerial && args.problem_size.split_k_slices > 1) {

      // Split-K serial: The user workspace is used to store semaphore and serialize writing the 
      // final reduced output to user's output tensor
      workspace_bytes = sizeof(int) * size_t(grid_tiled_shape.m()) * size_t(grid_tiled_shape.n());
    }

    return workspace_bytes;
  }

  /// Initializes convolution state from arguments.
  Status initialize(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {
   
    if (args.problem_size.split_k_slices > 1) {

      if (!workspace) {
        return Status::kErrorWorkspaceNull;
      }

      cudaError_t status = cudaMemsetAsync(workspace, 0, get_workspace_size(args), stream);

      if (status != cudaSuccess) {
        return Status::kErrorInternal;
      }
    }

    // Initialize the parameter structure from the arguments.
    params_ = typename UnderlyingKernel::Params(
      args,
      static_cast<int *>(workspace)
    );

    if constexpr (kEnableCudaHostAdapter) {
      CUTLASS_ASSERT(cuda_adapter);
      return Status::kSuccess;
    }
    else {
      int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));
  
      if (smem_size >= (48 << 10)) {
        cudaError_t result = cudaFuncSetAttribute(cutlass::Kernel<UnderlyingKernel>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);
  
        if (result != cudaSuccess) {
          return Status::kErrorInternal;
        }
      }
    }
    
    return Status::kSuccess;
  }

  /// Updates convolution state from arguments.
  Status update(Arguments const &args, void *workspace = nullptr) {

    // Update runtime tensor pointers and epilogue parameters.
    params_.ptr_A_real = args.ref_A_real.data();
    params_.ptr_A_imag = args.ref_A_imag.data();
    params_.ptr_B_real = args.ref_B_real.data();
    params_.ptr_B_imag = args.ref_B_imag.data();
    params_.ptr_C_real = args.ref_C_real.data();
    params_.ptr_C_imag = args.ref_C_imag.data();
    params_.ptr_D_real = args.ref_D_real.data();
    params_.ptr_D_imag = args.ref_D_imag.data();
    params_.output_op = args.output_op;
    params_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }

  /// Runs the initialized kernel.
  Status run(
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr,
    int32_t kernel_index = 0) {


    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
    dim3 block(32 * kWarpCount, 1, 1);

    int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));
    cutlass::Status launch_result = cutlass::Status::kSuccess;

    if constexpr (kEnableCudaHostAdapter) {
        //
        // Use the cuda host adapter
        //
        CUTLASS_ASSERT(cuda_adapter);
        if (cuda_adapter) {

          void* kernel_params[] = {&params_};
          launch_result = cuda_adapter->launch(
              grid, dim3(1, 1, 1), block, smem_size, stream, kernel_params, kernel_index
              );
        }
        else {
          launch_result = Status::kErrorInternal;
        }
    }
    else {
      cutlass::arch::synclog_setup();
      cutlass::Kernel<UnderlyingKernel><<<grid, block, smem_size, stream>>>(params_);      
    }

    cudaError_t result = cudaGetLastError();
    if (cudaSuccess == result && Status::kSuccess == launch_result) {
      return Status::kSuccess;
    }
    else {
      CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
      return Status::kErrorInternal;
    }
  }

  /// Runs the initialized kernel.
  Status operator()(
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr,
    int32_t kernel_index = 0) {
    return run(stream, cuda_adapter, kernel_index);
  }

  /// Initializes and runs the kernel.
  Status operator()(
    Arguments const &args, 
    void *workspace = nullptr, 
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr,
    int32_t kernel_index = 0) {
    
    Status status = initialize(args, workspace, stream, cuda_adapter);
    
    if (status == Status::kSuccess) {
      status = run(stream, cuda_adapter, kernel_index);
    }

    return status;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/semaphore.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/tensor_ref_planar_complex.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/conv/conv2d_problem_size.h"
#include "cutlass/conv/conv3d_problem_size.h"
#include "cutlass/epilogue/threadblock/output_iterator_parameter.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,                             ///! Epilogue
  typename ThreadblockSwizzle_,                   ///! Threadblock swizzling function
  conv::Operator ConvOperator,                    ///! Convolutional operator (Fprop, Dgrad, Wgrad, Deconv)
  typename ConvProblemSize_ = Conv2dProblemSize,  ///! Convolutional operator on 2D or 3D problem
  conv::GroupMode GroupMode_ = conv::GroupMode::kNone    ///! Group mode
>
struct ImplicitPlanarCGemmConvolution {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static Operator const kConvolutionalOperator = ConvOperator;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using ElementC = typename EpilogueOutputOp::ElementOutput;

  /// Set output tensor C layout
  using LayoutC = LayoutA;

  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  using ElementCompute = typename EpilogueOutputOp::ElementCompute;

  using WarpMmaOperator = typename Mma::Policy::Operator;

  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename ArchMmaOperator::Operator;
  
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;

  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename WarpMmaOperator::Shape;
  using InstructionShape = typename ArchMmaOperator::Shape;

  static int const kStages = Mma::kStages;
  static IteratorAlgorithm const kIteratorAlgorithm = Mma::IteratorA::kIteratorAlgorithm; 
  static StrideSupport const kStrideSupport = Mma::IteratorA::kStrideSupport;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using TensorRefA = typename Mma::IteratorA::TensorRef;
  using TensorRefB = typename Mma::IteratorB::TensorRef;
  using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;

  /// Check that iterators A and B target the same convolution dimensionality.
  static_assert(Mma::IteratorA::kConvDim == Mma::IteratorB::kConvDim, 
    "Convolution on different dimensions is not supported");
  static int const kConvDim = Mma::IteratorA::kConvDim;

  /// Conv dimension and problem size structure (Conv2d or Conv3d)
  using ConvProblemSize = ConvProblemSize_;

  static conv::GroupMode const kGroupMode = GroupMode_;

  /// Wgrad C stride idx for implicit gemm algorithm 
  // Conv2d row-major matrix C (KxRSC) 
  // Conv3d row-major matrix C (KxTRSC)
  static int const kWgradCStrideIdx = 
    platform::is_same<LayoutC, cutlass::layout::TensorNHWC>::value ? 2 : 3;

  /// This chooses the appropriate stride element of the C tensor.
  static int const kTensorCStrideIdx = 
    (kConvolutionalOperator == conv::Operator::kWgrad ? kWgradCStrideIdx : 0);

  //
  //
  //
  using ConvOutputIteratorParameter = epilogue::threadblock::ConvOutputIteratorParameter<
    LayoutC,
    typename Epilogue::OutputTileIterator::Layout, 
    TensorRefC,
    ConvOperator,
    ConvProblemSize
    >;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    ConvProblemSize problem_size;
    TensorRefA ref_A_real;
    TensorRefA ref_A_imag;
    TensorRefB ref_B_real;
    TensorRefB ref_B_imag;
    TensorRefC ref_C_real;
    TensorRefC ref_C_imag;
    TensorRefC ref_D_real;
    TensorRefC ref_D_imag;
    typename EpilogueOutputOp::Params output_op;
    int64_t imag_stride_A;
    int64_t imag_stride_B;
    SplitKMode split_k_mode;
    

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }
   
    CUTLASS_HOST_DEVICE 
    Arguments(
      ConvProblemSize const & problem_size
    ):
      problem_size(problem_size) { }

    CUTLASS_HOST_DEVICE
    Arguments(
      ConvProblemSize const & problem_size,
      TensorRefA const & ref_A_real,
      TensorRefA const & ref_A_imag,
      TensorRefB const & ref_B_real,
      TensorRefB const & ref_B_imag,
      TensorRefC const & ref_C_real,
      TensorRefC const & ref_C_imag,
      TensorRefC const & ref_D_real,
      TensorRefC const & ref_D_imag,
      typename EpilogueOutputOp::Params const & output_op,
      int64_t imag_stride_A = 0,
      int64_t imag_stride_B = 0,
      SplitKMode const & split_k_mode = SplitKMode::kSerial
    ):
      problem_size(problem_size),
      ref_A_real(ref_A_real),
      ref_A_imag(ref_A_imag),
      ref_B_real(ref_B_real),
      ref_B_imag(ref_B_imag),
      ref_C_real(ref_C_real),
      ref_C_imag(ref_C_imag),
      ref_D_real(ref_D_real),
      ref_D_imag(ref_D_imag),
      output_op(output_op),
      imag_stride_A(imag_stride_A),
      imag_stride_B(imag_stride_B),
      split_k_mode(split_k_mode)
    {

    }

  };

  /// Parameters structure
  struct Params {
    ConvProblemSize problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    gemm::GemmCoord implicit_gemm_problem_size;
    int swizzle_log_tile;

    int gemm_k_iterations;
    int gemm_k_iterations_per_channel;
    typename Mma::IteratorA::Params iterator_A_real;
    typename Mma::IteratorA::Params iterator_A_imag;
    typename Mma::IteratorA::Element const *ptr_A_real;
    typename Mma::IteratorA::Element const *ptr_A_imag;
    typename Mma::IteratorB::Params iterator_B_real;
    typename Mma::IteratorB::Params iterator_B_imag;
    typename Mma::IteratorB::Element const *ptr_B_real;
    typename Mma::IteratorB::Element const *ptr_B_imag;
    typename Epilogue::OutputTileIterator::Params iterator_C_real;
    typename Epilogue::OutputTileIterator::Params iterator_C_imag;
    typename Epilogue::OutputTileIterator::Element *ptr_C_real;
    typename Epilogue::OutputTileIterator::Element *ptr_C_imag;
    typename Epilogue::OutputTileIterator::Params iterator_D_real;
    typename Epilogue::OutputTileIterator::Params iterator_D_imag;
    typename Epilogue::OutputTileIterator::Element *ptr_D_real;
    typename Epilogue::OutputTileIterator::Element *ptr_D_imag;
    typename EpilogueOutputOp::Params output_op;
    int *semaphore;
    int64_t imag_stride_A;
    int64_t imag_stride_B;
    SplitKMode split_k_mode;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), gemm_k_iterations(0) { }

    /// 
    CUTLASS_HOST_DEVICE
    Params(
      Arguments const &args,
      int *semaphore = nullptr
    ):
      problem_size(args.problem_size),
      implicit_gemm_problem_size(cutlass::conv::implicit_gemm_problem_size(kConvolutionalOperator, args.problem_size)),
      iterator_A_real(Mma::IteratorA::getParams(args.problem_size, args.ref_A_real.layout())),
      iterator_A_imag(Mma::IteratorA::getParams(args.problem_size, args.ref_A_imag.layout())),
      ptr_A_real(args.ref_A_real.data()),
      ptr_A_imag(args.ref_A_imag.data()),
      iterator_B_real(args.problem_size, args.ref_B_real.layout()),
      iterator_B_imag(args.problem_size, args.ref_B_imag.layout()),
      ptr_B_real(args.ref_B_real.data()),
      ptr_B_imag(args.ref_B_imag.data()),
      iterator_C_real(ConvOutputIteratorParameter::layout(args.ref_C_real), implicit_gemm_tensor_c_extent(kConvolutionalOperator, args.problem_size)),
      iterator_C_imag(ConvOutputIteratorParameter::layout(args.ref_C_imag), implicit_gemm_tensor_c_extent(kConvolutionalOperator, args.problem_size)),
      ptr_C_real(args.ref_C_real.data()),
      ptr_C_imag(args.ref_C_imag.data()),
      iterator_D_real(ConvOutputIteratorParameter::layout(args.ref_D_real), implicit_gemm_tensor_c_extent(kConvolutionalOperator, args.problem_size)),
      iterator_D_imag(ConvOutputIteratorParameter::layout(args.ref_D_imag), implicit_gemm_tensor_c_extent(kConvolutionalOperator, args.problem_size)),
      ptr_D_real(args.ref_D_real.data()),
      ptr_D_imag(args.ref_D_imag.data()),
      output_op(args.output_op),
      semaphore(semaphore),
      imag_stride_A(args.imag_stride_A),
      imag_stride_B(args.imag_stride_B),
      split_k_mode(args.split_k_mode)
    {
      gemm_k_iterations = implicit_gemm_k_iterations(
        kConvolutionalOperator,
        ThreadblockShape::kK,
        args.problem_size,
        kIteratorAlgorithm,
        kGroupMode,
        ThreadblockShape::kN);

      gemm_k_iterations_per_channel = implicit_gemm_k_iterations_per_channel(
          kConvolutionalOperator, args.problem_size, kIteratorAlgorithm);

      ThreadblockSwizzle threadblock_swizzle;

      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        implicit_gemm_problem_size,
        {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
        args.problem_size.split_k_slices);

      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  ImplicitPlanarCGemmConvolution() { } 

  /// Executes one planar-complex implicit GEMM convolution.
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {

      return;
    }

    // Compute position within threadblock
    int thread_idx = threadIdx.x;
    int iterator_A_column_offset = threadblock_tile_idx.k() * Mma::Shape::kK;
    if (kGroupMode != GroupMode::kNone) {
      if (kGroupMode != GroupMode::kDepthwise) {
        int k_per_group = params.problem_size.K / params.problem_size.groups;
        int group_idx = threadblock_tile_idx.n() * Mma::Shape::kN / k_per_group;
        int channels_per_group = params.problem_size.C / params.problem_size.groups;
        iterator_A_column_offset += group_idx * channels_per_group;
      } else {
        iterator_A_column_offset += threadblock_tile_idx.n() * Mma::Shape::kN;
      }
    } 

    // Construct iterators to A and B operands
    typename Mma::IteratorA iterator_A_real(
      params.iterator_A_real,
      params.problem_size,
      params.ptr_A_real,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() * Mma::Shape::kM,
        iterator_A_column_offset
      )
    );

    typename Mma::IteratorA iterator_A_imag(
      params.iterator_A_imag,
      params.problem_size,
      params.ptr_A_imag,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.m() * Mma::Shape::kM,
        iterator_A_column_offset
      )
    );
    
    iterator_A_imag.add_pointer_offset(params.imag_stride_A);

    typename Mma::IteratorB iterator_B_real(
      params.iterator_B_real,
      params.problem_size,
      params.ptr_B_real,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.k() * Mma::Shape::kK,
        threadblock_tile_idx.n() * Mma::Shape::kN
      )
    );

    typename Mma::IteratorB iterator_B_imag(
      params.iterator_B_imag,
      params.problem_size,
      params.ptr_B_imag,
      thread_idx,
      MatrixCoord(
        threadblock_tile_idx.k() * Mma::Shape::kK,
        threadblock_tile_idx.n() * Mma::Shape::kN
      )
    );

    iterator_B_imag.add_pointer_offset(params.imag_stride_B);

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct the threadblock-scoped planar-complex MMA mainloop.
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;

    accumulators.clear();

    // Compute the threadblock-scoped complex matrix multiply-accumulate.
    mma(
      params.gemm_k_iterations,
      accumulators,
      iterator_A_real,
      iterator_A_imag,
      iterator_B_real,
      iterator_B_imag,
      accumulators,
      params.gemm_k_iterations_per_channel,
      params.imag_stride_A,
      params.imag_stride_B
    );

    //
    // Epilogue
    //

    EpilogueOutputOp output_op(params.output_op);

    // Construct the semaphore.
    int block_idx = threadblock_tile_idx.m() + threadblock_tile_idx.n() * params.grid_tiled_shape.m();

    Semaphore semaphore(params.semaphore + block_idx, thread_idx);
    
    // Compute logical position within grid
    threadblock_tile_idx =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    // If performing a reduction via split-K, fetch the initial synchronization
    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {
        
      // Fetch the synchronization lock initially but do not block.
      semaphore.fetch();

      // Indicate which position in a serial reduction the output operator is currently updating
      output_op.set_k_partition(threadblock_tile_idx.k(), params.grid_tiled_shape.k());
    }

    MatrixCoord threadblock_offset(
      threadblock_tile_idx.m() * Mma::Shape::kM,
      threadblock_tile_idx.n() * Mma::Shape::kN
    );

    // Tile iterator writing to destination tensor
    typename Epilogue::OutputTileIterator iterator_D_real(
      params.iterator_D_real,
      params.ptr_D_real,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );
    typename Epilogue::OutputTileIterator iterator_D_imag(
      params.iterator_D_imag,
      params.ptr_D_imag,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );
    
    // Tile iterator reading from source accumulator tensor
    typename Epilogue::OutputTileIterator iterator_C_real(
      params.iterator_C_real,
      params.ptr_C_real,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );
    typename Epilogue::OutputTileIterator iterator_C_imag(
      params.iterator_C_imag,
      params.ptr_C_imag,
      ConvOutputIteratorParameter::extent(params.problem_size),
      thread_idx,
      threadblock_offset
    );

    // Construct the epilogue
    Epilogue epilogue(
      shared_storage.epilogue, 
      thread_idx, 
      warp_idx, 
      lane_idx);

    // Wait on the semaphore - this latency may have been covered by iterator construction
    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) {
        
      // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
      if (threadblock_tile_idx.k()) {
        iterator_C_real = iterator_D_real;
        iterator_C_imag = iterator_D_imag;
      }

      semaphore.wait(threadblock_tile_idx.k());

    }
    // Each split-k-slice writes to a unique tensor location
    else if (params.split_k_mode == SplitKMode::kParallel) {
      iterator_D_real.add_pointer_offset(threadblock_tile_idx.k() * 
        cutlass::conv::implicit_gemm_tensor_c_size(ConvOperator, params.problem_size));
      iterator_D_imag.add_pointer_offset(threadblock_tile_idx.k() * 
        cutlass::conv::implicit_gemm_tensor_c_size(ConvOperator, params.problem_size));
    }

    // Run the planar-complex epilogue.
    epilogue(output_op, iterator_D_real, iterator_D_imag, accumulators, iterator_C_real, iterator_C_imag);
  
    //
    // Release the semaphore
    //

    if (params.split_k_mode == SplitKMode::kSerial && params.grid_tiled_shape.k() > 1) { 

      int lock = 0;
      if (params.grid_tiled_shape.k() == threadblock_tile_idx.k() + 1) {

        // The final threadblock resets the semaphore for subsequent grids.
        lock = 0;
      }
      else {
        // Otherwise, the semaphore is incremented
        lock = threadblock_tile_idx.k() + 1;
      }
      
      semaphore.release(lock);
    }
  } 
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace conv
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

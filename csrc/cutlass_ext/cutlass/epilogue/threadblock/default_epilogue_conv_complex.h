#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/array_planar_complex.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/arch/arch.h"

#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"

#include "cutlass/epilogue/threadblock/epilogue_planar_complex.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Defines defaults for a planar-complex convolution epilogue.
template <
  typename ThreadblockShape_,
  typename WarpMmaOperator_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess,
  bool ScatterD = false,
  typename PermuteDLayout = layout::NoPermute,
  conv::StrideSupport StrideSupport = conv::StrideSupport::kUnity,
  int Rank = 4
>
struct DefaultEpilogueConvComplex {

  // Reuse CUTLASS's real-valued Tensor Core epilogue configuration to obtain
  // the output/accumulator/shared-memory iterator types required by the
  // planar-complex epilogue.
  using RealEpilogue = DefaultEpilogueTensorOp<
    ThreadblockShape_,
    WarpMmaOperator_,
    PartitionsK,
    OutputOp_,
    ElementsPerAccess,
    ScatterD,
    PermuteDLayout,
    StrideSupport,
    Rank
  >;

  // Apply the output operation independently to the real and imaginary
  // accumulator fragments and write the two planar output tensors.
  using Epilogue = EpiloguePlanarComplex<
    ThreadblockShape_,
    WarpMmaOperator_,
    PartitionsK,
    typename RealEpilogue::OutputTileIterator,
    typename RealEpilogue::AccumulatorFragmentIterator,
    typename RealEpilogue::WarpTileIterator,
    typename RealEpilogue::SharedLoadIterator,
    OutputOp_,
    typename RealEpilogue::Padding
  >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

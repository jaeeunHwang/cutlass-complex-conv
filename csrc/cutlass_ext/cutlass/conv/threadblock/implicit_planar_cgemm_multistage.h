#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/cache_operation.h"
#include "cutlass/gemm/threadblock/mma_base.h"
#include "cutlass/gemm/threadblock/mma_planar_complex_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace conv {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages>

class ImplicitPlanarCGemmMultistage : 
  public gemm::threadblock::MmaPlanarComplexBase<Shape_, Policy_, Stages> {
public:
  ///< Base class
  using Base = gemm::threadblock::MmaPlanarComplexBase<Shape_, Policy_, Stages>;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Policy describing tuning details
  
  using Policy = Policy_;
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile

  using FragmentC = ArrayPlanarComplex<
    typename Policy::Operator::FragmentC::Element,
    Policy::Operator::FragmentC::kElements
  >;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;
  
  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    // Optional staged-accumulation (e.g., tf32x3 kernels) for improved numerical
    // accuracy, where each mainloop iteration first accumulates into a temporary
    // set of freshly-cleared accumulators, which are subsequently added to the
    // final accumulator set.
    static bool const kStagedAccumulation = false;
  };

 private:

  using WarpLoadedFragmentA = typename Operator::FragmentA;
  using WarpLoadedFragmentB = typename Operator::FragmentB;
  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

 private:

  //
  // Data members
  //

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  ImplicitPlanarCGemmMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B(shared_storage.operand_B_ref(), thread_idx)
  {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});

  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(
    IteratorA &iterator_A_real, IteratorA &iterator_A_imag, IteratorB &iterator_B_real, IteratorB &iterator_B_imag,
    int group_start_A = 0, int group_start_B = 0) {

    iterator_A_real.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    iterator_A_imag.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A.set_iteration_index(group_start_A);
      
    // Async Copy for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {

      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess /
                              IteratorA::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                  dst_ptr + v, iterator_A_real.get(), iterator_A_real.valid());
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                  dst_ptr + v + (Base::SharedStorage::kImaginaryStrideA / IteratorA::ThreadMap::kElementsPerAccess), iterator_A_imag.get(), iterator_A_imag.valid());
          ++iterator_A_real;
          ++iterator_A_imag;
        }

        ++this->smem_iterator_A;
      }
    }

    iterator_B_real.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    iterator_B_imag.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);

    this->smem_iterator_B.set_iteration_index(group_start_B);

    // Async Copy for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B.get());
        
        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess /
                              IteratorB::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                  dst_ptr + v, iterator_B_real.get(), iterator_B_real.valid());
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                  dst_ptr + v + (Base::SharedStorage::kImaginaryStrideB / IteratorB::ThreadMap::kElementsPerAccess), iterator_B_imag.get(), iterator_B_imag.valid());

          ++iterator_B_real;
          ++iterator_B_imag;
        }
        ++this->smem_iterator_B;
      }
    }
  }

  CUTLASS_DEVICE
  void add_planar(FragmentC &dst, FragmentC const &src) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < FragmentC::kElements; ++i) {
      dst.real[i] += src.real[i];
      dst.imag[i] += src.imag[i];
    }
  }

  CUTLASS_DEVICE
  void warp_mma_planar_complex(
    Operator & warp_mma,
    FragmentC &accum,
    WarpTransformedFragmentA const & real_A,
    WarpTransformedFragmentA const & imag_A,
    WarpTransformedFragmentB const & real_B,
    WarpTransformedFragmentB const & imag_B)
  {
    cutlass::negate<Array<typename WarpTransformedFragmentB::Element,
                          WarpTransformedFragmentB::kElements>> neg_op_B;

    WarpTransformedFragmentB neg_imag_B = neg_op_B(imag_B);

    // (Ar + iAi)(Br + iBi)
    //   real += Ar*Br - Ai*Bi
    //   imag += Ar*Bi + Ai*Br
    warp_mma(accum.real, real_A, real_B, accum.real);
    warp_mma(accum.real, imag_A, neg_imag_B, accum.real);
    warp_mma(accum.imag, real_A, imag_B, accum.imag);
    warp_mma(accum.imag, imag_A, real_B, accum.imag);
  }


  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A_real,
      IteratorA iterator_A_imag,
      ///< iterator over B operand in global memory
      IteratorB iterator_B_real,
      IteratorB iterator_B_imag,
      ///< initial value of accumulator
      FragmentC const &src_accum,
      ///< number of iterations per channel
      int gemm_k_iterations_per_channel = 0,
      ///< Imaginary strides used for planar-complex only - ignored here
      int64_t imag_stride_A = 0,
      int64_t imag_stride_B = 0) {

    //
    // Prologue
    //

    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1;
         ++stage, --gemm_k_iterations) {

      iterator_A_real.set_iteration_index(0);
      iterator_A_imag.set_iteration_index(0);
      this->smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr =
          reinterpret_cast<typename IteratorA::AccessType *>(
            this->smem_iterator_A.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          int const kSrcBytes =
            sizeof_bits<typename IteratorA::Element>::value *
            IteratorA::ThreadMap::kElementsPerAccess /
            IteratorA::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
            dst_ptr + v, iterator_A_real.get(), iterator_A_real.valid());
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
            dst_ptr + v + (Base::SharedStorage::kImaginaryStrideA / IteratorA::ThreadMap::kElementsPerAccess), iterator_A_imag.get(), iterator_A_imag.valid());

          ++iterator_A_real;
          ++iterator_A_imag;
        }

        ++this->smem_iterator_A;
      }

      iterator_B_real.set_iteration_index(0);
      iterator_B_imag.set_iteration_index(0);
      this->smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr =
          reinterpret_cast<typename IteratorB::AccessType *>(
              this->smem_iterator_B.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB::Element>::value *
              IteratorB::ThreadMap::kElementsPerAccess /
              IteratorB::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr + v, iterator_B_real.get(), iterator_B_real.valid());
          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr + v + (Base::SharedStorage::kImaginaryStrideB / IteratorB::ThreadMap::kElementsPerAccess), iterator_B_imag.get(), iterator_B_imag.valid());
  
          ++iterator_B_real;
          ++iterator_B_imag;
        }

        ++this->smem_iterator_B;
      }

      // Move to the next stage
      iterator_A_real.advance();
      iterator_A_imag.advance();
      iterator_B_real.advance();
      iterator_B_imag.advance();

      this->smem_iterator_A.add_tile_offset({0, 1});
      this->smem_iterator_B.add_tile_offset({1, 0});

      // Inserts a fence to group cp.async instructions into stages.
      cutlass::arch::cp_async_fence();
    }

    // Perform accumulation in the 'd' output operand
    accum = src_accum;

    // Waits until kStages-2 stages have committed. 
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();

    // Pair of fragments used to overlap shared memory loads and math
    // instructions
    WarpLoadedFragmentA warp_loaded_frag_real_A[2];
    WarpLoadedFragmentA warp_loaded_frag_imag_A[2];
    WarpLoadedFragmentB warp_loaded_frag_real_B[2];
    WarpLoadedFragmentB warp_loaded_frag_imag_B[2];
    WarpTransformedFragmentA warp_transformed_frag_real_A[2];
    WarpTransformedFragmentA warp_transformed_frag_imag_A[2];
    WarpTransformedFragmentB warp_transformed_frag_real_B[2];
    WarpTransformedFragmentB warp_transformed_frag_imag_B[2];

    Operator warp_mma;

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.set_kgroup_index(0);

    this->warp_tile_iterator_A_.load(warp_loaded_frag_real_A[0]);
    this->warp_tile_iterator_A_.load_with_pointer_offset(warp_loaded_frag_imag_A[0], Base::SharedStorage::kImaginaryStrideA);
    this->warp_tile_iterator_B_.load(warp_loaded_frag_real_B[0]);
    this->warp_tile_iterator_B_.load_with_pointer_offset(warp_loaded_frag_imag_B[0], Base::SharedStorage::kImaginaryStrideB);

    ++this->warp_tile_iterator_A_;
    ++this->warp_tile_iterator_B_;

    // Start issuing the first group of the next stage outside of the mainloop
    copy_tiles_and_advance(iterator_A_real, iterator_A_imag, iterator_B_real, iterator_B_imag);

    int smem_write_stage_idx = Base::kStages - 1;
    int smem_read_stage_idx = 0;

    warp_mma.transform(warp_transformed_frag_real_A[0], warp_transformed_frag_real_B[0],
                       warp_loaded_frag_real_A[0], warp_loaded_frag_real_B[0]);
    warp_mma.transform(warp_transformed_frag_imag_A[0], warp_transformed_frag_imag_B[0],
                       warp_loaded_frag_imag_A[0], warp_loaded_frag_imag_B[0]);

    // tf32x3 kernels use staging accumulation. warp_mma uses a temporary
    // accumulator and this temporary accumulator is added to the final
    // accumulator once in every mainloop iteration.
    FragmentC tmp_accum;

    if (Detail::kStagedAccumulation) {
      tmp_accum.clear();
    }

    //
    // Mainloop
    //

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
      //
      // Loop over GEMM K dimension
      //

      // Computes a warp-level GEMM on data held in shared memory
      // Each "warp_mma_k" refers to a warp-level matrix multiply-accumulate
      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
           ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if
        // this is the last group as the case may be.

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A_.load(warp_loaded_frag_real_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_A_.load_with_pointer_offset(warp_loaded_frag_imag_A[(warp_mma_k + 1) % 2], Base::SharedStorage::kImaginaryStrideA);
        this->warp_tile_iterator_B_.load(warp_loaded_frag_real_B[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load_with_pointer_offset(warp_loaded_frag_imag_B[(warp_mma_k + 1) % 2], Base::SharedStorage::kImaginaryStrideB);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k > 0) {
          warp_mma.transform(warp_transformed_frag_real_A[warp_mma_k % 2],
                             warp_transformed_frag_real_B[warp_mma_k % 2],
                             warp_loaded_frag_real_A[warp_mma_k % 2],
                             warp_loaded_frag_real_B[warp_mma_k % 2]);
          warp_mma.transform(warp_transformed_frag_imag_A[warp_mma_k % 2],
                             warp_transformed_frag_imag_B[warp_mma_k % 2],
                             warp_loaded_frag_imag_A[warp_mma_k % 2],
                             warp_loaded_frag_imag_B[warp_mma_k % 2]);

          }
        // Issue global->shared copies for the next stage
        int group_start_iteration_A, group_start_iteration_B;

        if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
          group_start_iteration_A = 0;
          group_start_iteration_B = 0;
        } else {
          group_start_iteration_A =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
          group_start_iteration_B =
              (warp_mma_k + 1) * Detail::kAccessesPerGroupB;
        }

        copy_tiles_and_advance(iterator_A_real, iterator_A_imag, iterator_B_real, iterator_B_imag, group_start_iteration_A,
                               group_start_iteration_B);

        if (Detail::kStagedAccumulation) {
          warp_mma_planar_complex(
            warp_mma,
            tmp_accum, 
            warp_transformed_frag_real_A[warp_mma_k % 2],
            warp_transformed_frag_imag_A[warp_mma_k % 2],
            warp_transformed_frag_real_B[warp_mma_k % 2],
            warp_transformed_frag_imag_B[warp_mma_k % 2]
          );
          

          if (warp_mma_k == 0) {
            add_planar(accum, tmp_accum);
            tmp_accum.clear();
          }
        } else {
          warp_mma_planar_complex(
            warp_mma,
            accum, 
            warp_transformed_frag_real_A[warp_mma_k % 2],
            warp_transformed_frag_imag_A[warp_mma_k % 2],
            warp_transformed_frag_real_B[warp_mma_k % 2],
            warp_transformed_frag_imag_B[warp_mma_k % 2]
          );
        }

        if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
          warp_mma.transform(warp_transformed_frag_real_A[(warp_mma_k + 1) % 2],
                             warp_transformed_frag_real_B[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_real_A[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_real_B[(warp_mma_k + 1) % 2]);
          
          warp_mma.transform(warp_transformed_frag_imag_A[(warp_mma_k + 1) % 2],
                             warp_transformed_frag_imag_B[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_imag_A[(warp_mma_k + 1) % 2],
                             warp_loaded_frag_imag_B[(warp_mma_k + 1) % 2]);
        }

        if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
          // Inserts a fence to group cp.async instructions into stages.
          cutlass::arch::cp_async_fence();

          // Waits until kStages-2 stages of cp.async have committed
          arch::cp_async_wait<Base::kStages - 2>();
          __syncthreads();

          // Move to the next stage
          iterator_A_real.advance();
          iterator_A_imag.advance();
          iterator_B_real.advance();
          iterator_B_imag.advance();

          this->smem_iterator_A.add_tile_offset({0, 1});
          this->smem_iterator_B.add_tile_offset({1, 0});

          // Add negative offsets to return iterators to the 'start' of the
          // circular buffer in shared memory
          if (smem_write_stage_idx == (Base::kStages - 1)) {
            this->smem_iterator_A.add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B.add_tile_offset({-Base::kStages, 0});
            smem_write_stage_idx = 0;
          } else {
            ++smem_write_stage_idx;
          }

          if (smem_read_stage_idx == (Base::kStages - 1)) {
            this->warp_tile_iterator_A_.add_tile_offset(
                {0, -Base::kStages * Policy::kPartitionsK *
                        Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_.add_tile_offset(
                {-Base::kStages * Policy::kPartitionsK *
                     Base::kWarpGemmIterations,
                 0});
            smem_read_stage_idx = 0;
          } else {
            ++smem_read_stage_idx;
          }

          --gemm_k_iterations;
        }
      }

    }

    if (Detail::kStagedAccumulation) {
      add_planar(accum, tmp_accum);
    }
  

    // Insert fence and wait for all outstanding cp.async operations to commit.
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

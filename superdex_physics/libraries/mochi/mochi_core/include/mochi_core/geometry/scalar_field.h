/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <mochi_core/geometry/any_shape.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>

#include <type_traits>

namespace mochi {

/** @brief Defines how the sampler should behave outside of the grid region. */
enum class GridExtrapolation {
  /** @brief Extrapolation is not supported. An assertion will be thrown if sampling outside of the
     grid region. */
  Unsupported,
  /** @brief Clamp sample to the grid region. */
  Clamp,
  /** @brief When outside of the grid region, provide an upper bound of the true distance. */
  UpperBound,
  /** @brief When outside of the grid region, provide a lower bound of the true distance. */
  LowerBound,
  /** @brief Number of grid extrapolation enum values. */
  Count
};

/** @brief Empty utility class to make the passing of compile-time sampler options to trilinear
 * sampler function easier. */
template <GridExtrapolation kExtrapolationType>
struct TrilinearSamplerOptions {};

/** @brief Scalar field implemented through a dense volumetric grid of samples interpolated using
tri-linear interpolation. Samples are locate at the CORNERS of each voxel. */
template <typename T>
class DenseGrid3D {
 public:
  using Scalar = T;
  using Scalar3 = NdArray<T, 3>;

  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "Only 32-bit and 64-bit types are supported");
  using IType = std::conditional_t<sizeof(T) == 4, int, int64_t>; // Same size as T

  /**
   * @brief Create a DenseGrid3D and initialize it
   *
   * @param gridDims Defines the grid of sample points. The minimum grid (one voxel) has eight
   * samples located at the corner points.
   * @param gridBounds Defines the volume of the grid. The sample with index (0,0,0) will be located
   * at gridBounds.GetMin(). The sample with index (gridDims[0]-1, gridDims[1]-1, gridDims[2]-1)
   * will be located at gridBounds.GetMax().
   * @param negativeValueBounds The volume within the grid bounds where all scalar values are known
   * to be <= zero. For example, in an SDF grid, this is the bounds of the source mesh (the surface
   * with signed distance of zero). If you don't know, then set this equal to gridBounds.
   * @param optionalValues If you already have the array of values, you can move it into the
   * DenseGrid3d. Size must be the product of gridDims.
   */
  DenseGrid3D(
      Int3 const& gridDims,
      Aabb const& gridBounds,
      Aabb const& negativeValueBounds,
      DynamicArray<T> optionalValues = {});

  /** @brief Get the bounding volume spanned by the 3D grid. */
  Aabb GetBounds() const {
    return _bounds;
  }

  /**
   * @brief Get the bounding volume within the grid that contains all points with a scalar value
   * that is <= zero. For example, in an SDF grid, this is the bounds of the source mesh (the
   * surface with signed distance of zero). This Aabb can be equal to the grid bounds, but it cannot
   * extend outside the grid bounds.
   */
  Aabb GetNegativeValueBounds() const {
    return _negativeValueBounds;
  }

  /** @brief Gets the grid sample nearest to the specified point. */
  Int3 GetNearestVoxelIndexAt(Real3 const& point) const;

  /** @brief Gets the value of a specified grid sample. */
  inline T& operator()(Int3 const& index);
  inline T const& operator()(Int3 const& index) const;
  inline T& operator()(int x, int y, int z);
  inline T const& operator()(int x, int y, int z) const;

  template <int kBatchSize>
  MOCHI_FORCE_INLINE Simd<T, kBatchSize>
  operator()(Simd<IType, kBatchSize> x, Simd<IType, kBatchSize> y, Simd<IType, kBatchSize> z) const;

  /** @brief Gets the spatial position corresponding to a grid sample. */
  Real3 GetPointOf(Int3 const& index) const;

  /** @brief The dimensions of the grid in samples (voxels) number. */
  Int3 GetDimensions() const {
    return _shape;
  }

  /** @brief The position of the extreme lower corner of the grid. */
  Real3 GetGridLowerCorner() const {
    return _bounds.GetMin();
  }

  /** @brief The position of the extreme upper corner of the grid. */
  Real3 GetGridUpperCorner() const {
    return _bounds.GetMax();
  }

  /** @brief The voxel size */
  Real3 GetVoxelSize() const {
    return _delta;
  }

  /** @brief Gets the raw data. Sample ijk is located at offset shape[2]*shape[1]*i + shape[1]*j + k
   */
  inline Span<T const> GetData() const {
    return _data;
  }
  inline Span<T> GetData() {
    return _data;
  }

  /** @brief Clamps the specified sample index to the boundaries of the grid. */
  Int3 ClampIndex(Int3 const& index) const;

  /** @brief Gets the offset at the internal data vector corresponding to given sample index. */
  int GetOffsetForIndex(Int3 const& index) const;
  template <typename Idx>
  Idx GetOffsetForIndex(Idx x, Idx y, Idx z) const;

  bool Contains(Real3 const& point) const;

  /**
   * @brief Sample a batch of points using trilinear interpolation.
   *
   * @tparam kBatchSize The number of points to sample. Type Simd<T, kBatchSize> must be supported.
   * @tparam kExtrapolationType The type of extrapolation to use outside of the grid region.
   * @tparam kComputeValues If true, the values are computed and returned via outValues.
   * @tparam kComputeGradients If true, the gradients are computed and returned via outGradients.
   * @param points The points to sample in transposed format (vectors of Xs, Ys, and Zs).
   * @param outValues Used to return the sampled values if kComputeValues is true.
   * @param outGradients Used to return the sampled gradients if kComputeGradients is true.
   */
  template <
      int kBatchSize,
      GridExtrapolation kExtrapolationType,
      bool kComputeValues = true,
      bool kComputeGradients = false>
  MOCHI_FORCE_INLINE void TrilinearSampleBatch(
      NdArray<Simd<real, kBatchSize>, 3> const& points,
      Simd<T, kBatchSize>* outValues,
      NdArray<Simd<T, kBatchSize>, 3>* outGradients = nullptr) const;

  /** @brief Sample using trilinear interpolation with the given options. */
  template <GridExtrapolation kExtrapolationType>
  void TrilinearSample(
      Span<Real3 const> points,
      Span<T> outValues,
      TrilinearSamplerOptions<kExtrapolationType>) const;

  /**
   * @brief Sample gradient using trilinear interpolation with the given options.
   * @note For points outside the grid region, the gradient is an approximation (the gradient of the
   * SDF at the closest point on the grid boundary is assumed to be zero).
   */
  template <GridExtrapolation kExtrapolationType>
  void TrilinearSampleGradient(
      Span<Real3 const> points,
      Span<Scalar3> outGradients,
      TrilinearSamplerOptions<kExtrapolationType>) const;

 private:
  /** @brief Computes the parametric coordinates within the grid volume of 'kBatchSize' points. */
  template <int kBatchSize, GridExtrapolation kExtrapolationType>
  MOCHI_FORCE_INLINE void GetClampedParametricCoordsAt(
      NdArray<Simd<real, kBatchSize>, 3> const& point,
      NdArray<Simd<T, kBatchSize>, 3>& outParametric,
      NdArray<Simd<IType, kBatchSize>, 3>& outLowerIndex,
      NdArray<Simd<IType, kBatchSize>, 3>& outUpperIndex) const;

  /** @brief SIMDify up to kBatchSize points. */
  template <int kBatchSize>
  MOCHI_FORCE_INLINE NdArray<Simd<T, kBatchSize>, 3> VectorizePoints(
      Span<Real3 const> points) const;

  Aabb _bounds = {};
  Aabb _negativeValueBounds = {};
  Real3 _delta = {};
  Real3 _deltaInv = {}; // 1 / _delta
  Int3 _shape = {};
  DynamicArray<T> _data;
};

/**
 * @brief Samples an SDF from a grid with support for extrapolation outside of the grid.
 * @note When outside the grid, it provides an upper bound of the true distance (not the true
 * distance).
 */
template <typename T>
struct TrilinearSdfGridUpperBoundSampler {
  using Scalar3 = typename DenseGrid3D<T>::Scalar3;
  using Options = TrilinearSamplerOptions<GridExtrapolation::UpperBound>;
  MOCHI_FORCE_INLINE void
  operator()(DenseGrid3D<T> const& grid, Span<Real3 const> points, Span<T> outValues) const {
    grid.TrilinearSample(points, outValues, Options{});
  }
  MOCHI_FORCE_INLINE void
  Gradient(DenseGrid3D<T> const& grid, Span<Real3 const> points, Span<Scalar3> outGradients) const {
    grid.TrilinearSampleGradient(points, outGradients, Options{});
  }
};

/** @brief Samples an SDF from a grid without support for extrapolation outside of the grid. */
template <typename T>
struct TrilinearSdfGridInteriorSampler {
  using Scalar3 = typename DenseGrid3D<T>::Scalar3;
  using Options = TrilinearSamplerOptions<GridExtrapolation::Unsupported>;
  MOCHI_FORCE_INLINE void
  operator()(DenseGrid3D<T> const& grid, Span<Real3 const> points, Span<T> outValues) const {
    grid.TrilinearSample(points, outValues, Options{});
  }
  MOCHI_FORCE_INLINE void
  Gradient(DenseGrid3D<T> const& grid, Span<Real3 const> points, Span<Scalar3> outGradients) const {
    grid.TrilinearSampleGradient(points, outGradients, Options{});
  }
};

} // namespace mochi

#include "scalar_field_inl.h"

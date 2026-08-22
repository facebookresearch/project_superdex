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

#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/lagrange_polynomials.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/profile.h>
#include <mochi_core/utils/simd.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/vmatrix.h>

#include <limits>
#include <type_traits>

namespace mochi {

template <typename T>
DenseGrid3D<T>::DenseGrid3D(
    Int3 const& gridDims,
    Aabb const& gridBounds,
    Aabb const& negativeValueBounds,
    DynamicArray<T> optionalValues) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ASSERT(
      (gridDims[0] >= 2) && (gridDims[1] >= 2) && (gridDims[2] >= 2),
      "Invalid grid dimensions. The smallest grid has 8 samples for the corners of one voxel.");
  _bounds = gridBounds;
  // Clamp negativeValueBounds to always be contained within gridBounds
  _negativeValueBounds = Aabb{
      Max(gridBounds.VGetMin(), negativeValueBounds.VGetMin()),
      Min(gridBounds.VGetMax(), negativeValueBounds.VGetMax())};
  _shape = gridDims;
  using D3 = NdArray<double, 3>;
  auto delta = StaticCast<D3>(gridBounds.GetSize()) / StaticCast<D3>(gridDims - 1);
  _delta = StaticCast<Real3>(delta);
  _deltaInv = StaticCast<Real3>(1.0 / delta);

  auto dataSize = gridDims[0] * gridDims[1] * gridDims[2];
  if (!optionalValues.empty()) {
    MOCHI_ASSERT(optionalValues.size() == dataSize, "Array size mismatch");
    _data = std::move(optionalValues);
  } else {
    _data.resize(dataSize);
  }
}

// Get the parametric coordinates and grid indices of 'kBatchSize' sample points.
//   outParametric has range (0,0,0) to (1,1,1)
//   outLowerIndex has range (0,0,0) to (_shape - 2)
//   outUpperIndex has range (1,1,1) to (_shape - 1)
template <typename T>
template <int kBatchSize, GridExtrapolation kExtrapolationType>
MOCHI_FORCE_INLINE void DenseGrid3D<T>::GetClampedParametricCoordsAt(
    NdArray<Simd<real, kBatchSize>, 3> const& points,
    NdArray<Simd<T, kBatchSize>, 3>& outParametric,
    NdArray<Simd<IType, kBatchSize>, 3>& outLowerIndex,
    NdArray<Simd<IType, kBatchSize>, 3>& outUpperIndex) const {
  using TVec = Simd<T, kBatchSize>;
  using IVec = Simd<IType, kBatchSize>;
  static_assert(
      TVec::kIsSupported && IVec::kIsSupported,
      "GetClampedParametricCoordsAt requires a type with SIMD support");
  static_assert(std::is_same_v<T const, real const>, "Configuration not supported");

  Real3 const minBound = _bounds.GetMin();
  for (int i = 0; i < 3; ++i) {
    TVec const coord = (points[i] - minBound[i]) * _deltaInv[i];
    TVec const coordFloor = Floor(coord);
    outLowerIndex[i] = StaticCast<IVec>(coordFloor);
    outParametric[i] = coord - coordFloor;

    // Clamp to the grid in case the points are outside or within floating point error of the
    // boundary. The lower clamp is only needed if extrapolation is supported. The upper clamp is
    // always needed (for the case points[i] == maxBound[i]).
    if constexpr (kExtrapolationType != GridExtrapolation::Unsupported) {
      // Lower clamp only needed if extrapolation is supported.
      IVec const clampLower = (outLowerIndex[i] < IVec{0});
      outLowerIndex[i] = Select(clampLower, IVec{0}, outLowerIndex[i]);
      outParametric[i] = Select(clampLower, TVec{0}, outParametric[i]);
    } else {
      MOCHI_ASSERT_VERBOSE(AllTrue(outLowerIndex[i] >= IVec{0}));
    }
    IVec const maxLowerIndex{_shape[i] - 2};
    IVec const clampUpper = (outLowerIndex[i] > maxLowerIndex);
    outLowerIndex[i] = Select(clampUpper, maxLowerIndex, outLowerIndex[i]);
    outParametric[i] = Select(clampUpper, TVec{1}, outParametric[i]);
    outUpperIndex[i] = outLowerIndex[i] + 1;
  }
}

template <typename T>
MOCHI_FORCE_INLINE Int3 DenseGrid3D<T>::GetNearestVoxelIndexAt(Real3 const& point) const {
  return ClampIndex(StaticCast<Int3>(Round((point - _bounds.GetMin()) * _deltaInv)));
}

template <typename T>
MOCHI_FORCE_INLINE Int3 DenseGrid3D<T>::ClampIndex(Int3 const& index) const {
  return mochi::Clamp(index, Int3{0, 0, 0}, _shape - Int3{1, 1, 1});
}

template <typename T>
template <typename Idx>
MOCHI_FORCE_INLINE Idx DenseGrid3D<T>::GetOffsetForIndex(Idx x, Idx y, Idx z) const {
  MOCHI_ASSERT_VERBOSE(AllTrue(x >= 0) && AllTrue(x < _shape[0]), "Coordinates out of bounds");
  MOCHI_ASSERT_VERBOSE(AllTrue(y >= 0) && AllTrue(y < _shape[1]), "Coordinates out of bounds");
  MOCHI_ASSERT_VERBOSE(AllTrue(z >= 0) && AllTrue(z < _shape[2]), "Coordinates out of bounds");
  return (x * _shape[1] * _shape[2]) + (y * _shape[2]) + (z);
}

template <typename T>
MOCHI_FORCE_INLINE int DenseGrid3D<T>::GetOffsetForIndex(Int3 const& index) const {
  return GetOffsetForIndex(index[0], index[1], index[2]);
}

template <typename T>
MOCHI_FORCE_INLINE T& DenseGrid3D<T>::operator()(Int3 const& index) {
  return _data[GetOffsetForIndex(index)];
}

template <typename T>
MOCHI_FORCE_INLINE T const& DenseGrid3D<T>::operator()(Int3 const& index) const {
  return _data[GetOffsetForIndex(index)];
}

template <typename T>
MOCHI_FORCE_INLINE T& DenseGrid3D<T>::operator()(int x, int y, int z) {
  return _data[GetOffsetForIndex(x, y, z)];
}

template <typename T>
MOCHI_FORCE_INLINE T const& DenseGrid3D<T>::operator()(int x, int y, int z) const {
  return _data[GetOffsetForIndex(x, y, z)];
}

template <typename T>
template <int kBatchSize>
MOCHI_FORCE_INLINE Simd<T, kBatchSize> DenseGrid3D<T>::operator()(
    Simd<IType, kBatchSize> x,
    Simd<IType, kBatchSize> y,
    Simd<IType, kBatchSize> z) const {
  using TVec = Simd<T, kBatchSize>;
  return LoadIndexed<TVec>(_data.data(), GetOffsetForIndex(x, y, z));
}

template <typename T>
MOCHI_FORCE_INLINE Real3 DenseGrid3D<T>::GetPointOf(Int3 const& index) const {
  return _bounds.GetMin() + (StaticCast<Real3>(index) * _delta);
}

template <typename T>
MOCHI_FORCE_INLINE bool DenseGrid3D<T>::Contains(Real3 const& point) const {
  return ContainsPoint(_bounds, point);
}

template <typename T>
template <int kBatchSize>
MOCHI_FORCE_INLINE NdArray<Simd<T, kBatchSize>, 3> DenseGrid3D<T>::VectorizePoints(
    Span<Real3 const> points) const {
  using TVec = Simd<T, kBatchSize>;
  using TVec3 = NdArray<TVec, 3>;
  auto const numPoints = isize(points);
  MOCHI_ASSERT_VERBOSE(numPoints > 0 && numPoints <= kBatchSize, "Invalid number of points.");
  if (numPoints == kBatchSize) {
    TVec3 out;
    LoadTransposed(&points[0][0], out[0], out[1], out[2]);
    return out;
  } else {
    int const i0 = 0;
    int const i1 = (1 < numPoints) ? 1 : i0;
    [[maybe_unused]] int const i2 = (2 < numPoints) ? 2 : i0;
    [[maybe_unused]] int const i3 = (3 < numPoints) ? 3 : i0;
    [[maybe_unused]] int const i4 = (4 < numPoints) ? 4 : i0;
    [[maybe_unused]] int const i5 = (5 < numPoints) ? 5 : i0;
    [[maybe_unused]] int const i6 = (6 < numPoints) ? 6 : i0;
    [[maybe_unused]] int const i7 = i0;
    if constexpr (kBatchSize == 8) {
      return {
          TVec{
              points[i0][0],
              points[i1][0],
              points[i2][0],
              points[i3][0],
              points[i4][0],
              points[i5][0],
              points[i6][0],
              points[i7][0]},
          TVec{
              points[i0][1],
              points[i1][1],
              points[i2][1],
              points[i3][1],
              points[i4][1],
              points[i5][1],
              points[i6][1],
              points[i7][1]},
          TVec{
              points[i0][2],
              points[i1][2],
              points[i2][2],
              points[i3][2],
              points[i4][2],
              points[i5][2],
              points[i6][2],
              points[i7][2]}};
    } else if constexpr (kBatchSize == 4) {
      return {
          TVec{points[i0][0], points[i1][0], points[i2][0], points[i3][0]},
          TVec{points[i0][1], points[i1][1], points[i2][1], points[i3][1]},
          TVec{points[i0][2], points[i1][2], points[i2][2], points[i3][2]}};
    } else {
      static_assert(kBatchSize == 2, "Unsupported batch size");
      return {
          TVec{points[i0][0], points[i1][0]},
          TVec{points[i0][1], points[i1][1]},
          TVec{points[i0][2], points[i1][2]}};
    }
  }
}

template <typename T>
template <
    int kBatchSize,
    GridExtrapolation kExtrapolationType,
    bool kComputeValues,
    bool kComputeGradients>
MOCHI_FORCE_INLINE void DenseGrid3D<T>::TrilinearSampleBatch(
    NdArray<Simd<real, kBatchSize>, 3> const& points,
    Simd<T, kBatchSize>* outValues,
    NdArray<Simd<T, kBatchSize>, 3>* outGradients) const {
  using TVec = Simd<T, kBatchSize>;
  using IVec = Simd<IType, kBatchSize>;
  using TVec3 = NdArray<TVec, 3>;
  using IVec3 = NdArray<IVec, 3>;

  static_assert(
      std::is_floating_point_v<T> && TVec::kIsSupported && IVec::kIsSupported,
      "Trilinear sampler only supported for floating point types with SIMD support");
  static_assert(std::is_same_v<T const, real const>, "Configuration not supported");
  static_assert(kComputeValues || kComputeGradients, "Must compute something");

  if constexpr (
      (kExtrapolationType == GridExtrapolation::Unsupported) && MOCHI_ASSERT_VERBOSE_ENABLED) {
    for (int i = 0; i < 3; ++i) {
      MOCHI_ASSERT_VERBOSE(
          AllTrue((points[i] >= _bounds.GetMin()[i]) && (points[i] <= _bounds.GetMax()[i])),
          "Point falls outside grid.");
    }
  }

  // Get grid indices and parametric coordinates in the range (0,0,0) to (1,1,1).
  TVec3 param;
  IVec3 indexLo, indexUp;
  GetClampedParametricCoordsAt<kBatchSize, kExtrapolationType>(points, param, indexLo, indexUp);

  // Get values at voxel coordinates. Indexing: v[x][y][z] with x,y,z ∈ {0,1}.
  TVec const v000 = operator()(indexLo[0], indexLo[1], indexLo[2]);
  TVec const v001 = operator()(indexLo[0], indexLo[1], indexUp[2]);
  TVec const v010 = operator()(indexLo[0], indexUp[1], indexLo[2]);
  TVec const v011 = operator()(indexLo[0], indexUp[1], indexUp[2]);
  TVec const v100 = operator()(indexUp[0], indexLo[1], indexLo[2]);
  TVec const v101 = operator()(indexUp[0], indexLo[1], indexUp[2]);
  TVec const v110 = operator()(indexUp[0], indexUp[1], indexLo[2]);
  TVec const v111 = operator()(indexUp[0], indexUp[1], indexUp[2]);

  // Z-direction differences. Reused for both interpolation and gradient.
  TVec const dv00dz = v001 - v000;
  TVec const dv01dz = v011 - v010;
  TVec const dv10dz = v101 - v100;
  TVec const dv11dz = v111 - v110;

  // First, interpolate along Z to get edge values.
  TVec const v00 = v000 + param[2] * dv00dz;
  TVec const v01 = v010 + param[2] * dv01dz;
  TVec const v10 = v100 + param[2] * dv10dz;
  TVec const v11 = v110 + param[2] * dv11dz;

  // Then interpolate along Y to get face values.
  TVec const v0 = v00 + param[1] * (v01 - v00);
  TVec const v1 = v10 + param[1] * (v11 - v10);

  if constexpr (kComputeValues) {
    // Finally interpolate along X.
    TVec values = v0 + param[0] * (v1 - v0);

    // Extrapolation outside of the grid bounds
    if constexpr (
        kExtrapolationType == GridExtrapolation::UpperBound ||
        kExtrapolationType == GridExtrapolation::LowerBound) {
      // Add additional term comprising of the distance of the point to the boundary of the grid
      auto const minBounds = BroadcastEach<TVec>(_bounds.GetMin());
      auto const maxBounds = BroadcastEach<TVec>(_bounds.GetMax());
      TVec3 const pointsClamped = Clamp(points, minBounds, maxBounds);
      TVec const extrapolatedDistSqr = NormSqr(points - pointsClamped);
      MOCHI_ASSERT_VERBOSE(
          AllTrue((values >= TVec{}) || VEqual(extrapolatedDistSqr, TVec{})),
          "Clamped extrapolated point has negative signed distance.");

      if constexpr (kExtrapolationType == GridExtrapolation::UpperBound) {
        // Note: Sqrt is expensive and could be avoided by using the L1 distance for the
        // extrapolation term. This would provide a looser upper bound but that may be acceptable
        // for collision purposes.
        values += Sqrt(extrapolatedDistSqr);
      } else {
        /**
        NOTE: A tighter lower bound is given by:

        TVec const extrDist0 = Abs(points[0] - pointsClamped[0]);
        TVec const extrDist1 = Abs(points[1] - pointsClamped[1]);
        TVec const extrDist2 = Abs(points[2] - pointsClamped[2]);
        values = Select(
            values < TVec{},
            values,
            Sqrt(
                Sqr(values) + Sqr(extrDist0) + Sqr(extrDist1) + Sqr(extrDist2) +
                TVec(2) * values * Min(extrDist0, extrDist1, extrDist2)));

        This is slightly more expensive and doesn't seem to reduce the number of culled points in
        practice.
        */

        values = Select(values < TVec{}, values, Sqrt(Sqr(values) + extrapolatedDistSqr));
      }
    } else {
      static_assert(
          kExtrapolationType == GridExtrapolation::Unsupported ||
              kExtrapolationType == GridExtrapolation::Clamp,
          "Extrapolation type not implemented");
    }

    MOCHI_ASSERT_VERBOSE(outValues != nullptr);
    *outValues = values;
  } else {
    MOCHI_ASSERT_VERBOSE(
        outValues == nullptr, "To return values, kComputeValues must be true at compile time.");
  }

  if constexpr (kComputeGradients) {
    // Gradient computation using chain rule:
    // For f(x,y,z) = lerp_x(v0(y,z), v1(y,z), x):
    // df/dx = v1 - v0 (since v0, v1 are independent of x)
    // df/dy = (1-x) * dv0/dy + x * dv1/dy = (1-x) * (v01 - v00) + x * (v11 - v10)
    // df/dz = (1-x) * dv0/dz + x * dv1/dz
    //       = (1-x) * [(1-y) * (v001-v000) + y * (v011-v010)]
    //           + x * [(1-y) * (v101-v100) + y * (v111-v110)]

    // Gradient in parametric space (then scale by deltaInv).
    TVec const one = TVec{1};
    TVec const dfdx = v1 - v0;
    TVec const dfdy = (one - param[0]) * (v01 - v00) + param[0] * (v11 - v10);

    TVec const dfdz = (one - param[0]) * ((one - param[1]) * dv00dz + param[1] * dv01dz) +
        param[0] * ((one - param[1]) * dv10dz + param[1] * dv11dz);

    TVec3 grad = {dfdx * _deltaInv[0], dfdy * _deltaInv[1], dfdz * _deltaInv[2]};

    if constexpr (
        kExtrapolationType == GridExtrapolation::Clamp ||
        kExtrapolationType == GridExtrapolation::UpperBound) {
      // This sampler is using extrapolation.
      auto const bounds = GetBounds();
      auto const minBounds = bounds.GetMin();
      auto const maxBounds = bounds.GetMax();

      // Indicator per coordinate of the set on which to evaluate the gradient using the grid.
      TVec3 const interiorIndicator = {
          (points[0] >= minBounds[0]) & (points[0] <= maxBounds[0]),
          (points[1] >= minBounds[1]) & (points[1] <= maxBounds[1]),
          (points[2] >= minBounds[2]) & (points[2] <= maxBounds[2])};

      if constexpr (kExtrapolationType == GridExtrapolation::Clamp) {
        grad = {
            Select(interiorIndicator[0], grad[0], SimdZero<TVec>()),
            Select(interiorIndicator[1], grad[1], SimdZero<TVec>()),
            Select(interiorIndicator[2], grad[2], SimdZero<TVec>())};

      } else if constexpr (kExtrapolationType == GridExtrapolation::UpperBound) {
        auto const halfExtents = bounds.GetHalfExtents();
        auto const center = bounds.GetCenter();

        // Direction from the closest point on the boundary of the grid.
        TVec3 exteriorGrad = {
            points[0] - center[0] +
                Clamp(center[0] - points[0], TVec{-halfExtents[0]}, TVec{halfExtents[0]}),
            points[1] - center[1] +
                Clamp(center[1] - points[1], TVec{-halfExtents[1]}, TVec{halfExtents[1]}),
            points[2] - center[2] +
                Clamp(center[2] - points[2], TVec{-halfExtents[2]}, TVec{halfExtents[2]})};
        TVec const exteriorGradNorm = Norm(exteriorGrad) + std::numeric_limits<T>::min();

        grad = {
            Select(interiorIndicator[0], grad[0], exteriorGrad[0] / exteriorGradNorm),
            Select(interiorIndicator[1], grad[1], exteriorGrad[1] / exteriorGradNorm),
            Select(interiorIndicator[2], grad[2], exteriorGrad[2] / exteriorGradNorm)};
      }
    } else {
      static_assert(
          kExtrapolationType == GridExtrapolation::Unsupported, "Unimplemented extrapolation type");
    }

    MOCHI_ASSERT_VERBOSE(outGradients != nullptr);
    *outGradients = grad;
  } else {
    MOCHI_ASSERT_VERBOSE(
        outGradients == nullptr,
        "To return gradients, kComputeGradients must be true at compile time.");
  }
}

template <typename T>
template <GridExtrapolation kExtrapolationType>
void DenseGrid3D<T>::TrilinearSample(
    Span<Real3 const> points,
    Span<T> outValues,
    TrilinearSamplerOptions<kExtrapolationType>) const {
  int constexpr kBatchSize = Simd<T>::kSize;
  using TVec = Simd<T, kBatchSize>;
  int const numPoints = isize(points);
  for (int i = 0; i < numPoints; i += kBatchSize) {
    int const count = Min(kBatchSize, numPoints - i);
    auto const pts = VectorizePoints<kBatchSize>(points.subspan(i, count));
    TVec values MOCHI_NO_INIT;
    TrilinearSampleBatch<kBatchSize, kExtrapolationType>(pts, &values);
    Store(&outValues[i], values, count);
  }
}

template <typename T>
template <GridExtrapolation kExtrapolationType>
void DenseGrid3D<T>::TrilinearSampleGradient(
    Span<Real3 const> points,
    Span<Scalar3> outGradients,
    TrilinearSamplerOptions<kExtrapolationType>) const {
  int constexpr kBatchSize = Simd<T>::kSize;
  using TVec = Simd<T, kBatchSize>;
  using TVec3 = NdArray<TVec, 3>;
  int const numPoints = isize(points);
  for (int i = 0; i < numPoints; i += kBatchSize) {
    int const count = Min(kBatchSize, numPoints - i);
    auto const pts = VectorizePoints<kBatchSize>(points.subspan(i, count));
    TVec3 grad MOCHI_NO_INIT;
    TrilinearSampleBatch<
        kBatchSize,
        kExtrapolationType,
        /*kComputeValues*/ false,
        /*kComputeGradients*/ true>(pts, nullptr, &grad);
    if (count == kBatchSize) {
      StoreTransposed(&outGradients[i][0], grad[0], grad[1], grad[2]);
    } else {
      for (int ii = 0; ii < count; ++ii) {
        outGradients[i + ii] = {Get(grad[0], ii), Get(grad[1], ii), Get(grad[2], ii)};
      }
    }
  }
}

} // namespace mochi

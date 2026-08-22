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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/matrix_operations.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/task_scheduler.h>

#include <algorithm>
#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace mochi::ai {

/*
 * f(z) = z
 */
template <typename T>
struct IdentityActivation {
  template <typename U>
  MOCHI_FORCE_INLINE void operator()(U& /*z*/) const {}

  template <typename U>
  MOCHI_FORCE_INLINE void operator()(U& /*z*/, U& dfdz) const {
    dfdz = U(1);
  }
};

/*
 * f(z) = z                   , if z >= 0
 * f(z) = alpha * (exp(z) - 1), otherwise
 */
template <typename T>
struct ELUActivation {
  T alpha = T(1);

  MOCHI_FORCE_INLINE void operator()(T& z) const {
    // Exponential is not evaluated when z is non-negative
    z = (z < T(0)) ? alpha * std::exp(z) - alpha : z;
  }

  template <int N>
  MOCHI_FORCE_INLINE void operator()(Simd<T, N>& z) const {
    static_assert(Simd<T, N>::kIsSupported);
    auto const vAlpha = Simd<T, N>(alpha);
    auto w = vAlpha * Exp(z) - vAlpha;
    auto const negMask = (z < 0);
    z = Simd<T, N>::Select(negMask, w, z);
  }

  MOCHI_FORCE_INLINE void operator()(T& z, T& dfdz) const {
    if (z < T(0)) {
      // Only one exponential is evaluated
      dfdz = alpha * std::exp(z);
      z = dfdz - alpha;
    } else {
      // Exponential is not evaluated when z is non-negative
      dfdz = T(1);
    }
  }

  template <int N>
  MOCHI_FORCE_INLINE void operator()(Simd<T, N>& z, Simd<T, N>& dfdz) const {
    static_assert(Simd<T, N>::kIsSupported);
    // Only one exponential is evaluated
    auto const negMask = (z < 0);
    dfdz = Select(negMask, alpha * Exp(z), Simd<T, N>(1));
    z = Select(negMask, dfdz - alpha, z);
  }
};

/*
 * f(z) = z, if z >= 0
 * f(z) = 0, otherwise
 */
template <typename T>
struct ReLUActivation {
  template <typename U>
  MOCHI_FORCE_INLINE void operator()(U& z) const {
    z = Select(z < 0, U(0), z);
  }

  template <typename U>
  MOCHI_FORCE_INLINE void operator()(U& z, U& dfdz) const {
    U const zero(0);
    auto const negMask = (z < 0);
    z = Select(negMask, zero, z);
    dfdz = Select(negMask, zero, U(1));
  }
};

namespace details {

template <typename T, typename SimdFunctor, typename ScalarFunctor>
MOCHI_FORCE_INLINE void SimdFor(T* start, T* end, SimdFunctor const& fv, ScalarFunctor const& fs) {
  MOCHI_ASSERT_VERBOSE(end >= start, "Invalid range.");
  size_t len = end - start;
  size_t pos = 0;
  if constexpr (Simd<T>::kIsSupported) {
    for (; pos + Simd<T>::kSize <= len; pos += Simd<T>::kSize) {
      auto vz = Load<Simd<T>>(start + pos);
      fv(vz);
      Store(start + pos, vz);
    }
  }
  for (; pos < len; ++pos) {
    fs(*(start + pos));
  }
}

template <typename MatType, typename T>
void ShiftColumnsInPlace(MatType&& mat, ColumnVector<T> const& shift) {
  if constexpr (mochi::details::MatTraits<MatType>::kMajorDir == krylov::Direction::ColMajor) {
    for (int c = 0; c < mat.Cols(); ++c) {
      mat.Col(c) += shift;
    }
  } else {
    static_assert(mochi::details::MatTraits<MatType>::kMajorDir == krylov::Direction::RowMajor);
    for (int r = 0; r < mat.Rows(); ++r) {
      auto sb = shift(r, 0);
      auto* ptr = &mat(r, 0);
      if constexpr (Simd<T>::kIsSupported) {
        auto vb = Simd<T>(sb);
        auto fv = [&vb](Simd<T>& v) { v += vb; };
        auto fs = [&sb](T& val) { val += sb; };
        SimdFor(ptr, ptr + mat.Cols(), fv, fs);
      } else {
        for (int c = 0; c < mat.Cols(); ++c) {
          mat(r, c) += sb;
        }
      }
    }
  }
}

template <typename MatrixType>
bool HasContiguousMemory(MatrixType const& Z) {
  if constexpr (mochi::details::MatTraits<MatrixType>::kMajorDir == krylov::Direction::ColMajor) {
    return Z.Rows() == Z.LeadDim();
  } else {
    static_assert(mochi::details::MatTraits<MatrixType>::kMajorDir == krylov::Direction::RowMajor);
    return Z.Cols() == Z.LeadDim();
  }
}

template <typename ActivationFunctor, typename Input>
void ActivationInPlace(ActivationFunctor const& f, Input&& Z) {
  if (HasContiguousMemory(Z)) {
    size_t len = Z.Rows() * Z.Cols();
    auto* ptr = Z.Data();
    SimdFor(ptr, ptr + len, f, f);
  } else {
    if constexpr (mochi::details::MatTraits<Input>::kMajorDir == krylov::Direction::ColMajor) {
      for (int c = 0; c < Z.Cols(); ++c) {
        auto* ptr = &Z(0, c);
        SimdFor(ptr, ptr + Z.Rows(), f, f);
      }
    } else {
      static_assert(mochi::details::MatTraits<Input>::kMajorDir == krylov::Direction::RowMajor);
      for (int r = 0; r < Z.Rows(); ++r) {
        auto* ptr = &Z(r, 0);
        SimdFor(ptr, ptr + Z.Cols(), f, f);
      }
    }
  }
}

template <typename ActivationFunctor, typename Input, typename ActivationDerivative>
void ActivationInPlace(ActivationFunctor const& f, Input&& Z, ActivationDerivative&& dfdZ) {
  static_assert(
      mochi::details::MatTraits<Input>::kMajorDir ==
      mochi::details::MatTraits<ActivationDerivative>::kMajorDir);
  MOCHI_ASSERT_VERBOSE(Z.Rows() == dfdZ.Rows() && Z.Cols() == dfdZ.Cols(), "Inconsistent sizes.");
  using T = typename mochi::details::MatTraits<Input>::Scalar;
  if ((HasContiguousMemory(Z)) && (HasContiguousMemory(dfdZ))) {
    size_t len = Z.Rows() * Z.Cols();
    size_t pos = 0;
    auto* zPtr = Z.Data();
    auto* dfPtr = dfdZ.Data();
    if constexpr (Simd<T>::kIsSupported) {
      for (; pos + Simd<T>::kSize <= len; pos += Simd<T>::kSize) {
        auto vz = Load<Simd<T>>(zPtr + pos);
        auto vdz = Load<Simd<T>>(dfPtr + pos);
        f(vz, vdz);
        Store(zPtr + pos, vz);
        Store(dfPtr + pos, vdz);
      }
    }
    for (; pos < len; ++pos) {
      f(*(zPtr + pos), *(dfPtr + pos));
    }
  } else {
    if constexpr (mochi::details::MatTraits<Input>::kMajorDir == krylov::Direction::ColMajor) {
      for (int c = 0; c < Z.Cols(); ++c) {
        int r = 0;
        if constexpr (Simd<T>::kIsSupported) {
          for (; r + Simd<T>::kSize <= Z.Rows(); r += Simd<T>::kSize) {
            auto vz = Load<Simd<T>>(&Z(r, c));
            auto vdz = Load<Simd<T>>(&dfdZ(r, c));
            f(vz, vdz);
            Store(&Z(r, c), vz);
            Store(&dfdZ(r, c), vdz);
          }
        }
        for (; r < Z.Rows(); ++r) {
          f(Z(r, c), dfdZ(r, c));
        }
      }
    } else {
      static_assert(mochi::details::MatTraits<Input>::kMajorDir == krylov::Direction::RowMajor);
      for (int r = 0; r < Z.Rows(); ++r) {
        int c = 0;
        if constexpr (Simd<T>::kIsSupported) {
          for (; c + Simd<T>::kSize <= Z.Cols(); c += Simd<T>::kSize) {
            auto vz = Load<Simd<T>>(&Z(r, c));
            auto vdz = Load<Simd<T>>(&dfdZ(r, c));
            f(vz, vdz);
            Store(&Z(r, c), vz);
            Store(&dfdZ(r, c), vdz);
          }
        }
        for (; c < Z.Cols(); ++c) {
          f(Z(r, c), dfdZ(r, c));
        }
      }
    }
  }
}

template <typename T>
using AnyActivation = std::variant<IdentityActivation<T>, ELUActivation<T>, ReLUActivation<T>>;

} // namespace details

/*
MlpLayer implements Y = f(W * X + b), where:
- Y is of shape (n,k)
- W is of shape (n,m)
- X is of shape (m,k)
- b is of shape (n,1)
- f is an activation function
*/
template <typename T>
class MlpLayer {
 public:
  static_assert(!std::is_const_v<T>, "Scalar type must be non-const");
  using WeightType = Matrix<T>;
  using BiasType = ColumnVector<T>;

  MlpLayer() = default;

  MlpLayer(WeightType&& W, BiasType&& b, details::AnyActivation<T> const& activation)
      : _inputDim(W.Cols()),
        _outputDim(W.Rows()),
        _W(std::move(W)),
        _b(std::move(b)),
        _activation(activation) {
    MOCHI_ASSERT_VERBOSE(
        _W.Rows() == _b.Rows(), "Weight and bias must have the same number of rows.");
  }

  auto const& GetActivation() const {
    return _activation;
  }

  auto WeightsView() {
    return AsView(_W);
  }

  auto WeightsView() const {
    return AsConstView(_W);
  }

  auto WeightsConstView() const {
    return AsConstView(_W);
  }

  auto BiasView() {
    return AsView(_b);
  }

  auto BiasView() const {
    return AsConstView(_b);
  }

  auto BiasConstView() const {
    return AsConstView(_b);
  }

  int InputDim() const {
    return _inputDim;
  }

  int OutputDim() const {
    return _outputDim;
  }

  /// @brief Randomize the weights and bias drawing from a uniform distribution in [minVal, maxVal).
  void Randomize(
      T minVal = T(-0.01),
      T maxVal = T(0.01),
      unsigned int seed_W = mochi::GetRandomSeed(),
      unsigned int seed_b = mochi::GetRandomSeed()) {
    _W.SetRandom(seed_W, minVal, maxVal);
    _b.SetRandom(seed_b, minVal, maxVal);
  }

  /// @brief Compute Y = f(G) = f(W * X + b), where f is the activation function. If
  /// kComputeActivationDerivative is true, the derivative of the activation function dYdG = df/dG
  /// is also computed.
  template <
      bool kComputeActivationDerivative = false,
      typename Input,
      typename Output,
      typename... ActivationDerivative>
  void Forward(Input const& X, Output&& Y, ActivationDerivative&&... dYdG) const {
    Y = _W * X;
    details::ShiftColumnsInPlace(Y, _b);
    if constexpr (kComputeActivationDerivative) {
      static_assert(sizeof...(ActivationDerivative) == 1);
      ApplyActivation(Y, dYdG...);
    } else {
      ApplyActivation(Y);
    }
  }

  /// @brief Compute dS/dX = dS/dY * dY/dG * dG/dX, where Y = f(W * X + b), G = W * X + b,
  /// and S = S(Y) is an arbitrary function of Y.
  /// @param[in,out] dSdY dS/dY at input and dS/dG at output. Size: (Sdim*batchSize) x _outputDim.
  /// @param[out] dSdX dS/dX. Size: (Sdim*batchSize) x _inputDim.
  /// @param[in] dYdG dY/dG. Size: _outputDim x batchSize.
  /// @param[in] Sdim Number of dimensions of S.
  /// @param[in] batchSize Batch size.
  template <int kSDimAtCT = krylov::kDynamic, typename DsDy, typename DsDx, typename DyDg>
  void BackwardJacobian(DsDy&& dSdY, DsDx&& dSdX, DyDg const& dYdG, int Sdim, int batchSize) const {
    MOCHI_ASSERT_VERBOSE(
        dSdY.Rows() == Sdim * batchSize && dYdG.Rows() == _outputDim && dYdG.Cols() == batchSize,
        "Inconsistent sizes.");

    // Jacobian w.r.t. activation inputs (dS/dG).
    static_assert(mochi::details::MatTraits<DyDg>::kMajorDir == krylov::Direction::ColMajor);
    for (int i = 0; i < batchSize; ++i) {
      krylov::ApplyBlockDiagonal<T>(
          MakeConstSpan(dYdG.Col(i)), // dYdG is stored col-major.
          dSdY.Transpose().template MiddleCols<kSDimAtCT>(i * Sdim, Sdim),
          dSdY.Transpose().template MiddleCols<kSDimAtCT>(i * Sdim, Sdim));
    }

    // Jacobian w.r.t. layer inputs (dS/dX).
    dSdX = dSdY * _W;
  }

 protected:
  template <typename MatType, class... Args>
  void ApplyActivation(MatType&& Y, Args&&... args) const {
    switch (_activation.index()) {
      case 0:
        details::ActivationInPlace(std::get<0>(_activation), Y, std::forward<Args>(args)...);
        break;
      case 1:
        details::ActivationInPlace(std::get<1>(_activation), Y, std::forward<Args>(args)...);
        break;
      case 2:
        details::ActivationInPlace(std::get<2>(_activation), Y, std::forward<Args>(args)...);
        break;
      default:
        MOCHI_ASSERT(false, "Unsupported activation type.");
        break;
    }
  }

 protected:
  int _inputDim = 0;
  int _outputDim = 0;
  WeightType _W = {};
  BiasType _b = {};
  details::AnyActivation<T> _activation = {};
};

template <typename T>
class Mlp {
  static_assert(!std::is_const_v<T>, "Scalar type must be non-const");
  using WeightType = typename MlpLayer<T>::WeightType;
  using BiasType = typename MlpLayer<T>::BiasType;

 public:
  explicit Mlp(DynamicArray<MlpLayer<T>>&& layersIn) : _layers(std::move(layersIn)) {
    MOCHI_ASSERT(!_layers.empty(), "MLP class requires at least 1 layer.");
#if MOCHI_ASSERT_VERBOSE_ENABLED
    for (int i = 0; i + 1 < isize(_layers); ++i) {
      MOCHI_ASSERT_VERBOSE(
          _layers[i + 1].InputDim() == _layers[i].OutputDim(), "Inconsistent layer dimensions.");
    }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
  }

  MlpLayer<T> const& GetLayer(int index) const {
    return _layers[index];
  }

  int InputDim() const {
    return _layers.front().InputDim();
  }

  int OutputDim() const {
    return _layers.back().OutputDim();
  }

  /// @brief Maximum output dimension of any of the layers.
  int MaxLayerOutputDim() const {
    int maxOutDim = 0;
    for (auto const& l : _layers) {
      maxOutDim = Max(maxOutDim, l.OutputDim());
    }
    return maxOutDim;
  }

  /// @brief FLOPs per point for forward evaluation.
  auto ForwardFlopsPerPoint() const {
    unsigned long long flops = 0;
    for (auto const& l : _layers) {
      MOCHI_ASSERT_VERBOSE(l.InputDim() > 0 && l.OutputDim() > 0, "Layer must not be empty.");
      flops += (2 * l.InputDim() - 1) * l.OutputDim();
    }
    return flops;
  }

  /// @brief FLOPs per point for Jacobian evaluation (using backward computation).
  auto JacobianFlopsPerPoint() const {
    unsigned long long flops = 0;
    for (auto const& l : _layers) {
      MOCHI_ASSERT_VERBOSE(l.InputDim() > 0 && l.OutputDim() > 0, "Layer must not be empty.");
      flops += (2 * l.OutputDim() - 1) * l.InputDim();
    }
    return flops * OutputDim();
  }

  /// @brief Perform forward evaluation Y = Y(X) on a batch of points.
  /// @param[in] X Inputs. Size: inputDim x batchSize.
  /// @param[out] Y Forward evaluation Y(X). Size: outputDim x batchSize.
  /// @details Store X and Y as col-major for best performance.
  template <typename Input, typename Output>
  void Forward(Input const& X, Output&& Y) const {
    MOCHI_PROFILE_SCOPE();
    MOCHI_ASSERT_VERBOSE(X.Cols() == Y.Cols(), "Inconsistent batch size.");
    auto const batchSize = X.Cols();
    if (batchSize == 0) {
      return;
    }

    auto const minTaskBatchSize = // At least 1e6 FLOPs per task.
        Clamp(static_cast<int>(1000000 / ForwardFlopsPerPoint()), 1, batchSize);
    ParallelForRange(
        "MlpForward", 0, batchSize, minTaskBatchSize, batchSize, [&](int pBegin, int pEnd) {
          MOCHI_PROFILE_SCOPE_N("MlpForwardWorkerTask");
          // Preserve compile-time batch size only if it's 1. If it's >1, some of the points may not
          // be in this task.
          constexpr auto kTaskBatchSizeAtCT = (mochi::details::MatTraits<Input>::kNumCols == 1 ||
                                               mochi::details::MatTraits<Output>::kNumCols == 1)
              ? 1
              : krylov::kDynamic;
          auto const taskBatchSize = pEnd - pBegin;
          BatchForward<false>(
              X.template MiddleCols<kTaskBatchSizeAtCT>(pBegin, taskBatchSize),
              Y.template MiddleCols<kTaskBatchSizeAtCT>(pBegin, taskBatchSize));
        });
  }

  /// @brief Perform forward Y = Y(X) and Jacobian dY(X)/dX evaluation on a batch of points.
  /// @param[in] X Inputs. Size: inputDim x batchSize.
  /// @param[out] Y Forward evaluation Y(X). Size: outputDim x batchSize.
  /// @param[out] dYdX Jacobian dY(X)/dX. Size: (outputDim*batchSize) x inputDim.
  /// @details Store X and Y as col-major and dYdX as row-major for best performance.
  template <typename Input, typename Output, typename Jacobian>
  void ForwardAndJacobian(Input const& X, Output&& Y, Jacobian&& dYdX) const {
    // TODO(T186289618): Compute the Jacobian via forward products instead of backward products if
    // inputDim < outputDim.
    MOCHI_PROFILE_SCOPE();
    [[maybe_unused]] auto const inputDim = X.Rows();
    auto const outputDim = Y.Rows();
    auto const batchSize = Y.Cols();
    MOCHI_ASSERT_VERBOSE(X.Cols() == Y.Cols(), "Inconsistent batch size.");
    MOCHI_ASSERT_VERBOSE(
        (dYdX.Rows() == outputDim * batchSize) && (dYdX.Cols() == inputDim), "Inconsistent sizes.");
    MOCHI_ASSERT(!_layers.empty(), "At least one layer required.");
    if (batchSize == 0) {
      return;
    }

    auto const minTaskBatchSize = Clamp( // At least 1e6 FLOPs per task.
        static_cast<int>(1000000 / (ForwardFlopsPerPoint() + JacobianFlopsPerPoint())),
        1,
        batchSize);
    ParallelForRange(
        "MlpForwardAndJacobian",
        0,
        batchSize,
        minTaskBatchSize,
        batchSize,
        [&](int pBegin, int pEnd) {
          MOCHI_PROFILE_SCOPE_N("MlpForwardAndJacobianWorkerTask");
          // Preserve compile-time batch size only if it's 1. If it's >1, some of the points may not
          // be in this task.
          constexpr auto kTaskBatchSizeAtCT = (mochi::details::MatTraits<Input>::kNumCols == 1 ||
                                               mochi::details::MatTraits<Output>::kNumCols == 1)
              ? 1
              : krylov::kDynamic;
          auto const taskBatchSize = pEnd - pBegin;

          // Perform forward evaluation.
          auto const dhdG = BatchForward<true>(
              X.template MiddleCols<kTaskBatchSizeAtCT>(pBegin, taskBatchSize),
              Y.template MiddleCols<kTaskBatchSizeAtCT>(pBegin, taskBatchSize));

          // Storage for the Jacobian w.r.t. the hidden state. Different points in the batch are
          // stored in different rows. Use row-major storage since 3D tensor operations that cannot
          // be expressed as matrix-matrix products are more efficient this way. Preallocate enough
          // columns to avoid reallocation (the downside is it may degrade cache hit rate in
          // matrix-matrix products).
          // TODO(T186289618): 'SetIdentity' call introduces non-negligible overhead.
          constexpr auto kOutputDimAtCT = mochi::details::MatTraits<Output>::kNumRows;
          RowMatrix<T> dYdhPrev(outputDim * taskBatchSize, MaxLayerOutputDim()),
              dYdh(outputDim * taskBatchSize, MaxLayerOutputDim());
          for (int i = 0; i < taskBatchSize; ++i) {
            dYdhPrev.Block(outputDim * i, 0, outputDim, outputDim).SetIdentity();
          }

          // Perform Jacobian evaluation (backward computation).
          for (int l = isize(_layers) - 1; l >= 0; --l) {
            auto const& layer = _layers[l];
            std::swap(dYdh, dYdhPrev);
            if (l == 0) {
              layer.template BackwardJacobian<kOutputDimAtCT>(
                  dYdh.LeftCols(layer.OutputDim()),
                  dYdX.MiddleRows(outputDim * pBegin, outputDim * taskBatchSize),
                  dhdG[l],
                  outputDim,
                  taskBatchSize);
            } else {
              layer.template BackwardJacobian<kOutputDimAtCT>(
                  dYdh.LeftCols(layer.OutputDim()),
                  dYdhPrev.LeftCols(layer.InputDim()),
                  dhdG[l],
                  outputDim,
                  taskBatchSize);
            }
          }
        });
  }

 protected:
  /// @brief Perform forward evaluation Y = Y(X) on a batch of points.
  /// @param[in] X Inputs. Size: inputDim x batchSize.
  /// @param[out] Y Forward evaluation Y(X). Size: outputDim x batchSize.
  /// @return If kComputeActivationDerivative is true, returns a vector with the derivative of the
  /// activation in each layer. If it's false, there is no return.
  template <bool kComputeActivationDerivative, typename Input, typename Output>
  [[nodiscard]] auto BatchForward(Input const& X, Output&& Y) const {
    MOCHI_ASSERT_VERBOSE(X.Cols() == Y.Cols(), "Inconsistent batch size.");
    MOCHI_ASSERT(!_layers.empty(), "At least one layer required.");
    constexpr auto kBatchSizeAtCT = Max(
        mochi::details::MatTraits<Input>::kNumCols, mochi::details::MatTraits<Output>::kNumCols);
    auto const batchSize = Y.Cols();
    int const numLayers = isize(_layers);

    // Storage for hidden state and activation derivative. Different points are stored in different
    // columns. Col-major storage is more efficient.
    Matrix<T, krylov::kDynamic, kBatchSizeAtCT> hPrev(MaxLayerOutputDim(), batchSize),
        h(MaxLayerOutputDim(),
          batchSize); // Preallocate enough rows to avoid reallocation. The downside is it may
                      // degrade cache hit rate in matrix-matrix products.
    DynamicArray<Matrix<T, krylov::kDynamic, kBatchSizeAtCT>> dhdG(numLayers);

    for (int l = 0; l < numLayers; ++l) {
      auto const& layer = _layers[l];
      auto& dhdGLayer = dhdG[l];
      if constexpr (kComputeActivationDerivative) {
        dhdGLayer.Resize(layer.OutputDim(), batchSize);
      }
      if (l == 0) {
        if (l + 1 == numLayers) {
          layer.template Forward<kComputeActivationDerivative>(X, Y, dhdGLayer);
        } else {
          layer.template Forward<kComputeActivationDerivative>(
              X, h.TopRows(layer.OutputDim()), dhdGLayer);
        }
      } else {
        std::swap(hPrev, h);
        if (l + 1 == numLayers) {
          layer.template Forward<kComputeActivationDerivative>(
              hPrev.TopRows(layer.InputDim()), Y, dhdGLayer);
        } else {
          layer.template Forward<kComputeActivationDerivative>(
              hPrev.TopRows(layer.InputDim()), h.TopRows(layer.OutputDim()), dhdGLayer);
        }
      }
    }

    if constexpr (kComputeActivationDerivative) {
      return dhdG;
    }
  }

 protected:
  /// @brief Vector with layers. The input layer is first and the output layer is last.
  DynamicArray<MlpLayer<T>> _layers;
};

} // namespace mochi::ai

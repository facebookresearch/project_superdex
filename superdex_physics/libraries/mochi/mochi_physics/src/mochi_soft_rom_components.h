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

#include "mochi_discretization_components.h"

#include <mochi_core/ai/mlp.h>
#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/rom/rom.h>
#include <mochi_core/utils/graph.h>
#include <mochi_physics/mochi_physics_experimental.h>

#include <deque>
#include <memory>
#include <optional>
#include <utility>

namespace mochi {

// ROM actors always have a rigid transform, but we differentiate between two categories:
// (a) ROMs with basis like PCA for which we must solve the rigid transform DOFs.
// (b) ROMs with basis like biharmonic for which we fix the rigid transform during the solve.
// This tag is used to make this distinction.
struct TagRomActorFixRigidTransformInSolve {};

struct CRomCommonProperties : NoCopy {
  explicit CRomCommonProperties(rom::ModelProperties const& properties) : value(properties) {}

  rom::ModelProperties value;
};

struct CRomProjectionStrategy : NoCopy {
  explicit CRomProjectionStrategy(experimental::RomProjectionStrategy const& strategy)
      : value(strategy) {}
  experimental::RomProjectionStrategy value = experimental::RomProjectionStrategy::Default;
};

/*
 * CRomLinearBasis is the component for managing ROM linear basis.
 *
 * - It is the single source of truth for the linear basis to use.
 *
 * - Supports using an alternative basis matrix (e.g., from adaptivity), while preserving a stable
 *   reference basis.
 *
 * - Use 'ShouldUseAlternativeBasis' to mark whether the alternative basis should be used the next
 *   time 'UpdateBasis' is called.
 *
 * - Use 'UpdateBasis' to actually update the active basis.
 *
 * The following conditions hold true at all times:
 *
 * 1. The reference basis is:
 *    - Initialized at construction.
 *    - Immutable after construction (never reassigned or modified).
 *
 * 2. The active basis is either the reference or the alternative, selected by a committed flag
 *    that 'UpdateBasis' advances from the staged selection set by 'ShouldUseAlternativeBasis'.
 *
 * 3. The alternative basis may or may not be valid at any time. If not valid, the active basis
 *    is the reference basis.
 *
 * 4. Any alternative basis must have the same number of rows as the default basis.
 *
 * Construction policy:
 * - The constructor only initializes the reference basis. This is intentional to enforce
 *   immutability of the reference basis and to require any alternative basis to be explicitly
 *   inserted and managed later.
 */
class CRomLinearBasis : public NoCopy {
 public:
  using MatrixT = RowMatrix<real>;

  explicit CRomLinearBasis(MatrixT&& M) : _referenceBasis(std::move(M)) {}

  explicit CRomLinearBasis(MatrixT const& M) : _referenceBasis(M) {}

  // Returns the number of rows in the current active basis matrix.
  int Rows() const {
    MOCHI_ASSERT(ActiveBasis().Rows() == _referenceBasis.Rows());
    return ActiveBasis().Rows();
  }

  // Returns the number of columns (modes) in the current active basis matrix.
  int NumModes() const {
    return ActiveBasis().Cols();
  }

  // Returns a const view of the current active basis matrix.
  auto ViewMatrix() const {
    return AsConstView(ActiveBasis());
  }

  // Returns a non-const reference to the alternative basis matrix.
  auto& GetAlternativeMatrix() {
    return _alternativeBasis;
  }

  // Marks whether the alternative basis should be used the next time 'UpdateBasis' is called.
  // NOTE: It does NOT update the active basis. 'UpdateBasis' does.
  void ShouldUseAlternativeBasis(bool value) {
    _stagingIsAlternative = value;
  }

  // Commits the staged selection set by 'ShouldUseAlternativeBasis' to the active basis.
  void UpdateBasis() {
    MOCHI_ASSERT(!_stagingIsAlternative || (_alternativeBasis.Rows() == _referenceBasis.Rows()));
    _activeIsAlternative = _stagingIsAlternative;
  }

 private:
  MatrixT const& ActiveBasis() const {
    return _activeIsAlternative ? _alternativeBasis : _referenceBasis;
  }

  MatrixT _referenceBasis = {};
  MatrixT _alternativeBasis = {};
  // Staged selection set by 'ShouldUseAlternativeBasis'. Takes effect on next 'UpdateBasis'.
  bool _stagingIsAlternative = false;
  // Committed selection advanced by 'UpdateBasis'. What the accessors return.
  bool _activeIsAlternative = false;
};

// Shift vector for affine ROMs.
struct CRomShiftVector : public NoCopy {
  using T = ColumnVector<real>;
  explicit CRomShiftVector(T&& vIn) : value(std::move(vIn)) {}
  explicit CRomShiftVector(T const& vIn) : value(vIn) {}

  T value;
};

struct CNeuralNetCromDecoderData : public NoCopy {
  explicit CNeuralNetCromDecoderData(
      Matrix<real>&& inputMatrixIn,
      Matrix<real>&& outputMatrixIn,
      RowMatrix<real>&& outputJacobianMatrixIn)
      : inputMatrix(std::move(inputMatrixIn)),
        networkOutput(std::move(outputMatrixIn)),
        networkOutputJacobian(std::move(outputJacobianMatrixIn)) {}

  Matrix<real> inputMatrix;
  Matrix<real> networkOutput;
  RowMatrix<real> networkOutputJacobian;
};

struct CNeuralNetCromModel : public NoCopy {
  std::optional<ai::Mlp<real>> encoder;
  std::optional<ColumnVector<real>> meanAndStdevForInputStandardize = {};
  ai::Mlp<real> decoder;
  ColumnVector<real> meanAndStdevForOutputInverseStandardize = {};

  explicit CNeuralNetCromModel(
      std::optional<ai::Mlp<real>> const& encoderNetworkIn,
      std::optional<ColumnVector<real>> const& inputStandardizeIn,
      ai::Mlp<real> const& decoderNetworkIn,
      ColumnVector<real> const& outputInverseStandardizeIn)
      : encoder(encoderNetworkIn),
        meanAndStdevForInputStandardize(inputStandardizeIn),
        decoder(decoderNetworkIn),
        meanAndStdevForOutputInverseStandardize(outputInverseStandardizeIn) {}

  explicit CNeuralNetCromModel(
      std::optional<ai::Mlp<real>>&& encoderNetworkIn,
      std::optional<ColumnVector<real>>&& inputStandardizeIn,
      ai::Mlp<real>&& decoderNetworkIn,
      ColumnVector<real>&& outputInverseStandardizeIn)
      : encoder(std::move(encoderNetworkIn)),
        meanAndStdevForInputStandardize(std::move(inputStandardizeIn)),
        decoder(std::move(decoderNetworkIn)),
        meanAndStdevForOutputInverseStandardize(std::move(outputInverseStandardizeIn)) {}
};

struct CRomModeAmplitudes {
  ColumnVector<real> value;
};

using RomVelocityMetadata = MatrixMetadata<MatrixSemantics::ReducedTangentSpaceVector>;
template <typename Scalar, TimeStep kStep>
using CRomVelocity = CDenseTimeSliceVector<Scalar, RomVelocityMetadata, kStep>;

/*
 * Auxiliary components that are needed for doing the rigid pivot transform for the ROMs
 * to avoid expensive allocations every single time
 */
struct CAuxiliaryPositionsForRomRigidTransform {
  ColumnVector<real> data = {};
  explicit CAuxiliaryPositionsForRomRigidTransform(ColumnVector<real>&& v) : data(std::move(v)) {}
};

namespace rom {
template <class StrategyType>
struct CDynamicSampleMeshStrategy : NoCopy {
  explicit CDynamicSampleMeshStrategy(StrategyType const& strategyIn) : strategy(strategyIn) {}
  StrategyType strategy;
};
} // namespace rom

// Component for basis adaptivity using contact forces.
struct CRomAdaptiveBasisContactForceInformed
    : experimental::ContactForceInformedRomAdaptivityParams,
      NoCopy {
  std::optional<RowMatrix<real>> requiredBasis = std::nullopt;
  RowMatrix<real> candidateBasis = {};
};

// Component for basis adaptivity using interpolation.
struct BasisInterpolationParameters {
  int step = 0;
  std::deque<ColumnVector<real>> onlineU;
  std::deque<ColumnVector<real>> onlineQ;
  real t = 0.2_r; // Interpolation parameter (0 = old basis, 1 = new basis)
  int dataDim = 20; // Data dimension for sliding window
};

// Component for basis adaptivity using neural network gradients.
struct CNeuralAffineRomStrategy : experimental::NeuralAffineRomParams, NoCopy {
  // Neural network components
  CNeuralNetCromModel neuralModel;
  ColumnVector<real> latentState;
  CNeuralNetCromDecoderData decoderData;
  BasisInterpolationParameters onlineBasisParams;

  explicit CNeuralAffineRomStrategy(
      experimental::NeuralAffineRomParams const& params,
      CNeuralNetCromModel&& model,
      ColumnVector<real>&& latentStateIn)
      : experimental::NeuralAffineRomParams(params),
        neuralModel(std::move(model)),
        latentState(std::move(latentStateIn)),
        decoderData(
            Matrix<real>(neuralModel.decoder.InputDim(), 0),
            Matrix<real>(3, 0),
            RowMatrix<real>(0, 0)) {}
};

/** @brief Component with ROM-FOM switching parameters. */
struct CRomFomSwitchingParams : NoCopy {
  CRomFomSwitchingParams() = delete;
  CRomFomSwitchingParams(experimental::RomFomSwitchingParams const& paramsIn) : params(paramsIn) {}
  experimental::RomFomSwitchingParams params;
};

/**
 * @brief Component to cache active elements/faces/nodes.
 * @details Needed to reuse the sample mesh when switching back from FOM to ROM.
 */
struct CSampleMeshCaching {
  std::unique_ptr<CActiveVolumeElements> activeVolumeElements;
  std::unique_ptr<CActiveBoundaryFaces> activeBoundaryFaces;
  std::unique_ptr<CActiveUniqueNodes> activeNodes;
};

namespace rom {
void InitializeOnce(entt::registry& reg);
}

} // namespace mochi

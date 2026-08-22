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

#include "mochi_soft_rom_init.h"
#include "mochi_actor_convergence.h"
#include "mochi_discretization_functions.h"
#include "mochi_hyper_reduction.h"
#include "mochi_rom_jacobian.h"
#include "mochi_soft_rom_components.h"
#include "mochi_soft_rom_polynomial_crom_systems.h"

#include <mochi_core/geometry/tetrahedral_map.h>
#include <mochi_core/linear_algebra/qr.h>
#include <mochi_core/mochi_config.h>
#include <mochi_core/rom/dynamic_sample_mesh_bsh_manager.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/profile.h>

#include <algorithm>
#include <charconv>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

using namespace mochi;
using namespace mochi::experimental;
using namespace mochi::rom;

static std::optional<RomData>
LoadRomData(std::string const& source, TetrahedralMeshShape const& shape, Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  // Polynomial CROMs are supported for any asset and therefore directly computed in Mochi.
  constexpr std::string_view kPolynomialCromPrefix = "polynomial_crom_order_";
  if (source.starts_with(kPolynomialCromPrefix)) {
    auto numberStr = source.substr(std::size(kPolynomialCromPrefix));
    int order = 0;
    char const* const begin = numberStr.data();
    char const* const end = begin + numberStr.size();
    auto const [ptr, ec] = std::from_chars(begin, end, order);
    MOCHI_ERROR_IF(
        numberStr.empty() || numberStr.front() < '0' || numberStr.front() > '9' ||
            ec != std::errc{} || ptr != end,
        error,
        "Failed to read the polynomial order from the ROM source string. Expected \"polynomial_crom_order_N\" where N is a non-negative integer.");
    MOCHI_ERROR_RETURN(error, {});
    return PolynomialCromData{.order = order};
  }

  auto data = [&]() -> std::optional<RomData> {
    // Load the data for this ROM
    auto const& romDatas = shape.GetRomData();
    auto it = romDatas.find(source);
    MOCHI_ERROR_IF(it == romDatas.end(), error, "Could not find specified ROM.");
    MOCHI_ERROR_RETURN(error, {});
    auto romData = it->second; // Copy data
    return romData;
  }();
  MOCHI_ERROR_RETURN(error, {});

  return data;
}

static std::pair<int, int> EmplaceRigidTransform(
    entt::registry& reg,
    entt::entity e,
    TetrahedralMesh const& mesh,
    bool addRigidDofs,
    int numFomDofs) {
  int transformDofsSize = 0;
  int transformPoseSize = 0;

  // Current and Previous rigid state are always needed, whether we solve for the rigid transform
  // or we fix it before the solve.
  reg.emplace<CRigidState<TimeStep::Current>>(e);
  reg.emplace<CRigidState<TimeStep::Previous>>(e);

  // Auxiliary positions needed when applying rigid transform.
  reg.emplace<CAuxiliaryPositionsForRomRigidTransform>(e, ColumnVector<real>::Zero(numFomDofs));

  if (addRigidDofs) {
    // The stage start rigid state is only needed if we solve for the rigid transform.
    reg.emplace<CRigidState<TimeStep::StageStart>>(e);
    transformDofsSize = RigidSize::kDAll;
    transformPoseSize = RigidSize::kAll;
  } else {
    reg.emplace<TagRomActorFixRigidTransformInSolve>(e);
  }

  Real3 rigidPivotPos = {};
  int rigidPivotEleIdx = -1;
  auto const& femLowVolDisc = reg.get<CFemVolumeDiscretizationP1Q1 const>(e);
  bool const grp = mesh.GetRigidPivot(rigidPivotEleIdx, rigidPivotPos);
  MOCHI_ASSERT(grp);
  MOCHI_ASSERT(rigidPivotEleIdx >= 0);
  MOCHI_ASSERT(rigidPivotEleIdx < isize(femLowVolDisc.femElements));
  reg.emplace_or_replace<CMeshPivot>(e, rigidPivotPos);

  auto const& pivotElement = femLowVolDisc.femElements[rigidPivotEleIdx];
  auto const& evalPt =
      reg.emplace_or_replace<CRigidTransformEvalPoint>(e, pivotElement, rigidPivotPos);
  MOCHI_ASSERT(
      evalPt.IsValid(), "The EvalPoint should be valid because the element index is valid.");

  return {transformDofsSize, transformPoseSize};
}

/* ////////////////////////////////////////////////////////////////////////////
 *
 * EMPLACE MODEL from LinearRomData
 *
 * //////////////////////////////////////////////////////////////////////////// */

static rom::ModelProperties EmplaceReducedModel(
    entt::registry& /*reg*/,
    entt::entity /*e*/,
    NeuralNetCromData const& /*data*/,
    std::shared_ptr<TetrahedralMeshShape const> /*shapePtr*/,
    bool /*addRigidDofs*/,
    Error& error) {
  MOCHI_ERROR_SET(error, "Neural CROMs are deprecated.");
  return {};
}

static rom::ModelProperties EmplaceReducedModel(
    entt::registry& reg,
    entt::entity e,
    LinearRomData const& data,
    std::shared_ptr<TetrahedralMeshShape const> shapePtr,
    bool addRigidDofs,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  rom::ModelProperties props;

  int const numModes = data.basis.Cols();
  int const numFomDofs = shapePtr->GetMesh()->GetNumNodes() * kSpaceDim3;

  auto const& basis = reg.emplace<CRomLinearBasis>(e, data.basis);
  MOCHI_ERROR_IF(
      basis.Rows() != numFomDofs,
      error,
      "The number of rows in the ROM basis matrix is inconsistent with the number of mesh nodes.");
  MOCHI_ERROR_RETURN(error, {});

  reg.emplace<CRomShiftVector>(e, ColumnVector<real>::Zero(numFomDofs));
  reg.emplace<CRomModeAmplitudes>(e, ColumnVector<real>::Zero(numModes));
  auto const [transformDofsSize, transformPoseSize] =
      EmplaceRigidTransform(reg, e, *shapePtr->GetMesh(), addRigidDofs, numFomDofs);
  props.baseDim = numModes;
  props.reducedDofsDim = numModes + transformDofsSize;
  props.reducedPoseDim = numModes + transformPoseSize;
  props.outputDim = numFomDofs;

  return props;
}

/* ////////////////////////////////////////////////////////////////////////////
 *
 * EMPLACE MODEL from PolynomialCromData
 *
 * //////////////////////////////////////////////////////////////////////////// */

static rom::ModelProperties EmplaceReducedModel(
    entt::registry& reg,
    entt::entity e,
    PolynomialCromData const& data,
    std::shared_ptr<TetrahedralMeshShape const> shapePtr,
    bool addRigidDofs,
    Error& error) {
  MOCHI_ERROR_RETURN(error, {});

  //
  // preconditions
  //
  MOCHI_ERROR_IF(addRigidDofs, error, "Polynomial CROM already includes a rigid transform.");
  MOCHI_ERROR_RETURN(error, {});
  MOCHI_ERROR_IF_NOT(
      (data.order >= 0 && data.order <= 4), error, "Polynomial CROM only supports order <=4");
  MOCHI_ERROR_RETURN(error, {});

  //
  // prepare and store model
  //
  LinearRomData linRomData;
  auto nodeCoordinates = shapePtr->GetMesh()->GetNodeCoordinates();

  // Orthogonalize monomial basis (via QR factorization) to improve numerical stability.
  ThinQR<real> qr(polynomial_crom::CreateBasisMatrix(data.order, nodeCoordinates));
  linRomData.basis.Reset(qr.Q());

  return EmplaceReducedModel(reg, e, linRomData, shapePtr, /* addRigidDofs */ false, error);
}

/* ////////////////////////////////////////////////////////////////////////////
 *
 * ADAPTIVITY
 *
 * //////////////////////////////////////////////////////////////////////////// */

static void EnableModelAdaptivity(
    entt::registry& reg,
    entt::entity e,
    RomAdaptivityParams const& adaptivityParams,
    std::shared_ptr<TetrahedralMeshShape const> shapePtr,
    Error& error) {
  MOCHI_ERROR_RETURN(error);

  //
  // 1. Check preconditions
  //
  MOCHI_ERROR_IF_NOT(
      reg.all_of<CRomLinearBasis>(e),
      error,
      "ROM adaptivity requires the starting model to be a linear ROM.");
  MOCHI_ERROR_RETURN(error);

  //
  // 2. Handle different adaptivity strategies using std::visit
  //
  std::visit(
      [&](auto const& s) {
        using ParamType = std::decay_t<decltype(s)>;
        if constexpr (std::is_same_v<ParamType, ContactForceInformedRomAdaptivityParams>) {
          //
          // Contact Force Informed Strategy
          //

          // Enable QueryType::NodeContactForces.
          AddRemoveOrRefComponentsForQuery(
              reg, e, QueryType::NodeContactForces, /*add*/ true, /*computeImmediately*/ false);

          // Load candidate source.
          auto romDataOpt = LoadRomData(std::string(s.candidateSource), *shapePtr, error);
          MOCHI_ERROR_RETURN(error);
          MOCHI_ASSERT(romDataOpt, "Invalid ROM data.");
          auto const& romDataVariantForSwap = romDataOpt.value();

          bool const candidateIsLinearRom =
              std::holds_alternative<LinearRomData>(romDataVariantForSwap);
          bool const candidateIsPolyCrom =
              std::holds_alternative<PolynomialCromData>(romDataVariantForSwap);
          MOCHI_ERROR_IF_NOT(
              candidateIsLinearRom || candidateIsPolyCrom,
              error,
              "ContactForceInformedRomAdaptivityParams requires the new model to be a linear ROM.");
          MOCHI_ERROR_RETURN(error);

          // Emplace adaptivity components.
          CRomAdaptiveBasisContactForceInformed adaptivityParamsComp;

          adaptivityParamsComp.numAdaptiveBasis = s.numAdaptiveBasis;
          auto const nodeCoordinates = shapePtr->GetMesh()->GetNodeCoordinates();

          // This strategy proposes a new set of modes by concatenating a basis of 1st order
          // polynomial modes (only if the rigid transform is FIXED during the solve) and a set of
          // modes adaptively chosen from a candidate basis. The 1st order polynomial modes are
          // needed to capture rotation and translation if the rigid transform is fixed during the
          // solve.
          bool const needBaselineModes = reg.all_of<TagRomActorFixRigidTransformInSolve>(e);
          if (needBaselineModes) {
            // Orthogonalize monomial basis (via QR factorization) to improve numerical stability.
            ThinQR<real> qrBaseline(
                polynomial_crom::CreateBasisMatrix(/*polyOrder*/ 1, nodeCoordinates));
            adaptivityParamsComp.requiredBasis = qrBaseline.Q();
          } else {
            adaptivityParamsComp.requiredBasis = std::nullopt;
          }

          if (candidateIsLinearRom) {
            auto const& romData = std::get<LinearRomData>(romDataVariantForSwap);
            adaptivityParamsComp.candidateBasis.Reset(romData.basis);
          } else if (candidateIsPolyCrom) {
            int const polyOrder = std::get<PolynomialCromData>(romDataVariantForSwap).order;
            MOCHI_ERROR_IF_NOT(
                polyOrder >= 2,
                error,
                "ContactForceInformedRomAdaptivityParams: Polynomial candidate basis order must be oder >= 2.");
            MOCHI_ERROR_RETURN(error);

            auto polyBasis = polynomial_crom::CreateBasisMatrix(polyOrder, nodeCoordinates);

            // For the candidate basis, we only keep the higher-order modes.
            int const lowOrderTerms =
                isize(polynomial_crom::ComputeMultiIndexSortedByTotalOrder(/*polyOrder*/ 1)) *
                kSpaceDim3;
            auto candidateBasis = polyBasis.RightCols(polyBasis.Cols() - lowOrderTerms);

            // Orthogonalize monomial basis (via QR factorization) to improve numerical stability.
            ThinQR<real> qrCandidate(candidateBasis);
            adaptivityParamsComp.candidateBasis.Reset(qrCandidate.Q());
          } else {
            MOCHI_ASSERT(false, "Unexpected case.");
          }

          auto const& adapBasisComp = reg.emplace<CRomAdaptiveBasisContactForceInformed>(
              e, std::move(adaptivityParamsComp));
          MOCHI_ERROR_IF(
              adapBasisComp.numAdaptiveBasis <= 0 ||
                  adapBasisComp.numAdaptiveBasis > adapBasisComp.candidateBasis.Cols(),
              error,
              "ContactForceInformedRomAdaptivityParams: numAdaptiveBasis must be > 0 and <= the num modes in the candidate basis.");
          MOCHI_ERROR_RETURN(error);

        } else if constexpr (std::is_same_v<ParamType, NeuralAffineRomParams>) {
          //
          // Neural Affine ROM Strategy
          //
          MOCHI_ERROR_IF(
              !MOCHI_USE_EIGEN && (s.method == NeuralAffineRomMethod::Interpolation),
              error,
              "NeuralAffineRomMethod::Interpolation is not supported in this build. To enable, include Eigen in your build setup and define MOCHI_USE_EIGEN=1");
          MOCHI_ERROR_RETURN(error);

          // Load neural CROM model from the specified source
          auto romDataOpt = LoadRomData(std::string(s.neuralCromSource), *shapePtr, error);
          MOCHI_ERROR_RETURN(error);
          MOCHI_ASSERT(romDataOpt, "Invalid ROM data.");
          auto const& romDataVariant = romDataOpt.value();

          MOCHI_ERROR_IF_NOT(
              std::holds_alternative<NeuralNetCromData>(romDataVariant),
              error,
              "NeuralAffineRomParams requires the neuralCromSource to be a neural network CROM model.");
          MOCHI_ERROR_RETURN(error);

          auto const& neuralData = std::get<NeuralNetCromData>(romDataVariant);

          // Validate that required optional fields are present for the selected method
          if (s.method == NeuralAffineRomMethod::CromEncoder ||
              s.method == NeuralAffineRomMethod::Interpolation) {
            MOCHI_ERROR_IF_NOT(
                neuralData.encoder.has_value(),
                error,
                "NeuralAffineRomMethod requires an encoder network, but the loaded model does not contain one.");
            MOCHI_ERROR_IF_NOT(
                neuralData.meanAndStdevForInputStandardize.has_value(),
                error,
                "NeuralAffineRomMethod requires input standardization data (mean and stdev), but the loaded model does not contain it.");
            MOCHI_ERROR_RETURN(error);
          }

          // Create neural model
          CNeuralNetCromModel neuralModel(
              neuralData.encoder,
              neuralData.meanAndStdevForInputStandardize,
              neuralData.decoder,
              neuralData.meanAndStdevForOutputInverseStandardize);

          // Emplace neural affine ROM strategy component with constructor
          ColumnVector<real> latentStateCopy = neuralData.latentStateForZeroDisplacement;
          reg.emplace<CNeuralAffineRomStrategy>(
              e, s, std::move(neuralModel), std::move(latentStateCopy));
        }
      },
      adaptivityParams);
}

/* ////////////////////////////////////////////////////////////////////////////
 *
 * OVERLOADS FOR PREPARING DYNAMIC HYPER-REDUCTION
 *
 * //////////////////////////////////////////////////////////////////////////// */

static void EmplaceDynamicSampleMeshStrategy(
    entt::registry& reg,
    entt::entity e,
    DynamicSampleMeshBsh const& s,
    Error& error) {
  MOCHI_ERROR_RETURN(error);
  MOCHI_ERROR_IF_NOT(
      s.maxSubsamplingDensity >= 0_r && s.maxSubsamplingDensity <= 1_r,
      error,
      "Invalid maximum subsampling density (maxSubsamplingDensity).");
  MOCHI_ERROR_RETURN(error);

  reg.emplace<rom::CDynamicSampleMeshStrategy<DynamicSampleMeshBsh>>(e, s);

  auto const& shapeComp = reg.get<CShape const>(e);
  auto const* tetMeshShape = assert_cast<TetrahedralMeshShape const*>(shapeComp.shape.get());
  auto const& bshs = tetMeshShape->GetBoundingSphereHierarchies();

  if (auto it = bshs.find(std::string(s.source)); it != bshs.end()) {
    reg.emplace<rom::hyper::CDynamicSampleMeshBshManager>(
        e, ContactSamplesBsh(it->second), s.anchorSelectionMode, s.maxColliderVelocity);

    // We require a far SDF evaluation for dynamic hyper-reduction.
    auto& farEval = reg.emplace_or_replace<CRequiresFarSdfEvaluation>(e);
    // Just to be safe if someone else requested this feature don't override them.
    farEval.maxDistance = Max(s.maxSdfCullDistance, farEval.maxDistance);
  } else {
    MOCHI_ERROR_SET(error, "Could not find requested BSH in the mesh.");
  }
}

static void InitializeHyperReductionIfApplicable(
    entt::registry& reg,
    entt::entity e,
    RomParams const& romParams,
    Error& error) {
  auto const hypRedParamsOpt = romParams.hyperReduction;
  if (!hypRedParamsOpt) {
    return;
  }

  // the CShape is emplaced by the soft actor initialization which is done before
  // this ROM initialization, so this component is/should be present.
  auto const& shapeComp = reg.get<CShape const>(e);
  auto const* tetMeshShape = assert_cast<TetrahedralMeshShape const*>(shapeComp.shape.get());

  /*
   * emplace the INITIAL active volume elements and boundary faces
   * these also compute/store the proper weighting for doing hyper-reduction
   */
  auto activeVolElemsAndBdFaces = std::visit(
      [&](auto& s) -> auto {
        auto const& activeVolElC =
            reg.emplace<CActiveVolumeElements>(e, CreateActiveVolumeElements(s, *tetMeshShape));
        MOCHI_ERROR_IF(activeVolElC.empty(), error, "Active volume elements must not be empty.");

        auto const& femBoundaryDisc = reg.get<CFemBoundaryDiscretization>(e);
        auto const& activeBoundaryFaces = reg.emplace<CActiveBoundaryFaces>(
            e, CreateActiveBoundaryFaces(s, *tetMeshShape, femBoundaryDisc));

        return std::make_pair(std::cref(activeVolElC), std::cref(activeBoundaryFaces));
      },
      hypRedParamsOpt->initializationStrategy);
  MOCHI_ERROR_RETURN(error);
  auto const& activeVolElems = activeVolElemsAndBdFaces.first;
  auto const& activeBdFaces = activeVolElemsAndBdFaces.second;

  /*
   * emplace the INITIAL active unique nodes
   */
  reg.emplace<CActiveUniqueNodes>(e, tetMeshShape->GetMesh(), activeVolElems, activeBdFaces);

  /*
   * IMPORTANT: updating the mass matrix needs to be done
   * AFTER emplacing active elements which internally store their weights,
   * and the same soft mass updating must be done every time the active
   * elements are changed
   */
  auto const& actorSnle = reg.get<CActorSnle const>(e);
  auto const& material = reg.get<CSoftMaterialParams const>(e);
  auto& femHighVolDisc = reg.get<CFemVolumeDiscretizationP1Q4>(e);
  auto const& l2g = reg.get<CLocal2GlobalMap const>(e);
  auto const& nbs = reg.get<CNodalBasedStructure const>(e);
  auto& massMatrix = reg.get<CMassMatrix>(e);
  auto const& sparsity = reg.get<CFullSparsityPattern const>(e);
  // rather than hardwiring the template, we can deduce it so it is more generic
  auto& perElemMass =
      reg.get<CPerElementMassMatrix<std::remove_cvref_t<decltype(femHighVolDisc)>>>(e);
  auto& lumpedMass = reg.get<CLumpedMassMatrix>(e);

  soft::UpdateSoftMass(
      actorSnle,
      l2g,
      nbs,
      material,
      femHighVolDisc,
      sparsity,
      massMatrix,
      perElemMass,
      lumpedMass,
      &activeVolElems);

  /*
   * if applicable, emplace the dynamic sample mesh strategy
   */
  if (hypRedParamsOpt->dynamicStrategy) {
    std::visit(
        [&](auto const& s) { EmplaceDynamicSampleMeshStrategy(reg, e, s, error); },
        hypRedParamsOpt->dynamicStrategy.value());
  }
}

/* ////////////////////////////////////////////////////////////////////////////
 *
 * OVERLOADS FOR PREPARING ROM/FOM SWITCHING
 *
 * //////////////////////////////////////////////////////////////////////////// */

static void EmplaceRomFomSwitchingParams(
    entt::registry& reg,
    entt::entity e,
    RomFomSwitchingTestOnlyParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(
      (params.fomToRomSwappingStep > params.romToFomSwappingStep) &&
          (params.romToFomSwappingStep > 0),
      error,
      "Invalid RomFomSwitchingTestOnlyParams.");
  MOCHI_ERROR_RETURN(error);

  reg.emplace<CRomFomSwitchingParams>(e, params);
}

static void EmplaceRomFomSwitchingParams(
    entt::registry& reg,
    entt::entity e,
    RomFomSwitchingContactInformedParams const& params,
    Error& error) {
  MOCHI_ERROR_IF_NOT(
      params.numCollisionPtsThreshold > 0, error, "Invalid RomFomSwitchingContactInformedParams.");
  MOCHI_ERROR_RETURN(error);

  reg.emplace<CRomFomSwitchingParams>(e, params);
}

/* ////////////////////////////////////////////////////////////////////////////
 *
 * INIT FUNCTION ENTRY POINT
 *
 * //////////////////////////////////////////////////////////////////////////// */

void mochi::rom::InitSoftActorRom(
    entt::registry& reg,
    entt::entity e,
    RomParams const& romParams,
    std::shared_ptr<TetrahedralMeshShape const> shapePtr,
    std::shared_ptr<DeepFlowShape const> flow,
    bool hasExternalRigidDofs,
    Error& error) {
  MOCHI_PROFILE_SCOPE();
  MOCHI_ERROR_RETURN(error);
  //
  // preconditions (todo: this is not complete)
  //
  MOCHI_ASSERT(shapePtr->GetMesh(), "Cannot initialize ROM actor without valid mesh.");
  MOCHI_ERROR_IF_NOT(
      static_cast<int>(romParams.romProjectionStrategy) >= 0 &&
          romParams.romProjectionStrategy < experimental::RomProjectionStrategy::Count,
      error,
      "Invalid ROM projection strategy (RomParams::romProjectionStrategy).");
  MOCHI_ERROR_RETURN(error);

  // -------------------------------------------------------------------
  // tag(s)
  // -------------------------------------------------------------------
  reg.emplace<TagRomActor>(e);
  reg.emplace<CRomProjectionStrategy>(e, romParams.romProjectionStrategy);

  // -------------------------------------------------------------------
  // 1. emplace ROM model
  // -------------------------------------------------------------------
  MOCHI_ERROR_IF(
      flow && romParams.adaptivity, error, "ROM adaptivity not supported with deep flow.");
  MOCHI_ERROR_RETURN(error);

  //
  // Step a: emplace the baseline model
  //
  // This defines the kind of ROM (linear biharmonic, linear polynomial, etc.)
  //
  auto romData = LoadRomData(std::string(romParams.source), *shapePtr, error);
  MOCHI_ERROR_RETURN(error);
  MOCHI_ASSERT(romData.has_value());
  bool const addRigidDofs = !hasExternalRigidDofs && romData->NeedsRigidTransformLayer();
  auto lambda = [&](auto const& v) -> rom::ModelProperties {
    return EmplaceReducedModel(reg, e, v, shapePtr, addRigidDofs, error);
  };
  auto const romProperties = std::visit(lambda, romData->GetVariant());
  MOCHI_ERROR_RETURN(error);

  /*
   After emplacing the ROM model, we now know the actual reduced dimension of this actor.
   This information is stored in "romProperties.{reducedPoseDim,reducedDofsDim}" which is returned
   above. Since the ROM initialization is effectively taking an underlying
   already-partially-initialized soft actor, at this point CActorDofInfo, if present, contains the
   DoF count of the full regular soft actor. We need to modify CActorDofInfo to store the reduced
   dimensions of the ROM actor.
   */
  CActorDofInfo& dofInfo = reg.emplace_or_replace<CActorDofInfo>(e);
  dofInfo.poseSize = romProperties.reducedPoseDim;
  dofInfo.dofsSize = romProperties.reducedDofsDim;

  //
  // Step b: Enable adaptivity of the model emplaced in (a), if needed
  //
  if (romParams.adaptivity) {
    auto const* recenteringParams = reg.try_get<CRecenteringParams>(e);
    MOCHI_ERROR_IF(
        recenteringParams && recenteringParams->useRecentering,
        error,
        "ROM adaptivity is not supported with recentering.");
    MOCHI_ERROR_RETURN(error);

    EnableModelAdaptivity(reg, e, romParams.adaptivity.value(), shapePtr, error);
  }

  // -------------------------------------------------------------------
  // 2. deal with sample mesh/hyper-reduction if needed
  // -------------------------------------------------------------------
  InitializeHyperReductionIfApplicable(reg, e, romParams, error);
  MOCHI_ERROR_RETURN(error);

  // Check the number of DoFs does not cause an under-determined problem.
  auto const* activeNodes = reg.try_get<CActiveUniqueNodes>(e);
  MOCHI_ERROR_IF(
      (activeNodes && (activeNodes->Count() * kSpaceDim3 < romProperties.reducedDofsDim)),
      error,
      "ROM problem is underdetermined. Please increase the number of active nodes or reduce the ROM "
      "dimensions so that the latter is smaller than or equal to the number of active node DoFs.");
  MOCHI_ERROR_RETURN(error);

  // -------------------------------------------------------------------
  // 3. more storage things
  // -------------------------------------------------------------------
  reg.emplace_or_replace<CReducedSparsityPattern>(
      e, MakeDenseSparsityGraph(romProperties.reducedDofsDim, romProperties.reducedDofsDim));

  // Enabled reduced SNLE.
  reg.get<CActorSnle>(e).EnableReduced(
      Matrix<real>::Zero(romProperties.reducedDofsDim, romProperties.reducedDofsDim));

  // Invalidate convergence weights from InitSoftActor.
  InvalidateActorConvergenceWeights(reg, e);

  reg.emplace<CRomCommonProperties>(e, romProperties);

  reg.emplace<CRomJacobian>(
      e, CRomJacobian::DenseT::Zero(romProperties.outputDim, romProperties.reducedDofsDim));

  // Emplace ROM velocity components. Only Previous and Current are needed. Used to compute sample
  // point velocities with deep flow.
  if (flow) {
    reg.emplace<CRomVelocity<real, TimeStep::Current>>(
        e, ColumnVector<real>::Zero(romProperties.reducedDofsDim));
    reg.emplace<CRomVelocity<real, TimeStep::Previous>>(
        e, ColumnVector<real>::Zero(romProperties.reducedDofsDim));
  }

  // -------------------------------------------------------------------
  // rendering
  if (romParams.surfaceMeshColor) {
    reg.emplace_or_replace<CMeshColor>(e, romParams.surfaceMeshColor.value());
  }

  // -------------------------------------------------------------------
  // ROM/FOM switching capability
  if (romParams.romFomSwitching) {
    MOCHI_ERROR_IF(flow, error, "ROM/FOM switching is not supported with deep flow.");
    MOCHI_ERROR_RETURN(error);

    // If there are active nodes, there is hyper-reduction. Store the sample mesh caching component
    // to recycle the sample mesh when switching back to ROM.
    if (activeNodes) {
      reg.emplace<CSampleMeshCaching>(e);
    }
    std::visit(
        [&](auto const& params) { EmplaceRomFomSwitchingParams(reg, e, params, error); },
        *romParams.romFomSwitching);
    MOCHI_ERROR_RETURN(error);
  }

  if (romParams.adaptivity) {
    // Unreachable unless initialization invariants regress. Fail with a clear message.
    using CurrentRomVelocity = CRomVelocity<real, TimeStep::Current>;
    using PreviousRomVelocity = CRomVelocity<real, TimeStep::Previous>;
    bool const hasRomVelocity = reg.any_of<CurrentRomVelocity, PreviousRomVelocity>(e);
    MOCHI_ASSERT(!hasRomVelocity, "Adaptivity not supported with ROM velocity.");
  }
}

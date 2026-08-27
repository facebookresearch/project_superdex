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

#include "mochi_attributes.h"
#include "mochi_ecs.h"

#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/geometry/any_shape.h>
#include <mochi_core/integration/integration_utils.h>
#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/utils/dtransform.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_physics/mochi_physics.h>

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace mochi {

// 3D space dimension, commonly used in this library.
constexpr int kSpaceDim3 = 3;

enum class TimeStep { Current = 1, StageStart = 0, Previous = -1 };

} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::TimeStep);
MOCHI_ENUM_ITEM(Current);
MOCHI_ENUM_ITEM(StageStart);
MOCHI_ENUM_ITEM(Previous);
MOCHI_ENUM_END();

namespace mochi {

/**************************************************************************
  ECS Tags
*/

struct TagStaticActor {};
struct TagRigidActor {};
struct TagSoftActor {};
struct TagShellActor {};
struct TagRodActor {};
struct TagDeformableActor {}; // Soft (including soft-skinned), Shell, and Rod actors
struct TagRomActor {};
struct TagArticulatedActor {};
struct TagArticulatedLinkActor {};
struct TagSoftSkinnedActor {};
struct TagBlendedActor {};
struct TagCompoundActor {};
struct TagFullyInitialized {}; // Added last when initializing an actor (for reative systems)
struct TagHasDeepFlowCollider {};

// Indicates that an actor participates in the point-cloud contact system.
struct TagUsePointCloudContact {};

// Indicates that a rod actor uses a triangular surface for contact instead of centerline contact.
// The surface may be the rod's visual mesh or a dedicated contact skin.
struct TagRodSurfaceContact {};

// Indicates that an actor's contact samples need a non-trivial Jacobian to back-propagate async
// forces through skinning/embedding (i.e. the actor produces a CSkinnedContactSnle and is
// dispatched to AssembleAsyncSkinnedContact). Currently emplaced for articulated actors with
// skinned contact meshes, soft-skinned actors, and rod actors that use a contact skin.
struct TagSkinnedContact {};

// Hides an actor from debug draw systems. Emplaced by DebugDraw::EnableActor.
struct TagExcludedFromDebugDraw {};

/**************************************************************************
  ECS Components for Any Actor Type
*/

/// @brief ECS component for NodalBasedStructure.
struct CNodalBasedStructure final : public NodalBasedStructure, public NoCopy {};

/// @brief Nodal-based assembly structure for soft actor boundary contact.
/// @details Element order matches @ref CBoundaryLocal2GlobalMap and @ref
/// CFemBoundaryDiscretization. Each boundary face assembles over the 4 volume nodes of its parent
/// tet so contact residuals and dresiduals scatter into the soft actor's volume DoFs. Sparse
/// indices target the actor's volume dresidual sparsity.
struct CBoundaryNodalBasedStructure final : public NodalBasedStructure, public NoCopy {
  explicit CBoundaryNodalBasedStructure(NodalBasedStructure&& other)
      : NodalBasedStructure(std::move(other)) {}
};

/// @brief Nodal-based assembly structure for codimensional deformable actor contact.
/// @details Element order matches @ref CContactLocal2GlobalMap and the contact FEM discretization.
/// Elements are the actor's contact elements, such as shell triangles or rod centerline segments.
/// Each element must expose the node count expected by the contact FEM element. Variable-width
/// connectivity must be padded consistently with the L2G stencil. Sparse indices target the actor's
/// full dresidual sparsity.
struct CContactNodalBasedStructure final : public NodalBasedStructure, public NoCopy {
  explicit CContactNodalBasedStructure(NodalBasedStructure&& other)
      : NodalBasedStructure(std::move(other)) {}
};

// Basic info stored on all actors
struct CActorInfo : public NoCopy {
  CActorInfo(std::string name, ActorType type) : name(std::move(name)), type(type) {}
  std::string name;
  ActorType type;
};

// The convergence status of an actor during the current time step.
struct CConvergenceStatus : public NoCopy {
  ConvergenceStatus stageStatus = ConvergenceStatus::None; // Per-stage status
  ConvergenceStatus stepStatus = ConvergenceStatus::None; // Step status (worst across stages)

  MOCHI_STRUCT_BEGIN(mochi::CConvergenceStatus);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_FIELD(stageStatus);
  MOCHI_FIELD(stepStatus);
  MOCHI_STRUCT_END();
};

// Component storing the actor's root transform at different times during the step.
struct CRootTransform : public NoCopy {
  CRootTransform() = default;
  explicit CRootTransform(TransformRT const& x)
      : worldFromLocal(x), worldFromLocalPrev(x), worldFromLocalStageStart(x) {}
  // For all actors, it stores the current root transform.
  TransformRT worldFromLocal;
  // For all actors, it stores the root transform at the beginning of the step, i.e., the end of the
  // previous step.
  TransformRT worldFromLocalPrev;
  // Used only by dynamic rigid actors (because their root transform represents state), it stores
  // the root transform at the beginning of the stage.
  TransformRT worldFromLocalStageStart;

  MOCHI_STRUCT_BEGIN(mochi::CRootTransform);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_FIELD(worldFromLocal);
  MOCHI_FIELD(worldFromLocalPrev);
  MOCHI_FIELD(worldFromLocalStageStart);
  MOCHI_STRUCT_END();
};

// Helper function to resolve the root transform of an actor at a given time.
template <TimeStep kTimeStep>
TransformRT const& GetRootTransform(entt::registry const& reg, entt::entity entity) {
  auto const& crt = reg.get<CRootTransform const>(entity);
  if constexpr (kTimeStep == TimeStep::Current) {
    return crt.worldFromLocal;
  } else if constexpr (kTimeStep == TimeStep::Previous) {
    return crt.worldFromLocalPrev;
  } else {
    static_assert(kTimeStep == TimeStep::StageStart);
    // Note: Order matters. Static links have both TagStaticActor and TagRigidActor.
    if (reg.all_of<TagStaticActor>(entity)) {
      // "Static" actors are not solved, but they can move with prescribed transforms. To correctly
      // evaluate friction against "moving" static actors, their stage-start transform is evaluated
      // at the beginning of the step.
      return crt.worldFromLocalPrev;
    } else if (reg.all_of<TagRigidActor>(entity)) {
      // Rigid actors store the stage-start transform in its own field.
      return crt.worldFromLocalStageStart;
    } else {
      // For all other actors, the stage-start transform is the same as the current transform.
      // This is particularly relevant for soft actors with recentering, whose stage-start and
      // current displacements are both expressed wrt the current root transform.
      return crt.worldFromLocal;
    }
  }
}

// Contains the inertia information of a dynamic rigid body.
// Defined in this file because it is accessed in contact assembly.
struct CRigidBodyInertia : public NoCopy, RigidBodyInertia {
  using RigidBodyInertia::RigidBodyInertia;
};

// Bounding volume that contains the entire actor at a given time.
template <TimeStep kStep>
struct CBoundingVolume : public NoCopy {
  CBoundingVolume() = default;

  // Construct from AnyShape, Sphere, Obb, etc...
  template <typename ShapeT>
  explicit CBoundingVolume(ShapeT&& s)
    requires(std::is_constructible_v<AnyShape, ShapeT>)
      : localShape(std::forward<ShapeT>(s)) {}

  // TODO(T225595100): Replace by AnyBoundingVolume.
  AnyShape localShape; // actor space

  MOCHI_TEMPLATE_BEGIN(mochi::CBoundingVolume, kStep);
  // CBoundingVolume is captured for deformable actors because it is used to compute the
  // CConservativeStepBounds for the next step, which in turn affects island formation, and thus
  // determinism. We could avoid capturing the current bounds if it was recomputed before
  // UpdateConservativeStepBounds.
  MOCHI_ATTRIBUTE_IF(
      kStep == TimeStep::Current || kStep == TimeStep::Previous,
      CaptureState(ecs::RequiredTag<TagDeformableActor>{}));
  MOCHI_FIELD(localShape);
  MOCHI_TEMPLATE_END();
};

// Every dynamic actor has CConservativeStepBounds. It stores a conservative world-space bounding
// volume which must be large enough to contain the actor's movement for the current time step,
// assuming reasonable constraints on acceleration, etc... This information is used to sort actors
// into islands. If the volume is not conservative enough, then incorrect behavior will result.
struct CConservativeStepBounds : NoCopy {
  Aabb worldAabb;

  // True when the next conservative step-bounds update should apply a one-step relaxation. This is
  // needed when previous-step history is unavailable or no longer trustworthy, e.g. immediately
  // after actor creation or after external state changes. Defaults to true so a freshly-created
  // actor is relaxed on its next step. Captured so save/restore is deterministic.
  bool needsNextStepRelaxation = true;

  MOCHI_STRUCT_BEGIN(mochi::CConservativeStepBounds);
  MOCHI_ATTRIBUTE(CaptureState);
  // `worldAabb` is a per-step derived state and intentionally omitted from reflection.
  MOCHI_FIELD(needsNextStepRelaxation);
  MOCHI_STRUCT_END();
};

inline void RelaxConservativeStepBoundsOnNextStep(entt::registry& reg, entt::entity actor) {
  if (auto* stepBounds = reg.try_get<CConservativeStepBounds>(actor)) {
    stepBounds->needsNextStepRelaxation = true;
  }
}

// Contact parameters.
using CContactParams = ContactParams;

// Global context component that stores the acceleration of gravity.
// Gravity will be applied to actors with TagUseGravity.
struct CSceneGravity : public NoCopy {
  Vec4r accel = SimdZero(); // world space
};

// Global context component that stores the current simulation time and time step (seconds).
struct CSceneTime : public NoCopy {
  static double constexpr kDefaultTimeStep = 1e-2;

 private:
  // Initialize a non-zero time step to avoid divide-by-zero in velocity initializations.
  double _deltaSeconds = kDefaultTimeStep;
  // Previous time-step is irrelevant; it is overwritten on the first step.
  double _deltaSecondsPrev = 0.0;
  // Initialize the total time s.t. the first step starts at time 0.0.
  double _totalSeconds = -kDefaultTimeStep;

 public:
  CSceneTime() = default;

  // Total simulation time (seconds) accumulated up to the START of the current step.
  // May not match the real-time that has elapsed.
  inline double StepStartTime() const {
    return _totalSeconds;
  }

  // Total simulation time (seconds) accumulated up through the END of the current step.
  // May not match the real-time that has elapsed.
  inline double StepEndTime() const {
    return _totalSeconds + _deltaSeconds;
  }

  // The current time step (seconds), which will advance time from StepStartTime() to StepEndTime().
  inline double DeltaTime() const {
    return _deltaSeconds;
  }

  // The previous time step (seconds)
  inline double DeltaTimePrev() const {
    return _deltaSecondsPrev;
  }

  // Advances time (seconds) for the next simulation step
  inline void Advance(double deltaSeconds) {
    _totalSeconds += _deltaSeconds;
    _deltaSecondsPrev = _deltaSeconds;
    _deltaSeconds = deltaSeconds;
  }

  // Reset the current time and the deltas. This is needed when the scene state is externally reset.
  inline void Reset(double totalSeconds, double deltaSecondsPrev, double deltaSeconds) {
    _totalSeconds = totalSeconds;
    _deltaSecondsPrev = deltaSecondsPrev;
    _deltaSeconds = deltaSeconds;
  }

  MOCHI_STRUCT_BEGIN(mochi::CSceneTime);
  MOCHI_ATTRIBUTE(CaptureStateCtx);
  MOCHI_FIELD(_totalSeconds);
  MOCHI_FIELD(_deltaSecondsPrev);
  MOCHI_FIELD(_deltaSeconds);
  MOCHI_STRUCT_END();
};

// Global context component that stores the current simulation step number.
struct CSceneStepCounter : public NoCopy {
  uint64_t value = 0;

  MOCHI_STRUCT_BEGIN(mochi::CSceneStepCounter);
  MOCHI_ATTRIBUTE(CaptureStateCtx);
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};

// Global context component that stores the scene's handle.
struct CSceneHandle : public NoCopy {
  SceneHandle value;
  explicit CSceneHandle(SceneHandle handle) : value(handle) {}
};

// Stores the current state of the time integrator.
struct CTimeIntegratorState : public TimeIntegratorState, NoCopy {};

using CRecenteringParams = RecenteringParams;

enum class MatrixSemantics {
  // Use these transform semantics to ensure a vector is transformed relative to
  // a set of reference positions.
  DisplacementVector,
  // Transforms like positions
  PositionVector,
  // Use this to ensure that the vector is transformed as a member of the tangent
  // space at a point (i.e., velocity acceleration).
  TangentSpaceVector,
  // Use this to ensure that the vector is transformed as a member of the cotangent
  // space at a point (i.e., dual to tangent space, momentum).
  CotangentSpaceVector,
  // Use this to ensure that the vector is transformed as normal to the tangent space.
  NormalVector,
  // For ROMs, designates the vector as having coordinates in reduced space.
  ReducedCoordinatesVector,
  // For ROMs, designates the vector as having velocities/accelerations in reduced space.
  ReducedTangentSpaceVector,
  // For ROMs, designates the vector as having momentum in reduced space.
  ReducedCotangentSpaceVector,
  // For ROMs, designates this as a Jacobian matrix
  ReducedModelJacobian
};

} // namespace mochi

MOCHI_ENUM_BEGIN(mochi::MatrixSemantics);
MOCHI_ENUM_ITEM(DisplacementVector);
MOCHI_ENUM_ITEM(PositionVector);
MOCHI_ENUM_ITEM(TangentSpaceVector);
MOCHI_ENUM_ITEM(CotangentSpaceVector);
MOCHI_ENUM_ITEM(NormalVector);
MOCHI_ENUM_ITEM(ReducedCoordinatesVector);
MOCHI_ENUM_ITEM(ReducedTangentSpaceVector);
MOCHI_ENUM_ITEM(ReducedCotangentSpaceVector);
MOCHI_ENUM_ITEM(ReducedModelJacobian);
MOCHI_ENUM_END();

namespace mochi {

// Concept to ensure T has a member called 'value'
template <typename T>
concept ValueContainer = requires(T t) { t.value; };

// General class to implement time integration on some data type. The data type must be enclosed as
// a member called 'value' within a container class.
template <ValueContainer T>
struct IntegrationBundle {
  // Default constructor only if T is default constructible
  IntegrationBundle()
    requires(std::is_default_constructible_v<T>)
  {
    prevSteps.reserve(kMaxIntegrationSteps);
    stages.reserve(kMaxIntegrationStages);
  }

  // Constructor from numDofs only if T is constructible from int
  explicit IntegrationBundle(int numDofs)
    requires(std::is_constructible_v<T, int>)
      : stepStart(numDofs) {
    prevSteps.reserve(kMaxIntegrationSteps);
    stages.reserve(kMaxIntegrationStages);
  }

  T stepStart{};
  DynamicArray<T> prevSteps;
  DynamicArray<T> stages;

  // NOTE: Step-internal state (stepStart, stages) is not serialized. Only the state that carries
  // over across steps (prevSteps) is serialized.
  MOCHI_TEMPLATE_BEGIN(mochi::IntegrationBundle, T);
  MOCHI_FIELD(stepStart) MOCHI_ATTRIBUTE(NoSerialize);
  MOCHI_FIELD(prevSteps);
  MOCHI_FIELD(stages) MOCHI_ATTRIBUTE(NoSerialize);
  MOCHI_TEMPLATE_END();
};

// Rigid body state at a given time.
struct TransformRTContainer {
  TransformRTContainer() = default;
  explicit TransformRTContainer(TransformRT const& valueIn) : value(valueIn) {}

  TransformRT value;

  MOCHI_STRUCT_BEGIN(mochi::TransformRTContainer);
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};

template <TimeStep kStep>
struct CRigidState : public TransformRTContainer {
  using TransformRTContainer::TransformRTContainer;
  MOCHI_TEMPLATE_BEGIN(mochi::CRigidState, kStep);
  MOCHI_ATTRIBUTE_IF(kStep == TimeStep::Current, CaptureState);
  MOCHI_BASE_CLASS(mochi::TransformRTContainer);
  MOCHI_TEMPLATE_END();
};

/// @brief Component for time integration of rigid body pose.
struct CIntegrationRigidStates : public IntegrationBundle<TransformRTContainer>, NoCopy {
  MOCHI_STRUCT_BEGIN(mochi::CIntegrationRigidStates);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_BASE_CLASS(IntegrationBundle<TransformRTContainer>);
  MOCHI_STRUCT_END();
};

// Traits that define metadata for different vector types that might be part of an
// actor's state.
template <MatrixSemantics kSemanticsType>
struct MatrixMetadata {
  static constexpr MatrixSemantics kSemantics = kSemanticsType;
  MOCHI_TEMPLATE(mochi::MatrixMetadata, kSemanticsType);
};

/// @brief Data structure that holds vector data as well as various metadata. Used to define various
/// components.
template <typename Scalar, typename MetadataType>
struct VectorComponent {
  static constexpr MatrixSemantics kSemantics = MetadataType::kSemantics;

  static constexpr bool kIsDisplacement = kSemantics == MatrixSemantics::DisplacementVector;
  static constexpr bool kIsTangentOrCotangent = kSemantics == MatrixSemantics::TangentSpaceVector ||
      kSemantics == MatrixSemantics::CotangentSpaceVector;
  static constexpr bool kIsReduced = kSemantics == MatrixSemantics::ReducedCoordinatesVector ||
      kSemantics == MatrixSemantics::ReducedCotangentSpaceVector ||
      kSemantics == MatrixSemantics::ReducedTangentSpaceVector;

  ColumnVector<Scalar> value;

  VectorComponent() = default;
  explicit VectorComponent(int numDof) : value(ColumnVector<Scalar>::Zero(numDof)) {}
  explicit VectorComponent(ColumnVector<Scalar> const& value) : value(value) {}
  explicit VectorComponent(ColumnVector<Scalar>&& value) : value(std::move(value)) {}

  void Initialize(int numDof) {
    Resize(numDof);
  }

  template <typename OtherMetadata>
  void CopyFrom(VectorComponent<Scalar, OtherMetadata> const& other) {
    static_assert(
        kSemantics == OtherMetadata::kSemantics,
        "Copying one vector component to another requires them to have the same semantics!");
    value = other.value;
  }

  template <typename OtherMetadata>
  void CopyTo(VectorComponent<Scalar, OtherMetadata>& other) const {
    other.value = value;
  }

  void Resize(int numDof) {
    value.Reset(numDof);
    value.SetZero();
  }

  void ApplyTranslation(Real3 const& x) {
    if constexpr (kIsDisplacement) {
      using Scalar3 = NdArray<Scalar, 3>;
      auto srcCoords = Unflatten<Scalar3 const>(value.GetConstSpan());
      auto dstCoords = Unflatten<Scalar3>(value.GetSpan());
      ArrayAdd<Scalar>(dstCoords, srcCoords, StaticCast<Scalar3>(x));
    } else {
      static_assert(!kIsReduced, "Cannot apply transformations to reduced coordinates!");
    }
  }

  void ApplyTransform(TransformRT const& rt, Span<real const> refCoords) {
    VMatrix4x4r mat4T = ToVMatrix4x4Transpose(rt);

    if constexpr (kIsDisplacement) {
      ArrayTransformDisplacements_MatT<true>(
          AsView(value), AsConstView(value), AsConstView(refCoords), mat4T);
    } else if constexpr (kIsTangentOrCotangent) {
      VMatrix3x3r mat3T{mat4T[0], mat4T[1], mat4T[2]};
      ArrayRotateVectors_MatT<true>(AsView(value), AsConstView(value), mat3T);
    } else {
      static_assert(!kIsReduced, "Cannot apply transformations to reduced coordinates!");
      static_assert(
          kIsDisplacement || !kIsTangentOrCotangent || kIsReduced,
          "Invalid semantics for transformation");
    }
  }

  MOCHI_TEMPLATE_BEGIN(mochi::VectorComponent, Scalar, MetadataType);
  MOCHI_FIELD(value);
  MOCHI_TEMPLATE_END();
};

template <typename Scalar, typename MetadataType>
struct CVectorComponent : public VectorComponent<Scalar, MetadataType>, NoCopy {
  using BaseType = VectorComponent<Scalar, MetadataType>;
  using BaseType::BaseType;

  MOCHI_TEMPLATE_BEGIN(mochi::CVectorComponent, Scalar, MetadataType);
  MOCHI_BASE_CLASS(BaseType);
  MOCHI_TEMPLATE_END();
};

// Same as CVectorComponent but uses Matrix / ColumnVector instead
template <
    typename Scalar,
    typename MetadataType,
    int kRowsAtCompileTime = krylov::kDynamic,
    int kColsAtCompileTime = krylov::kDynamic,
    krylov::Direction kMajorDirection = krylov::Direction::ColMajor,
    int kLeadingDim = krylov::kAutomaticLeadDim>
struct CDenseMatrixComponent : public NoCopy {
  static constexpr MatrixSemantics kSemantics = MetadataType::kSemantics;

  static constexpr bool kIsDisplacement = kSemantics == MatrixSemantics::DisplacementVector;
  static constexpr bool kIsTangentOrCotangent = kSemantics == MatrixSemantics::TangentSpaceVector ||
      kSemantics == MatrixSemantics::CotangentSpaceVector;
  static constexpr bool kIsReducedDisplacement =
      kSemantics == MatrixSemantics::ReducedCoordinatesVector;
  static constexpr bool kIsReducedTangentOrCotangent =
      kSemantics == MatrixSemantics::ReducedCotangentSpaceVector ||
      kSemantics == MatrixSemantics::ReducedTangentSpaceVector;

  using MatrixType = Matrix<
      Scalar,
      kRowsAtCompileTime,
      kColsAtCompileTime,
      kMajorDirection,
      krylov::Ownership::Owner,
      kLeadingDim>;

  MatrixType value;

  explicit CDenseMatrixComponent(int rows) : value(rows) {
    value.SetZero();
  }
  explicit CDenseMatrixComponent(int rows, int cols) : value(rows, cols) {
    value.SetZero();
  }
  explicit CDenseMatrixComponent(MatrixType const& value) : value(value) {}
  explicit CDenseMatrixComponent(MatrixType&& value) : value(std::move(value)) {}

  template <typename OtherMetadata>
  void CopyFrom(CDenseMatrixComponent<Scalar, OtherMetadata> const& other) {
    static_assert(
        kSemantics == OtherMetadata::kSemantics,
        "Copying one vector component to another requires them to have the same semantics!");
    value = other.value;
  }

  template <typename OtherMetadata>
  void CopyTo(CDenseMatrixComponent<Scalar, OtherMetadata>& other) const {
    other.CopyFrom(*this);
  }

  void ApplyTranslation(ColumnVectorView<real const> translation) {
    static_assert(kColsAtCompileTime == 1, "Cannot apply translations to matrix data!");
    if constexpr (kIsReducedDisplacement) {
      value += translation;
    }
  }
};

/// @brief A component to hold a time slice of a specific type of data
template <
    typename Scalar,
    typename MetadataType,
    TimeStep kRelTime,
    int kRowsAtCompileTime = krylov::kDynamic,
    int kColsAtCompileTime = krylov::kDynamic,
    krylov::Direction kMajorDirection = krylov::Direction::ColMajor,
    int kLeadingDim = krylov::kAutomaticLeadDim>
struct CDenseTimeSlice : public CDenseMatrixComponent<
                             Scalar,
                             MetadataType,
                             kRowsAtCompileTime,
                             kColsAtCompileTime,
                             kMajorDirection,
                             kLeadingDim> {
  using BaseType = CDenseMatrixComponent<
      Scalar,
      MetadataType,
      kRowsAtCompileTime,
      kColsAtCompileTime,
      kMajorDirection,
      kLeadingDim>;

  using MatrixType = typename BaseType::MatrixType;

  explicit CDenseTimeSlice(int rows) : BaseType(rows) {}
  explicit CDenseTimeSlice(int rows, int cols) : BaseType(rows, cols) {}
  explicit CDenseTimeSlice(MatrixType const& value) : BaseType(value) {}
  explicit CDenseTimeSlice(MatrixType&& value) : BaseType(std::move(value)) {}
};

template <
    typename Scalar,
    typename MetadataType,
    TimeStep kRelTime,
    int kRowsAtCompileTime = krylov::kDynamic>
using CDenseTimeSliceVector = CDenseTimeSlice<
    Scalar,
    MetadataType,
    kRelTime,
    kRowsAtCompileTime,
    1,
    krylov::Direction::ColMajor>;

template <typename Scalar, typename MetadataType, TimeStep kRelTime>
struct CTimeSlice : public CVectorComponent<Scalar, MetadataType> {
  using BaseType = CVectorComponent<Scalar, MetadataType>;
  CTimeSlice() = default;
  explicit CTimeSlice(int numDofs) : BaseType(numDofs) {}
  explicit CTimeSlice(ColumnVector<Scalar> const& value) : BaseType(value) {}
  explicit CTimeSlice(ColumnVector<Scalar>&& value) : BaseType(std::move(value)) {}

  MOCHI_TEMPLATE_BEGIN(mochi::CTimeSlice, Scalar, MetadataType, kRelTime);
  MOCHI_ATTRIBUTE_IF(kRelTime == TimeStep::Current, CaptureState);
  MOCHI_BASE_CLASS(BaseType);
  MOCHI_TEMPLATE_END()
};

// enum class to templatize components and systems of soft actors according to function composition
// layer.
// DisplacementLayer::Default refers to the layer of soft displacements wrt rest coordinates.
// DisplacementLayer::Skinned refers to the layer of skinned positions wrt rest coordinates.
enum class DisplacementLayer { Default, Skinned };

} // namespace mochi
MOCHI_ENUM_BEGIN(mochi::DisplacementLayer);
MOCHI_ENUM_ITEM(Default);
MOCHI_ENUM_ITEM(Skinned);
MOCHI_ENUM_END();
namespace mochi {

/// @brief Components for soft-actor displacements and velocities
template <DisplacementLayer kLayer>
struct DisplacementVectorMetadata : public MatrixMetadata<MatrixSemantics::DisplacementVector> {
  using BaseType = MatrixMetadata<MatrixSemantics::DisplacementVector>;
  MOCHI_TEMPLATE_BEGIN(mochi::DisplacementVectorMetadata, kLayer);
  MOCHI_BASE_CLASS(BaseType);
  MOCHI_TEMPLATE_END();
};
template <typename Scalar, TimeStep kRelTime, DisplacementLayer kLayer = DisplacementLayer::Default>
using CDisplacementSlice = CTimeSlice<Scalar, DisplacementVectorMetadata<kLayer>, kRelTime>;

template <DisplacementLayer kLayer>
struct VelocityVectorMetadata : public MatrixMetadata<MatrixSemantics::TangentSpaceVector> {
  using BaseType = MatrixMetadata<MatrixSemantics::TangentSpaceVector>;
  MOCHI_TEMPLATE_BEGIN(mochi::VelocityVectorMetadata, kLayer);
  MOCHI_BASE_CLASS(BaseType);
  MOCHI_TEMPLATE_END();
};
template <typename Scalar, TimeStep kRelTime, DisplacementLayer kLayer = DisplacementLayer::Default>
using CVelocitySlice = CTimeSlice<Scalar, VelocityVectorMetadata<kLayer>, kRelTime>;

/// @brief Component for time integration of displacement slices.
using DefaultDisplacementSlice =
    VectorComponent<real, DisplacementVectorMetadata<DisplacementLayer::Default>>;
struct CIntegrationDisplacementSlices : public IntegrationBundle<DefaultDisplacementSlice>, NoCopy {
  using IntegrationBundle<DefaultDisplacementSlice>::IntegrationBundle;

  MOCHI_STRUCT_BEGIN(mochi::CIntegrationDisplacementSlices);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_BASE_CLASS(IntegrationBundle<DefaultDisplacementSlice>);
  MOCHI_STRUCT_END();
};

/// @brief Component for time integration of velocity slices.
template <DisplacementLayer kLayer = DisplacementLayer::Default>
struct CIntegrationVelocitySlices
    : public IntegrationBundle<VectorComponent<real, VelocityVectorMetadata<kLayer>>>,
      NoCopy {
  using BaseClass = IntegrationBundle<VectorComponent<real, VelocityVectorMetadata<kLayer>>>;
  using BaseClass::BaseClass;

  MOCHI_TEMPLATE_BEGIN(mochi::CIntegrationVelocitySlices, kLayer);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_BASE_CLASS(BaseClass);
  MOCHI_TEMPLATE_END();
};

template <typename... Ts>
struct CVariant {
  using variant_t = std::variant<Ts...>;

  variant_t value;

  CVariant(variant_t&& value) : value(std::move(value)) {}

  template <typename T>
    requires(std::disjunction_v<std::is_same<std::remove_cvref_t<T>, Ts>...>)
  CVariant(T&& value) : value(variant_t(std::forward<T>(value))) {}

  template <typename T>
  bool Is() const {
    return std::get_if<T>(&value) != nullptr;
  }
  template <typename T>
  T& Get() {
    return std::get<T>(value);
  }
  template <typename T>
  T const& Get() const {
    return std::get<T>(value);
  }
  template <typename T>
  T* TryGet() {
    return std::get_if<T>(&value);
  }
  template <typename T>
  T const* TryGet() const {
    return std::get_if<T>(&value);
  }
  template <typename Visitor>
  auto Visit(Visitor&& vis) {
    return std::visit(vis, value);
  }
  template <typename Visitor>
  auto Visit(Visitor&& vis) const {
    return std::visit(vis, value);
  }
};

/// @brief Structure that allows defining components that store pointers to vectors in other
/// components.
struct VectorComponentRef {
  explicit VectorComponentRef(ColumnVectorView<real const> v) : value(v) {}

  ~VectorComponentRef() = default;

  MOCHI_DECLARE_NO_COPY(VectorComponentRef);

  VectorComponentRef(VectorComponentRef&&) = default;

  // This struct must support move assignment to be used as an ECS component, however,
  // ColumnVectorView<real>::operator= would not do what we want. The default behavior for non-owned
  // vectors is to copy the values, while we need to copy the pointer to the underlying storage.
  VectorComponentRef& operator=(VectorComponentRef&& rhs) noexcept {
    if (this != &rhs) {
      value.Reset(rhs.value);
    }
    return *this;
  }

  ColumnVectorView<real const> value;
};

/// @brief Component to store a reference to the final displacement of an actor.
template <TimeStep kStep>
struct CFinalDisplacementRef : public VectorComponentRef {
  using VectorComponentRef::VectorComponentRef;
};

/// @brief Optional component to store the color used in the debug draw.
struct CMeshColor {
  Color value = colors::kWhite;
};

namespace common_components {
void InitializeOnce(entt::registry& reg);
}

} // namespace mochi

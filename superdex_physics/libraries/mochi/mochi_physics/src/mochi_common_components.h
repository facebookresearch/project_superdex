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
struct TagDeformableActor {}; // Soft (including nested soft), Shell, and Rod actors
struct TagRomActor {};
struct TagArticulatedActor {};
struct TagArticulatedLinkActor {};
struct TagNestedSoftActor {};
struct TagBlendedActor {};
struct TagCompoundActor {};
struct TagFullyInitialized {}; // Added last when initializing an actor (for reative systems)
struct TagHasDeepFlowCollider {};

// Global (registry-context) tag for preserving previous-step quantities without advancing
// integration history. This is used by zero-step solve-free assembly.
struct TagAssemblyPreserveHistory {};

// Indicates that an actor participates in the point-cloud contact system.
struct TagUsePointCloudContact {};

// Indicates that a deformable actor uses its separately authored embedded contact-skin mesh,
// rather than its physics discretization, to generate colliding samples.
struct TagUseDeformableContactSkin {};

// Selects async contact assembly through CSkinnedContactSnle when the sample Jacobian maps through
// skinning or embedding. Used by skinned articulated, nested-soft, blended, and contact-skinned
// deformable actors.
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
  // For all actors, it stores the root transform at the beginning of the step. Recentering moves it
  // along with the previous-step state, so for deformable actors it is always the frame in which
  // that state is expressed.
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

// Helper function to resolve the root transform of an actor at a given time, given its
// CRootTransform.
template <TimeStep kTimeStep>
[[nodiscard]] TransformRT const& GetRootTransform(
    [[maybe_unused]] ecs::
        PartialRegistry<CRootTransform const, TagStaticActor const, TagRigidActor const> reg,
    [[maybe_unused]] entt::entity entity,
    CRootTransform const& crt) {
  MOCHI_ASSERT_VERBOSE(
      &crt == reg.try_get<CRootTransform const>(entity),
      "Expected the CRootTransform of the given entity.");
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

// Helper function to resolve the root transform of an actor at a given time.
template <TimeStep kTimeStep>
[[nodiscard]] TransformRT const& GetRootTransform(entt::registry const& reg, entt::entity entity) {
  return GetRootTransform<kTimeStep>(reg, entity, reg.get<CRootTransform const>(entity));
}

// Contains the inertia information of a dynamic rigid body.
// Defined in this file because it is accessed in contact assembly.
struct CRigidBodyInertia : public NoCopy, RigidBodyInertia {
  using RigidBodyInertia::RigidBodyInertia;
};

/**
 * @brief For soft actors, world-space velocity of the center-of-mass at the end of the previous
 * step.
 *
 * @note This value is approximated based on the center-of-bounding-volume.
 */
struct CPrevRigidVelocity : NoCopy {
  Vec4r linearVelocityWorld = {}; // World-space
  Vec4r angularVelocityWorld = {}; // World-space
  Vec4r centerOfMassLocal = {}; // Actor local-space

  MOCHI_STRUCT_BEGIN(mochi::CPrevRigidVelocity);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_STRUCT_END();
};

// Every dynamic actor has CConservativeStepBounds. It stores a conservative world-space bounding
// volume which must be large enough to contain the actor's movement for the current time step,
// assuming reasonable constraints on acceleration, etc... This information is used to sort actors
// into islands. If the volume is not conservative enough, then incorrect behavior will result.
struct CConservativeStepBounds : NoCopy {
  Aabb worldAabb;

  // Maximum Euclidean world-space speed [m/s] of any point represented by the actor's current
  // bounding volume. This is derived at the beginning of each simulation step.
  real maxGeometrySpeed = 0_r;

  // True when the next conservative step-bounds update should apply a one-step relaxation, e.g.
  // immediately after actor creation or an external state change. Defaults to true so a
  // freshly-created actor gets the relaxation on its first step.
  bool needsNextStepRelaxation = true;

  MOCHI_STRUCT_BEGIN(mochi::CConservativeStepBounds);
  MOCHI_ATTRIBUTE(CaptureState);
  // `worldAabb` and `maxGeometrySpeed` are per-step derived state and intentionally omitted from
  // reflection.
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

// Index of the node whose motion drives recentering for shell and rod actors.
struct CRecenteringPivotNode {
  int index = 0;
};

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

  // Forward value-specific construction arguments to the step-start state.
  template <typename... Args>
  explicit IntegrationBundle(Args&&... args)
    requires(
        sizeof...(Args) > 0 && std::is_constructible_v<T, Args...> &&
        !(sizeof...(Args) == 1 &&
          (std::is_base_of_v<IntegrationBundle<T>, std::remove_cvref_t<Args>> && ...)))
      : stepStart(std::forward<Args>(args)...) {
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

// Macros for defining ECS components of numerically-integrated positions/velocities. Invoke them
// directly in namespace mochi. ValueType must not contain a top-level comma; use a type alias for
// multi-argument template types.
#define MOCHI_DEFINE_INTEGRATION_COMPONENT(Component, ValueType)   \
  struct Component : public IntegrationBundle<ValueType>, NoCopy { \
    using IntegrationBundle<ValueType>::IntegrationBundle;         \
                                                                   \
    MOCHI_STRUCT_BEGIN(mochi::Component);                          \
    MOCHI_ATTRIBUTE(CaptureState);                                 \
    MOCHI_BASE_CLASS(IntegrationBundle<ValueType>);                \
    MOCHI_STRUCT_END();                                            \
  }

#define MOCHI_DEFINE_INTEGRATION_COMPONENT_TEMPLATE(Component, ValueType, ...) \
  struct Component : public IntegrationBundle<ValueType>, NoCopy {             \
    using IntegrationBundle<ValueType>::IntegrationBundle;                     \
                                                                               \
    MOCHI_TEMPLATE_BEGIN(mochi::Component, __VA_ARGS__);                       \
    MOCHI_ATTRIBUTE(CaptureState);                                             \
    MOCHI_BASE_CLASS(IntegrationBundle<ValueType>);                            \
    MOCHI_TEMPLATE_END();                                                      \
  }

// Rigid body state at a given time.
struct TransformRTContainer {
  TransformRTContainer() = default;
  explicit TransformRTContainer(TransformRT const& valueIn) : value(valueIn) {}

  TransformRT value;

  MOCHI_STRUCT_BEGIN(mochi::TransformRTContainer);
  MOCHI_FIELD(value);
  MOCHI_STRUCT_END();
};

/// @brief Component for time integration of rigid body pose.
MOCHI_DEFINE_INTEGRATION_COMPONENT(CIntegrationRigidStates, TransformRTContainer);

template <TimeStep kStep>
struct CRigidState : public TransformRTContainer {
  using TransformRTContainer::TransformRTContainer;
  MOCHI_TEMPLATE_BEGIN(mochi::CRigidState, kStep);
  // This component is captured only when it represents true state, which is signaled by the
  // existence of a corresponding integration component.
  MOCHI_ATTRIBUTE_IF(
      kStep == TimeStep::Current,
      CaptureState(ecs::Included<CIntegrationRigidStates>{}));
  MOCHI_BASE_CLASS(mochi::TransformRTContainer);
  MOCHI_TEMPLATE_END();
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
  using BaseType::BaseType;

  MOCHI_TEMPLATE_BEGIN(mochi::CTimeSlice, Scalar, MetadataType, kRelTime);
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
struct CDisplacementSlice
    : public CTimeSlice<Scalar, DisplacementVectorMetadata<kLayer>, kRelTime> {
  using BaseType = CTimeSlice<Scalar, DisplacementVectorMetadata<kLayer>, kRelTime>;
  using BaseType::BaseType;

  MOCHI_TEMPLATE_BEGIN(mochi::CDisplacementSlice, Scalar, kRelTime, kLayer);
  MOCHI_ATTRIBUTE_IF(
      kRelTime == TimeStep::Current && kLayer == DisplacementLayer::Default,
      CaptureState);
  MOCHI_BASE_CLASS(BaseType);
  MOCHI_TEMPLATE_END()
};

template <DisplacementLayer kLayer>
struct VelocityVectorMetadata : public MatrixMetadata<MatrixSemantics::TangentSpaceVector> {
  using BaseType = MatrixMetadata<MatrixSemantics::TangentSpaceVector>;
  MOCHI_TEMPLATE_BEGIN(mochi::VelocityVectorMetadata, kLayer);
  MOCHI_BASE_CLASS(BaseType);
  MOCHI_TEMPLATE_END();
};

template <DisplacementLayer kLayer>
using VelocityIntegrationValue = VectorComponent<real, VelocityVectorMetadata<kLayer>>;
/// @brief Component for time integration of velocity slices.
template <DisplacementLayer kLayer = DisplacementLayer::Default>
MOCHI_DEFINE_INTEGRATION_COMPONENT_TEMPLATE(
    CIntegrationVelocitySlices,
    VelocityIntegrationValue<kLayer>,
    kLayer);

template <typename Scalar, TimeStep kRelTime, DisplacementLayer kLayer = DisplacementLayer::Default>
struct CVelocitySlice : public CTimeSlice<Scalar, VelocityVectorMetadata<kLayer>, kRelTime> {
  using BaseType = CTimeSlice<Scalar, VelocityVectorMetadata<kLayer>, kRelTime>;
  using BaseType::BaseType;

  MOCHI_TEMPLATE_BEGIN(mochi::CVelocitySlice, Scalar, kRelTime, kLayer);
  MOCHI_ATTRIBUTE_IF(
      kRelTime == TimeStep::Current && kLayer == DisplacementLayer::Default,
      CaptureState);
  MOCHI_ATTRIBUTE_IF(
      kRelTime == TimeStep::Current && kLayer == DisplacementLayer::Skinned,
      CaptureState(ecs::Included<CIntegrationVelocitySlices<kLayer>>{}));
  MOCHI_BASE_CLASS(BaseType);
  MOCHI_TEMPLATE_END()
};

using DefaultDisplacementSlice =
    VectorComponent<real, DisplacementVectorMetadata<DisplacementLayer::Default>>;
/// @brief Component for time integration of displacement slices.
MOCHI_DEFINE_INTEGRATION_COMPONENT(CIntegrationDisplacementSlices, DefaultDisplacementSlice);

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

// Bounding volume that contains the entire actor. For actors with a deforming surface, this
// component is scratch state overwritten with bounds for the state being evaluated. For actors with
// a rigid surface, the actor-local bounds are immutable. Not captured: deforming bounds are derived
// state, recomputed after restore (see step::InitializeOnce).
struct CBoundingVolume : public NoCopy {
  CBoundingVolume() = default;

  // Construct from AnyShape, Sphere, Obb, etc...
  template <typename ShapeT>
  explicit CBoundingVolume(ShapeT&& s)
    requires(std::is_constructible_v<AnyShape, ShapeT>)
      : localShape(std::forward<ShapeT>(s)) {}

  // TODO(T225595100): Replace by AnyBoundingVolume.
  AnyShape localShape; // actor space
};

/// @brief Optional component to store the color used in the debug draw.
struct CMeshColor {
  Color value = colors::kWhite;
};

namespace common_components {
void InitializeOnce(entt::registry& reg);
}

} // namespace mochi

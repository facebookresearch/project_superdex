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

// Include experimental header first to ensure PointCloudColliderParams is defined before other
// includes that might bring in mochi_contact.h transitively.
#include <mochi_physics/mochi_physics_experimental.h>

#include "mochi_common_components.h"
#include "mochi_contact_pair_params.h"
#include "mochi_discretization_components.h"
#include "mochi_ecs.h"
#include "mochi_query.h"
#include "mochi_shape.h"
#include "mochi_simulation.h"
#include "mochi_snle.h"

#include <mochi_core/contact/contact_correspondence.h>
#include <mochi_core/contact/contact_partition.h>
#include <mochi_core/contact/contact_utils.h>
#include <mochi_core/element_operations/element_assembler.h>
#include <mochi_core/geometry/aabb.h>
#include <mochi_core/geometry/base_map.h>
#include <mochi_core/geometry/bvh_tree.h>
#include <mochi_core/geometry/geometry_utils.h>
#include <mochi_core/linear_algebra/block_sparse_matrix.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/constants.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/reflection.h>
#include <mochi_core/utils/task_scheduler.h>
#include <mochi_core/utils/vmatrix.h>

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mochi {

// Forwards
struct CGroupMemberInfo;
struct CIslandContactSnle;
struct CIslandDescendants;
struct CIslandDofInfo;
struct CIslandMemberInfo;
struct CQueryContactSamples;
struct CQuerySdfSurface;

/** @brief Contact type for collision interactions. */
enum class ContactType {
  /** @brief No contact. */
  None,

  /**
   * @brief Asynchronous contact.
   *
   * @note Applies contact forces only to the colliding actors (NOT to the collider actors).
   * @note Uses the colliders' state at the previous time step and the colliding's state at the
   * current time step.
   */
  Async,

  /**
   * @brief Synchronous contact (recommended).
   *
   * @note Applies contact forces to both colliding and collider actors.
   * @note Uses the colliders' and colliding's states at the current time step.
   */
  Sync,

  /** @brief Number of contact type enum values. */
  Count
};

/**************************************************************************
  ECS Collider Components
*/

/** @brief Contains information on the collider of an actor. */
struct CColliderInfo final : NoCopy {
  ColliderType type = ColliderType::None;
};

/** @brief A box collider component implemented using an Obb shape. */
struct CBoxCollider final : BoxCollider, NoCopy {};

/** @brief A sphere collider component implemented using a Sphere shape. */
struct CSphereCollider final : SphereCollider, NoCopy {};

/** @brief A mesh collider component implemented using a TriangularMesh. */
struct CMeshCollider final : MeshCollider, NoCopy {
  CMeshCollider(std::shared_ptr<TriangularMesh const> const& meshIn) : MeshCollider(meshIn) {}
};

/** @brief A collider component implemented using an infinite plane. */
struct CPlaneCollider final : PlaneCollider, NoCopy {};

/** @brief A collider defined by a signed distance field. */
struct CSdfCollider final : SdfCollider, NoCopy {};

/** @brief Component for point-cloud collider properties. */
struct CPointCloudColliderParams final : experimental::PointCloudColliderParams, NoCopy {
  /// @brief Dimension of the collider-side integral (2 for shell, 1 for rods). Set during actor
  /// initialization.
  int integralDim = 0;
};

/**
 * @brief Any actor with async contact can use this component. The semaphore will be incremented at
 * the start of assembly. It will be decremented (marked "done") once async contact is up-to-date
 * for that actor.
 */
struct CActorAsyncContactSemaphore final : NoCopy {
  std::unique_ptr<TaskSemaphore> asyncContactUpToDate = std::make_unique<TaskSemaphore>();
};

/**
 * @brief Denotes an entity as requiring SDF samples from all colliders at least maxDistance away.
 * This is used by dynamic hyper reduction (and will likely be used by other systems in the future)
 * to query the SDF (or a lower bound thereof) in order to determine distance to closest surface.
 */
struct CRequiresFarSdfEvaluation {
  real maxDistance = 0.0_r;
};

/** @brief Indicates that a @ref GridSdf collider is being computed asynchronously. */
struct CSdfColliderPending final : SdfCollider, NoCopy {
  MOCHI_DECLARE_MOVE_ONLY(CSdfColliderPending);
  CSdfColliderPending() = default;

  // Get the resulting GridSdf:
  //    gridSdfShape->GetGridSdfSemaphore().Wait();
  //    auto sdf = gridSdfShape->GetGridSdf();
  std::shared_ptr<GridSdfShape const> gridSdfShape;
};

/** @brief Mappings for an SDF collider. */
template <TimeStep kStep>
struct CSdfMapping : std::unique_ptr<BaseMap>, NoCopy {};

namespace details {
template <ContactType kContactType, TimeStep kTimeStep>
struct BoundingVolumeForImpl {
  static_assert(kTimeStep == TimeStep::Current || kTimeStep == TimeStep::StageStart);
  using Type = std::conditional_t<
      kContactType == ContactType::Async && kTimeStep == TimeStep::StageStart,
      CBoundingVolume<TimeStep::Previous>,
      CBoundingVolume<TimeStep::Current>>;
};
} // namespace details

/**
 * @brief Alias to get the appropriate CBoundingVolume component of a collider during collision
 * detection, depending on contact type (sync = dynamic collider / async = static collider) and time
 * step (current / stage start). Static colliders at stage start use previous bounds. All other
 * cases use current bounds.
 */
template <ContactType kContactType, TimeStep kTimeStep>
using CBoundingVolumeFor = typename details::BoundingVolumeForImpl<kContactType, kTimeStep>::Type;

/**************************************************************************
  ECS Broadphase Components
*/

/** @brief Stores data for each potential collider. */
struct PotentialColliderData {
  entt::entity entity = {};

  PotentialColliderData(entt::entity e) : entity(e) {}
};

/**
 * @brief Stores a list of entities that might contact the actor during any iteration of the current
 * step. This list is generated by finding actors with overlapping @ref CConservativeStepBounds.
 * Templatized according to async or sync contact.
 */
template <ContactType kContactType>
struct CConservativePotentialColliders : public std::vector<PotentialColliderData> {};

/**
 * @brief Stores a list of entities that might contact the actor during the current iteration. This
 * list is generated by finding actors with overlapping @ref CBoundingVolume. Templatized according
 * to async or sync contact.
 */
template <ContactType kContactType>
struct CPotentialColliders : public std::vector<PotentialColliderData> {};

/**************************************************************************
  ECS Narrowphase Components
*/

/** @brief Information of a collision pair with a collider actor. */
struct ActiveCollision {
  entt::entity colliderEntity; // Entity of the corresponding collider for this collision pair
  ContactDetectionResult collisionResult; // All the contacts with this collider
  int collidingJacId = 0; // Id of the colliding actor's contact Jacobian for implicit contact
  int colliderJacId = 0; // Id of the collider actor's contact Jacobian for implicit contact

  // Comparison operators for ordering by (colliderEntity, collidingPartitionId).
  MOCHI_FORCE_INLINE bool operator<(ActiveCollision const& other) const {
    return (colliderEntity != other.colliderEntity)
        ? (colliderEntity < other.colliderEntity)
        : (collisionResult.collidingPartitionId < other.collisionResult.collidingPartitionId);
  }

  MOCHI_FORCE_INLINE bool operator>(ActiveCollision const& other) const {
    return (colliderEntity != other.colliderEntity)
        ? (colliderEntity > other.colliderEntity)
        : (collisionResult.collidingPartitionId > other.collisionResult.collidingPartitionId);
  }

  MOCHI_FORCE_INLINE bool operator==(ActiveCollision const& other) const {
    return colliderEntity == other.colliderEntity &&
        collisionResult.collidingPartitionId == other.collisionResult.collidingPartitionId;
  }

  // Clear the contact detection results. For performance reasons, the memory is not deallocated so
  // that it's reused across assemblies.
  void Clear() {
    collisionResult.Clear();
  }
};

/**
 * @brief Collection of collider entities with which the actor is currently colliding (or
 * potentially colliding), and the corresponding contact result. The size is equal to the number of
 * potential colliders times the number of partitions. Templatized according to async or sync
 * contact, and according to the time step (Current or StageStart).
 */
template <ContactType kContactType, TimeStep kTimeStep>
struct CActiveCollisions : public std::vector<ActiveCollision> {
  // Set up based on the potential colliders and number of partitions. Existing memory from the
  // previous assembly is reused for performance reasons.
  void SetUp(CPotentialColliders<kContactType> const& potentialColls, int numPartitions) {
    MOCHI_ASSERT_VERBOSE(size() % numPartitions == 0, "Inconsistent size.");

    // Remove colliders that were previously potentially in contact but are no longer potentially in
    // contact.
    erase(
        std::remove_if(
            begin(),
            end(),
            [&](auto const& x) {
              for (auto const& potentialColl : potentialColls) {
                if (x.colliderEntity == potentialColl.entity) {
                  return false;
                }
              }
              return true;
            }),
        end());

    // Clear the data of the colliders that were previously potentially in contact and continue to
    // be potentially in contact. For performance, this operations clears the data without releasing
    // memory so that memory can be recycled across assemblies.
    for (auto& col : *this) {
      col.Clear();
    }

    // Add colliders that previously were not potentially in contact but are now potentially in
    // contact.
    for (int c = 0; c < isize(potentialColls); ++c) {
      bool prevInContact = false;
      for (auto const& col : *this) {
        if (col.colliderEntity == potentialColls[c].entity) {
          prevInContact = true;
          break;
        }
      }
      if (!prevInContact) {
        for (int p = 0; p < numPartitions; ++p) {
          emplace_back(
              ActiveCollision{
                  potentialColls[c].entity, ContactDetectionResult{.collidingPartitionId = p}});
        }
      }
    }

    // Sort colliders by colliderEntity first, then by collidingPartitionId. This allows
    // deterministic assembly of contact under scene resetting.
    std::sort(begin(), end());
  }
};

/**
 * @brief Collection of response data for all points in
 * CActiveCollisions<ContactType::Async, TimeStep::Current>, used by deformable actors in async
 * contact assembly.
 */
struct CDeformablePointAsyncCollisionsResponse {
  static constexpr int kInvalidIndex = kSentinelIndex;
  static_assert(kInvalidIndex < 0, "Invalid index must be negative.");

 private:
  /** @brief Storage for the evaluation of contact response samples in SoA format. Some arrays may
   * be empty if the corresponding data was not computed. */
  DynamicArray<double> _energies;
  DynamicArray<Real3> _gradients;
  DynamicArray<Matrix3x3r> _hessians;

  /** @brief Contact sample indices that produced response data. */
  DynamicArray<int> _sampleIndices;

  /** @brief Lookup from contact sample index to response data index. Entries are @ref kInvalidIndex
   * for samples that produced no response. */
  DynamicArray<int> _responseIndexFromSampleIndex;

  /** @brief Number of contact quadrature samples associated with each contact integration element.
   */
  int _numQuadsPerContactElement = 0;

  /**
   * @brief Contact integration elements with at least one sample that produced async contact
   * response.
   *
   * @note A contact element is an element of the contact integration discretization, e.g., a
   * boundary trace for soft actors, a surface triangle for shell actors, or a centerline segment
   * for rods using centerline contact. For soft actors, this discretization is not the same as the
   * volume discretization.
   * @note "Active" here means active for the current contact response. This is distinct from
   * hyper-reduction active elements/faces, which are preselected sample-mesh elements used to
   * limit assembly. For ROMs with hyper-reduction, active contact elements are a subset of the
   * hyper-reduction active boundary faces.
   * @note Indices are unique but not necessarily sorted.
   */
  DynamicArray<int> _activeContactElementIndices;

  /**
   * @brief Membership mask for @ref _activeContactElementIndices.
   *
   * @note _isActiveContactElement[e] is true iff contact integration element e has at least one
   * sample that produced async contact response in the current assembly.
   */
  DynamicArray<bool> _isActiveContactElement;

 public:
  [[nodiscard]] MOCHI_FORCE_INLINE bool Empty() const {
    return _sampleIndices.empty();
  }

  [[nodiscard]] MOCHI_FORCE_INLINE int GetResponseIndexFromSampleIndex(int sampleIndex) const {
    return _responseIndexFromSampleIndex[sampleIndex];
  }

  [[nodiscard]] MOCHI_FORCE_INLINE double GetEnergy(int responseIndex) const {
    return _energies[responseIndex];
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Real3 const& GetGradient(int responseIndex) const {
    return _gradients[responseIndex];
  }

  [[nodiscard]] MOCHI_FORCE_INLINE Matrix3x3r const& GetHessian(int responseIndex) const {
    return _hessians[responseIndex];
  }

  void Clear() {
    _numQuadsPerContactElement = 0;
    _responseIndexFromSampleIndex.clear();
    _sampleIndices.clear();
    _energies.clear();
    _gradients.clear();
    _hessians.clear();
    _activeContactElementIndices.clear();
    _isActiveContactElement.clear();
  }

  void Reset(int numContactElements, int numQuadsPerContactElement) {
    MOCHI_ASSERT_VERBOSE(
        numContactElements >= 0, "Number of contact elements must be non-negative.");
    MOCHI_ASSERT_VERBOSE(
        numQuadsPerContactElement > 0,
        "Number of quadrature points per contact element must be positive.");
    int const numSamples = numContactElements * numQuadsPerContactElement;
    _numQuadsPerContactElement = numQuadsPerContactElement;

    if (isize(_responseIndexFromSampleIndex) == numSamples) {
      // Clear only the entries touched by the previous collision response.
      for (int sampleIndex : _sampleIndices) {
        _responseIndexFromSampleIndex[sampleIndex] = kInvalidIndex;
      }
    } else {
      _responseIndexFromSampleIndex.clear();
      _responseIndexFromSampleIndex.resize(numSamples, kInvalidIndex);
    }
    _sampleIndices.clear();
    _energies.clear();
    _gradients.clear();
    _hessians.clear();

    if (isize(_isActiveContactElement) == numContactElements) {
      // Clear only the entries touched by the previous collision response.
      for (int e : _activeContactElementIndices) {
        _isActiveContactElement[e] = false;
      }
      MOCHI_ASSERT_VERBOSE(
          std::ranges::all_of(_isActiveContactElement, [](bool active) { return !active; }),
          "Active contact element mask must be fully cleared before reuse.");
    } else {
      _isActiveContactElement.clear();
      _isActiveContactElement.resize(numContactElements, false);
    }
    _activeContactElementIndices.clear();
    _activeContactElementIndices.reserve(numContactElements);
  }

  // Add response data for a contact sample. If data is provided more than once for the same sample
  // index, the values are accumulated.
  void AddContactSampleResponse(
      int sampleIndex,
      double const* energy,
      Real3 const* gradient,
      VMatrix3x3r const* hessian) {
    MOCHI_ASSERT_VERBOSE(sampleIndex >= 0, "Contact sample index must be non-negative.");
    MOCHI_ASSERT_VERBOSE(
        sampleIndex < isize(_responseIndexFromSampleIndex),
        "Contact sample index is outside the response lookup range.");
    MOCHI_ASSERT_VERBOSE(
        _numQuadsPerContactElement > 0,
        "Number of quadrature points per contact element must be positive.");

    int responseIndex = _responseIndexFromSampleIndex[sampleIndex];
    if (responseIndex == kInvalidIndex) {
      responseIndex = isize(_sampleIndices);
      _sampleIndices.push_back(sampleIndex);
      _responseIndexFromSampleIndex[sampleIndex] = responseIndex;
      if (energy) {
        _energies.push_back(*energy);
      }
      if (gradient) {
        _gradients.push_back(*gradient);
      }
      if (hessian) {
        _hessians.push_back(ToNdArray3x3(*hessian));
      }
    } else {
      if (energy) {
        _energies[responseIndex] += *energy;
      }
      if (gradient) {
        _gradients[responseIndex] += *gradient;
      }
      if (hessian) {
        _hessians[responseIndex] += ToNdArray3x3(*hessian);
      }
    }

    int const contactElementIndex = sampleIndex / _numQuadsPerContactElement;
    MOCHI_ASSERT_VERBOSE(
        contactElementIndex < isize(_isActiveContactElement),
        "Contact sample index maps outside the contact element range.");
    if (!_isActiveContactElement[contactElementIndex]) {
      _isActiveContactElement[contactElementIndex] = true;
      _activeContactElementIndices.push_back(contactElementIndex);
    }
  }

  void ValidateInvariants([[maybe_unused]] Span<bool const> allowedContactElementMask = {}) const {
#if MOCHI_ASSERT_VERBOSE_ENABLED
    MOCHI_ASSERT_VERBOSE(
        IsUnique(MakeConstSpan(_activeContactElementIndices)),
        "Active contact element indices must be unique.");
    auto const activeCount = static_cast<int>(std::ranges::count(_isActiveContactElement, true));
    MOCHI_ASSERT_VERBOSE(
        activeCount == isize(_activeContactElementIndices),
        "Active contact element mask is inconsistent with active contact element indices.");
    MOCHI_ASSERT_VERBOSE(
        _sampleIndices.empty() == _activeContactElementIndices.empty(),
        "Contact response samples and active contact elements must be either both empty or both non-empty.");
    MOCHI_ASSERT_VERBOSE(
        _energies.empty() || isize(_energies) == isize(_sampleIndices),
        "Contact response energies must be empty or match the number of response samples.");
    MOCHI_ASSERT_VERBOSE(
        _gradients.empty() || isize(_gradients) == isize(_sampleIndices),
        "Contact response gradients must be empty or match the number of response samples.");
    MOCHI_ASSERT_VERBOSE(
        _hessians.empty() || isize(_hessians) == isize(_sampleIndices),
        "Contact response hessians must be empty or match the number of response samples.");
    if (!allowedContactElementMask.empty()) {
      MOCHI_ASSERT_VERBOSE(
          isize(allowedContactElementMask) == isize(_isActiveContactElement),
          "Allowed contact element mask size must match the number of contact elements.");
      for (int contactElementIndex : _activeContactElementIndices) {
        MOCHI_ASSERT_VERBOSE(
            allowedContactElementMask[contactElementIndex],
            "Active contact element is outside the allowed contact element set.");
      }
    }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
  }

  /**
   * @brief Returns the contact-active element subset (active element indices and membership mask)
   * used to restrict assembly to elements with at least one active contact sample.
   *
   * @warning An empty @ref AssemblyActiveSubset is interpreted by the assembler as "assemble all
   * elements", not "assemble none". This method must therefore never be called on an empty
   * response. Otherwise assembly would silently process every element instead of skipping them.
   * Callers must check @ref Empty() first.
   */
  [[nodiscard]] AssemblyActiveSubset ViewActiveContactElementSubset() const {
    MOCHI_ASSERT(
        !_activeContactElementIndices.empty(),
        "Cannot view an empty active contact element subset: empty AssemblyActiveSubset means all elements.");
    return {MakeConstSpan(_activeContactElementIndices), MakeConstSpan(_isActiveContactElement)};
  }
};

template <typename BvType>
class ContactSamplesBvh : public NoCopy {
  // BvhTree stores a raw pointer to PointSetBvhObject. Use unique_ptr's to avoid dangling pointers.
  std::unique_ptr<PointSetBvhObject<BvType>> _object;
  std::unique_ptr<BvhTree<BvType>> _bvh;

 public:
  ContactSamplesBvh(Span<Real3 const> points)
      : _object(std::make_unique<PointSetBvhObject<BvType>>(points)),
        _bvh(
            std::make_unique<BvhTree<BvType>>(
                _object.get(),
                BvhTreeParams{.splittingAlgorithm = BvhSplittingAlgorithm::TopDown_Mean})) {}

  template <typename BvOther>
  void FindIntersectingSamples(BvOther const& bv, DynamicArray<int>& outIntersectingSamples) const {
    _bvh->template FindIntersectingElements</*kSkipElementBvCheck*/ true>(
        bv, outIntersectingSamples);
  }

  void FindIntersectingSamples(
      AnyBoundingVolume const& anyBv,
      DynamicArray<int>& outIntersectingSamples) const {
    std::visit([&](auto const& bv) { FindIntersectingSamples(bv, outIntersectingSamples); }, anyBv);
  }

  void Refit() {
    _bvh->Refit();
  }

  int NumSamplePoints() const {
    return _object->GetNumElements();
  }

  /// @brief Read-only access to the underlying bounding volume hierarchy (used for debug drawing).
  BvhTree<BvType> const& GetBvh() const {
    return *_bvh;
  }
};

/// @brief Contains a vector of potential contact points of an actor in local space.
struct ContactSamples : public NoCopy {
  /// @brief Non-default constructor for sizing of members used by all actor types.
  ContactSamples(int numSamples) {
    weights.resize(numSamples);
    positions.resize(numSamples);
  }

  /// @brief Quadrature weights for surface integration
  /// The weights carry the area info.
  std::vector<real> weights;

  /// @brief Physical position of sample point
  std::vector<Real3> positions;

  // Rigid actors store their contact sample normals because they never change. Soft actors don't
  // because computing all the normals would be expensive. Instead, soft actors compute the
  // normals on-demand after determining which points are actually in contact.
  std::optional<std::vector<Real3>> normals;

  /// @brief Physical positions and indices of the sample points on active boundary faces,
  /// satisfying activePositions[i] = positions[activeIndices[i]]. Only populated if the actor has
  /// CActiveBoundaryFaces component. Empty otherwise.
  std::vector<Real3> activePositions;
  std::vector<int> activeIndices;

  /// @brief Optional BSH tree to accelerate collision detection by culling only sample points that
  /// are potentially in contact.
  std::optional<ContactSamplesBvh<Sphere>> bsh;
};

/**
 * @brief Component that contains a vector of potential contact points of an actor in local space,
 * templatized according to TimeStep.
 *
 * @note Actors whose sample points are fixed (i.e. rigid actors) use
 * CContactSamples<TimeStep::Current> for all time slices. Actors with a deforming surface need
 * CContactSamples<TimeStep::Current> and CContactSamples<TimeStep::StageStart>. However,
 * CContactSamples<TimeStep::StageStart> is updated only if ExperimentalEvalParams.explicitNormals =
 * true.
 */
template <TimeStep kStep>
struct CContactSamples : public ContactSamples {};

/** @brief All partitions of the contact points of an actor. */
struct CContactPartitions : public NoCopy {
  CContactPartitions(std::vector<ContactPartition>&& partitions)
      : _partitions(std::move(partitions)) {
    int numSamples = 0;
    for (int p = 0; p < isize(_partitions); ++p) {
      numSamples += isize(_partitions[p].GetIndices());
    }
    _sampleIdxToPartitionIdx.resize(numSamples, -1);
    for (int p = 0; p < isize(_partitions); ++p) {
      for (int sampleIdx : _partitions[p].GetIndices()) {
        MOCHI_ASSERT_VERBOSE(_sampleIdxToPartitionIdx[sampleIdx] == -1, "Not a partition.");
        _sampleIdxToPartitionIdx[sampleIdx] = p;
      }
    }
#if MOCHI_ASSERT_VERBOSE_ENABLED
    for (int sampleIdx : _sampleIdxToPartitionIdx) {
      MOCHI_ASSERT_VERBOSE(sampleIdx >= 0, "Not a partition.");
    }
#endif // MOCHI_ASSERT_VERBOSE_ENABLED
  }

  int SampleIdxToPartitionIdx(int sampleIdx) const {
    return _sampleIdxToPartitionIdx[sampleIdx];
  }

  auto const& operator[](int p) const {
    MOCHI_ASSERT_VERBOSE(p >= 0 && p < isize(_partitions), "Index out of range.");
    return _partitions[p];
  }

  auto size() const {
    return _partitions.size();
  }

 private:
  std::vector<ContactPartition> _partitions;
  DynamicArray<int> _sampleIdxToPartitionIdx;
};

/**
 * @brief Components for finding correspondences between two ContactDetectionResult at different
 * time slices.
 *
 * @note The purpose of the components is to avoid dynamic memory allocation, but their data is
 * volatile during the problem assembly. In the function AddStageStartCollisionDetection(), the
 * colliding actor uses the component every time it needs to check correspondence between current
 * and stage-start contacts for a particular collider. There are two components, for sync and async
 * contact, to avoid race conditions because they run in parallel. Parallelization across colliders
 * is not possible, because it would turn into race conditions.
 */
template <ContactType kContactType>
class CContactCorrespondence : public ContactCorrespondence {
 public:
  using ContactCorrespondence::ContactCorrespondence;
};

/**
 * @brief World-space velocity of the center-of-mass at the end of the previous step.
 *
 * @note For soft actors, this value is approximated based on the center-of-bounding-volume.
 * @note For static actors, this value is computed based on change in root transform.
 */
struct CPrevRigidVelocity : NoCopy {
  Vec4r linearVelocityWorld = {}; // World-space
  Vec4r angularVelocityWorld = {}; // World-space
  Vec4r centerOfMassLocal = {}; // Actor local-space

  MOCHI_STRUCT_BEGIN(mochi::CPrevRigidVelocity);
  MOCHI_ATTRIBUTE(CaptureState);
  MOCHI_STRUCT_END();
};

/*
  For each contact pair, collection of data needed for contact assembly using contact Jacobians,
  either from the colliding or collider perspective. In sync contact, all actors (both colliding and
  collider) need JacData. In async contact, only some colliding actors need JacData. To reduce
  memory allocation and deallocation of the contact Jacobians, the JacData of a pair of actors in
  contact is recycled across assemblies until the actors are no longer in contact.
  - Contact Jacobians of the contact pair for the owning entity. They are used for assembly of
  implicit (sync) contact for all actors, and for async contact for some actors (i.e., articulated
  bodies with a skinned surface). The size of the array is defined by the maximum number of
  fundamental actors that may define a contact point (currently 2, in soft skinned actors).
  - Contact type.
  - A pointer to the result of the contact detection query.
  - The other entity in the contact pair (not the owner).
  - The ID of the colliding actor's contact sample partition.
  - A flag indicating if both actors are rigid, to signal dedicated collision response functions.
*/
struct JacData {
  static constexpr int kMaxJacs = 2; // Current max is 2. Needed for soft skinned actors.
  ContactType type = ContactType::None;
  ContactDetectionResult* query = nullptr;
  std::unique_ptr<std::array<ContactJac, kMaxJacs>> jacs =
      std::make_unique<std::array<ContactJac, kMaxJacs>>();
  bool bothRigid = false; // True if both colliding and collider actors are rigid.
  entt::entity otherEntity = {}; // The other entity in the contact pair (not the owner).
  int collidingPartitionId = 0; // ID of the colliding actor's contact sample partition.

  JacData(
      ContactType typeIn,
      ContactDetectionResult* queryIn,
      bool bothRigidIn,
      entt::entity otherEntityIn,
      int collidingPartitionIdIn)
      : type(typeIn),
        query(queryIn),
        bothRigid(bothRigidIn),
        otherEntity(otherEntityIn),
        collidingPartitionId(collidingPartitionIdIn) {}

  MOCHI_DECLARE_MOVE_ONLY(JacData); // Avoid copies for performance reasons.

  template <class ContainerT>
  void GetJacs(ContainerT& outJacs) const {
    for (auto const& jac : *jacs) {
      if (jac.nContacts > 0) {
        outJacs.push_back(&jac);
      }
    }
  }
};

enum class CollRole { Colliding = 0, Collider = 1 };

/**
 * @brief Component storing a vector of @ref JacData, templatized according to the collision role
 * (colliding or collider).
 */
template <CollRole kRole>
struct CCollJacs : public std::vector<JacData>, NoCopy {
  // Returns a vector of pointers to the non-empty JacData
  DynamicArray<JacData*> GetPtrsNonEmpty(Allocator* allocator) {
    DynamicArray<JacData*> out(allocator);
    out.reserve(size());
    for (auto& jac : *this) {
      if (!jac.query->sampleIndices.empty()) {
        out.emplace_back(&jac);
      }
    }
    return out;
  }
};

inline void ValidateContactParams(ContactParams const& params, Error& error) {
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.penaltyCoefficient) && params.penaltyCoefficient > 0_r,
      error,
      "Contact penalty coefficient (penaltyCoefficient) must be finite and positive.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.penaltySmoothingHalfDistance) && params.penaltySmoothingHalfDistance >= 0_r,
      error,
      "Contact penalty smoothing half-distance (penaltySmoothingHalfDistance) must be finite and not "
      "negative.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.penaltyThresholdDefault),
      error,
      "Contact penalty threshold (penaltyThresholdDefault) must be finite.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.penaltyThresholdExtraPadding) && params.penaltyThresholdExtraPadding >= 0_r,
      error,
      "Penalty threshold extra padding (penaltyThresholdExtraPadding) must be finite and not "
      "negative.");
  MOCHI_ERROR_IF_NOT(
      params.maxAlignmentNormals >= -1_r && params.maxAlignmentNormals <= 1_r,
      error,
      "Maximum normal alignment (maxAlignmentNormals) must be in [-1, 1].");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.coulombFrictionCoefficient) && params.coulombFrictionCoefficient >= 0_r,
      error,
      "Coulomb friction coefficient (coulombFrictionCoefficient) must be finite and not negative.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.viscousFrictionCoefficient) && params.viscousFrictionCoefficient >= 0_r,
      error,
      "Viscous friction coefficient (viscousFrictionCoefficient) must be finite and not negative.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.frictionFalloffVel) && params.frictionFalloffVel >= 0_r,
      error,
      "Friction falloff velocity (frictionFalloffVel) must be finite and not negative.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.normalViscousDampingCoefficient) &&
          params.normalViscousDampingCoefficient >= 0_r,
      error,
      "Normal viscous damping coefficient (normalViscousDampingCoefficient) must be finite and not "
      "negative.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.distanceErrorBound),
      error,
      "Contact distance error bound (distanceErrorBound) must be finite.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.objScale) && params.objScale > 0_r,
      error,
      "Contact object scale (objScale) must be finite and strictly positive.");
  MOCHI_ERROR_IF_NOT(
      IsFinite(params.collidingPenaltyLengthScale) && params.collidingPenaltyLengthScale > 0_r,
      error,
      "Colliding penalty length scale (collidingPenaltyLengthScale) must be finite and positive.");
}

using ContactAssemblyReg = ecs::PartialRegistry<
    CContactPairParamsOverrideTable const,
    CContactParams const,
    CPointCloudColliderParams const,
    CColliderInfo const,
    CFemSurfaceDiscretization const,
    TagStaticActor const,
    TagShellActor const,
    TagRodSurfaceContact const,
    TagRodActor const,
    CRootTransform const,
    CRigidState<TimeStep::Current> const,
    CRigidState<TimeStep::StageStart> const,
    CRigidBodyInertia const>;

[[nodiscard]] inline int CollidingIntegralDim(
    ContactAssemblyReg const& reg,
    entt::entity colliding) {
  // Rod actors with a contact skin integrate over a 2D surface. Rod actors using centerline
  // contact lump contact traction on the centerline, taking a 1D line integral. All other actor
  // types currently integrate contact on 2D surfaces.
  if (reg.all_of<TagRodActor>(colliding)) {
    return reg.all_of<TagRodSurfaceContact>(colliding) ? 2 : 1;
  }
  return 2;
  // NOTE: Contact with point masses would return 0 here, but it's not supported.
}

/**
 * @brief Returns the dimension of the collider-side integral for a given collider entity.
 *
 * Point-cloud colliders store their integral dimension in @ref CPointCloudColliderParams.
 * All other actor types evaluate the collider SDF at points without any collider-side
 * integration, so the dimension is 0.
 */
[[nodiscard]] inline int ColliderIntegralDim(ContactAssemblyReg const& reg, entt::entity collider) {
  auto const* pcParams = reg.try_get<CPointCloudColliderParams const>(collider);
  int const dim = pcParams ? pcParams->integralDim : 0;
  return dim;
}

/**
 * @brief Returns the collider-side penalty length scale for a given collider entity.
 *
 * Point-cloud colliders use their contact radius as the length scale.
 * All other collider types return 0, which has no effect when the collider integral dimension
 * is also 0.
 */
[[nodiscard]] inline real ColliderPenaltyLengthScale(
    ContactAssemblyReg const& reg,
    entt::entity collider) {
  auto const* pcParams = reg.try_get<CPointCloudColliderParams const>(collider);
  return pcParams ? pcParams->radius : 0_r;
}

/**
 * @brief Combines contact parameters from two actors into a single set of contact parameters.
 *
 * This function implements the combination logic for friction coefficients, penalty coefficient,
 * and friction falloff velocity using geometric mean. Additional parameters can modify the
 * behavior for static colliders and rod actors.
 *
 * @param collidingParams Contact parameters from the colliding actor.
 * @param colliderParams Contact parameters from the collider actor.
 * @param paramsOverride Optional pair-specific parameter replacements.
 * @param isStaticCollider If true, use the colliding actor's penalty and falloff values directly
 * instead of taking the geometric mean.
 * @param collidingIntegralDim The dimension of the colliding-side contact integral (2 for surfaces,
 * 1 for rods).
 * @param colliderIntegralDim The dimension of the collider-side integral (stored in point-cloud
 * collider params, 0 for standard colliders with no collider-side integration).
 * @param colliderPenaltyLengthScale Length scale [m] for the collider-side dimensional correction.
 * For point-cloud colliders, this is the contact radius.
 * @return Combined contact parameters for the pair.
 */
inline ContactParams CombineContactParams(
    ContactParams const& collidingParams,
    ContactParams const& colliderParams,
    ContactPairParamsOverride const* paramsOverride,
    bool isStaticCollider,
    int collidingIntegralDim,
    int colliderIntegralDim,
    real colliderPenaltyLengthScale) {
  ContactParams pairParams = colliderParams;
  if (paramsOverride != nullptr) {
    pairParams = ApplyContactPairParamsOverride(pairParams, *paramsOverride);
  }

  // Friction coefficients: Use the geometric mean of the colliding and collider actors.
  if (paramsOverride == nullptr || !paramsOverride->coulombFrictionCoefficient) {
    pairParams.coulombFrictionCoefficient = Sqrt(
        collidingParams.coulombFrictionCoefficient * colliderParams.coulombFrictionCoefficient);
  }
  if (paramsOverride == nullptr || !paramsOverride->viscousFrictionCoefficient) {
    pairParams.viscousFrictionCoefficient = Sqrt(
        collidingParams.viscousFrictionCoefficient * colliderParams.viscousFrictionCoefficient);
  }
  if (paramsOverride == nullptr || !paramsOverride->normalViscousDampingCoefficient) {
    pairParams.normalViscousDampingCoefficient = Sqrt(
        collidingParams.normalViscousDampingCoefficient *
        colliderParams.normalViscousDampingCoefficient);
  }
  // Penalty coefficient and friction falloff velocity: Use the geometric mean if both actors are
  // dynamic, and the colliding if the collider is static.
  if (paramsOverride == nullptr || !paramsOverride->penaltyCoefficient) {
    pairParams.penaltyCoefficient = isStaticCollider
        ? collidingParams.penaltyCoefficient
        : Sqrt(collidingParams.penaltyCoefficient * colliderParams.penaltyCoefficient);
  }
  if (paramsOverride == nullptr || !paramsOverride->frictionFalloffVel) {
    pairParams.frictionFalloffVel = isStaticCollider
        ? collidingParams.frictionFalloffVel
        : Sqrt(collidingParams.frictionFalloffVel * colliderParams.frictionFalloffVel);
  }

  bool const hasFrictionOverride = paramsOverride != nullptr &&
      (paramsOverride->coulombFrictionCoefficient || paramsOverride->viscousFrictionCoefficient);
  if (!hasFrictionOverride &&
      Max(collidingParams.coulombFrictionCoefficient, collidingParams.viscousFrictionCoefficient) >
          0_r &&
      Max(colliderParams.coulombFrictionCoefficient, colliderParams.viscousFrictionCoefficient) >
          0_r &&
      Max(pairParams.coulombFrictionCoefficient, pairParams.viscousFrictionCoefficient) == 0_r) {
    MOCHI_LOG_WARNING_ONCE(
        "Inconsistent friction coefficients between the contact pair. No friction will be applied.");
  }

  // If penalty tractions are integrated on some colliding manifold other than a 2D surface (e.g.,
  // lumping contact tractions on a thin rod's centerline), we need to correct the penalty factor
  // with a length scale raised to some power. Because quadrature occurs on the colliding body, the
  // colliding body determines the value of the integral's dimension and length scale used for
  // correction.
  real const collidingDimCorrection =
      Pow(collidingParams.collidingPenaltyLengthScale, 2 - collidingIntegralDim);
  pairParams.penaltyCoefficient *= collidingDimCorrection;

  // For point-cloud colliders, the penalty traction is integrated over the collider surface via
  // quadrature, requiring a dimensional correction analogous to the colliding-side correction
  // above. The collider body determines the value of the collider integral's dimension and length
  // scale used for correction.
  MOCHI_ASSERT_VERBOSE(
      colliderIntegralDim == 0 || colliderPenaltyLengthScale > 0_r,
      "Non-zero collider integral dimension requires a positive penalty length scale.");
  real const colliderDimCorrection = Pow(colliderPenaltyLengthScale, -colliderIntegralDim);
  pairParams.penaltyCoefficient *= colliderDimCorrection;

  // All other parameters are those of the collider.
  // TODO: Penalty distance threshold and smoothing distance are those of the collider. For static
  // colliders, it may be beneficial to use those of the colliding actor. That would require changes
  // to (at least) the collision detection pipelines.

  return pairParams;
}

inline ContactParams
GetContactPairParams(ContactAssemblyReg const& reg, entt::entity colliding, entt::entity collider) {
  MOCHI_ASSERT(
      reg.all_of<CContactParams>(colliding) && reg.all_of<CContactParams>(collider),
      "Colliding and collider actors must have contact params.");

  auto const& collidingParams = reg.get<CContactParams const>(colliding);
  auto const& colliderParams = reg.get<CContactParams const>(collider);
  bool const isStaticCollider = reg.all_of<TagStaticActor>(collider);
  int const collidingIntegralDim = CollidingIntegralDim(reg, colliding);
  int const colliderIntegralDim = ColliderIntegralDim(reg, collider);
  real const colliderPenaltyLengthScale = ColliderPenaltyLengthScale(reg, collider);
  auto const& overrideTable = reg.ctx<CContactPairParamsOverrideTable const>();
  ContactPairParamsOverride const* const paramsOverride =
      overrideTable.Empty() ? nullptr : overrideTable.Find(colliding, collider);

  return CombineContactParams(
      collidingParams,
      colliderParams,
      paramsOverride,
      isStaticCollider,
      collidingIntegralDim,
      colliderIntegralDim,
      colliderPenaltyLengthScale);
}

inline bool ValidCollidingNormals(ContactAssemblyReg const& reg, entt::entity colliding) {
  // Tags for actor types without valid normals at colliding samples should be added here.
  return !reg.any_of<TagShellActor, TagRodActor>(colliding);
}

/**************************************************************************
  ECS contact-related pipelines
*/

void UpdateStageStartDataPipeline(entt::registry& reg, CIslandDescendants const& descendants);

template <TimeStep kTimeStep>
void CollisionDetectionPipeline(entt::registry& reg, CIslandDescendants const& descendants);

void ContactJacobiansPipeline(
    entt::registry& reg,
    GradTarget gradTarget,
    CIslandDescendants const& descendants,
    TaskSemaphore const& updateJacobianSem);

// Collision detection for far SDF queries.
// Handles both ContactType::Async and ContactType::Sync.
void FarSdfCollisionDetection(
    ecs::Included<TagUseContact, CRequiresFarSdfEvaluation>,
    entt::registry& reg,
    entt::entity ent);

/**************************************************************************
  ECS contact-related systems
*/

template <bool kUpdateOnlyActiveFaces, typename DiscretizationType, int kNumFields>
void UpdateCollisionSamplePositionsImpl(
    ColumnVectorView<real const> currSol,
    DiscretizationType const& boundaryDiscrVariant,
    CActiveBoundaryFaces const* activeBoundaryFaces,
    ContactSamples& outSamples);

// Updates the collision sample positions, having them match 1:1 the quadrature points of the
// given discretization. Used for colliding objects with a deforming surface.
template <typename DiscretizationType, TimeStep kTimeStep, int kNumFields>
void UpdateCollisionSamplePositions(
    ecs::RequiredTag<TagUseContact>,
    CFinalDisplacementRef<kTimeStep> const& currSol,
    DiscretizationType const& discretization,
    CActiveBoundaryFaces const* activeBoundaryFaces,
    CContactSamples<kTimeStep>& outSamples);

// Adds colliders found in TimeStep::StageStart to TimeStep::Current collisions.
template <ContactType kContactType>
void AddMissingStageStartCollisions(
    CActiveCollisions<kContactType, TimeStep::StageStart> const& stageStartCollisions,
    CActiveCollisions<kContactType, TimeStep::Current>& outCurrentCollisions);

// Adds to TimeStep::Current collisions and contacts found in TimeStep::StageStart.
// This function is needed to ensure continuity of dissipative contact forces within a solve if
// implicitNormalForceForDissipation = false. For performance reasons, currently it is called only
// if explicitNormals = true, hence discontinuities may exist with implicitNormalForceForDissipation
// = false and explicitNormals = false.
template <ContactType kContactType>
void AddStageStartCollisionDetection(
    entt::registry const& reg,
    entt::entity e,
    CColliderInfo const& colliderInfo,
    CContactSamples<TimeStep::Current> const& samples,
    CActiveCollisions<kContactType, TimeStep::StageStart> const& stageStartCollisions,
    CActiveCollisions<kContactType, TimeStep::Current>& outCurrentCollisions,
    CContactCorrespondence<kContactType>& correspondence);

// Update CQuerySdfSurface which is used for debug drawing of the SDF surface and its normals.
void UpdateQuerySdfSurface(
    CSdfCollider const& collider,
    CBoundingVolume<TimeStep::Current> const& bounds,
    CQuerySdfSurface& outQuery);

// Update CQueryContactSamples which is used for debug drawing of contact samples.
void UpdateQueryContactSamples(
    CContactSamples<TimeStep::Current> const& samplePositions,
    CRootTransform const& rootTransform,
    CRecenteringParams const* recenterParams,
    ecs::OptionalTag<TagSoftActor> hasSoftActorTag,
    CQueryContactSamples& outQuery);

// Update CQuerySdfFarDistances which is used for recording distance to far away objects.
void UpdateQuerySdfDistances(
    CContactSamples<TimeStep::Current> const& collidingSamples,
    CRootTransform const& rootTransform,
    CActiveCollisions<ContactType::Async, TimeStep::Current> const* collisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current> const* collisionsSync,
    CRequiresFarSdfEvaluation const* farSdfEval,
    CConvergenceStatus const* convergenceStatus,
    CQuerySdfDistances& outQuerySdfDistances);

// Update components that store results for QueryType::ContactPoints and
// QueryType::NodeContactForces
void UpdateQueryActiveContactsWorldSpace(
    entt::registry const& reg, // Used for contact points with ContactType::Sync
    entt::entity e,
    ecs::RequiredTag<TagQueryActiveContacts>,
    // System is incompatible with things that require a final SDF pass
    ecs::Excluded<CRequiresFarSdfEvaluation>,
    CActiveCollisions<ContactType::Async, TimeStep::Current> const& collisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current> const& collisionsSync,
    CCollJacs<CollRole::Collider> const* colliderJacs,
    CConvergenceStatus const* convergenceStatus,
    CQueryContactPoints* outQueryActiveContacts,
    CQueryNodeContactForces* outQueryNodeForces);

// Update the component that stores results for QueryType::ActorContactForces
void UpdateQueryActorContactForces(
    entt::registry const& reg,
    entt::entity e,
    // System is incompatible with things that require a final SDF pass
    ecs::Excluded<CRequiresFarSdfEvaluation>,
    CContactSamples<TimeStep::Current> const& samples,
    CActiveCollisions<ContactType::Async, TimeStep::Current> const& collisionsAsync,
    CActiveCollisions<ContactType::Sync, TimeStep::Current> const& collisionsSync,
    CCollJacs<CollRole::Collider> const* colliderJacs,
    CRigidState<TimeStep::Current> const* rigidState,
    CConvergenceStatus const* convergenceStatus,
    ecs::OptionalTag<TagRigidActor> hasRigidActorTag,
    CQueryActorContactForces& outQueryActorForces);

// Assemble collision response into DoFs
void AssembleCollisionResponse(
    ContactAssemblyReg reg,
    entt::entity colliding,
    entt::entity collider,
    ContactDetectionResult const& contactQuery,
    CollisionResponseResult const& collisionResponse,
    Span<real const> intWeights,
    Span<ContactJac const*> jacs,
    Allocator* filoAllocator, // Will be used in first-in-last-out order
    double* outObj,
    ColumnVectorView<real> outRes,
    AnyMatrixView<real> outDRes,
    bool isSyncRigid = false); // Optionally set to 'true' to improve performance when assembling
                               // sync contact between rigid (including articulated rigid) actors.

// Assemble sync contact for all pairs of colliding actors within the island.
// Results are written to CIslandContactSnle and possibly to
// CActiveCollisions<ContactType::Sync, TimeStep::Current>.
void AssembleIslandSyncContact(
    AssemblyParams const& params,
    bool useBlockSparse3x3,
    entt::registry& reg,
    CIslandDofInfo const& islandDofInfo,
    CIslandDescendants const& descendants,
    CIslandContactSnle& outContactSnle);

// Assemble async contact for a single colliding actor whose contact samples are tied to its DoFs
// through skinning/embedding (i.e. it carries CSkinnedContactSnle). Currently used for articulated
// actors with skinned contact meshes, nested soft actors configured as colliding actors, and rod
// actors that use visual-mesh contact. Results are written to CSkinnedContactSnle.
void AssembleAsyncSkinnedContact(
    AssemblyParams const& params,
    bool useBlockSparse3x3,
    ecs::RequiredTag<TagSkinnedContact>,
    ecs::OptionalTag<TagQueryActiveContacts> queryActiveContacts,
    entt::registry const& reg,
    entt::entity e,
    ecs::CtxGlobal<CSimulationParams const> simParams,
    CActiveCollisions<ContactType::Async, TimeStep::Current>& activeCollisions,
    CIslandMemberInfo const& islandMember, // TODO: Need a better way to determine the size of
                                           // the interaction residual
    CTimeIntegratorState const& intState,
    CContactSamples<TimeStep::Current> const& samples,
    CColliderInfo const& colliderInfo,
    CCollJacs<CollRole::Colliding> const& collJacs,
    CSkinnedContactSnle& outContactSnle);

Aabb ExpandConservativeBoundsWithContactPadding(
    Aabb bounds,
    ecs::PartialRegistry<
        CContactParams const,
        CRequiresFarSdfEvaluation const,
        CPointCloudColliderParams const> reg,
    entt::entity e);

inline real GetColliderPadding(ContactParams const& contactParams) {
  return contactParams.GetPenaltyThresholdDist(/* addPadding */ true) +
      contactParams.distanceErrorBound;
}

/// @brief Returns `true` if the collider type warrants extra penalty-threshold padding.
inline constexpr bool ShouldAddPenaltyPadding(ColliderType type) {
  return type == ColliderType::None || type == ColliderType::PointCloud;
}

// Expand a collider's bounding volume to include contact penalty falloff. Any point outside of this
// volume should receive zero contact penalty.
template <class ShapeT>
auto ExpandColliderBoundsForContact(
    ShapeT const& colliderBounds,
    ContactParams const& colliderContactParams) {
  return ExpandShape(colliderBounds, GetColliderPadding(colliderContactParams));
}

// Warn if an entity has moved outside of its own CConservativeStepBounds.
// If so, collision detection may be incorrect because CPotentialColliders may be incomplete.
void CheckConservativeStepBounds(entt::registry const& reg, entt::entity e);

/** @brief Build the dof-to-dof connectivity induced by contact between Actors.
 *
 * @param reg
 * @param actors Members of the compound entity.
 * @return The dof-to-dof connectivity from contact.
 */
template <int kBlockSize, ContactType kContactType>
Graph<int, int> MakeContactGraph(entt::registry const& reg, Span<entt::entity const> actors);

extern template Graph<int, int> MakeContactGraph<1, ContactType::Async>(
    entt::registry const& reg,
    Span<entt::entity const> actors);
extern template Graph<int, int> MakeContactGraph<3, ContactType::Async>(
    entt::registry const& reg,
    Span<entt::entity const> actors);

/**************************************************************************
  ECS collider-related systems
*/

// If we have an CSdfColliderPending then wait for initialization to complete, then transfer the
// GridSdf to a CSdfCollider.
void WaitForPendingSdfCollider(
    mochi::CActorInfo const& actorInfo,
    mochi::CSdfColliderPending& pendingComp,
    mochi::CSdfCollider& finalComp);

// Log warnings for SdfCollider diagnostics (e.g. low resolution, non-closed mesh).
void LogSdfColliderDiagnostics(SdfCollider const& collider, std::string const& actorName);

// Log warnings for MeshCollider diagnostics (e.g. non-closed mesh).
void LogMeshColliderDiagnostics(MeshCollider const& collider, std::string const& actorName);

} // namespace mochi

namespace mochi::contact {

/**
  Contact Simulation
*/

// Call this at the beginning of each simulation step to update CConservativePotentialColliders for
// all actors. Must come AFTER CConservativeStepBounds has been updated.
void UpdateConservativePotentialColliders(entt::registry& reg);

void InitializeOnce(entt::registry& reg);

} // namespace mochi::contact

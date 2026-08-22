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

#include "mochi_constraint_interface.h"

#include "mochi_constraint.h"
#include "mochi_ecs_utils.h"
#include "mochi_query.h"
#include "mochi_scene.h"

#include <memory>

using namespace mochi;

namespace {
class ConstraintInterfaceImpl final : public ConstraintInterface {
 public:
  entt::registry& reg;
  entt::entity e = {};
  SceneImpl* scene = nullptr;

  explicit ConstraintInterfaceImpl(entt::registry& reg_, entt::entity e_, SceneImpl* scene_)
      : reg(reg_), e(e_), scene(scene_) {
    MOCHI_ASSERT(e != entt::null);
    MOCHI_ASSERT(scene != nullptr);
  }

  ConstraintType GetType() const override {
    return reg.get<CConstraintInfo const>(e).type;
  }

  ConstraintHandle GetHandle() const override {
    return GetConstraintHandle(e, scene->GetHandle());
  }

  real GetStiffness() const override {
    return reg.get<CConstraintInfo const>(e).stiffness;
  }

  void SetStiffness(real stiffness, Error& error) override {
    MOCHI_ERROR_IF(
        stiffness < 0_r || !IsFinite(stiffness),
        error,
        "Stiffness must be non-negative and finite.");
    MOCHI_ERROR_RETURN(error);
    reg.get<CConstraintInfo>(e).stiffness = stiffness;
  }

  real GetDamping() const override {
    return reg.get<CConstraintInfo const>(e).damping;
  }

  void SetDamping(real damping, Error& error) override {
    MOCHI_ERROR_IF(
        damping < 0_r || !IsFinite(damping), error, "Damping must be non-negative and finite.");
    MOCHI_ERROR_RETURN(error);
    reg.get<CConstraintInfo>(e).damping = damping;
  }

  real GetSaturation() const override {
    return reg.get<CConstraintInfo const>(e).saturation;
  }

  void SetSaturation(real saturation, Error& error) override {
    MOCHI_ERROR_IF_NOT(IsFinite(saturation), error, "Saturation must be finite.");
    MOCHI_ERROR_IF(
        saturation == 0_r,
        error,
        "Saturation must not be zero. Use any negative value to disable saturation or a positive value to enable it.");
    MOCHI_ERROR_RETURN(error);
    reg.get<CConstraintInfo>(e).saturation = saturation;
  }

  DynamicArray<real> GetDeviation() const override {
    DynamicArray<real> c(GetConstraintSize(GetType()));
    bool isActive{};
    EvalConstraint<TimeStep::Current>(reg, e, c, {}, {}, isActive);
    return c;
  }

  Span<real const> GetForce(Error& error) const override {
    MOCHI_ERROR_RETURN(error, {});
    auto const* query = reg.try_get<CQueryConstraintForce const>(e);
    MOCHI_ERROR_IF_NOT(
        query && query->force.Rows() > 0,
        error,
        "If you registered for the query, then the results should be available after the next simulation step.");
    MOCHI_ERROR_RETURN(error, {});
    return query->force;
  }

  int GetNumActors() const override {
    return isize(reg.get<CConstraintInfo const>(e).actors);
  }

  Actor* GetActor(int actorIndex) override {
    auto const& actors = reg.get<CConstraintInfo const>(e).actors;
    MOCHI_ASSERT(actorIndex >= 0 && actorIndex < isize(actors), "Actor index out of bounds");
    auto* actor = scene->GetActor(GetActorHandle(actors[actorIndex], scene->GetHandle()));
    MOCHI_ASSERT(
        actor, "If the actor was destroyed, then this constraint should have been destroyed too");
    return actor;
  }

  Actor const* GetActor(int actorIndex) const override {
    // Shares code with the non-const implementation
    return const_cast<ConstraintInterfaceImpl*>(this)->GetActor(actorIndex);
  }

  Span<int const> GetDofIndicesForActor(int actorIndex) const override {
    auto const& info = reg.get<CConstraintInfo const>(e);
    auto const numActors = isize(info.actors);
    MOCHI_ASSERT(actorIndex >= 0 && actorIndex < numActors, "Actor index out of bounds");
    return info.actorDofs[actorIndex];
  }

  void SetTargetPosition(Real3 const& position, Error& error) override {
    MOCHI_ERROR_IF_NOT(IsFinite(position), error, "Target position must be finite.");
    MOCHI_ERROR_RETURN(error);
    bool invoked =
        ecs::TryInvokeOnEntity(SetConstraintTargetPosition<Real3>, reg, e, std::cref(position)) ||
        ecs::TryInvokeOnEntity(
            SetConstraintTargetPosition<TransformRT>, reg, e, std::cref(position));
    MOCHI_ERROR_IF(!invoked, error, "Constraint type has no position target");
  }

  void SetTargetRotation(Quaternion const& rotation, Error& error) override {
    MOCHI_ERROR_IF_NOT(IsFinite(rotation), error, "Target quaternion must be finite.");
    MOCHI_ERROR_IF(NearEqual(Norm(rotation), 0_r), error, "Target quaternion must be non-zero.");
    MOCHI_ERROR_RETURN(error);
    auto rot = Normalize(rotation);
    bool invoked =
        ecs::TryInvokeOnEntity(SetConstraintTargetRotation<Quaternion>, reg, e, std::cref(rot)) ||
        ecs::TryInvokeOnEntity(SetConstraintTargetRotation<TransformRT>, reg, e, std::cref(rot));
    MOCHI_ERROR_IF(!invoked, error, "Constraint type has no rotation target");
  }

  void SetTargetDof(real val, Error& error) override {
    MOCHI_ERROR_IF_NOT(IsFinite(val), error, "Target DoF value must be finite.");
    using Component = CConstraintTarget<real, TimeStep::Current>;
    auto* target = MOCHI_TRY_GET(Component, reg, e, error);
    MOCHI_ERROR_RETURN(error);
    target->value = val;
  }

  void UpdateOldTarget(Error& error) override {
    MOCHI_ERROR_RETURN(error);
    bool invoked = ecs::TryInvokeOnEntity(UpdateConstraintOldTarget<real>, reg, e) ||
        ecs::TryInvokeOnEntity(UpdateConstraintOldTarget<Real3>, reg, e) ||
        ecs::TryInvokeOnEntity(UpdateConstraintOldTarget<Quaternion>, reg, e) ||
        ecs::TryInvokeOnEntity(UpdateConstraintOldTarget<TransformRT>, reg, e);
    MOCHI_ERROR_IF(!invoked, error, "Constraint type has no old target");
  }

  void SetRefRelativeRotation(Quaternion const& qA, Quaternion const& qB, Error& error) override {
    MOCHI_ERROR_IF_NOT(IsFinite(qA) && IsFinite(qB), error, "Reference rotations must be finite.");
    MOCHI_ERROR_IF(
        NearEqual(Norm(qA), 0_r) || NearEqual(Norm(qB), 0_r),
        error,
        "Reference quaternions must be non-zero.");
    MOCHI_ERROR_RETURN(error);
    bool const hasRotationTarget = reg.any_of<
        CConstraintData<ConstraintType::JointRotationTracking>,
        CConstraintData<ConstraintType::RodElementRotationToRigid>>(e);
    MOCHI_ERROR_IF(!hasRotationTarget, error, "Constraint must have a relative rotation target");
    MOCHI_ERROR_RETURN(error);
    auto& target = reg.get<CConstraintTarget<Quaternion, TimeStep::Current>>(e).value;
    target = Normalize(Normalize(qA).GetConjugate() * Normalize(qB));
  }

  Span<real const> GetLimitMinValues(Error& error) const override {
    MOCHI_ERROR_RETURN(error, {});
    auto const* articulated3dRotationRangeData =
        reg.try_get<CConstraintData<ConstraintType::Articulated3dRotationRange> const>(e);
    if (articulated3dRotationRangeData) {
      return articulated3dRotationRangeData->minValues;
    }
    auto const* articulatedSingleDofRangeData =
        reg.try_get<CConstraintData<ConstraintType::ArticulatedSingleDofRange> const>(e);
    if (articulatedSingleDofRangeData) {
      return MakeSingletonConstSpan(articulatedSingleDofRangeData->minValue);
    }
    auto const* jointRotationRangeData =
        reg.try_get<CConstraintData<ConstraintType::JointRotationRange> const>(e);
    if (jointRotationRangeData) {
      return jointRotationRangeData->minRotVec;
    }
    MOCHI_ERROR_SET(error, "Constraint type does not support min values");
    return {};
  }

  Span<real const> GetLimitMaxValues(Error& error) const override {
    MOCHI_ERROR_RETURN(error, {});
    auto const* articulated3dRotationRangeData =
        reg.try_get<CConstraintData<ConstraintType::Articulated3dRotationRange> const>(e);
    if (articulated3dRotationRangeData) {
      return articulated3dRotationRangeData->maxValues;
    }
    auto const* articulatedSingleDofRangeData =
        reg.try_get<CConstraintData<ConstraintType::ArticulatedSingleDofRange> const>(e);
    if (articulatedSingleDofRangeData) {
      return MakeSingletonConstSpan(articulatedSingleDofRangeData->maxValue);
    }
    auto const* jointRotationRangeData =
        reg.try_get<CConstraintData<ConstraintType::JointRotationRange> const>(e);
    if (jointRotationRangeData) {
      return jointRotationRangeData->maxRotVec;
    }
    MOCHI_ERROR_SET(error, "Constraint type does not support max values");
    return {};
  }

  QueryHandle RegisterQuery(QueryType type, Error& error) override {
    constexpr bool kComputeImmediately = false;
    return mochi::RegisterQuery(reg, e, type, kComputeImmediately, error);
  }

  void CancelQuery(QueryHandle handle) override {
    mochi::CancelQuery(reg, e, handle);
  }

  bool IsQuerySupported(QueryType type) const override {
    Error error;
    SetErrorIfQueryNotSupported(reg, e, type, error);
    return error.IsOK();
  }
};
} // namespace

ConstraintInterfacePtr
mochi::CreateConstraintInterface(entt::registry& reg, entt::entity e, SceneImpl* scene) {
  return std::make_unique<ConstraintInterfaceImpl>(reg, e, scene);
}

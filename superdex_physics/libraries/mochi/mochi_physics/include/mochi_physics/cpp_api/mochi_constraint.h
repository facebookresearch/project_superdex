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

#include "mochi_enums.h"
#include "mochi_handle.h"
#include "mochi_scene.h"
#include "mochi_structs.h"

/********************************************************************************
 IMPORTANT: PLEASE KEEP HEADER INCLUDES TO A MINIMUM.
    If you must include a mochi_core header, then please make sure that it only
    declares the data types (not containing other implementation details).
*********************************************************************************/
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/error.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/span.h>

namespace mochi {

class Actor;

class Constraint {
 public:
  [[nodiscard]] virtual ConstraintType GetType() const = 0;

  [[nodiscard]] virtual ConstraintHandle GetHandle() const = 0;

  [[nodiscard]] virtual real GetStiffness() const = 0;

  virtual void SetStiffness(real stiffness, Error& error) = 0;

  [[nodiscard]] virtual real GetDamping() const = 0;

  virtual void SetDamping(real damping, Error& error) = 0;

  [[nodiscard]] virtual real GetSaturation() const = 0;

  virtual void SetSaturation(real saturation, Error& error) = 0;

  [[nodiscard]] virtual DynamicArray<real> GetDeviation() const = 0;

  [[nodiscard]] virtual Span<real const> GetForce(Error& error) const = 0;

  [[nodiscard]] virtual int GetNumActors() const = 0;

  [[nodiscard]] virtual Actor* GetActor(int actorIndex) = 0;

  [[nodiscard]] virtual Actor const* GetActor(int actorIndex) const = 0;

  [[nodiscard]] virtual Span<int const> GetDofIndicesForActor(int actorIndex) const = 0;

  virtual void SetTargetPosition(Real3 const& position, Error& error) = 0;

  virtual void SetTargetRotation(Quaternion const& rotation, Error& error) = 0;

  virtual void SetTargetDof(real target, Error& error) = 0;

  virtual void UpdateOldTarget(Error& error) = 0;

  virtual void SetRefRelativeRotation(
      Quaternion const& rotationA,
      Quaternion const& rotationB,
      Error& error) = 0;

  [[nodiscard]] virtual Span<real const> GetLimitMinValues(Error& error) const = 0;

  [[nodiscard]] virtual Span<real const> GetLimitMaxValues(Error& error) const = 0;

  virtual QueryHandle RegisterQuery(QueryType type, Error& error) = 0;

  virtual void CancelQuery(QueryHandle handle) = 0;

  [[nodiscard]] virtual bool IsQuerySupported(QueryType type) const = 0;

 protected:
  // Don't delete the Constraint pointer. Call Scene::DestroyConstraint.
  virtual ~Constraint() = default;
};

} // namespace mochi

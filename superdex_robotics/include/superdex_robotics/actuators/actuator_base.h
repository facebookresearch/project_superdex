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

#include <superdex_physics.h>
#include <superdex_robotics/core/component_base.h>

#include <string_view>

namespace superdex::robotics {

/* @brief Base class for robot actuators.
 *
 * Mirrors @ref ControllerBase and @ref SensorBase: derives from ComponentBase for common lifecycle
 * (validity, actor, destroy). No concrete actuators exist yet; this establishes the contract early
 * so the first actuator follows the same shape as controllers and sensors.
 *
 * Registration contract (a RoboticsContext::RegisterActuator<T> template lands with the first
 * concrete actuator, mirroring RegisterController/RegisterSensor): every actuator must declare
 *   static constexpr std::string_view TypeName() { return "..."; }
 * (used by the registration templates and for the generated Python type_name() binding), implement
 *   std::string_view GetTypeName() const override; // returns TypeName()
 * and provide a uniform constructor
 *   ActuatorX(Actor* actor, std::string_view paramArgs, Error& error);
 * that loads its own params from the path or inline JSON. An (Actor*, Params const&, Error&)
 * overload is recommended for programmatic / non-filesystem construction.
 *
 * Teardown contract (mirrors @ref ControllerBase / @ref SensorBase): @ref GetActor() resolves the
 * live actor on demand and yields nullptr once the owning mochi Scene (or the actor) is gone, so an
 * actuator that outlives its Scene (Scene destroyed before the RoboticsContext) sees @ref
 * GetActor() == nullptr in its destructor. Any concrete actuator whose destructor releases
 * actor-held resources must therefore guard on @ref GetActor() != nullptr (or, for context-owned
 * resources, resolve via
 * @ref GetContext()) so teardown never dereferences a dangling Actor*. */
// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class MOCHI_API ActuatorBase : public ComponentBase {
 public:
  /* @brief Reset internal state between episodes, keeping params. Pure virtual so every actuator
   * states its answer explicitly: a stateful actuator that silently inherited a no-op would leave
   * stale state alive across a reset, which is only discoverable in the collected data. An actuator
   * that holds no per-episode state implements this as an empty body. */
  void Reset() override = 0;

 protected:
  /* @brief Common actuator constructor archetype: stores the actor via ComponentBase and validates
   * it is non-null. Every actuator delegates to this, then loads its own params. */
  ActuatorBase(Actor* actor, superdex::Error& error) : ComponentBase(actor) {
    MOCHI_ERROR_IF(actor == nullptr, error, "ActuatorBase: actor is null");
  }
  ~ActuatorBase() override;
};

} // namespace superdex::robotics

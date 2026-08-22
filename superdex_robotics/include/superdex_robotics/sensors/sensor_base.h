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

#include <mochi_core/utils/transform_rt.h>

#include <string_view>

namespace superdex::robotics {

/* @brief Base class for robot sensors.
 * Derives from ComponentBase for common lifecycle (validity, actor, destroy).
 * Each derived sensor keeps its own specialized ComputeSignal signature. */
// NOLINTNEXTLINE(cppcoreguidelines-virtual-class-destructor)
class MOCHI_API SensorBase : public ComponentBase {
 public:
  /* @brief Reset internal state between episodes, keeping params. Pure virtual so every sensor
   * states its answer explicitly: a stateful sensor that silently inherited a no-op would leave
   * stale state alive across a reset, which is only discoverable in the collected data. A sensor
   * that holds no per-episode state implements this as an empty body. */
  void Reset() override = 0;

  /* @brief The sensor's pose relative to its parent frame: the associated link's actor for
   * bot-owned sensors, or the scene root for scene-level / actor-less sensors. aFromB convention
   * (maps sensor-frame coordinates into the parent frame).
   * @return Sensor pose relative to its parent frame. */
  [[nodiscard]] TransformRT const& GetParentFromSensor() const {
    return _parentFromSensor;
  }

  /* @brief Set the sensor's pose relative to its parent frame.
   * @param parentFromSensor The new parent-from-sensor transform (aFromB convention). */
  void SetParentFromSensor(TransformRT const& parentFromSensor) {
    _parentFromSensor = parentFromSensor;
  }

  /* @brief The sensor's pose in world space.
   * When the sensor is bound to a Mochi @ref Actor, that actor's transform is already
   * world-resolved by the simulation, so world = actorWorldTransform * parentFromSensor. When the
   * sensor has no actor (a fusion sensor that consumes other sensors, or a scene sensor standing in
   * for a scene root that has no Mochi actor), the scene root is the parent: world =
   * parentFromSensor, i.e. the scene-root-to-world transform is treated as identity (a surrogate
   * for a real scene-root actor) until scene spawn transforms are supported.
   * @return Sensor pose in world space. */
  [[nodiscard]] TransformRT GetWorldTransform() const;

 protected:
  /* @brief Common sensor constructor archetype: stores the (optionally null) @p actor via
   * ComponentBase. A null actor is permitted so a sensor can run without a live Mochi body -- e.g.
   * a fusion sensor that consumes other sensors' outputs and re-emits fused data, mirroring how a
   * controller can run without an Actor. Sensors that require an actor enforce that themselves. The
   * @p error argument is retained for archetype/delegation uniformity even though the base no
   * longer validates the actor. */
  SensorBase(Actor* actor, [[maybe_unused]] superdex::Error& error) : ComponentBase(actor) {}

  /* The sensor's pose relative to its parent frame (see GetParentFromSensor). */
  TransformRT _parentFromSensor;

  ~SensorBase() override;
};

} // namespace superdex::robotics

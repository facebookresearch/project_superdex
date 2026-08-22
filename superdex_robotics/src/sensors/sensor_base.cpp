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

#include <superdex_robotics/sensors/sensor_base.h>

using namespace superdex::robotics;
using namespace mochi;

SensorBase::~SensorBase() = default;

TransformRT SensorBase::GetWorldTransform() const {
  if (Actor* const actor = GetActor()) {
    // The Mochi actor's transform is already world-resolved by the simulation (the articulation
    // hierarchy up to the world is baked in), so it must NOT be composed with the scene transform.
    return actor->GetRootTransform() * _parentFromSensor;
  }
  // Actor-less sensor: the scene root is the parent frame. Its scene-root-to-world transform is a
  // surrogate for a real scene-root actor and is identity until scene spawn transforms are set.
  return _parentFromSensor;
}

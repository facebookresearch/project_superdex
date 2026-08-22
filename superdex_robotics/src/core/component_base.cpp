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

#include <superdex_robotics/core/component_base.h>

namespace superdex::robotics {
using namespace mochi;

ComponentBase::ComponentBase(Actor* actor) {
  if (actor != nullptr) {
    _actorHandle = actor->GetHandle();
    if (Scene* const scene = actor->GetScene()) {
      _sceneHandle = scene->GetHandle();
      _context = scene->GetContext();
    }
  }
}

/* Out-of-line destructor anchors the vtable in this translation unit.
 * This ensures RTTI works correctly across shared library boundaries. */
ComponentBase::~ComponentBase() = default;

Scene* ComponentBase::GetScene() const {
  return _context != nullptr ? _context->GetScene(_sceneHandle) : nullptr;
}

Actor* ComponentBase::GetActor() const {
  Scene* const scene = GetScene();
  return scene != nullptr ? scene->GetActor(_actorHandle) : nullptr;
}

void ComponentBase::Destroy(ComponentBase* component) {
  if (component) {
    /* Mark invalid before deletion. There is no cached actor to refresh: destructors that release
     * actor-held resources (contact queries, articulated pose controllers, Newton-Euler terms) call
     * GetActor(), which resolves the live actor on demand and yields nullptr if the owning scene
     * (or the actor) is already gone, so they skip that work instead of touching a dangling Actor*.
     */
    component->SetInvalid();
  }
  delete component;
}

} // namespace superdex::robotics

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

#include "mochi_handle.h"
#include "mochi_structs.h"

/********************************************************************************
 IMPORTANT: PLEASE KEEP HEADER INCLUDES TO A MINIMUM.
    If you must include a mochi_core header, then please make sure that it only
    declares the data types (not containing other implementation details).
*********************************************************************************/
#include <mochi_core/utils/error.h>

#include <string_view>

namespace mochi {

class DebugDraw {
 public:
  [[nodiscard]] virtual bool IsEnabled() const = 0;

  virtual void Enable(bool enable) = 0;

  [[nodiscard]] virtual int GetNumFeatures() const = 0;

  [[nodiscard]] virtual int FindFeature(std::string_view name) const = 0;

  [[nodiscard]] virtual std::string_view GetFeatureName(int index) const = 0;

  [[nodiscard]] virtual std::string_view GetFeatureDescription(int index) const = 0;

  [[nodiscard]] virtual bool IsFeatureEnabled(int index) const = 0;

  virtual void EnableFeature(int index, bool enable) = 0;

  [[nodiscard]] virtual DebugDrawData GatherData() = 0;

  virtual void EnableActor(ActorHandle actor, bool enable, Error& error) = 0;

 protected:
  virtual ~DebugDraw() = default; // Owned by mochi::Scene
};

} // namespace mochi

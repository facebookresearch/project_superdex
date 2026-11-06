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

#include <mochi_renderer/resource.h>

#include <mochi_core/utils/span.h>

#include <memory>

namespace mochi_renderer {

class DeformableModelInstance : public RenderModelInstance, public IInstanceable {
 public:
  ~DeformableModelInstance() override;

  // Sets every vertex in the model root's local frame. The vertex count and topology stay fixed;
  // tangent frames for normal maps are derived from @p normals and the GLB's TEXCOORD_0.
  void SetVertices(mochi::Span<float const> positions, mochi::Span<float const> normals);
  int GetVertexCount() const;

  DeformableModelInstance* AsDeformableModelInstance() override {
    return this;
  }
  IInstanceable* GetInstanceable() override {
    return this;
  }
  // Copies start at this instance's current pose and deform independently afterward.
  std::unique_ptr<SceneObject> GetInstance() override;
  int GetInstanceCount() const override;

 private:
  friend class RenderModel;
  DeformableModelInstance(
      RenderModel* model,
      filament::gltfio::FilamentInstance* instance,
      int slot);

  int _slot = -1;
};

} // namespace mochi_renderer

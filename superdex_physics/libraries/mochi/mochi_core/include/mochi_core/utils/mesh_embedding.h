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

#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>
#include <vector>

namespace mochi {

/*********************************************************************************
  Embeddings.
*/

/**
 * Describes a (possibly non-linear) geometric embedding between
 * two meshes. Specifically designed for visualization purposes.
 */
class MeshEmbedding {
 public:
  virtual ~MeshEmbedding() = default;

 public:
  virtual void Update(Span<real const> src, Span<real> dst) const = 0;
  virtual void Update(Span<Real3 const> src, Span<Real3> dst) const = 0;
};

/**
 * Describes a simple linear embedding.
 */
class LinearMeshEmbedding final : public MeshEmbedding {
 public:
  explicit LinearMeshEmbedding(
      size_t entriesPerValue,
      Span<int const> indices,
      Span<real const> weights);

 public:
  void Update(Span<real const> src, Span<real> dst) const override;
  void Update(Span<Real3 const> src, Span<Real3> dst) const override;
  Span<int const> GetIndices() const {
    return MakeSpan(_indices);
  };
  Span<real const> GetWeights() const {
    return MakeSpan(_weights);
  };
  size_t GetNumSkinningWeightsPerEntry() const {
    return _entriesPerValue;
  };

 private:
  size_t _entriesPerValue = 0;
  size_t _expectedDstSize = 0;
  std::vector<int> _indices;
  std::vector<real> _weights;
};

} // namespace mochi

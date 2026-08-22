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

#include <mochi_core/contact/contact_types.h>
#include <mochi_core/geometry/any_shape.h>
#include <mochi_core/geometry/scalar_field.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/transform_rt.h>

namespace mochi {

/******************************************************************************
  Sdf - Signed Distance Field
*/

class Sdf {
 public:
  // Find contacts between a span of points and the 0-level set of the SDF (i.e., the collider).
  virtual void FindPointContacts(
      Span<Real3 const> points,
      TransformRT const& pointsFromActor,
      ContactDetectionParams const& params,
      DynamicArray<int>& outIndices,
      DynamicArray<Real3>& outContacts,
      SdfInfo& outSdf,
      bool& outIsSdfGradUnitary) const = 0;

  // Get the bounding volume containing the 0-level set of the SDF (i.e., the collider).
  virtual AnyShape GetColliderBounds() const = 0;

  virtual ~Sdf() = default;
};

} // namespace mochi

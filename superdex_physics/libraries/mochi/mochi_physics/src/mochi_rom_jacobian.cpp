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

#include "mochi_rom_jacobian.h"
#include "mochi_scene_recorder.h"

#include <mochi_core/contact/dmap.h>

#include <optional>
#include <type_traits>
#include <variant>

using namespace mochi;
using namespace mochi::rom;
using namespace mochi::dmap;

int CRomJacobian::Cols() const {
  return Visit(OverloadVisitor{[](auto const& item) { return item.Cols(); }});
}

void CRomJacobian::ApplyTranspose(ColumnVectorView<real const> input, ColumnVectorView<real> output)
    const {
  return Visit(
      OverloadVisitor{
          [&](DenseT const& dense) { output = dense.Transpose() * input; },
      });
}

void CRomJacobian::Apply(ColumnVectorView<real const> input, ColumnVectorView<real> output) const {
  Visit(
      OverloadVisitor{
          [&](DenseT const& dense) { output = dense * input; },
      });
}

namespace mochi::rom::jacobian {
void InitializeOnce(entt::registry& reg) {
  ecs::RegisterComponent<CIntermediateRomJacobian>(reg);
  ecs::RegisterComponent<CRomJacobian>(reg);
  ecs::RegisterComponent<CRomJacobianTypes>(reg);
}
} // namespace mochi::rom::jacobian

void mochi::rom::SetupCollidingJacobians(
    ecs::Included<TagRomActor>,
    ecs::Excluded<TagNestedSoftActor>,
    CFemBoundaryDiscretization const& discretization,
    CRootTransform const& transform,
    CDofOffset const& dofOffset,
    CRomJacobian const& jacobianRom,
    CCollJacs<CollRole::Colliding>& outJacobians) {
  MOCHI_PROFILE_SCOPE();

  discretization.Visit([&](auto const& discretizationImpl) {
    using DiscretizationT = std::decay_t<decltype(discretizationImpl)>;
    using ElementT = typename DiscretizationT::ElementT;

    // Create shared differentiable maps. Consider the case where the ROM has no root transform.
    bool useTransform = transform.worldFromLocal != TransformRT::Identity();
    using DQuad = DMapQuad<ElementT>;
    using DNoTransform = DMap<DQuad, DMapRom>;
    using DWithTransform = DMap<DQuad, DMapRTConst, DMapRom>;
    using DMapThis = std::variant<DNoTransform, DWithTransform>;
    DMapRom drom(0, jacobianRom.value, dofOffset.dofsOffset);
    std::optional<DMapRTConst> dtransform;
    if (useTransform) {
      dtransform.emplace(transform.worldFromLocal);
    }

    // Compute Jacobians
    for (auto& jac : outJacobians) {
      if (jac.type == ContactType::Sync) {
        // Create differentiable map.
        DQuad dquad(discretizationImpl.femElements, jac.query->jacColliderFromWorld);
        DMapThis dmap(
            useTransform ? DMapThis{DWithTransform(&dquad, &dtransform.value(), &drom)}
                         : DMapThis{DNoTransform(&dquad, &drom)});
        std::visit(
            [&jac](auto const& dmap) { dmap.GetJac(jac.query->sampleIndices, *jac.jacs); }, dmap);
        (*jac.jacs)[0].CompressIndices();
      }
    }
  });
}

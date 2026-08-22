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

#include <mochi_core/elements/transforms.h>
#include <mochi_core/materials/alt_smith_neo_hookean.h>
#include <mochi_core/materials/smith_neo_hookean_params.h>

namespace mochi {

struct SmithNeoHookeanPseudoLame {
  SmithNeoHookeanPseudoLame(auto E, auto nu) {
    // Parameterization of Lame parameters for linear materials
    lambda = (E * nu / ((1_r + nu) * (1_r - 2_r * nu)));
    mu = (E / (2_r * (1_r + nu)));
    alpha = (1_r + mu / lambda - mu / (lambda * 4_r));
    // Reparameterization to be consistent with linear elasticity. This reparameterization
    // is mentioned in Stable Neo-Hookean Flesh Simulation [Smith et al. 2018, Sec. 3.4]
    lambda += mu * (5.0_r / 6.0_r);
    mu = mu * (4.0_r / 3.0_r);
    alpha = (1_r + mu / lambda - mu / (lambda * 4_r));
  }
  explicit SmithNeoHookeanPseudoLame(SmithNeoHookeanMaterialParams const& params)
      : SmithNeoHookeanPseudoLame(params.youngsModulus, params.poissonRatio) {}

  real lambda;
  real mu;
  real alpha;
};

/** @brief Integrate an element's stiffness contribution.
 *
 * The material tensor is stored as a 9x9 matrix C. Each Gauss point has contribution given by:
 * \f$ B^T C B |J| w_g \f$
 * In tensor form, each entry is:
 * \f$ K_{ij} = \frac{\partial F}{\partial q_i} : C : \frac{\partial F}{\partial q_j} \f$
 * where
 * \f$ F = \frac{\partial x}{\partial X} = \frac{\partial x}{\partial \xi}\frac{\partial
 * xi}{\partial X} \f$ Each term \f$ \frac{\partial F}{\partial q_i} \f$ is flattened into the ith
 * column of \f$ B \f$. The flattening's order works column by column. Effectively that is
 * equivalent to stacking the columns of of \f$ \frac{\partial F}{\partial q_i} \F$ into a vector of
 * 9 components.
 *
 * Remarkably, with this flattening, \f$ B \f$ is made of 3 by number of nodes diagonal blocks.
 * Each block is a single factor times the 3x3 identity matrix.
 *
 * @tparam nNodes Number of nodes in the element
 * @param params
 * @param K
 * @param disp
 * @param refCoords
 * @param dBari_dxi
 * @param weights
 */
template <int nNodes>
MOCHI_ANY void Integrate(
    SmithNeoHookeanPseudoLame const& params,
    auto&& energy,
    auto&& force,
    auto&& K,
    StridedMatrix<real, 3, nNodes>& disp,
    StridedMatrix<real, 3, nNodes>& refCoords,
    Span<StridedMatrixView<real const, nNodes, 3>, int> const& dBari_dxi,
    Span<real const> weights) {
  using namespace mochi::materials;
  constexpr bool kEnergy = IsNotVoidObject<decltype(energy)>;
  constexpr bool kStress = IsNotVoidObject<decltype(force)>;
  constexpr bool kHessian = IsNotVoidObject<decltype(K)>;
  auto numGP = weights.size();
  for (int gpIndex = 0; gpIndex < numGP; ++gpIndex) {
    auto const& dBdxi = dBari_dxi[gpIndex];
    auto map = ComputeDefGradient(disp, refCoords, dBdxi);
    auto gpCoef = weights[gpIndex] * map.J;
    auto state = AltSmithNeoHookeanState<kEnergy, kStress, kHessian>(
        params.mu, params.lambda, params.alpha, map.F);
    // The material tensor is stored as a 9x9 matrix C.
    // We need to form B^T C B.
    if constexpr (kEnergy) {
      energy += gpCoef * state.energy;
    }
    if constexpr (kStress) {
      StridedMatrix<real, 3, nNodes> nodalB = map.dxi_dX.Transpose() * dBdxi.Transpose();
      force += gpCoef * (state.stress * nodalB);
    }
    if constexpr (kHessian) {
      auto CBlock3x3 = [&C = state.hessian](int rowDof, int colDof) {
        return StridedMatrixView<real, 3, 3, 1, Direction::ColMajor, 9>{
            C.Data() + 3 * rowDof + 27 * colDof};
      };
      // Compact representation of B. (1/9th the size) of the full B
      StridedMatrix<real, 3, nNodes> nodalB = map.dxi_dX.Transpose() * dBdxi.Transpose();
      for (int colNd = 0; colNd < nNodes; ++colNd) {
        for (int rowDof = 0; rowDof < 3; ++rowDof) {
          // One term of C*B : rowDof, colNd. Summed over colDof
          StridedMatrix<real, 3, 3> T = nodalB(0, colNd) * CBlock3x3(rowDof, 0) +
              nodalB(1, colNd) * CBlock3x3(rowDof, 1) + nodalB(2, colNd) * CBlock3x3(rowDof, 2);
          for (int rowNd = 0; rowNd < nNodes; ++rowNd) {
            auto Kblock = K.template Block<3, 3>(3 * rowNd, 3 * colNd);
            Kblock += (weights[gpIndex] * map.J) * nodalB(rowDof, rowNd) * T;
          }
        }
      }
    }
  }
}

} // namespace mochi

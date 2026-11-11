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

#include "mochi_physics_test_fixture.h"

#include <mochi_core/element_operations/fem_rod.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>
#include <mochi_physics/src/mochi_actor_convergence.h>

#include <gtest/gtest.h>

#include <array>

using namespace mochi;
using namespace mochi::experimental;

static_assert(
    static_cast<int>(ActorType::Count) == 6,
    "Please update the tests below if adding new actor types");

/**
 * @brief Tests that per-actor convergence weights produce O(1) weighted residual norms under
 * characteristic loading.
 */
class ActorConvergenceWeightsTest : public test::MochiSceneTestBase {
 protected:
  static constexpr real kDt = 0.01_r;
  static constexpr real kEps = 1e-4_r;
  static constexpr real kRomEps = 0.1_r;
  static constexpr real kRomHrEps = 0.2_r;

  static constexpr Real3 kGravityDirections[] = {
      Real3{0_r, -9.81_r, 0_r},
      Real3{0_r, 0_r, 9.81_r},
      Real3{3_r, -5_r, 7_r}};
  static constexpr real kDensities[] = {100_r, 1000_r};
  static constexpr real kScales[] = {0.3_r, 2_r};
  static constexpr int kTetDims[] = {3, 6};

  void ResetScene() {
    TearDown();
    SetUp();
  }

  void SetConvergenceTestParams(real absTol) {
    auto p = _scene->GetSolverParams();
    p.nonLinearSolver.convergenceMode = NonLinearSolverConvergenceMode::PerActorWeighted;
    p.nonLinearSolver.absTol = absTol;
    p.nonLinearSolver.relTol = 0_r;
    p.nonLinearSolver.relStepTol = 0_r;
    p.nonLinearSolver.maxIter = 50;
    _scene->SetSolverParams(p, test::ExpectOK{});
  }

  // Verifies that the initial weighted residual norm |r0|_W lies in the interval (lo, hi]:
  //   absTol = hi  →  |r0|_W <= hi  →  solver converges at iteration 0.
  //   absTol = lo  →  |r0|_W >  lo  →  solver requires at least one iteration.
  void TestWeightedNormBound(std::function<Actor*()> setup, real hi, real lo) {
    ResetScene();
    auto* actor = setup();
    SetConvergenceTestParams(hi);
    _scene->Step(kDt);
    EXPECT_EQ(0, _scene->GetSolverStats().maxNonLinearIters);
    EXPECT_EQ(ConvergenceStatus::Converged, actor->GetConvergenceStatus());

    ResetScene();
    actor = setup();
    SetConvergenceTestParams(lo);
    _scene->Step(kDt);
    EXPECT_GT(_scene->GetSolverStats().maxNonLinearIters, 0);
    EXPECT_EQ(ConvergenceStatus::Converged, actor->GetConvergenceStatus());
  }

  // --- Actor creation helpers ---
  ShapeHandle CreateTetShape(Real3 scale = {1_r, 1_r, 1_r}, int dim = 4) {
    auto [c, t] = test::CreateMinimalTetMeshUnitGrid(scale, /*dims*/ Int3{dim, dim, dim});
    return _mochiContext->CreateTetMeshShape(
        Flatten(MakeSpan(c)), Flatten(MakeSpan(t)), test::ExpectOK{});
  }

  ShapeHandle CreateTriShape(Real3 scale = {1_r, 1_r, 1_r}) {
    auto [c, t] = test::CreateMinimalTriMeshUnitCube(scale);
    return _mochiContext->CreateTriMeshShape(
        Flatten(MakeSpan(c)), Flatten(MakeSpan(t)), test::ExpectOK{});
  }

  Actor* CreateSoftActor_(Real3 scale = {1_r, 1_r, 1_r}, int dim = 4, real density = 1000_r) {
    SoftActorParams p;
    p.shape = CreateTetShape(scale, dim);
    p.material.density = density;
    return _scene->CreateSoftActor(p, test::ExpectOK{});
  }

  Actor* CreateShellActor_(Real3 scale = {1_r, 1_r, 1_r}, real density = 1_r) {
    ShellActorParams p;
    p.shape = CreateTriShape(scale);
    p.material.density = density;
    return CreateShellActor(_scene, p, test::ExpectOK{});
  }

  Actor* CreateRigidCube(Real3 scale = {1_r, 1_r, 1_r}, int dim = 4, real density = 1000_r) {
    RigidActorParams p;
    p.shape = CreateTetShape(scale, dim);
    p.density = density;
    p.colliderType = ColliderType::None;
    return _scene->CreateRigidActor(p, test::ExpectOK{});
  }

  Actor* CreateStraightRod(real linearDensity = 1_r, real spacing = 0.1_r, int numNodes = 10) {
    DynamicArray<Real3> nodes;
    for (int i = 0; i < numNodes; ++i) {
      nodes.push_back(Real3{static_cast<real>(i) * spacing, 0_r, 0_r});
    }

    DynamicArray<Real3> axes;
    for (int i = 0; i < numNodes - 1; ++i) {
      axes.push_back(Real3{0_r, 1_r, 0_r});
    }
    auto shape =
        CreatePolylineShape(_mochiContext, nodes, axes, /*isClosedLoop*/ false, test::ExpectOK{});

    RodActorParams p;
    p.shape = shape;
    p.material.linearDensity = linearDensity;
    return CreateRodActor(_scene, p, test::ExpectOK{});
  }

  // Closed-loop circular rod centered at the origin in the XZ plane.
  Actor* CreateCircularRod(real linearDensity = 1_r, int numNodes = 32, real radius = 1_r) {
    MOCHI_ASSERT(numNodes >= 3, "Closed-loop rod needs at least 3 nodes.");
    DynamicArray<Real3> nodes;
    for (int i = 0; i < numNodes; ++i) {
      real const theta = 2_r * kPI * static_cast<real>(i) / static_cast<real>(numNodes);
      nodes.push_back(Real3{radius * Cos(theta), 0_r, radius * Sin(theta)});
    }
    auto shape =
        CreatePolylineShape(_mochiContext, nodes, {}, /*isClosedLoop*/ true, test::ExpectOK{});

    RodActorParams p;
    p.shape = shape;
    p.material.linearDensity = linearDensity;
    return CreateRodActor(_scene, p, test::ExpectOK{});
  }

  Actor* CreateRomActor(
      Real3 scale = {1_r, 1_r, 1_r},
      int dim = 4,
      real density = 1000_r,
      std::optional<HyperReductionParams> hr = std::nullopt) {
    SoftActorParams p;
    p.shape = CreateTetShape(scale, dim);
    p.material.density = density;
    ExperimentalSoftActorParams ep;
    ep.rom = RomParams{.source = "polynomial_crom_order_1"};
    ep.rom->hyperReduction = hr;
    return CreateSoftActor(_scene, p, ep, test::ExpectOK{});
  }

  Actor* CreateArticulatedChain(
      Real3 linkScale = {0.2_r, 0.2_r, 0.2_r},
      int linkDim = 4,
      real density = 1000_r) {
    auto linkShape = CreateTetShape(linkScale, linkDim);

    // Three-link serial chain: a free root joint followed by two revolute joints, with each child
    // link offset one unit along +X from its parent.
    constexpr int kNumLinks = 3;
    ArticulatedActorParams ap;
    ap.joints.resize(kNumLinks);
    ap.links.resize(kNumLinks);
    for (int i = 0; i < kNumLinks; ++i) {
      auto& joint = ap.joints[i];
      if (i == 0) {
        joint.type = ArticulatedJointType::Free;
      } else {
        joint.type = ArticulatedJointType::Revolute;
        joint.axis = Real3{0_r, 0_r, 1_r};
        joint.parentLinkFromJoint = TransformRT{Real3{1_r, 0_r, 0_r}};
      }

      auto& link = ap.links[i];
      link.parentLink = i - 1;
      link.shape = linkShape;
      link.density = density;
      link.colliderType = ColliderType::None;
    }
    return _scene->CreateArticulatedActor(ap, test::ExpectOK{});
  }
};

// Rigid actors under uniform load: |r|_W = 1, invariant w.r.t. gravity direction, density, scale,
// and mesh resolution.
TEST_F(ActorConvergenceWeightsTest, Rigid) {
  for (auto const& g : kGravityDirections) {
    for (real const density : kDensities) {
      for (real const scale : kScales) {
        for (int const dim : kTetDims) {
          TestWeightedNormBound(
              [&]() -> Actor* {
                _scene->SetGravity(g);
                return CreateRigidCube(Real3{scale, scale, scale}, dim, density);
              },
              1_r + kEps,
              1_r - kEps);
        }
      }
    }
  }
}

// Rigid actors under torque: |r|_W = O(1), direction-invariant.
TEST_F(ActorConvergenceWeightsTest, RigidWithTorque) {
  // Torques are O(I·α) ≈ 100–300 N·m for the unit cube (I ≈ 167 kg·m² per axis), producing O(1
  // rad/s²) angular acceleration and O(1 m/s²) linear acceleration.
  for (auto const& torque :
       {Real3{300_r, 0_r, 0_r}, Real3{0_r, -200_r, 100_r}, Real3{100_r, -200_r, 300_r}}) {
    TestWeightedNormBound(
        [&]() -> Actor* {
          _scene->SetGravity({0_r, 0_r, 0_r});
          auto* actor = CreateRigidCube();
          DynamicArray<int> dofs = {3, 4, 5};
          DynamicArray<real> vals = {torque[0], torque[1], torque[2]};
          actor->SetExternalForcesOnDofs(dofs, vals, test::ExpectOK{});
          return actor;
        },
        1e1_r,
        1e-1_r);
  }
}

// Soft actors under uniform load: |r|_W = 1, invariant w.r.t. gravity direction, density, scale,
// and mesh resolution.
TEST_F(ActorConvergenceWeightsTest, Soft) {
  for (auto const& g : kGravityDirections) {
    for (real const density : kDensities) {
      for (real const scale : kScales) {
        for (int const dim : kTetDims) {
          TestWeightedNormBound(
              [&]() -> Actor* {
                _scene->SetGravity(g);
                return CreateSoftActor_(Real3{scale, scale, scale}, dim, density);
              },
              1_r + kEps,
              1_r - kEps);
        }
      }
    }
  }
}

// Shell actors under uniform load: |r|_W = 1, invariant w.r.t. gravity direction, density, and
// scale.
TEST_F(ActorConvergenceWeightsTest, Shell) {
  for (auto const& g : kGravityDirections) {
    for (real const density : kDensities) {
      for (real const scale : kScales) {
        TestWeightedNormBound(
            [&]() -> Actor* {
              _scene->SetGravity(g);
              return CreateShellActor_(Real3{scale, scale, scale}, density);
            },
            1_r + kEps,
            1_r - kEps);
      }
    }
  }
}

// Rod actors (no twist) under uniform load: |r|_W = 1, invariant w.r.t. gravity direction, linear
// density, element length, and number of nodes.
TEST_F(ActorConvergenceWeightsTest, RodNoTwist) {
  for (auto const& g : kGravityDirections) {
    for (real const linearDensity : {0.3_r, 3_r}) {
      for (real const spacing : {0.05_r, 0.3_r}) {
        for (int const numNodes : {5, 15}) {
          TestWeightedNormBound(
              [&]() -> Actor* {
                _scene->SetGravity(g);
                return CreateStraightRod(linearDensity, spacing, numNodes);
              },
              1_r + kEps,
              1_r - kEps);
        }
      }
    }
  }
}

// Rod with twist: |r|_W = O(1).
TEST_F(ActorConvergenceWeightsTest, RodWithTwist) {
  // Applied torque (1e-3 N·m) sized so the weighted twist residual is O(1), which the wide
  // (1e-1, 1e1] bound below checks. Rod uses default material parameters.
  constexpr int kNumRodNodes = 10;
  TestWeightedNormBound(
      [&]() -> Actor* {
        _scene->SetGravity({0_r, 0_r, 0_r});
        auto* actor = CreateStraightRod(/*linearDensity*/ 1_r, /*spacing*/ 0.1_r, kNumRodNodes);
        int const lastTwistDof = fem::kNumRodFields * (kNumRodNodes - 2) + fem::kRodThetaDofOffset;
        DynamicArray<int> dofs = {lastTwistDof};
        DynamicArray<real> vals = {1e-3_r};
        actor->SetExternalForcesOnDofs(dofs, vals, test::ExpectOK{});
        return actor;
      },
      1e1_r,
      1e-1_r);
}

// Closed-loop (periodic) rod under uniform load: |r|_W = 1, invariant w.r.t. gravity direction,
// linear density, radius, and number of nodes. Exercises the wraparound path in convergence
// weights and reference-curvature computation that is unique to closed-loop rods.
TEST_F(ActorConvergenceWeightsTest, ClosedLoopRod) {
  for (auto const& g : kGravityDirections) {
    for (real const linearDensity : {0.3_r, 3_r}) {
      for (real const radius : {1_r, 10_r, 100_r}) {
        for (int const numNodes : {8, 32}) {
          TestWeightedNormBound(
              [&]() -> Actor* {
                _scene->SetGravity(g);
                return CreateCircularRod(linearDensity, numNodes, radius);
              },
              1_r + kEps,
              1_r - kEps);
        }
      }
    }
  }
}

// ROM actors under uniform load: |r|_W = O(1), invariant w.r.t. gravity direction, density, scale,
// and mesh resolution.
TEST_IF_F(MOCHI_ENABLE_ROM_ACTORS, ActorConvergenceWeightsTest, RomNoHyperReduction) {
  for (auto const& g : kGravityDirections) {
    for (real const density : kDensities) {
      for (real const scale : kScales) {
        for (int const dim : kTetDims) {
          TestWeightedNormBound(
              [&]() -> Actor* {
                _scene->SetGravity(g);
                return CreateRomActor(Real3{scale, scale, scale}, dim, density);
              },
              1_r + kRomEps,
              1_r - kRomEps);
        }
      }
    }
  }
}

// ROM with hyper-reduction under uniform load: |r|_W = O(1), invariant w.r.t. gravity direction,
// subsampling, density, scale, and mesh resolution.
TEST_IF_F(MOCHI_ENABLE_ROM_ACTORS, ActorConvergenceWeightsTest, RomHyperReduction) {
  for (auto const& g : kGravityDirections) {
    for (int const stepSize : {3, 10}) {
      for (real const density : kDensities) {
        for (real const scale : kScales) {
          for (int const dim : kTetDims) {
            TestWeightedNormBound(
                [&]() -> Actor* {
                  _scene->SetGravity(g);
                  return CreateRomActor(
                      Real3{scale, scale, scale},
                      dim,
                      density,
                      HyperReductionParams{SampleMeshInitRandomSampling{stepSize, stepSize}, {}});
                },
                1_r + kRomHrEps,
                1_r - kRomHrEps);
          }
        }
      }
    }
  }
}

// A fixed root (link 0) carries a tree of links. Each link's joint lies L from its parent's along
// the link's direction, and its center of mass L/2 further. Links 1 to 4 are point masses with a
// small isotropic MOI: links 1 to 3 on revolute joints about different axes, one of them oblique,
// and link 4 on a free joint. Links 5 and 6 are massless frames on hard joints. Link 1 branches
// into links 2 and 3, so the subtree of link 2, {2, 4, 5, 6}, skips link 3. At rest, a joint's mass
// block is the mass of the links it moves for a translation, and their MOI about its axis for a
// rotation. The weights follow w = 1/(Q_type · Q_c), with Q_c = aRef·√m_subtree·√M_block and
// aRef = 1 without gravity.
TEST_F(ActorConvergenceWeightsTest, ArticulatedTreeOfPointMasses) {
  constexpr real kLength = 1_r;
  constexpr real kMoi = 0.01_r;
  constexpr int kFreeLink = 4; // The last moving link.

  // Per link; the root's entries are unused.
  constexpr std::array kParents{-1, 0, 1, 1, 2, 4, 5};
  constexpr std::array kMasses{0_r, 1_r, 2_r, 3_r, 4_r, 0_r, 0_r};
  constexpr std::array kDirections{
      kReal3Zeros, kReal3XAxis, kReal3XAxis, kReal3YAxis, kReal3ZAxis, kReal3XAxis, kReal3YAxis};
  constexpr std::array kAxes{
      kReal3Zeros,
      kReal3ZAxis,
      Real3{2_r / 3_r, 2_r / 3_r, 1_r / 3_r},
      kReal3XAxis,
      kReal3Zeros,
      kReal3Zeros,
      kReal3Zeros};
  constexpr int kNumLinks = isize(kParents);

  _scene->SetGravity(Real3{});

  // Build the tree, and the world positions at rest of each link's joint and center of mass.
  ArticulatedActorParams params;
  params.joints.push_back().type = ArticulatedJointType::Hard;
  params.links.push_back();
  std::array<Real3, kNumLinks> jointPositions{};
  std::array<Real3, kNumLinks> centersOfMass{};
  for (int i = 1; i < kNumLinks; ++i) {
    Real3 const offset = kLength * kDirections[i];
    auto& joint = params.joints.push_back();
    joint.parentLinkFromJoint = TransformRT{offset};
    auto& link = params.links.push_back();
    link.parentLink = kParents[i];
    if (kMasses[i] > 0_r) {
      joint.type = i == kFreeLink ? ArticulatedJointType::Free : ArticulatedJointType::Revolute;
      joint.axis = kAxes[i];
      link.mass = kMasses[i];
      link.centerOfMass = 0.5_r * offset;
      link.momentOfInertia = Real6{kMoi, 0_r, 0_r, kMoi, 0_r, kMoi};
    } else {
      joint.type = ArticulatedJointType::Hard;
    }

    jointPositions[i] = jointPositions[kParents[i]] + offset;
    centersOfMass[i] = jointPositions[i] + 0.5_r * offset;
  }

  Actor* actor = _scene->CreateArticulatedActor(params, test::ExpectOK{});
  _scene->Step(kDt);

  // Joint k moves link j if link k is link j or one of its ancestors.
  std::array<real, kFreeLink> Qc{};
  real Qrot = 0_r;
  for (int k = 1; k < kFreeLink; ++k) {
    real subtreeMass = 0_r;
    real massBlock = 0_r;
    for (int j = k; j < kNumLinks; ++j) {
      int ancestor = j;
      while (ancestor > k) {
        ancestor = kParents[ancestor];
      }
      if (ancestor == k && kMasses[j] > 0_r) {
        Real3 const r = centersOfMass[j] - jointPositions[k];
        subtreeMass += kMasses[j];
        massBlock += kMasses[j] * (Dot(r, r) - Sqr(Dot(kAxes[k], r))) + kMoi;
      }
    }
    Qc[k] = Sqrt(subtreeMass) * Sqrt(massBlock);
    Qrot += Qc[k];
  }

  // The free joint moves link 4 alone, with its center of mass L/2 along z: Q_c = m for the
  // translations, and √m·√(m·(L/2)² + kMoi) about x and y and √m·√kMoi about z for the rotations.
  real const m = kMasses[kFreeLink];
  real const lateralQc = Sqrt(m) * Sqrt(m * Sqr(0.5_r * kLength) + kMoi);
  std::array const freeQc{m, m, m, lateralQc, lateralQc, Sqrt(m) * Sqrt(kMoi)};
  Qrot += freeQc[3] + freeQc[4] + freeQc[5];

  auto const& weights = GetRegistry().get<CActorConvergenceWeights const>(GetEntity(actor)).values;
  ASSERT_EQ(kFreeLink - 1 + isize(freeQc), weights.Rows());
  for (int k = 1; k < kFreeLink; ++k) {
    EXPECT_NEAR_RTOL(1_r / (Qrot * Qc[k]), weights[k - 1], 1e-5_r);
  }
  for (int d = 0; d < isize(freeQc); ++d) {
    real const Qtype = d < 3 ? 3_r * m : Qrot;
    EXPECT_NEAR_RTOL(1_r / (Qtype * freeQc[d]), weights[kFreeLink - 1 + d], 1e-5_r);
  }
}

// Articulated actors under uniform load: |r|_W = O(1), invariant w.r.t. gravity direction, link
// density, link scale, and link mesh resolution.
TEST_F(ActorConvergenceWeightsTest, ArticulatedActor) {
  for (auto const& g : kGravityDirections) {
    for (real const density : kDensities) {
      for (real const linkScale : {0.05_r, 0.5_r}) {
        for (int const linkDim : kTetDims) {
          TestWeightedNormBound(
              [&]() -> Actor* {
                _scene->SetGravity(g);
                return CreateArticulatedChain(
                    Real3{linkScale, linkScale, linkScale}, linkDim, density);
              },
              1e1_r,
              1e-1_r);
        }
      }
    }
  }
}

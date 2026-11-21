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

/* The openxr hand bots must use the OpenXR XR_EXT_hand_tracking joint frames and reproduce the
 * oculus_xr hand bots they are derived from (assets/bots/hands/openxr/README.md). */

#include <mochi_core/geometry/model_data.h>
#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/math_utils.h>
#include <superdex_robotics/superdex_robotics.h>
#include <superdex_robotics/utils/bot_utils.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

using namespace mochi;
using namespace superdex::robotics;
using namespace mochi::test;

namespace {

constexpr real kRotTolerance = 1e-4_r; // [rad]
constexpr real kPosTolerance = 1e-5_r; // [m]

/* @brief openxr link -> oculus_xr link whose mesh and joint it reuses. */
constexpr std::array<std::pair<char const*, char const*>, 17> kReusedLinks{{
    {"xr_01_wrist", "bone_00_wrist_root"},
    {"xr_02_thumb_metacarpal", "bone_03_thumb_proximal"},
    {"xr_03_thumb_proximal", "bone_04_thumb_medial"},
    {"xr_04_thumb_distal", "bone_05_thumb_distal"},
    {"xr_07_index_proximal", "bone_06_index_proximal"},
    {"xr_08_index_intermediate", "bone_07_index_medial"},
    {"xr_09_index_distal", "bone_08_index_distal"},
    {"xr_12_middle_proximal", "bone_09_middle_proximal"},
    {"xr_13_middle_intermediate", "bone_10_middle_medial"},
    {"xr_14_middle_distal", "bone_11_middle_distal"},
    {"xr_17_ring_proximal", "bone_12_ring_proximal"},
    {"xr_18_ring_intermediate", "bone_13_ring_medial"},
    {"xr_19_ring_distal", "bone_14_ring_distal"},
    {"xr_21_little_metacarpal", "bone_15_pinky_metacarpal"},
    {"xr_22_little_proximal", "bone_16_pinky_proximal"},
    {"xr_23_little_intermediate", "bone_17_pinky_medial"},
    {"xr_24_little_distal", "bone_18_pinky_distal"},
}};

/* @brief Phalanx links and the joint at the distal end of their bone. */
constexpr std::array<std::pair<char const*, char const*>, 16> kBones{{
    {"xr_02_thumb_metacarpal", "xr_03_thumb_proximal"},
    {"xr_03_thumb_proximal", "xr_04_thumb_distal"},
    {"xr_07_index_proximal", "xr_08_index_intermediate"},
    {"xr_08_index_intermediate", "xr_09_index_distal"},
    {"xr_12_middle_proximal", "xr_13_middle_intermediate"},
    {"xr_13_middle_intermediate", "xr_14_middle_distal"},
    {"xr_17_ring_proximal", "xr_18_ring_intermediate"},
    {"xr_18_ring_intermediate", "xr_19_ring_distal"},
    {"xr_21_little_metacarpal", "xr_22_little_proximal"},
    {"xr_22_little_proximal", "xr_23_little_intermediate"},
    {"xr_23_little_intermediate", "xr_24_little_distal"},
    {"xr_09_index_distal", "xr_10_index_tip"},
    {"xr_14_middle_distal", "xr_15_middle_tip"},
    {"xr_19_ring_distal", "xr_20_ring_tip"},
    {"xr_04_thumb_distal", "xr_05_thumb_tip"},
    {"xr_24_little_distal", "xr_25_little_tip"},
}};

std::string BotPath(
    std::string const& family,
    std::string const& side,
    std::string const& variant = "highpoly") {
  return (std::filesystem::path(GetAssetsDir()) / "bots" / "hands" / family / side /
          (family + "_hand_" + variant + "_" + side + ".superdex_bot"))
      .string();
}

int FindLink(BotPrefab const& bot, std::string_view name) {
  return FindLinkIndexByName(bot, name);
}

/* @brief F_openxr = F_oculus * R for every reused link. */
Quaternion OculusToOpenXr(std::string const& side) {
  return Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, (side == "right" ? 0.5_r : -0.5_r) * kPI);
}

/* @brief Rotation vector of a joint rotation, from its DOF values. */
Real3 JointRotationVector(BotJointPrefab const& joint, Real3 const& dofs) {
  return joint.type == ArticulatedJointType::Spherical ? dofs : Normalize(joint.axis) * dofs[0];
}

/* @brief Offset of each link's joint in the bot-space pose (root joint excluded), or -1. */
DynamicArray<int> PoseOffsets(BotPrefab const& bot) {
  DynamicArray<int> offsets(bot.links.size(), -1);
  int offset = 0;
  for (int i = 1; i < isize(bot.joints); ++i) {
    ArticulatedJointType const type = bot.joints[i].type;
    if (type == ArticulatedJointType::Revolute || type == ArticulatedJointType::Prismatic) {
      offsets[i] = offset;
      offset += 1;
    } else if (type == ArticulatedJointType::Spherical) {
      offsets[i] = offset;
      offset += 3;
    }
  }
  return offsets;
}

DynamicArray<TransformRT> RootFromLinks(BotPrefab const& bot, DynamicArray<real> const& pose) {
  DynamicArray<TransformRT> transforms;
  ComputeLinkTransformsFromPose(
      bot, MakeConstSpan(pose), transforms, LinkTransformSpace::RootFromParent, ExpectOK{});
  return transforms;
}

void ExpectNearTransform(TransformRT const& a, TransformRT const& b, char const* what) {
  Quaternion const d = a.GetRotation().GetConjugate() * b.GetRotation();
  real const angle = 2_r * std::atan2(Norm(Real3{d.data[0], d.data[1], d.data[2]}), Abs(d.data[3]));
  EXPECT_LT(angle, kRotTolerance) << what;
  EXPECT_LT(Norm(a.GetTranslation() - b.GetTranslation()), kPosTolerance) << what;
}

/* @brief A model's grid SDF and the locations of its samples in the model's frame. */
class SampledSdf {
 public:
  explicit SampledSdf(std::string const& path) {
    ModelData model = model::LoadFromFile(path, ExpectOK{});
    MOCHI_ASSERT(model.sdf.has_value());
    _sdf = std::move(*model.sdf);
    _scale = _sdf.scale.value_or(Real3{1_r, 1_r, 1_r});
    MOCHI_ASSERT(_scale[0] == _scale[1] && _scale[1] == _scale[2]);
    _modelFromGrid = TransformRT(
        _sdf.rotation.value_or(Quaternion::Identity()), _sdf.translation.value_or(Real3{}));
  }

  [[nodiscard]] GridSdfData const& Data() const {
    return _sdf;
  }

  /* @brief Location of grid sample (i, j, k) in the model's frame. */
  [[nodiscard]] Real3 SamplePoint(int i, int j, int k) const {
    Real3 const lower = _sdf.bounds.GetMin();
    Real3 const size = _sdf.bounds.GetMax() - lower;
    std::array<int, 3> const ijk{i, j, k};
    Real3 p;
    for (int a = 0; a < 3; ++a) {
      p[a] = lower[a] + size[a] * static_cast<real>(ijk[a]) / static_cast<real>(_sdf.dims[a] - 1);
    }
    return _modelFromGrid.TransformPoint(p * _scale);
  }

  [[nodiscard]] real Value(int i, int j, int k) const {
    return _sdf.values[(static_cast<size_t>(i) * _sdf.dims[1] + j) * _sdf.dims[2] + k];
  }

  /* @brief Largest grid spacing in the model's frame. */
  [[nodiscard]] real Spacing() const {
    Real3 const size = _sdf.bounds.GetMax() - _sdf.bounds.GetMin();
    real spacing = 0_r;
    for (int a = 0; a < 3; ++a) {
      spacing = Max(spacing, size[a] / static_cast<real>(_sdf.dims[a] - 1));
    }
    return spacing * _scale[0];
  }

 private:
  GridSdfData _sdf;
  Real3 _scale;
  TransformRT _modelFromGrid;
};

class OpenXrHandBotTest : public ::testing::TestWithParam<std::tuple<std::string, std::string>> {
 protected:
  void SetUp() override {
    side = std::get<0>(GetParam());
    std::string const& variant = std::get<1>(GetParam());
    oculus = LoadBotPrefabFromFile(BotPath("oculus_xr", side, variant), ExpectOK{});
    openxr = LoadBotPrefabFromFile(BotPath("openxr", side, variant), ExpectOK{});
    r = OculusToOpenXr(side);
  }

  std::string side;
  BotPrefab oculus;
  BotPrefab openxr;
  Quaternion r;
};

TEST_P(OpenXrHandBotTest, HasTheOpenXrJointSet) {
  ASSERT_EQ(openxr.links.size(), 26u);
  EXPECT_EQ(std::string(openxr.links[0].name.c_str()), "xr_01_wrist");
  EXPECT_EQ(std::string(openxr.links[1].name.c_str()), "xr_00_palm");
  for (int i = 2; i < 26; ++i) {
    char prefix[8];
    std::snprintf(prefix, sizeof(prefix), "xr_%02d_", i);
    EXPECT_EQ(std::string(openxr.links[i].name.c_str()).rfind(prefix, 0), 0u) << i;
  }
  EXPECT_EQ(openxr._numDofs, 26);
  // Placed with the wrist frame at the world frame (README).
  ExpectNearTransform(openxr.worldFromRoot, TransformRT{}, "worldFromRoot");
}

TEST_P(OpenXrHandBotTest, PosedLinksAreOculusLinksTimesR) {
  DynamicArray<int> const oculusOffsets = PoseOffsets(oculus);
  DynamicArray<int> const openxrOffsets = PoseOffsets(openxr);

  std::mt19937 rng(7);
  std::uniform_real_distribution<real> angle(-0.6_r, 0.6_r);
  Quaternion const rInv = r.GetConjugate();
  for (int trial = 0; trial < 20; ++trial) {
    // The oculus_xr trapezium has no openxr link, so it stays at rest (zero).
    DynamicArray<real> oculusPose(static_cast<size_t>(oculus._numDofs), 0_r);
    DynamicArray<real> openxrPose(static_cast<size_t>(openxr._numDofs), 0_r);
    for (auto const& [xrName, ocName] : kReusedLinks) {
      int const oc = FindLink(oculus, ocName);
      int const xr = FindLink(openxr, xrName);
      ASSERT_GE(oc, 0);
      ASSERT_GE(xr, 0);
      if (oculusOffsets[oc] < 0) {
        continue;
      }
      BotJointPrefab const& ocJoint = oculus.joints[oc];
      // A rotation vector in the oculus joint frame is R^-1 * w in the openxr joint frame.
      Real3 w{};
      if (ocJoint.type == ArticulatedJointType::Spherical) {
        w = Real3{angle(rng), angle(rng), angle(rng)};
        for (int k = 0; k < 3; ++k) {
          oculusPose[oculusOffsets[oc] + k] = w[k];
        }
      } else {
        real const theta = angle(rng);
        oculusPose[oculusOffsets[oc]] = theta;
        w = Normalize(ocJoint.axis) * theta;
      }
      Real3 const wXr = rInv * w;
      if (openxr.joints[xr].type == ArticulatedJointType::Spherical) {
        for (int k = 0; k < 3; ++k) {
          openxrPose[openxrOffsets[xr] + k] = wXr[k];
        }
      } else {
        openxrPose[openxrOffsets[xr]] = Dot(Normalize(openxr.joints[xr].axis), wXr);
      }
    }
    DynamicArray<TransformRT> const oc = RootFromLinks(oculus, oculusPose);
    DynamicArray<TransformRT> const xr = RootFromLinks(openxr, openxrPose);
    TransformRT const rT(r);
    for (auto const& [xrName, ocName] : kReusedLinks) {
      // Root frames: openxr root = oculus root * R.
      ExpectNearTransform(
          oc[FindLink(oculus, ocName)] * rT, rT * xr[FindLink(openxr, xrName)], xrName);
    }
  }
}

TEST_P(OpenXrHandBotTest, LinkFramesFollowTheOpenXrConvention) {
  DynamicArray<real> const rest(static_cast<size_t>(openxr._numDofs), 0_r);
  DynamicArray<TransformRT> const g = RootFromLinks(openxr, rest);

  // +Z runs along the bone, away from the finger tip.
  for (auto const& [link, next] : kBones) {
    TransformRT const& a = g[FindLink(openxr, link)];
    Real3 const backward =
        Normalize(a.GetTranslation() - g[FindLink(openxr, next)].GetTranslation());
    EXPECT_GT(Dot(a.TransformDirection(Real3{0_r, 0_r, 1_r}), backward), 0.995_r) << link;
  }

  // +Y is dorsal: the back-of-hand normal of the plane through the wrist and the index and little
  // proximal joints. Looking down the fingers (-Z) with +Y up, a right thumb is on -X.
  Real3 const wrist = g[FindLink(openxr, "xr_01_wrist")].GetTranslation();
  Real3 const index = g[FindLink(openxr, "xr_07_index_proximal")].GetTranslation() - wrist;
  Real3 const little = g[FindLink(openxr, "xr_22_little_proximal")].GetTranslation() - wrist;
  Real3 const dorsal = Normalize(side == "right" ? Cross(little, index) : Cross(index, little));
  for (char const* link :
       {"xr_01_wrist",
        "xr_00_palm",
        "xr_07_index_proximal",
        "xr_12_middle_proximal",
        "xr_17_ring_proximal",
        "xr_22_little_proximal"}) {
    EXPECT_GT(
        Dot(g[FindLink(openxr, link)].TransformDirection(Real3{0_r, 1_r, 0_r}), dorsal), 0.9_r)
        << link;
  }
  Real3 const thumb = g[FindLink(openxr, "xr_02_thumb_metacarpal")].GetTranslation();
  EXPECT_EQ(thumb[0] < 0_r, side == "right");
}

/* Joint axes and limits of reused joints are the oculus_xr ones re-expressed by R; the thumb
 * metacarpal's box bounds the trapezium and CMC motions it replaces. */
TEST_P(OpenXrHandBotTest, JointLimitsAreOculusLimitsTimesR) {
  Quaternion const rInv = r.GetConjugate();
  for (auto const& [xrName, ocName] : kReusedLinks) {
    BotJointPrefab const& xr = openxr.joints[FindLink(openxr, xrName)];
    BotJointPrefab const& oc = oculus.joints[FindLink(oculus, ocName)];
    if (std::string_view(xrName) == "xr_02_thumb_metacarpal" ||
        oc.type == ArticulatedJointType::Free) {
      continue;
    }
    ASSERT_EQ(xr.type, oc.type) << xrName;
    ASSERT_EQ(xr.minLimit.has_value(), oc.minLimit.has_value()) << xrName;
    ASSERT_EQ(xr.maxLimit.has_value(), oc.maxLimit.has_value()) << xrName;
    if (oc.type == ArticulatedJointType::Revolute) {
      EXPECT_TRUE(NearEqual(xr.axis, rInv * oc.axis, 1e-6_r)) << xrName;
      if (oc.minLimit) {
        EXPECT_TRUE(NearEqual(*xr.minLimit, rInv * *oc.minLimit, 1e-6_r)) << xrName;
        EXPECT_TRUE(NearEqual(*xr.maxLimit, rInv * *oc.maxLimit, 1e-6_r)) << xrName;
      }
    } else if (oc.type == ArticulatedJointType::Spherical && oc.minLimit && oc.maxLimit) {
      // A per-axis box: R^-1 permutes the axes and may flip their signs.
      Real3 const a = rInv * *oc.minLimit;
      Real3 const b = rInv * *oc.maxLimit;
      for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR((*xr.minLimit)[i], Min(a[i], b[i]), 1e-6_r) << xrName << " axis " << i;
        EXPECT_NEAR((*xr.maxLimit)[i], Max(a[i], b[i]), 1e-6_r) << xrName << " axis " << i;
      }
    }
  }

  // Random trapezium (spherical) and CMC (revolute) poses, as rotations of the openxr thumb
  // metacarpal relative to its rest, must lie in its box (to within its sampling resolution).
  BotJointPrefab const& trapezium = oculus.joints[FindLink(oculus, "bone_02_thumb_metacarpal")];
  BotJointPrefab const& cmc = oculus.joints[FindLink(oculus, "bone_03_thumb_proximal")];
  BotJointPrefab const& thumb = openxr.joints[FindLink(openxr, "xr_02_thumb_metacarpal")];
  ASSERT_EQ(thumb.type, ArticulatedJointType::Spherical);
  ASSERT_TRUE(trapezium.minLimit && trapezium.maxLimit && cmc.minLimit && cmc.maxLimit);
  ASSERT_TRUE(thumb.minLimit && thumb.maxLimit);
  Quaternion const qTrapezium = trapezium.parentLinkFromJoint.GetRotation();
  Quaternion const qCmc = cmc.parentLinkFromJoint.GetRotation();
  Quaternion const restInv = (qTrapezium * qCmc * r).GetConjugate();
  real const cmcA = Dot(Normalize(cmc.axis), *cmc.minLimit);
  real const cmcB = Dot(Normalize(cmc.axis), *cmc.maxLimit);
  std::mt19937 rng(11);
  std::uniform_real_distribution<real> unit(0_r, 1_r);
  constexpr real kSamplingTolerance = 0.02_r; // [rad]
  for (int trial = 0; trial < 500; ++trial) {
    Real3 w;
    for (int i = 0; i < 3; ++i) {
      w[i] = (*trapezium.minLimit)[i] +
          unit(rng) * ((*trapezium.maxLimit)[i] - (*trapezium.minLimit)[i]);
    }
    real const theta = cmcA + unit(rng) * (cmcB - cmcA);
    Quaternion const q = qTrapezium * Quaternion::FromRotationVector(w) * qCmc *
        Quaternion::FromRotationVector(JointRotationVector(cmc, Real3{theta, 0_r, 0_r})) * r;
    Real3 const rv = (restInv * q).ToRotationVector();
    for (int i = 0; i < 3; ++i) {
      EXPECT_GE(rv[i], (*thumb.minLimit)[i] - kSamplingTolerance) << "trial " << trial;
      EXPECT_LE(rv[i], (*thumb.maxLimit)[i] + kSamplingTolerance) << "trial " << trial;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Hands,
    OpenXrHandBotTest,
    ::testing::Combine(
        ::testing::Values("left", "right"),
        ::testing::Values("highpoly", "lowpoly")));

/* @brief Generalized winding number of a closed triangle mesh about a point: 1 inside, 0 outside.
 */
real WindingNumber(MeshData const& mesh, Real3 const& point) {
  auto vertex = [&](int index) {
    return Real3{
               mesh.coordinates[3 * index],
               mesh.coordinates[3 * index + 1],
               mesh.coordinates[3 * index + 2]} -
        point;
  };
  double solidAngle = 0.0;
  for (size_t t = 0; t + 2 < mesh.connectivity.size(); t += 3) {
    Real3 const a = vertex(mesh.connectivity[t]);
    Real3 const b = vertex(mesh.connectivity[t + 1]);
    Real3 const c = vertex(mesh.connectivity[t + 2]);
    real const la = Norm(a);
    real const lb = Norm(b);
    real const lc = Norm(c);
    real const numerator = Dot(a, Cross(b, c));
    real const denominator = la * lb * lc + Dot(a, b) * lc + Dot(b, c) * la + Dot(c, a) * lb;
    solidAngle += 2.0 * std::atan2(numerator, denominator);
  }
  return static_cast<real>(solidAngle / (4.0 * kPI));
}

/* The wrist link merges the oculus_xr wrist root, forearm stub and trapezium. Its SDF must be the
 * SDF of their union -- the same inside/outside as the pieces' meshes -- and resolve the wrist as
 * finely as the finest source link. */
class OpenXrHandBotWristSdfTest
    : public ::testing::TestWithParam<std::tuple<std::string, std::string>> {};

TEST_P(OpenXrHandBotWristSdfTest, IsTheUnionOfTheMergedPieces) {
  auto const& [side, variant] = GetParam();
  BotPrefab const oculus = LoadBotPrefabFromFile(BotPath("oculus_xr", side, variant), ExpectOK{});
  BotPrefab const openxr = LoadBotPrefabFromFile(BotPath("openxr", side, variant), ExpectOK{});
  Quaternion const r = OculusToOpenXr(side);
  int const root = FindLink(oculus, "bone_00_wrist_root");
  ASSERT_GE(root, 0);
  TransformRT const wristFromOldRoot =
      Invert(oculus.links[root]._rootFromLink * TransformRT(r, Real3{}));

  // The merged pieces' meshes, in the wrist link frame.
  std::vector<MeshData> pieces;
  for (char const* name :
       {"bone_00_wrist_root", "bone_01_wrist_stub", "bone_02_thumb_metacarpal"}) {
    BotLinkPrefab const& link = oculus.links[FindLink(oculus, name)];
    ASSERT_FALSE(link.shapeFile.empty()) << name;
    ModelData model = model::LoadFromFile(std::string(link.shapeFile.c_str()), ExpectOK{});
    TransformRT const wristFromModel = wristFromOldRoot * link._rootFromLink *
        TransformRT(link.shapeRotation, link.shapeTranslation);
    MeshData& mesh = pieces.emplace_back(std::move(*model.mesh));
    for (size_t v = 0; v + 2 < mesh.coordinates.size(); v += 3) {
      Real3 const p = wristFromModel.TransformPoint(
          Real3{mesh.coordinates[v], mesh.coordinates[v + 1], mesh.coordinates[v + 2]});
      for (int a = 0; a < 3; ++a) {
        mesh.coordinates[v + a] = p[a];
      }
    }
  }

  BotLinkPrefab const& wristLink = openxr.links[FindLink(openxr, "xr_01_wrist")];
  ASSERT_EQ(wristLink.shapeRotation, Quaternion{});
  ASSERT_EQ(wristLink.shapeTranslation, Real3{});
  SampledSdf const wrist(std::string(wristLink.shapeFile.c_str()));

  real finestSourceSpacing = std::numeric_limits<real>::infinity();
  for (BotLinkPrefab const& link : oculus.links) {
    if (!link.shapeFile.empty()) {
      finestSourceSpacing =
          Min(finestSourceSpacing, SampledSdf(std::string(link.shapeFile.c_str())).Spacing());
    }
  }
  EXPECT_LE(wrist.Spacing(), finestSourceSpacing * 1.01_r);

  // Compare away from the surface, where interpolation cannot flip the sign. Subsample the grid to
  // about 10k samples.
  real const margin = 1.5_r * wrist.Spacing();
  Int3 const dims = wrist.Data().dims;
  int const stride =
      Max(1,
          static_cast<int>(
              std::ceil(std::cbrt(static_cast<double>(dims[0]) * dims[1] * dims[2] / 10000.0))));
  int compared = 0;
  int wrongSide = 0;
  for (int i = 0; i < dims[0]; i += stride) {
    for (int j = 0; j < dims[1]; j += stride) {
      for (int k = 0; k < dims[2]; k += stride) {
        real const value = wrist.Value(i, j, k);
        if (Abs(value) < margin) {
          continue;
        }
        Real3 const p = wrist.SamplePoint(i, j, k);
        bool inside = false;
        for (MeshData const& piece : pieces) {
          inside = inside || WindingNumber(piece, p) > 0.5_r;
        }
        ++compared;
        wrongSide += (value < 0_r) != inside ? 1 : 0;
      }
    }
  }
  ASSERT_GT(compared, 500);
  EXPECT_EQ(wrongSide, 0) << "of " << compared << " samples";
}

INSTANTIATE_TEST_SUITE_P(
    Hands,
    OpenXrHandBotWristSdfTest,
    ::testing::Combine(
        ::testing::Values("left", "right"),
        ::testing::Values("highpoly", "lowpoly")));

/* The link frames and joint types of the left hand mirror the right's. (The joint dynamics are
 * those of the oculus_xr hands, whose friction differs between left and right.) */
class OpenXrHandBotMirrorTest : public ::testing::TestWithParam<std::string> {};

TEST_P(OpenXrHandBotMirrorTest, LeftIsTheMirrorOfRight) {
  BotPrefab const left = LoadBotPrefabFromFile(BotPath("openxr", "left", GetParam()), ExpectOK{});
  BotPrefab const right = LoadBotPrefabFromFile(BotPath("openxr", "right", GetParam()), ExpectOK{});
  ASSERT_EQ(left.links.size(), right.links.size());
  for (size_t i = 0; i < left.joints.size(); ++i) {
    EXPECT_EQ(left.joints[i].type, right.joints[i].type) << left.links[i].name.c_str();
  }
  for (size_t i = 1; i < left.links.size(); ++i) {
    // Mirroring across X = 0 negates x positions and the y, z quaternion components.
    TransformRT const& l = left.links[i]._parentFromLink;
    TransformRT const& r = right.links[i]._parentFromLink;
    Quaternion const& q = r.GetRotation();
    Real3 const t = r.GetTranslation();
    TransformRT const mirrored(
        Quaternion(q.data[0], -q.data[1], -q.data[2], q.data[3]), Real3{-t[0], t[1], t[2]});
    ExpectNearTransform(l, mirrored, left.links[i].name.c_str());
  }
}

INSTANTIATE_TEST_SUITE_P(Hands, OpenXrHandBotMirrorTest, ::testing::Values("highpoly", "lowpoly"));

/* Each left hand is generated as the exact mirror of the right one
 * (scale_openxr_hand_geometry.py --mirror-left), joint parameters included: axes and limit rotation
 * vectors negate y and z, spherical box limits reordered per component. */
class OpenXrHandBotExactMirrorTest : public ::testing::TestWithParam<std::string> {};

TEST_P(OpenXrHandBotExactMirrorTest, LeftJointsAndLinksMirrorRight) {
  BotPrefab const left = LoadBotPrefabFromFile(BotPath("openxr", "left", GetParam()), ExpectOK{});
  BotPrefab const right = LoadBotPrefabFromFile(BotPath("openxr", "right", GetParam()), ExpectOK{});
  ASSERT_EQ(left.joints.size(), right.joints.size());
  auto mirror = [](Real3 const& v) { return Real3{v[0], -v[1], -v[2]}; };
  auto expectNear = [](Real3 const& a, Real3 const& b, char const* what, char const* joint) {
    for (int k = 0; k < 3; ++k) {
      EXPECT_NEAR(a[k], b[k], 1e-6_r) << what << " " << joint << " [" << k << "]";
    }
  };
  for (size_t i = 0; i < left.joints.size(); ++i) {
    auto const& l = left.joints[i];
    auto const& r = right.joints[i];
    char const* name = l.name.c_str();
    EXPECT_EQ(l.type, r.type) << name;
    expectNear(l.axis, mirror(r.axis), "axis", name);
    ASSERT_EQ(l.minLimit.has_value(), r.minLimit.has_value()) << name;
    ASSERT_EQ(l.maxLimit.has_value(), r.maxLimit.has_value()) << name;
    if (r.minLimit && r.maxLimit) {
      Real3 lo = mirror(*r.minLimit);
      Real3 hi = mirror(*r.maxLimit);
      if (r.type == ArticulatedJointType::Spherical) {
        for (int k = 0; k < 3; ++k) {
          real const a = lo[k];
          lo[k] = Min(a, hi[k]);
          hi[k] = Max(a, hi[k]);
        }
      }
      expectNear(*l.minLimit, lo, "minLimit", name);
      expectNear(*l.maxLimit, hi, "maxLimit", name);
    }
    EXPECT_EQ(l.limitStiffness, r.limitStiffness) << name;
    EXPECT_EQ(l.limitDamping, r.limitDamping) << name;
  }
  for (size_t i = 0; i < left.links.size(); ++i) {
    EXPECT_EQ(left.links[i].density, right.links[i].density) << left.links[i].name.c_str();
  }
}

INSTANTIATE_TEST_SUITE_P(
    Hands,
    OpenXrHandBotExactMirrorTest,
    ::testing::Values("highpoly", "lowpoly"));

} // namespace

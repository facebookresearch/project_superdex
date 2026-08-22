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

#include <mochi_core/contact/dmap.h>
#include <mochi_core/elements/triangular/finite_element.h>
#include <mochi_core/linear_algebra/sparse_matrix.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/rand_utils.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <numeric>
#include <unordered_set>
#include <vector>

using namespace mochi;
using namespace dmap;

/************************************************************************
 * Declaration of sizes and common defines
 ************************************************************************/

namespace {
using VolumeElement = tetrahedral::Pk3DElement<1, 1>;
using BoundaryElement = tetrahedral::Pk3DElementTrace<VolumeElement, 3>;
using DQuad = DMapQuad<BoundaryElement>;
using TriElement = triangular::Pk2DElement<1, 3>;
using DQuadTri = DMapQuad<TriElement>;
constexpr real kEps = 1e-4_r;
constexpr real kTol = 8e-3_r;
// The size of the input varies depending on the total map. The test uses a cube mesh with 8 nodes,
// 12 triangles and 36 collision samples. If the map includes quadrature sampling, then the size of
// the output is 36, otherwise it is 8. The test selects 6 points from the output.
constexpr size_t kOutput = 6;
constexpr size_t kRomSize = 4;
constexpr size_t kArticulatedSize = 7; // Free joint + revolute joint
constexpr size_t kSoftSize = 24;
constexpr size_t kBones = 2;
} // namespace

/************************************************************************
 * Common transformations, used both at initialization and during finite-difference testing
 ************************************************************************/

static std::array<TransformRT, kBones> ArticulatedToSkeleton(
    ColumnVectorView<real const, kArticulatedSize> articulatedState) {
  std::array<TransformRT, kBones> skeletonState{};
  // Free joint
  skeletonState[0] =
      TransformFromRawDofs(articulatedState.TopRows<RigidSize::kDAll>(RigidSize::kDAll));
  // Revolute joint
  skeletonState[1] =
      TransformRT{Quaternion::RotationY(articulatedState(kArticulatedSize - 1))} * skeletonState[0];
  return skeletonState;
}

static ColumnVector<real> SkeletonToVector(Span<TransformRT const> skeletonState) {
  ColumnVector<real> result(isize(skeletonState) * RigidSize::kDAll);
  for (int i = 0, row = 0; i < skeletonState.size(); i++, row += RigidSize::kDAll) {
    TransformToRawDofs(skeletonState[i], result.Slice<RigidSize::kDAll>(row, RigidSize::kDAll));
  }
  return result;
}

static void ApplyRigid(TransformRT const& transform, Span<Real3 const> input, Span<Real3> out) {
  for (int i = 0; i < out.size(); i++) {
    out[i] = transform.TransformPoint(input[i]);
  }
}

static void ApplyRom(
    ColumnVectorView<real const> state,
    MatrixView<real const> basis,
    Span<Real3 const> input,
    Span<Real3> out) {
  auto outView = AsView(Flatten(out));
  outView = basis * state + AsConstView(Flatten(input));
}

template <typename ElementT>
static void
ApplyQuadrature(Span<ElementT const> femElements, Span<Real3 const> input, Span<Real3> out) {
  for (int i = 0; i < out.size(); i++) {
    out[i] = {};
    auto weights = GetWeightInfo(femElements, i);
    auto nodes = GetNodeInfo(femElements, i);
    for (int k = 0; k < ElementT::kNumNodes; k++) {
      out[i] += weights[k] * input[nodes[k]];
    }
  }
}

// Overload for containers to enable template deduction
template <typename Container>
static void
ApplyQuadrature(Container const& femElements, Span<Real3 const> input, Span<Real3> out) {
  ApplyQuadrature(MakeConstSpan(femElements), input, out);
}

static void ApplySkinning(
    ColumnVectorView<real const, kArticulatedSize> articulatedState,
    DSkinningTransform const& skinningTransform,
    Span<Real3 const> input,
    Span<Real3> out) {
  auto skeletonState = ArticulatedToSkeleton(articulatedState);
  skinningTransform.Transform(
      MakeConstSpan(skeletonState), AsConstView(Flatten(input)), AsView(Flatten(out)));
}

static void
ApplySoft(ColumnVectorView<real const> softState, Span<Real3 const> input, Span<Real3> out) {
  AsView(Flatten(out)) = AsConstView(Flatten(input)) + softState;
}

static void ApplyBarycentricWeighting(
    Span<Real3 const> points,
    Span<Int4 const> inds,
    Span<Real4 const> weights,
    Span<Real3> out,
    bool negate) {
  for (int i = 0; i < isize(out); ++i) {
    Real3 result = {};
    for (int j = 0; j < 4; ++j) {
      result += points[inds[i][j]] * weights[i][j];
    }
    out[i] = negate ? -result : result;
  }
}

static void ApplyBlending(
    BlendingDataSourceMesh const& blending,
    Span<Real3 const> rest,
    Span<Real3 const> input,
    Span<Real3> out) {
  for (int i = 0; i < out.size(); i++) {
    out[i] = {};
    real weight = 0_r;
    int ind = blending.mappingTargetToSource[i];
    if (ind != -1) {
      weight = blending.weightsSource[ind];
      out[i] += weight * input[ind];
    }
    out[i] += (1_r - weight) * rest[i];
  }
}

static void
SelectOutput(Span<Real3 const> in, Span<int const> inds, Span<Real3>& out, bool negate = false) {
  for (int i = 0; i < out.size(); i++) {
    out[i] = negate ? -in[inds[i]] : in[inds[i]];
  }
}

// Transform points to the collider's local frame
static void TransformToCollider(Span<VMatrix3x3r const> toCollider, Span<Real3> out) {
  if (toCollider.empty()) {
    return; // Nothing to do
  }
  for (int i = 0, j = 0, dj = (toCollider.size() == 1 ? 0 : 1); i < out.size(); i++, j += dj) {
    out[i] = ToReal3(DotMatVec3x3(toCollider[j], ToSimd(out[i])));
  }
}

/************************************************************************
 * Implementations of state increments for finite-difference testing
 ************************************************************************/

static void AddEpsRigid(int i, real eps, TransformRT& out) {
  if (i < RigidSize::kDTrans) {
    auto trans = out.GetTranslation();
    trans[i] += eps;
    out.SetTranslation(trans);
  } else {
    Real3 inc{};
    inc[i - RigidSize::kDTrans] = eps;
    out.SetRotation(Quaternion::FromRotationVector(inc) * out.GetRotation());
  }
}

static void AddEpsEuclidean(int i, real eps, ColumnVectorView<real> out) {
  out(i) += eps;
}

static void AddEpsArticulated(int i, real eps, ColumnVectorView<real> out) {
  if (i < 6) {
    auto rootState = TransformFromRawDofs(out.TopRows<RigidSize::kDAll>(RigidSize::kDAll));
    AddEpsRigid(i, eps, rootState);
    TransformToRawDofs(rootState, out.TopRows<RigidSize::kDAll>(RigidSize::kDAll));
  } else {
    // Revolute joint
    MOCHI_ASSERT(i == 6);
    out(i) += eps;
  }
}

/************************************************************************
 * Functions to create test data
 ************************************************************************/

static TetrahedralMesh CreateTetMesh() {
  auto&& [coords, connectivity] = test::CreateMinimalTetMeshUnitCube();
  TetrahedralMesh mesh = {coords, connectivity};
  return mesh;
}

static std::vector<VolumeElement> CreateVolumeElements(TetrahedralMesh const& mesh) {
  std::vector<VolumeElement> femElementsVolume;
  femElementsVolume.reserve(mesh.GetNumElements());
  for (int i = 0; i < mesh.GetNumElements(); ++i) {
    femElementsVolume.emplace_back(
        i,
        mesh.GetNodeCoordinates(),
        mesh.GetElementConnectivity(),
        tetrahedral::kTetrahedralQuadrature1);
  }
  return femElementsVolume;
}

static DynamicArray<TriElement> CreateTriElements(
    Span<Real3 const> coordinates,
    Span<Int3 const> connectivity) {
  DynamicArray<TriElement> elements;
  elements.reserve(connectivity.size());
  for (int i = 0; i < isize(connectivity); ++i) {
    elements.emplace_back(i, coordinates, connectivity);
  }
  return elements;
}

static std::vector<BoundaryElement> CreateBoundaryElements(
    TetrahedralMesh const& mesh,
    Span<VolumeElement const> femElementsVolume) {
  std::vector<BoundaryElement> femElementsBoundary;
  femElementsBoundary.reserve(mesh.GetNumBoundaryFaces());
  for (auto const& bdface : mesh.GetBoundaryFaces()) {
    femElementsBoundary.emplace_back(
        femElementsVolume[bdface.element],
        bdface.faceNum,
        tetrahedral::kTetrahedralTraceQuadrature3[bdface.faceNum]);
  }
  return femElementsBoundary;
}

static TransformRT CreateRigidState() {
  return TransformRT{
      Quaternion::FromRotationVector(Real3{0.5_r, -1_r, -0.5_r}), Real3{1_r, -3_r, 4_r}};
}

static real Lerp(int num, int den, real min, real max) {
  real weight = static_cast<real>(num) / static_cast<real>(den);
  return weight * max + (1_r - weight) * min;
}

static Matrix<real> CreateRomBasis(int numPoints) {
  Matrix<real> result(3 * numPoints, kRomSize);
  for (int i = 0; i < result.Rows(); i++) {
    for (int j = 0; j < result.Cols(); j++) {
      int den = result.Rows() * result.Cols() / 3;
      int num = (i * result.Cols() + j) % den;
      result(i, j) = Lerp(num, den, -5_r, 5_r);
    }
  }
  return result;
}

static ColumnVector<real> CreateVectorState(int size) {
  ColumnVector<real> result(size);
  for (int i = 0; i < result.Rows(); i++) {
    result(i) = Lerp(i, result.Rows(), -1_r, 1_r);
  }
  return result;
}

static std::vector<int> CreateArticulatedDofs(int size) {
  std::vector<int> result(size);
  std::iota(result.begin(), result.end(), 0);
  return result;
}

static SkinningData CreateSkinning(int size) {
  SkinningData result;
  result.weightsPerNode = kBones;
  result.indices.reserve(size * kBones);
  result.weights.reserve(size * kBones);
  for (int i = 0; i < size; i++) {
    result.indices.emplace_back(0);
    result.indices.emplace_back(1);
    real weight = (1_r / size) * i;
    result.weights.emplace_back(weight);
    result.weights.emplace_back(1_r - weight);
  }
  return result;
}

static DSkinningTransform CreateSkinningTransform(SkinningData const& skinningData) {
  SkinningWeightsByBone weights{
      skinningData.indices, skinningData.weights, skinningData.weightsPerNode, kBones};
  return DSkinningTransform{weights};
}

static std::array<VMatrix3x3r, kBones> CreateSkeletonRotations(
    ColumnVectorView<real const, kArticulatedSize> articulatedState,
    DSkinningTransform const& skinningTransform) {
  std::array<VMatrix3x3r, kBones> result{};
  auto skeletonPose = ArticulatedToSkeleton(articulatedState);
  for (int i = 0; i < result.size(); i++) {
    auto preTransform =
        Rodrigues(skinningTransform.GetBonePreTransform(i).GetRotation().VToRotationVector());
    result[i] = Dot3x3(VGetRotationMatrix(skeletonPose[i]), preTransform);
  }
  return result;
}

static BlendingDataSourceMesh CreateBlending() {
  BlendingDataSourceMesh result;
  result.mappingTargetToSource = {0, 1, -1, -1, -1, -1, 6, 7};
  result.weightsSource = {0.8_r, 0.2_r, 0_r, 0_r, 0_r, 0_r, 0.4_r, 0.3_r};
  return result;
}

static ContactDetectionResult CreateDetectionResultAfterRigid(
    Span<Real3 const> points,
    Span<int const> inds) {
  ContactDetectionResult result;
  result.posColliding.resize_noinit(inds.size());
  for (int i = 0; i < result.posColliding.size(); i++) {
    result.posColliding[i] = points[inds[i]];
  }
  return result;
}

static ContactDetectionResult CreateDetectionResultAfterRomMap(
    MatrixView<real const> basis,
    Span<int const> inds,
    Span<VMatrix3x3r const> toCollider) {
  ContactDetectionResult result;
  result.posColliding.resize_noinit(inds.size());
  result.ndofs = basis.Cols();
  result.jacColliderFromWorld.resize_noinit(inds.size());
  result.jacWorldFromDofs.resize_noinit(inds.size());
  for (int i = 0; i < inds.size(); i++) {
    result.jacColliderFromWorld[i] = toCollider[i];
    for (int j = 0; j < basis.Cols(); j++) {
      AsColumnVectorView<3>(result.jacWorldFromDofs[i].jac[j]) =
          basis.Col(j).MiddleRows<3>(3 * inds[i], 3);
    }
    std::iota(result.jacWorldFromDofs[i].inds.begin(), result.jacWorldFromDofs[i].inds.end(), 0);
  }
  return result;
}

static ContactDetectionResult CreateDetectionResultAfterSoftMap(
    Span<Int4 const> softColliderInds,
    Span<Real4 const> softColliderWeights,
    Span<int const> inds,
    Span<VMatrix3x3r const> toCollider) {
  ContactDetectionResult result;
  result.posColliding.resize_noinit(inds.size());
  result.ndofs = 12;
  result.jacColliderFromWorld.resize_noinit(inds.size());
  result.jacWorldFromDofs.resize_noinit(inds.size());
  for (int i = 0; i < inds.size(); ++i) {
    result.jacColliderFromWorld[i] = toCollider[i];
    Int4 const& nodeInds = softColliderInds[i];
    Real4 const& nodeWeights = softColliderWeights[i];
    for (int j = 0; j < 4; ++j) {
      for (int k = 0; k < 3; ++k) {
        Real3 weight = {};
        weight[k] = nodeWeights[j];
        result.jacWorldFromDofs[i].jac[3 * j + k] = ToSimd(weight);
        result.jacWorldFromDofs[i].inds[3 * j + k] = 3 * nodeInds[j] + k;
      }
    }
  }
  return result;
}

static std::vector<Real3> CreatePointsAfterRom(
    ColumnVectorView<real const> state,
    MatrixView<real const> basis,
    Span<Real3 const> points) {
  std::vector<Real3> result(points.size());
  ApplyRom(state, basis, points, result);
  return result;
}

static std::vector<Real3> CreatePointsAfterSoft(
    ColumnVectorView<real const> state,
    Span<Real3 const> points) {
  std::vector<Real3> result(points.size());
  ApplySoft(state, points, result);
  return result;
}

static std::vector<Real3> CreatePointsAfterBlending(
    BlendingDataSourceMesh blending,
    Span<Real3 const> rest,
    Span<Real3 const> input) {
  std::vector<Real3> result(rest.size());
  ApplyBlending(blending, rest, input, result);
  return result;
}

static RowMatrix<real> CreateSkinningJacobian(
    DSkinningTransform const& skinningTransform,
    ColumnVectorView<real const, kArticulatedSize> articulatedStateIn,
    Span<Real3 const> points) {
  Matrix<real> articulatedJacobian(kBones * RigidSize::kDAll, articulatedStateIn.Rows());
  for (int i = 0; i < articulatedStateIn.Rows(); i++) {
    ColumnVector<real, kArticulatedSize> articulatedState = articulatedStateIn;
    AddEpsArticulated(i, kEps, articulatedState);
    auto skeletonStatep = ArticulatedToSkeleton(articulatedState);
    articulatedState = articulatedStateIn;
    AddEpsArticulated(i, -kEps, articulatedState);
    auto skeletonStatem = ArticulatedToSkeleton(articulatedState);
    std::array<TransformRT, kBones> skeletonStateDiff;
    for (int j = 0; j < kBones; ++j) {
      skeletonStateDiff[j].SetTranslation(
          skeletonStatep[j].GetTranslation() - skeletonStatem[j].GetTranslation());
      skeletonStateDiff[j].SetRotation(
          skeletonStatep[j].GetRotation() * skeletonStatem[j].GetRotation().GetConjugate());
    }
    articulatedJacobian.Col(i) = (1_r / (2_r * kEps)) * SkeletonToVector(skeletonStateDiff);
  }

  SparseMatrix<real> jacobianBones = skinningTransform.CreateDBones();
  auto skeletonState = ArticulatedToSkeleton(articulatedStateIn);
  skinningTransform.DTransformDBones(
      MakeConstSpan(skeletonState), AsConstView(Flatten(points)), jacobianBones);
  auto result = jacobianBones * articulatedJacobian;
  return result;
}

// Create a single to-collider matrix as the transpose of a rigid transform
static VMatrix3x3r CreateToRigidCollider(TransformRT const& transform) {
  return ToVMatrix3x3Transpose(transform.GetRotation());
}

// Create a different to-collider matrix for each contact
static std::vector<VMatrix3x3r> CreateToSoftCollider(int numContacts) {
  std::vector<VMatrix3x3r> result(numContacts);
  for (int i = 0; i < numContacts; i++) {
    // Create different matrices per contact by varying the coefficients
    real scale = 0.8_r + 0.4_r * (static_cast<real>(i) / numContacts);
    result[i] = VMatrix3x3r{
        Vec4r{scale * 1.2_r, scale * 0.3_r - 0.1_r * i, -0.1_r + 0.05_r * i},
        Vec4r{-0.2_r + 0.1_r * i, scale * 1.1_r, 0.4_r - 0.1_r * i},
        Vec4r{0.1_r + 0.05_r * i, -0.3_r + 0.1_r * i, scale * 0.9_r}};
  }
  return result;
}

/************************************************************************
 * Test Fixture Class
 ************************************************************************/
class DMapTest : public testing::Test {
 public:
  std::array<int, kOutput> const kInds{0, 1, 2, 3, 5, 7};

  TetrahedralMesh const kMesh{CreateTetMesh()};
  Span<Real3 const> const kPoints = kMesh.GetNodeCoordinates();

  TransformRT const kRigidState{CreateRigidState()};
  TransformRT const kRigidTransform = kRigidState;

  std::vector<VolumeElement> const kFemElementsVol{CreateVolumeElements(kMesh)};
  std::vector<BoundaryElement> const kFemElements{CreateBoundaryElements(kMesh, kFemElementsVol)};
  test::TriMeshParams const kTriMeshData{test::CreateMinimalTriMeshUnitCube()};
  DynamicArray<TriElement> const kTriElements{
      CreateTriElements(kTriMeshData.first, kTriMeshData.second)};

  Matrix<real> const kBasis{CreateRomBasis(isize(kPoints))};
  DMapRom::VariantJacobian const kRomJacobian{kBasis};
  ColumnVector<real> const kRomState{CreateVectorState(kRomSize)};
  std::vector<Real3> const kPointsAfterRom{CreatePointsAfterRom(kRomState, kBasis, kPoints)};

  ColumnVector<real, kArticulatedSize> const kArticulatedState{CreateVectorState(kArticulatedSize)};
  std::vector<int> const kArticulatedDofs{CreateArticulatedDofs(kArticulatedSize)};
  SkinningData const kSkinning{CreateSkinning(isize(kPoints))};
  DSkinningTransform const kSkinningTransform{CreateSkinningTransform(kSkinning)};
  std::array<VMatrix3x3r, kBones> const kSkeletonRotations{
      CreateSkeletonRotations(kArticulatedState, kSkinningTransform)};

  ColumnVector<real> const kSoftState{CreateVectorState(kSoftSize)};
  std::vector<Real3> const kPointsAfterSoft{CreatePointsAfterSoft(kSoftState, kPoints)};
  std::array<Int4, kOutput> const kSoftColliderInds{{
      {0, 1, 2, 4},
      {1, 3, 2, 5},
      {2, 3, 6, 7},
      {4, 5, 6, 0},
      {5, 7, 6, 1},
      {0, 2, 4, 6},
  }};
  std::array<Real4, kOutput> const kSoftColliderWeights{{
      {0.25_r, 0.25_r, 0.25_r, 0.25_r},
      {0.1_r, 0.2_r, 0.3_r, 0.4_r},
      {0.4_r, 0.3_r, 0.2_r, 0.1_r},
      {0.5_r, 0.2_r, 0.2_r, 0.1_r},
      {0.1_r, 0.1_r, 0.4_r, 0.4_r},
      {0.15_r, 0.35_r, 0.35_r, 0.15_r},
  }};

  BlendingDataSourceMesh kBlending{CreateBlending()};

  // To-collider transforms for rigid and soft colliders
  VMatrix3x3r const kToRigidCollider = CreateToRigidCollider(kRigidState);
  std::vector<VMatrix3x3r> const kToSoftCollider{CreateToSoftCollider(kOutput)};
  std::array<Span<VMatrix3x3r const>, 2> const kToCollider{
      MakeSingletonConstSpan(kToRigidCollider),
      MakeConstSpan(kToSoftCollider)};

  // Main function for finite-difference consistency testing
  using AddEpsFunc = std::function<void(int, real, Span<Real3 const>, Span<Real3>)>;
  template <int kNumSlices, typename D>
  void TestConsistency(
      D const& dmap,
      std::array<int const, kNumSlices> const& dstateSizes,
      Span<int const> inds,
      std::array<AddEpsFunc const, kNumSlices> const& addEpsAndMap,
      Span<VMatrix3x3r const> toCollider,
      real tol = kTol) {
    // Evaluate the Jacobian
    std::array<ContactJac, kNumSlices> jacs;
    dmap.GetJac(inds, jacs);

    // Test per state slice
    for (int slice = 0; slice < kNumSlices; slice++) {
      // Evaluate the Jacobian through finite differences
      std::vector<Real3> resultp(kOutput);
      std::vector<Real3> resultm(kOutput);
      std::vector<Matrix<real, 3>> jacFD(kOutput, Matrix<real, 3>(3, dstateSizes[slice]));
      for (int i = 0; i < dstateSizes[slice]; i++) {
        addEpsAndMap[slice](i, kEps, kPoints, resultp);
        TransformToCollider(toCollider, resultp);
        addEpsAndMap[slice](i, -kEps, kPoints, resultm);
        TransformToCollider(toCollider, resultm);
        for (int j = 0; j < kOutput; j++) {
          auto valp = AsView(resultp[j]);
          auto valm = AsView(resultm[j]);
          Matrix<real> valdiff = (1_r / (2_r * kEps)) * (valp - valm);
          jacFD[j].Col(i) = valdiff;
        }
      }

      // Test per output point
      for (int i = 0; i < kInds.size(); i++) {
        // Get the analytical Jacobian and compress columns based on indices
        auto jacA = jacs[slice].Jac(i);
        auto indsA = jacs[slice].Inds(i);
        std::unordered_set<int> testInds(indsA.begin(), indsA.end());
        auto jacACompressed = Matrix<real, 3>::Zero(3, isize(testInds));
        for (int j = 0; j < indsA.size(); j++) {
          auto dst = static_cast<int>(std::distance(testInds.begin(), testInds.find(indsA[j])));
          jacACompressed.Col(dst) += jacA.Col(j);
        }

        // Get the finite-difference Jacobian and select colums based on indices
        auto jacBFull = jacFD.at(i);
        Matrix<real, 3> jacBSelect(3, isize(testInds));
        for (auto ind = testInds.begin(); ind != testInds.end(); ind++) {
          auto dst = static_cast<int>(std::distance(testInds.begin(), ind));
          jacBSelect.Col(dst) = jacBFull.Col(*ind);
        }

        // Test
        real normA = jacACompressed.Norm();
        real normB = jacBSelect.Norm();
        Matrix<real> diff = jacACompressed - jacBSelect;
        real normDiff = diff.Norm();
        if (std::max(normA, normB) > 1e-9_r) {
          EXPECT_NEAR(normDiff / std::max(normA, normB), 0_r, tol);
        }
      }
    }
  }
};

/************************************************************************
 * The actual tests, with diverse combinations of differentiable maps
 ************************************************************************/

TEST_F(DMapTest, RigidCollider) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    TransformRT state = kRigidState;
    AddEpsRigid(i, eps, state);
    std::vector<Real3> afterRigid(points.size());
    ApplyRigid(state, points, afterRigid);
    SelectOutput(afterRigid, kInds, out, true);
  };

  auto detectionResultAfterRigid = CreateDetectionResultAfterRigid(kPoints, kInds);
  DMapRTOutput dtransform(0, kRigidState, {}, 0);
  DMap<DMapRTOutput> dmap(&dtransform);
  dtransform.SetData(&detectionResultAfterRigid);
  std::vector<int> inds(detectionResultAfterRigid.posColliding.size());
  std::iota(inds.begin(), inds.end(), 0);
  TestConsistency<1>(
      dmap, {RigidSize::kDAll}, inds, {addEpsAndMap}, MakeSingletonConstSpan(kToRigidCollider));
}

TEST_F(DMapTest, RigidColliding) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    TransformRT state = DMapTest::kRigidState;
    AddEpsRigid(i, eps, state);
    std::vector<Real3> afterRigid(points.size());
    ApplyRigid(state, points, afterRigid);
    SelectOutput(afterRigid, kInds, out);
  };

  for (auto toCollider : kToCollider) {
    DMapRTInput dtransform(0, kRigidState, kRigidTransform, 0, kPoints, toCollider);
    DMap<DMapRTInput> dmap(&dtransform);
    TestConsistency<1>(dmap, {RigidSize::kDAll}, kInds, {addEpsAndMap}, toCollider);
  }
}

TEST_F(DMapTest, Rom) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kRomState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterRom(points.size());
    ApplyRom(state, kBasis, points, afterRom);
    SelectOutput(afterRom, kInds, out);
  };

  DMapRom drom(0, kRomJacobian, 0);
  DMap<DMapRom> dmap(&drom);
  TestConsistency<1>(dmap, {kRomSize}, kInds, {addEpsAndMap}, {});
}

TEST_F(DMapTest, TransformRom) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kRomState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterRom(points.size());
    ApplyRom(state, kBasis, points, afterRom);
    std::vector<Real3> afterRigid(afterRom.size());
    ApplyRigid(kRigidTransform, afterRom, afterRigid);
    SelectOutput(afterRigid, kInds, out);
  };

  DMapRom drom(0, kRomJacobian, 0);
  DMapRTConst dtransform(kRigidTransform);
  DMap<DMapRTConst, DMapRom> dmap(&dtransform, &drom);
  TestConsistency<1>(dmap, {kRomSize}, kInds, {addEpsAndMap}, {});
}

TEST_F(DMapTest, QuadratureRom) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kRomState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterRom(points.size());
    ApplyRom(state, kBasis, points, afterRom);
    std::vector<Real3> afterQuadrature(3 * kFemElements.size());
    ApplyQuadrature(kFemElements, afterRom, afterQuadrature);
    SelectOutput(afterQuadrature, kInds, out);
  };

  for (auto toCollider : kToCollider) {
    DMapRom drom(0, kRomJacobian, 0);
    DQuad dquad(kFemElements, toCollider);
    DMap<DQuad, DMapRom> dmap(&dquad, &drom);
    TestConsistency<1>(dmap, {kRomSize}, kInds, {addEpsAndMap}, toCollider);
  }
}

TEST_F(DMapTest, QuadratureTransformRom) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kRomState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterRom(points.size());
    ApplyRom(state, kBasis, points, afterRom);
    std::vector<Real3> afterRigid(afterRom.size());
    ApplyRigid(kRigidTransform, afterRom, afterRigid);
    std::vector<Real3> afterQuadrature(3 * kFemElements.size());
    ApplyQuadrature(kFemElements, afterRigid, afterQuadrature);
    SelectOutput(afterQuadrature, kInds, out);
  };

  for (auto toCollider : kToCollider) {
    DMapRom drom(0, kRomJacobian, 0);
    DMapRTConst dtransform(kRigidTransform);
    DQuad dquad(kFemElements, toCollider);
    DMap<DQuad, DMapRTConst, DMapRom> dmap(&dquad, &dtransform, &drom);
    TestConsistency<1>(dmap, {kRomSize}, kInds, {addEpsAndMap}, toCollider);
  }
}

TEST_F(DMapTest, Skinning) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real, kArticulatedSize> state = kArticulatedState;
    AddEpsArticulated(i, eps, state);
    std::vector<Real3> afterSkinning(points.size());
    ApplySkinning(state, kSkinningTransform, points, afterSkinning);
    SelectOutput(afterSkinning, kInds, out);
  };

  auto skinningJacobian = CreateSkinningJacobian(kSkinningTransform, kArticulatedState, kPoints);
  DMapSkinNoInput dskinning(0, skinningJacobian, kArticulatedDofs, 0);
  DMap<DMapSkinNoInput> dmap(&dskinning);
  TestConsistency<1>(dmap, {kArticulatedSize}, kInds, {addEpsAndMap}, {});
}

TEST_F(DMapTest, SkinningRom) {
  auto addEpsAndMapRom = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kRomState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterRom(points.size());
    ApplyRom(state, kBasis, points, afterRom);
    std::vector<Real3> afterSkinning(points.size());
    ApplySkinning(kArticulatedState, kSkinningTransform, afterRom, afterSkinning);
    SelectOutput(afterSkinning, kInds, out);
  };

  auto addEpsAndMapSkeleton = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real, kArticulatedSize> state = kArticulatedState;
    AddEpsArticulated(i, eps, state);
    std::vector<Real3> afterRom(points.size());
    ApplyRom(kRomState, kBasis, points, afterRom);
    std::vector<Real3> afterSkinning(points.size());
    ApplySkinning(state, kSkinningTransform, afterRom, afterSkinning);
    SelectOutput(afterSkinning, kInds, out);
  };

  auto skinningJacRom =
      CreateSkinningJacobian(kSkinningTransform, kArticulatedState, kPointsAfterRom);
  DMapRom drom(0, kRomJacobian, 0);
  DMapSkinInput dskinning(1, skinningJacRom, kArticulatedDofs, 0, kSkinning, kSkeletonRotations);
  DMap<DMapSkinInput, DMapRom> dmap(&dskinning, &drom);
  TestConsistency<2>(
      dmap, {kRomSize, kArticulatedSize}, kInds, {addEpsAndMapRom, addEpsAndMapSkeleton}, {});
}

TEST_F(DMapTest, QuadratureSkinningRom) {
  auto addEpsAndMapRom = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kRomState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterRom(points.size());
    ApplyRom(state, kBasis, points, afterRom);
    std::vector<Real3> afterSkinning(points.size());
    ApplySkinning(kArticulatedState, kSkinningTransform, afterRom, afterSkinning);
    std::vector<Real3> afterQuadrature(3 * kFemElements.size());
    ApplyQuadrature(kFemElements, afterSkinning, afterQuadrature);
    SelectOutput(afterQuadrature, kInds, out);
  };

  auto addEpsAndMapSkeleton = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real, kArticulatedSize> state = kArticulatedState;
    AddEpsArticulated(i, eps, state);
    std::vector<Real3> afterRom(points.size());
    ApplyRom(kRomState, kBasis, points, afterRom);
    std::vector<Real3> afterSkinning(points.size());
    ApplySkinning(state, kSkinningTransform, afterRom, afterSkinning);
    std::vector<Real3> afterQuadrature(3 * kFemElements.size());
    ApplyQuadrature(kFemElements, afterSkinning, afterQuadrature);
    SelectOutput(afterQuadrature, kInds, out);
  };

  auto skinningJacRom =
      CreateSkinningJacobian(kSkinningTransform, kArticulatedState, kPointsAfterRom);
  for (auto toCollider : kToCollider) {
    DMapRom drom(0, kRomJacobian, 0);
    DMapSkinInput dskinning(1, skinningJacRom, kArticulatedDofs, 0, kSkinning, kSkeletonRotations);
    DQuad dquad(kFemElements, toCollider);
    DMap<DQuad, DMapSkinInput, DMapRom> dmap(&dquad, &dskinning, &drom);
    TestConsistency<2>(
        dmap,
        {kRomSize, kArticulatedSize},
        kInds,
        {addEpsAndMapRom, addEpsAndMapSkeleton},
        toCollider);
  }
}

TEST_F(DMapTest, Soft) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kSoftState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterSoft(points.size());
    ApplySoft(state, points, afterSoft);
    SelectOutput(afterSoft, kInds, out);
  };

  DMapSoft dsoft(0, 0);
  DMap<DMapSoft> dmap(&dsoft);
  TestConsistency<1>(dmap, {kSoftSize}, kInds, {addEpsAndMap}, {});
}

TEST_F(DMapTest, TransformSoft) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kSoftState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterSoft(points.size());
    ApplySoft(state, points, afterSoft);
    std::vector<Real3> afterRigid(afterSoft.size());
    ApplyRigid(kRigidTransform, afterSoft, afterRigid);
    SelectOutput(afterRigid, kInds, out);
  };

  DMapSoft dsoft(0, 0);
  DMapRTConst dtransform(kRigidTransform);
  DMap<DMapRTConst, DMapSoft> dmap(&dtransform, &dsoft);
  TestConsistency<1>(dmap, {kSoftSize}, kInds, {addEpsAndMap}, {});
}

TEST_F(DMapTest, QuadratureSoft) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kSoftState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterSoft(points.size());
    ApplySoft(state, points, afterSoft);
    std::vector<Real3> afterQuadrature(afterSoft.size());
    ApplyQuadrature(kFemElements, afterSoft, afterQuadrature);
    SelectOutput(afterQuadrature, kInds, out);
  };

  for (auto toCollider : kToCollider) {
    DMapSoft dsoft(0, 0);
    DQuad dquad(kFemElements, toCollider);
    DMap<DQuad, DMapSoft> dmap(&dquad, &dsoft);
    TestConsistency<1>(dmap, {kSoftSize}, kInds, {addEpsAndMap}, toCollider);
  }
}

TEST_F(DMapTest, QuadratureTransformSoft) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kSoftState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterSoft(points.size());
    ApplySoft(state, points, afterSoft);
    std::vector<Real3> afterRigid(afterSoft.size());
    ApplyRigid(kRigidTransform, afterSoft, afterRigid);
    std::vector<Real3> afterQuadrature(afterRigid.size());
    ApplyQuadrature(kFemElements, afterRigid, afterQuadrature);
    SelectOutput(afterQuadrature, kInds, out);
  };

  for (auto toCollider : kToCollider) {
    DMapSoft dsoft(0, 0);
    DMapRTConst dtransform(kRigidTransform);
    DQuad dquad(kFemElements, toCollider);
    DMap<DQuad, DMapRTConst, DMapSoft> dmap(&dquad, &dtransform, &dsoft);
    TestConsistency<1>(dmap, {kSoftSize}, kInds, {addEpsAndMap}, toCollider);
  }
}

TEST_F(DMapTest, SkinningSoft) {
  auto addEpsAndMapSoft = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kSoftState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterSoft(points.size());
    ApplySoft(state, points, afterSoft);
    std::vector<Real3> afterSkinning(afterSoft.size());
    ApplySkinning(kArticulatedState, kSkinningTransform, afterSoft, afterSkinning);
    SelectOutput(afterSkinning, kInds, out);
  };

  auto addEpsAndMapSkeleton = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real, kArticulatedSize> state = kArticulatedState;
    AddEpsArticulated(i, eps, state);
    std::vector<Real3> afterSoft(points.size());
    ApplySoft(kSoftState, points, afterSoft);
    std::vector<Real3> afterSkinning(afterSoft.size());
    ApplySkinning(state, kSkinningTransform, afterSoft, afterSkinning);
    SelectOutput(afterSkinning, kInds, out);
  };

  auto skinningJacSoft =
      CreateSkinningJacobian(kSkinningTransform, kArticulatedState, kPointsAfterSoft);
  DMapSoft dsoft(0, 0);
  DMapSkinInput dskinning(1, skinningJacSoft, kArticulatedDofs, 0, kSkinning, kSkeletonRotations);
  DMap<DMapSkinInput, DMapSoft> dmap(&dskinning, &dsoft);
  TestConsistency<2>(
      dmap, {kSoftSize, kArticulatedSize}, kInds, {addEpsAndMapSoft, addEpsAndMapSkeleton}, {});
}

TEST_F(DMapTest, QuadratureSkinningSoft) {
  auto addEpsAndMapSoft = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kSoftState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterSoft(points.size());
    ApplySoft(state, points, afterSoft);
    std::vector<Real3> afterSkinning(afterSoft.size());
    ApplySkinning(kArticulatedState, kSkinningTransform, afterSoft, afterSkinning);
    std::vector<Real3> afterQuadrature(3 * kFemElements.size());
    ApplyQuadrature(kFemElements, afterSkinning, afterQuadrature);
    SelectOutput(afterQuadrature, kInds, out);
  };

  auto addEpsAndMapSkeleton = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real, kArticulatedSize> state = kArticulatedState;
    AddEpsArticulated(i, eps, state);
    std::vector<Real3> afterSoft(points.size());
    ApplySoft(kSoftState, points, afterSoft);
    std::vector<Real3> afterSkinning(afterSoft.size());
    ApplySkinning(state, kSkinningTransform, afterSoft, afterSkinning);
    std::vector<Real3> afterQuadrature(3 * kFemElements.size());
    ApplyQuadrature(kFemElements, afterSkinning, afterQuadrature);
    SelectOutput(afterQuadrature, kInds, out);
  };

  auto skinningJacSoft =
      CreateSkinningJacobian(kSkinningTransform, kArticulatedState, kPointsAfterSoft);
  for (auto toCollider : kToCollider) {
    DMapSoft dsoft(0, 0);
    DMapSkinInput dskinning(1, skinningJacSoft, kArticulatedDofs, 0, kSkinning, kSkeletonRotations);
    DQuad dquad(kFemElements, toCollider);
    DMap<DQuad, DMapSkinInput, DMapSoft> dmap(&dquad, &dskinning, &dsoft);
    TestConsistency<2>(
        dmap,
        {kSoftSize, kArticulatedSize},
        kInds,
        {addEpsAndMapSoft, addEpsAndMapSkeleton},
        toCollider);
  }
}

TEST_F(DMapTest, BlendingSoft) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kSoftState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterSoft(points.size());
    ApplySoft(state, points, afterSoft);
    std::vector<Real3> afterBlending(afterSoft.size());
    ApplyBlending(kBlending, points, afterSoft, afterBlending);
    SelectOutput(afterBlending, kInds, out);
  };

  DMapSoft dsoft(0, 0);
  DMapBlending dblending(kBlending);
  DMap<DMapBlending, DMapSoft> dmap(&dblending, &dsoft);
  TestConsistency<1>(dmap, {kSoftSize}, kInds, {addEpsAndMap}, {});
}

TEST_F(DMapTest, BlendingRom) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kRomState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterRom(points.size());
    ApplyRom(state, kBasis, points, afterRom);
    std::vector<Real3> afterBlending(afterRom.size());
    ApplyBlending(kBlending, points, afterRom, afterBlending);
    SelectOutput(afterBlending, kInds, out);
  };

  DMapRom drom(0, kRomJacobian, 0);
  DMapBlending dblending(kBlending);
  DMap<DMapBlending, DMapRom> dmap(&dblending, &drom);
  TestConsistency<1>(dmap, {kRomSize}, kInds, {addEpsAndMap}, {});
}

TEST_F(DMapTest, QuadratureSkinningBlendingSoft) {
  auto addEpsAndMapSoft = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kSoftState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterSoft(points.size());
    ApplySoft(state, points, afterSoft);
    std::vector<Real3> afterBlending(afterSoft.size());
    ApplyBlending(kBlending, points, afterSoft, afterBlending);
    std::vector<Real3> afterSkinning(afterBlending.size());
    ApplySkinning(kArticulatedState, kSkinningTransform, afterBlending, afterSkinning);
    std::vector<Real3> afterQuadrature(3 * kFemElements.size());
    ApplyQuadrature(kFemElements, afterSkinning, afterQuadrature);
    SelectOutput(afterQuadrature, kInds, out);
  };

  auto addEpsAndMapSkeleton = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real, kArticulatedSize> state = kArticulatedState;
    AddEpsArticulated(i, eps, state);
    std::vector<Real3> afterSoft(points.size());
    ApplySoft(kSoftState, points, afterSoft);
    std::vector<Real3> afterBlending(afterSoft.size());
    ApplyBlending(kBlending, points, afterSoft, afterBlending);
    std::vector<Real3> afterSkinning(afterBlending.size());
    ApplySkinning(state, kSkinningTransform, afterBlending, afterSkinning);
    std::vector<Real3> afterQuadrature(3 * kFemElements.size());
    ApplyQuadrature(kFemElements, afterSkinning, afterQuadrature);
    SelectOutput(afterQuadrature, kInds, out);
  };

  auto pointsAfterBlendingSoft = CreatePointsAfterBlending(kBlending, kPoints, kPointsAfterSoft);
  auto skinningJacBlendingSoft =
      CreateSkinningJacobian(kSkinningTransform, kArticulatedState, pointsAfterBlendingSoft);
  for (auto toCollider : kToCollider) {
    DMapSoft dsoft(0, 0);
    DMapBlending dblending(kBlending);
    DMapSkinInput dskinning(
        1, skinningJacBlendingSoft, kArticulatedDofs, 0, kSkinning, kSkeletonRotations);
    DQuad dquad(kFemElements, toCollider);
    DMap<DQuad, DMapSkinInput, DMapBlending, DMapSoft> dmap(&dquad, &dskinning, &dblending, &dsoft);
    TestConsistency<2>(
        dmap,
        {kSoftSize, kArticulatedSize},
        kInds,
        {addEpsAndMapSoft, addEpsAndMapSkeleton},
        toCollider,
        1.1e-2_r);
  }
}

TEST_F(DMapTest, QuadratureSkinningBlendingRom) {
  auto addEpsAndMapRom = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kRomState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterRom(points.size());
    ApplyRom(state, kBasis, points, afterRom);
    std::vector<Real3> afterBlending(afterRom.size());
    ApplyBlending(kBlending, points, afterRom, afterBlending);
    std::vector<Real3> afterSkinning(afterBlending.size());
    ApplySkinning(kArticulatedState, kSkinningTransform, afterBlending, afterSkinning);
    std::vector<Real3> afterQuadrature(3 * kFemElements.size());
    ApplyQuadrature(kFemElements, afterSkinning, afterQuadrature);
    SelectOutput(afterQuadrature, kInds, out);
  };

  auto addEpsAndMapSkeleton = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real, kArticulatedSize> state = kArticulatedState;
    AddEpsArticulated(i, eps, state);
    std::vector<Real3> afterRom(points.size());
    ApplyRom(kRomState, kBasis, points, afterRom);
    std::vector<Real3> afterBlending(afterRom.size());
    ApplyBlending(kBlending, points, afterRom, afterBlending);
    std::vector<Real3> afterSkinning(afterBlending.size());
    ApplySkinning(state, kSkinningTransform, afterBlending, afterSkinning);
    std::vector<Real3> afterQuadrature(3 * kFemElements.size());
    ApplyQuadrature(kFemElements, afterSkinning, afterQuadrature);
    SelectOutput(afterQuadrature, kInds, out);
  };

  auto pointsAfterBlendingRom = CreatePointsAfterBlending(kBlending, kPoints, kPointsAfterRom);
  auto skinningJacBlendingRom =
      CreateSkinningJacobian(kSkinningTransform, kArticulatedState, pointsAfterBlendingRom);
  for (auto toCollider : kToCollider) {
    DMapRom drom(0, kRomJacobian, 0);
    DMapBlending dblending(kBlending);
    DMapSkinInput dskinning(
        1, skinningJacBlendingRom, kArticulatedDofs, 0, kSkinning, kSkeletonRotations);
    DQuad dquad(kFemElements, toCollider);
    DMap<DQuad, DMapSkinInput, DMapBlending, DMapRom> dmap(&dquad, &dskinning, &dblending, &drom);
    TestConsistency<2>(
        dmap,
        {kRomSize, kArticulatedSize},
        kInds,
        {addEpsAndMapRom, addEpsAndMapSkeleton},
        toCollider);
  }
}

TEST_F(DMapTest, InverseRom) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kRomState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterRom(points.size());
    ApplyRom(state, kBasis, points, afterRom);
    SelectOutput(afterRom, kInds, out, true);
  };

  auto detectionResult = CreateDetectionResultAfterRomMap(kBasis, kInds, kToSoftCollider);
  DMapInverse dinverse(0, 0, true);
  dinverse.SetData(&detectionResult);
  DMap<DMapInverse> dmap(&dinverse);
  TestConsistency<1>(dmap, {kRomSize}, kInds, {addEpsAndMap}, kToSoftCollider);
}

TEST_F(DMapTest, InverseSoft) {
  auto addEpsAndMap = [this](int i, real eps, Span<Real3 const> points, Span<Real3> out) {
    ColumnVector<real> state = kSoftState;
    AddEpsEuclidean(i, eps, state);
    std::vector<Real3> afterSoft(points.size());
    ApplySoft(state, points, afterSoft);
    ApplyBarycentricWeighting(afterSoft, kSoftColliderInds, kSoftColliderWeights, out, true);
  };

  auto detectionResult = CreateDetectionResultAfterSoftMap(
      kSoftColliderInds, kSoftColliderWeights, kInds, kToSoftCollider);
  DMapInverse dinverse(0, 0, false);
  dinverse.SetData(&detectionResult);
  DMap<DMapInverse> dmap(&dinverse);
  TestConsistency<1>(dmap, {kSoftSize}, kInds, {addEpsAndMap}, kToSoftCollider);
}

TEST_F(DMapTest, SparseSkinning) {
  // Test DMapSparseSkinning with synthetic CSR data.
  // Each node depends on a variable number of DoFs with simple linear Jacobians.
  constexpr int kNumNodes = 8;
  constexpr int kNumDofs = 12;

  // Build a SparseMatrix<Real3> with 1 row per node.
  // Each non-zero entry is a Real3 holding the (x, y, z) Jacobian contribution.
  DynamicArray<int> rowPtrs;
  DynamicArray<int> colIndices;
  DynamicArray<Real3> values;

  rowPtrs.push_back(0);
  for (int i = 0; i < kNumNodes; ++i) {
    int const numDepsForNode = 2 + (i % 3); // 2, 3, or 4 DoFs per node
    for (int j = 0; j < numDepsForNode; ++j) {
      colIndices.push_back((i + j) % kNumDofs);
      // x: 0.1*(i+1), y: 0.2*(j+1), z: 0.3*(i+j+1)
      values.push_back(Real3{0.1_r * (i + 1), 0.2_r * (j + 1), 0.3_r * (i + j + 1)});
    }
    rowPtrs.push_back(isize(colIndices));
  }

  SparseMatrix<Real3> sparseJacobian(
      kNumDofs, std::move(rowPtrs), std::move(colIndices), std::move(values));
  auto sparseJacView = AsConstView(sparseJacobian);

  // Create initial state (zeros)
  std::array<real, kNumDofs> state = {};

  auto computePosition = [&](int nodeIdx, Span<real const> q) -> Real3 {
    Real3 pos = {};
    auto colInds = sparseJacView.Indices(nodeIdx);
    for (int k = 0; k < isize(colInds); ++k) {
      int const dofIdx = colInds[k];
      for (int r = 0; r < 3; ++r) {
        pos[r] += sparseJacView.Values(nodeIdx)[k][r] * q[dofIdx];
      }
    }
    return pos;
  };

  auto addEpsAndMap = [&](int i, real eps, Span<Real3 const> /*points*/, Span<Real3> out) {
    std::array<real, kNumDofs> perturbedState = state;
    perturbedState[i] += eps;

    for (size_t p = 0; p < kOutput; ++p) {
      out[p] = computePosition(kInds[p], MakeConstSpan(perturbedState));
    }
  };

  DMapSparseSkinning dsparseSkinning(0, 0, sparseJacView);
  DMap<DMapSparseSkinning> dmap(&dsparseSkinning);
  TestConsistency<1>(dmap, {kNumDofs}, kInds, {addEpsAndMap}, {});

  // Test composition DMap<DMapQuad<TriElement>, DMapSparseSkinning>.
  // This exercises the non-shared-DoF code path in DMapQuad::PropagateJacobianSlice,
  // which is the production path for rod visual mesh contact.
  auto addEpsAndMapQuad = [&](int i, real eps, Span<Real3 const> /*points*/, Span<Real3> out) {
    std::array<real, kNumDofs> perturbedState = state;
    perturbedState[i] += eps;

    std::array<Real3, kNumNodes> nodePositions{};
    for (int n = 0; n < kNumNodes; ++n) {
      nodePositions[n] = computePosition(n, MakeConstSpan(perturbedState));
    }

    DynamicArray<Real3> afterQuadrature(3 * isize(kTriElements));
    ApplyQuadrature(MakeConstSpan(kTriElements), MakeConstSpan(nodePositions), afterQuadrature);
    SelectOutput(afterQuadrature, kInds, out);
  };

  for (auto toCollider : kToCollider) {
    DMapSparseSkinning dsparseSkinningQ(0, 0, sparseJacView);
    DQuadTri dquadTri(kTriElements, toCollider);
    DMap<DQuadTri, DMapSparseSkinning> dmapQ(&dquadTri, &dsparseSkinningQ);
    TestConsistency<1>(dmapQ, {kNumDofs}, kInds, {addEpsAndMapQuad}, toCollider);
  }

  // Test composition DMap<DMapQuad<TriElement>, DMapRTConst, DMapSparseSkinning>.
  // This exercises propagation through DMapRTConst sandwiched between sparse skinning and
  // quadrature, mirroring the QuadratureTransformSoft pattern.
  auto addEpsAndMapTransformQuad =
      [&](int i, real eps, Span<Real3 const> /*points*/, Span<Real3> out) {
        std::array<real, kNumDofs> perturbedState = state;
        perturbedState[i] += eps;

        std::array<Real3, kNumNodes> nodePositions{};
        for (int n = 0; n < kNumNodes; ++n) {
          nodePositions[n] = computePosition(n, MakeConstSpan(perturbedState));
        }

        std::array<Real3, kNumNodes> afterRigid{};
        ApplyRigid(kRigidTransform, MakeConstSpan(nodePositions), MakeSpan(afterRigid));

        DynamicArray<Real3> afterQuadrature(3 * isize(kTriElements));
        ApplyQuadrature(MakeConstSpan(kTriElements), MakeConstSpan(afterRigid), afterQuadrature);
        SelectOutput(afterQuadrature, kInds, out);
      };

  for (auto toCollider : kToCollider) {
    DMapSparseSkinning dsparseSkinningRTQ(0, 0, sparseJacView);
    DMapRTConst dtransformRTQ(kRigidTransform);
    DQuadTri dquadTriRTQ(kTriElements, toCollider);
    DMap<DQuadTri, DMapRTConst, DMapSparseSkinning> dmapRTQ(
        &dquadTriRTQ, &dtransformRTQ, &dsparseSkinningRTQ);
    // Slightly looser tolerance than kTol: the three-stage chain (sparse skinning + rigid
    // transform + tri quadrature) accumulates slightly more single-precision noise in the FD
    // estimate than the two-stage compositions above.
    TestConsistency<1>(
        dmapRTQ, {kNumDofs}, kInds, {addEpsAndMapTransformQuad}, toCollider, 1.5 * kTol);
  }
}

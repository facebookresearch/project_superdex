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

#include <mochi_core/geometry/model_utils.h>
#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/coordinate_space.h>
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/nd_array_utils.h>
#include <mochi_core/utils/task_scheduler.h>

using namespace mochi;

static constexpr std::string_view kTetMeshCubeJson = R"(
{
  "mesh": {
    "nodesPerElement": 4,
    "coordinates": [
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      1.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
      1.0, 0.0, 1.0,
      0.0, 1.0, 1.0,
      1.0, 1.0, 1.0
    ],
    "connectivity": [
      0, 1, 2, 4,
	    6, 7, 4, 2,
	    5, 4, 7, 1,
	    3, 2, 1, 7,
	    1, 2, 4, 7
    ]
  },
  "constrainedNodes": [1, 2, 3]
})";

static constexpr std::string_view kTriMeshCubeJson = R"(
{
  "mesh": {
    "nodesPerElement": 3,
    "coordinates": [
      0.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      0.0, 1.0, 0.0,
      1.0, 1.0, 0.0,
      0.0, 0.0, 1.0,
      1.0, 0.0, 1.0,
      0.0, 1.0, 1.0,
      1.0, 1.0, 1.0
    ],
    "connectivity": [
      0, 2, 1,
      2, 3, 1,
      1, 3, 5,
      3, 7, 5,
      5, 7, 4,
      7, 6, 4,
      4, 6, 0,
      6, 2, 0,
      2, 6, 3,
      6, 7, 3,
      0, 1, 4,
      4, 1, 5
    ],
    "skinning": {
      "weightsPerNode": 4,
      "indices": [
         0,  1,  2,  3,
         4,  5,  6,  7,
         8,  9, 10, 11,
        12, 13, 14, 15,
        16, 17, 18, 19,
        20, 21, 22, 23,
        24, 25, 26, 27,
        28, 29, 30, 31
      ],
      "weights": [
        1.00, 0.00, 0.00, 0.00,
        0.00, 1.00, 0.00, 0.00,
        0.00, 0.00, 1.00, 0.00,
        0.00, 0.00, 0.00, 1.00,
        0.50, 0.50, 0.00, 0.00,
        0.00, 0.00, 0.50, 0.50,
        0.25, 0.25, 0.00, 0.50,
        0.25, 0.25, 0.25, 0.25
      ]
    }
  }
})";

static constexpr std::string_view kPolylineMeshJson = R"(
{
  "mesh": {
    "nodesPerElement": 2,
    "coordinates": [
      0.0, 0.0, 0.0,
      0.1, 0.0, 0.0,
      0.2, 0.0, 0.0,
      0.3, 0.0, 0.0
    ],
    "connectivity": [
      0, 1,
      1, 2,
      2, 3
    ]
  },
  "elementFrameAxes": [
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0
  ]
})";

static constexpr std::string_view kBoxJson = R"(
{
  "box": {
    "center": [1.0, 2.0, 3.0],
    "halfExtents": [0.5, 1.0, 1.5],
    "rotation": [0.0, 0.0, 0.0, 1.0]
  }
})";

static constexpr std::string_view kPlaneJson = R"(
{
  "plane": {
    "normal": [0.0, 1.0, 0.0],
    "distance": 0.5
  }
})";

static constexpr std::string_view kSphereJson = R"(
{
  "sphere": {
    "center": [1.0, 2.0, 3.0],
    "radius": 0.5
  }
})";

static constexpr real kNonFiniteValues[] = {
    std::numeric_limits<real>::infinity(),
    -std::numeric_limits<real>::infinity(),
    std::numeric_limits<real>::quiet_NaN(),
    std::numeric_limits<real>::signaling_NaN()};

static void TestSerializationRoundTrip(ModelData const& model, bool shouldValidate = true) {
  // Round trip via SaveToFile (JSON), LoadFromFile, and LoadFromFileUnchecked`
  for (bool saveUsingDataView : {false, true}) {
    // SaveToFile
    auto file = CreateTempFile("model_utils_test", ".json", test::ExpectOK{});
    if (saveUsingDataView) {
      model::SaveToFile(
          ModelDataView{model}, file.Path().string(), FileFormat::JSON, test::ExpectOK{});
    } else {
      model::SaveToFile(model, file.Path().string(), FileFormat::JSON, test::ExpectOK{});
    }

    // LoadFromFile
    if (shouldValidate) {
      ModelData loadedModel = model::LoadFromFile(file.Path().string(), test::ExpectOK{});
      EXPECT_EQ(model, loadedModel); // Expect exact equality
    }

    // LoadFromFileUnchecked
    {
      ModelData loadedModel = model::LoadFromFileUnchecked(file.Path().string(), test::ExpectOK{});
      if (shouldValidate) {
        model::AutoCorrect(loadedModel, test::ExpectOK{});
        model::Validate(loadedModel, test::ExpectOK{});
      }
      EXPECT_EQ(model, loadedModel); // Expect exact equality
    }
  }

  // Round trip via SaveToJsonString, LoadFromBytes, and LoadFromBytesUnchecked
  {
    // SaveToJsonString
    auto json = model::SaveToJsonString(model, test::ExpectOK{});

    // LoadFromBytes
    if (shouldValidate) {
      ModelData loadedModel = model::LoadFromBytes(json, test::ExpectOK{});
      EXPECT_EQ(model, loadedModel); // Expect exact equality
    }

    // LoadFromBytesUnchecked
    {
      ModelData loadedModel = model::LoadFromBytesUnchecked(json, test::ExpectOK{});
      if (shouldValidate) {
        model::AutoCorrect(loadedModel, test::ExpectOK{});
        model::Validate(loadedModel, test::ExpectOK{});
      }
      EXPECT_EQ(model, loadedModel); // Expect exact equality
    }
  }

  // Round trip via SaveToFile (H5), LoadFromFile, and LoadFromFileUnchecked
  for (bool saveUsingDataView : {false, true}) {
    // Save to H5 file
    auto tempH5 = CreateTempFile("model_utils_test", ".h5", test::ExpectOK{});

    Error error;
    if (saveUsingDataView) {
      model::SaveToFile(ModelDataView{model}, tempH5.Path().string(), FileFormat::H5, error);
    } else {
      model::SaveToFile(model, tempH5.Path().string(), FileFormat::H5, error);
    }
#if MOCHI_USE_HDF5
    EXPECT_OK(error);
#else
    EXPECT_NOT_OK(error);
#endif

    // Load From H5 file
    error = {};
    ModelData loadedModel = model::LoadFromFileUnchecked(tempH5.Path().string(), error);
#if MOCHI_USE_HDF5
    EXPECT_OK(error);
    if (shouldValidate) {
      model::AutoCorrect(loadedModel, test::ExpectOK{});
      model::Validate(loadedModel, test::ExpectOK{});
    }
    EXPECT_EQ(model, loadedModel);
#else
    EXPECT_NOT_OK(error);
#endif

    // Load from H5 file bytes in memory
    error = {};
    auto fileBytes = ReadFileBytes(tempH5.Path(), test::ExpectOK{});
    loadedModel = model::LoadFromBytesUnchecked(fileBytes, error);
#if MOCHI_USE_HDF5
    EXPECT_OK(error);
    if (shouldValidate) {
      model::AutoCorrect(loadedModel, test::ExpectOK{});
      model::Validate(loadedModel, test::ExpectOK{});
    }
    EXPECT_EQ(model, loadedModel);
#else
    EXPECT_NOT_OK(error);
#endif
  }

  // This isn't technically "serialization", but we should also test ModelData -> ModelDataView ->
  // ModelData.
  {
    ModelDataView info = model; // View of the ModelData
    ModelData model2{info}; // Copy from ModelDataView
    EXPECT_EQ(model, model2); // Expect exact equality
  }
}

TEST(ModelUtils, SaveLoad_TetMesh) {
  // Load JSON using LoadFromBytes
  ModelData model = model::LoadFromBytes(kTetMeshCubeJson, test::ExpectOK{});
  EXPECT_TRUE(model.mesh.has_value());
  auto& mesh = *model.mesh;

  // Check it
  EXPECT_EQ(4, mesh.nodesPerElement);
  EXPECT_EQ(8, mesh.GetNumNodes());
  EXPECT_EQ(5, mesh.GetNumElements());
  EXPECT_EQ(8 * 3, isize(mesh.coordinates));
  EXPECT_EQ(5 * 4, isize(mesh.connectivity));
  EXPECT_FALSE(mesh.skinning.has_value());
  EXPECT_TRUE(model.constrainedNodes.has_value());
  EXPECT_EQ(3, isize(*model.constrainedNodes));

  // AutoCorrect and Validate should succeed and change nothing.
  {
    ModelData model2 = model;
    model::AutoCorrect(model2, test::ExpectOK());
    model::Validate(model2, test::ExpectOK{});
    EXPECT_EQ(model, model2); // No change
  }

  // Round trip serialization
  TestSerializationRoundTrip(model);
}

TEST(ModelUtils, SaveLoad_TriMesh) {
  // Load JSON using LoadFromBytes
  ModelData model = model::LoadFromBytes(kTriMeshCubeJson, test::ExpectOK{});
  EXPECT_TRUE(model.mesh.has_value());
  auto& mesh = *model.mesh;

  // Check it
  EXPECT_EQ(3, mesh.nodesPerElement);
  EXPECT_EQ(8, mesh.GetNumNodes());
  EXPECT_EQ(12, mesh.GetNumElements());
  EXPECT_EQ(8 * 3, isize(mesh.coordinates));
  EXPECT_EQ(12 * 3, isize(mesh.connectivity));
  EXPECT_TRUE(mesh.skinning.has_value());
  EXPECT_EQ(4, mesh.skinning->weightsPerNode);
  EXPECT_EQ(8 * 4, isize(mesh.skinning->weights));
  EXPECT_EQ(8 * 4, isize(mesh.skinning->indices));

  // AutoCorrect and Validate should succeed and change nothing.
  {
    ModelData model2 = model;
    model::AutoCorrect(model2, test::ExpectOK());
    model::Validate(model2, test::ExpectOK{});
    EXPECT_EQ(model, model2); // No change
  }

  // Round-trip serialization
  TestSerializationRoundTrip(model);
}

TEST(ModelUtils, SaveLoad_PolylineMesh) {
  // Load JSON using LoadFromBytes
  ModelData model = model::LoadFromBytes(kPolylineMeshJson, test::ExpectOK{});
  EXPECT_TRUE(model.mesh.has_value());
  auto& mesh = *model.mesh;

  // Check it
  EXPECT_EQ(2, mesh.nodesPerElement);
  EXPECT_EQ(4, mesh.GetNumNodes());
  EXPECT_EQ(4 * 3, isize(mesh.coordinates));
  EXPECT_EQ(3, mesh.GetNumElements());
  EXPECT_EQ(3 * 2, isize(mesh.connectivity));
  EXPECT_TRUE(model.elementFrameAxes.has_value());
  EXPECT_EQ(3 * 3, isize(*model.elementFrameAxes));

  // AutoCorrect and Validate should succeed and change nothing.
  {
    ModelData model2 = model;
    model::AutoCorrect(model2, test::ExpectOK());
    model::Validate(model2, test::ExpectOK{});
    EXPECT_EQ(model, model2); // No change
  }

  // Round-trip serialization
  TestSerializationRoundTrip(model);

  // If we were to omit the connectivity array, that should be OK too.
  model.mesh->connectivity.clear();
  model::Validate(model, test::ExpectOK{}); // no change
  TestSerializationRoundTrip(model); // no change

  // If we omit the elementFrameAxes, that's OK too.
  model.elementFrameAxes = std::nullopt;
  model::Validate(model, test::ExpectOK{}); // no change
  TestSerializationRoundTrip(model); // no change
}

TEST(ModelUtils, SaveLoad_ClosedLoopPolylineMesh) {
  // Create a closed-loop polyline (triangle loop)
  ModelData model;
  model.mesh = MeshData{};
  model.mesh->nodesPerElement = 2;
  model.mesh->coordinates = DynamicArray<real>{
      0.0,
      0.0,
      0.0, //
      1.0,
      0.0,
      0.0, //
      0.5,
      1.0,
      0.0};
  // Closed-loop topology is encoded by an explicit closing-segment connectivity array.
  model.mesh->connectivity = DynamicArray<int>{0, 1, 1, 2, 2, 0};

  // Validate should succeed
  model::Validate(model, test::ExpectOK{});

  // Round-trip serialization
  TestSerializationRoundTrip(model);

  // Also test with elementFrameAxes
  model.elementFrameAxes = DynamicArray<real>{
      0.0,
      0.0,
      1.0, //
      0.0,
      0.0,
      1.0, //
      0.0,
      0.0,
      1.0};
  model::Validate(model, test::ExpectOK{});
  TestSerializationRoundTrip(model);
}

static void TestValidateMeshCoordinates(ModelData const& model, MeshData& mesh) {
  // Modify the size of the coordinates array so it is not a multiple of 3
  mesh.coordinates.push_back(0_r);
  model::Validate(model, test::ExpectNotOK{});
  mesh.coordinates.pop_back();
  model::Validate(model, test::ExpectOK{}); // Valid again

  // Any of these non-finite coordinate values should fail validation
  for (real val : kNonFiniteValues) {
    for (int i = 0; i < isize(mesh.coordinates); ++i) {
      real prev = mesh.coordinates[i];
      mesh.coordinates[i] = val;
      model::Validate(model, test::ExpectNotOK{});
      mesh.coordinates[i] = prev; // Restore
    }
  }
  model::Validate(model, test::ExpectOK{}); // Valid again
}

TEST(ModelUtils, Validate_Mesh_Coordinates) {
  ModelData model = model::LoadFromBytes(kTetMeshCubeJson, test::ExpectOK{});
  TestValidateMeshCoordinates(model, *model.mesh);
}

TEST(ModelUtils, Validate_PolylineMesh_Coordinates) {
  // This function tests validation rules that are unique to a polyline mesh.
  ModelData model;
  model.mesh = MeshData{};
  model.mesh->nodesPerElement = 2;
  model.mesh->coordinates =
      DynamicArray<real>{0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0, 0.3, 0.0, 0.0};
  model::Validate(model, test::ExpectOK{}); // Starts valid

  // Each element (line segment) must have non-zero length.
  for (int i = 1; i < model.mesh->GetNumNodes(); ++i) {
    // Make this node equal to the previous node
    auto backup = model.mesh->coordinates[3 * i];
    model.mesh->coordinates[3 * i] =
        model.mesh->coordinates[3 * (i - 1)]; // Same X as previous node
    model::Validate(model, test::ExpectNotOK{});
    model.mesh->coordinates[3 * i] = backup;
    model::Validate(model, test::ExpectOK{}); // Valid again
  }

  // Each element must not point in the opposite direction (180 degrees) of the previous element.
  for (int i = 1; i < model.mesh->GetNumNodes(); ++i) {
    model.mesh->coordinates[3 * i] *=
        -1_r; // Opposite direction. Other elements increase along the x-axis.
    model::Validate(model, test::ExpectNotOK{});
    model.mesh->coordinates[3 * i] *= -1_r;
    model::Validate(model, test::ExpectOK{}); // Valid again
  }
}

static void TestValidateMeshConnectivity(ModelData const& model, MeshData& mesh) {
  // Modify the size of the connectivity array so it is not a multiple of nodesPerElement
  mesh.connectivity.push_back(0_r);
  model::Validate(model, test::ExpectNotOK{});
  mesh.connectivity.pop_back();
  model::Validate(model, test::ExpectOK{}); // Valid again

  // Now modify nodesPerElement
  mesh.nodesPerElement++;
  model::Validate(model, test::ExpectNotOK{}); // connectivity array not a multiple of this value
  mesh.nodesPerElement--;
  model::Validate(model, test::ExpectOK{}); // Valid again

  // Any of these out-of-bounds node indices should fail validation
  int const kTestValues[] = {-1, mesh.GetNumNodes()};
  for (int val : kTestValues) {
    for (int i = 0; i < isize(mesh.connectivity); ++i) {
      int prev = mesh.connectivity[i];
      mesh.connectivity[i] = val;
      model::Validate(model, test::ExpectNotOK{});
      mesh.connectivity[i] = prev; // Restore
      model::Validate(model, test::ExpectOK{}); // Valid again
    }
  }
}

TEST(ModelUtils, Validate_Mesh_Connectivity) {
  ModelData model = model::LoadFromBytes(kTetMeshCubeJson, test::ExpectOK{});
  TestValidateMeshConnectivity(model, *model.mesh);
}

TEST(ModelUtils, Validate_PolylineMesh_Connectivity) {
  // The connectivity array is optional on a Polyline mesh. If non-empty, the connectivity is
  // required to be of the form: [[0, 1], [1, 2], [2, 3], ...]
  ModelData model = model::LoadFromBytes(kPolylineMeshJson, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{}); // starts valid
  EXPECT_TRUE(model.mesh.has_value());
  auto& mesh = *model.mesh;
  EXPECT_EQ(2, mesh.nodesPerElement);

  // Validation fails if the indices are in a different order.
  for (int i = 0; i < isize(mesh.connectivity); ++i) {
    for (int j = i + 1; j < isize(mesh.connectivity); ++j) {
      if (mesh.connectivity[i] != mesh.connectivity[j]) {
        std::swap(mesh.connectivity[i], mesh.connectivity[j]);
        model::Validate(model, test::ExpectNotOK{});
        std::swap(mesh.connectivity[i], mesh.connectivity[j]);
        model::Validate(model, test::ExpectOK{}); // Valid again
      }
    }
  }
}

TEST(ModelUtils, Validate_ClosedLoopPolylineMesh_Coordinates) {
  // A valid closed-loop polyline (triangle loop): three nodes forming a triangle
  ModelData model;
  model.mesh = MeshData{};
  model.mesh->nodesPerElement = 2;
  model.mesh->coordinates = DynamicArray<real>{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 1.0, 0.0};
  model.mesh->connectivity = DynamicArray<int>{0, 1, 1, 2, 2, 0};
  model::Validate(model, test::ExpectOK{});

  // Zero-length closing edge: last node == first node
  {
    auto backup = model.mesh->coordinates;
    model.mesh->coordinates = DynamicArray<real>{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    model::Validate(model, test::ExpectNotOK{});
    model.mesh->coordinates = backup;
    model::Validate(model, test::ExpectOK{}); // Valid again
  }

  // 180-degree tangent at closing edge: closing edge reverses relative to previous edge.
  // Nodes: [0,0,0], [1,0,0], [2,0,0] — closing edge from [2,0,0] to [0,0,0] reverses
  // the direction of edge from [1,0,0] to [2,0,0].
  {
    auto backup = model.mesh->coordinates;
    model.mesh->coordinates = DynamicArray<real>{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0};
    model::Validate(model, test::ExpectNotOK{});
    model.mesh->coordinates = backup;
    model::Validate(model, test::ExpectOK{}); // Valid again
  }

  // 180-degree tangent wrap-around: first edge reverses relative to closing edge.
  // Nodes: [1,0,0], [0,0,0], [0.5,1,0] — closing edge from [0.5,1,0] to [1,0,0]
  // has tangent ~[0.45, -0.89, 0], first edge from [1,0,0] to [0,0,0] has tangent [-1,0,0].
  // These are not 180 degrees, so use a different configuration.
  // Instead: [2,0,0], [1,0,0], [0,0,0] — all collinear.
  // Edge 0: [1,0,0]-[2,0,0] = [-1,0,0], Edge 1: [0,0,0]-[1,0,0] = [-1,0,0],
  // Closing: [2,0,0]-[0,0,0] = [2,0,0] (tangent [1,0,0]) vs first edge tangent [-1,0,0] → 180°.
  {
    auto backup = model.mesh->coordinates;
    model.mesh->coordinates = DynamicArray<real>{2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    model::Validate(model, test::ExpectNotOK{});
    model.mesh->coordinates = backup;
    model::Validate(model, test::ExpectOK{}); // Valid again
  }
}

TEST(ModelUtils, Validate_ClosedLoopPolylineMesh_Connectivity) {
  // A valid closed-loop polyline (square loop): 4 nodes with closed-loop connectivity.
  // ValidateMesh infers closed-loop topology from the connectivity size (2 * numNodes).
  ModelData model;
  model.mesh = MeshData{};
  model.mesh->nodesPerElement = 2;
  model.mesh->coordinates =
      DynamicArray<real>{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0};
  model.mesh->connectivity = DynamicArray<int>{0, 1, 1, 2, 2, 3, 3, 0};
  model::Validate(model, test::ExpectOK{});

  // AutoCorrect must be a no-op for the polyline topology now that connectivity is the
  // single source of truth.
  {
    ModelData modelCopy = model;
    model::AutoCorrect(modelCopy, test::ExpectOK{});
    EXPECT_EQ(model, modelCopy);
  }

  // Invalid connectivity size: neither 2*numNodes nor 2*(numNodes-1)
  {
    auto backup = model.mesh->connectivity;
    model.mesh->connectivity = DynamicArray<int>{0, 1, 1, 2, 2, 3, 3, 0, 0, 1};
    model::Validate(model, test::ExpectNotOK{});
    model.mesh->connectivity = backup;
    model::Validate(model, test::ExpectOK{}); // Valid again
  }

  // Wrong closing element: [3,1] instead of [3,0]
  {
    auto backup = model.mesh->connectivity;
    model.mesh->connectivity = DynamicArray<int>{0, 1, 1, 2, 2, 3, 3, 1};
    model::Validate(model, test::ExpectNotOK{});
    model.mesh->connectivity = backup;
    model::Validate(model, test::ExpectOK{}); // Valid again
  }

  // Swapping indices should also fail
  {
    for (int i = 0; i < isize(model.mesh->connectivity); ++i) {
      for (int j = i + 1; j < isize(model.mesh->connectivity); ++j) {
        if (model.mesh->connectivity[i] != model.mesh->connectivity[j]) {
          std::swap(model.mesh->connectivity[i], model.mesh->connectivity[j]);
          model::Validate(model, test::ExpectNotOK{});
          std::swap(model.mesh->connectivity[i], model.mesh->connectivity[j]);
          model::Validate(model, test::ExpectOK{}); // Valid again
        }
      }
    }
  }

  // Open connectivity inferred from size 2*(numNodes-1)
  {
    model.mesh->connectivity = DynamicArray<int>{0, 1, 1, 2, 2, 3};
    model::Validate(model, test::ExpectOK{});
  }

  // Empty connectivity is treated as an open polyline.
  {
    model.mesh->connectivity.clear();
    model::Validate(model, test::ExpectOK{});
  }
}

static void TestValidateSkinningData(
    ModelData const& model,
    SkinningData& skinning,
    Span<int const> invalidSkinningIndices) {
  // Weights array size must be a multiple of weightsPerNode
  skinning.weights.push_back(0_r);
  model::Validate(model, test::ExpectNotOK{});
  skinning.weights.pop_back();
  model::Validate(model, test::ExpectOK{}); // Valid again

  // Indices array size must be a multiple of weightsPerNode
  skinning.indices.push_back(0);
  model::Validate(model, test::ExpectNotOK{});
  skinning.indices.pop_back();
  model::Validate(model, test::ExpectOK{}); // Valid again

  // weightsPerNode must be >= 1
  int prevWeightsPerNode = skinning.weightsPerNode;
  skinning.weightsPerNode = 0;
  model::Validate(model, test::ExpectNotOK{});
  skinning.weightsPerNode = -1;
  model::Validate(model, test::ExpectNotOK{});
  skinning.weightsPerNode = prevWeightsPerNode;
  model::Validate(model, test::ExpectOK{}); // Valid again

  // Weights must be finite and non-negative
  for (real badVal : kNonFiniteValues) {
    for (int i = 0; i < isize(skinning.weights); ++i) {
      real prev = skinning.weights[i];
      skinning.weights[i] = badVal;
      model::Validate(model, test::ExpectNotOK{});
      skinning.weights[i] = -1_r;
      model::Validate(model, test::ExpectNotOK{});
      skinning.weights[i] = prev; // restore
      model::Validate(model, test::ExpectOK{}); // Valid again
    }
  }

  // Check for invalid skinning indices
  for (int badIdx : invalidSkinningIndices) {
    for (int i = 0; i < isize(skinning.indices); ++i) {
      int prev = skinning.indices[i];
      skinning.indices[i] = badIdx;
      model::Validate(model, test::ExpectNotOK{});
      skinning.indices[i] = prev; // Restore
    }
  }
  model::Validate(model, test::ExpectOK{}); // Valid again
}

TEST(ModelUtils, Validate_Mesh_Skinning) {
  // Load model with a mesh + skinning
  ModelData model = model::LoadFromBytes(kTriMeshCubeJson, test::ExpectOK{});
  EXPECT_TRUE(model.mesh.has_value());
  EXPECT_TRUE(model.mesh->skinning.has_value());

  // Unfortunately, we cannot validate the range of indices because we do not yet know
  // what other mesh it will be skinned to. But negative indices are definitely wrong.
  int constexpr kInvalidSkinningIndices[] = {-1};
  TestValidateSkinningData(model, *model.mesh->skinning, kInvalidSkinningIndices);
}

TEST(ModelUtils, AutoCorrect_Mesh_Skinning) {
  // Load model with a mesh + skinning
  ModelData model = model::LoadFromBytes(kTriMeshCubeJson, test::ExpectOK{});
  EXPECT_TRUE(model.mesh.has_value());
  EXPECT_TRUE(model.mesh->skinning.has_value());
  auto& skinning = *model.mesh->skinning;
  model::AutoCorrect(model, test::ExpectOK{}); // Starts valid

  // AutoCorrect should normalize weights
  for (int i = 0; i < isize(skinning.weights); i += skinning.weightsPerNode) {
    ModelData model2 = model;
    model2.mesh->skinning->weights[i + 0] = 0.5_r;
    model2.mesh->skinning->weights[i + 1] = 1.0_r;
    model2.mesh->skinning->weights[i + 2] = 0.25_r;
    model2.mesh->skinning->weights[i + 3] = 2.25_r;
    real constexpr kTotalWeight = 4_r;
    model::AutoCorrect(model2, test::ExpectOK{});
    EXPECT_NEAR_EQ(model2.mesh->skinning->weights[i + 0], 0.5_r / kTotalWeight);
    EXPECT_NEAR_EQ(model2.mesh->skinning->weights[i + 1], 1.0_r / kTotalWeight);
    EXPECT_NEAR_EQ(model2.mesh->skinning->weights[i + 2], 0.25_r / kTotalWeight);
    EXPECT_NEAR_EQ(model2.mesh->skinning->weights[i + 3], 2.25_r / kTotalWeight);
  }

  // Each node needs at least one non-zero weight.
  for (int i = 0; i < isize(skinning.weights); i += skinning.weightsPerNode) {
    ModelData model2 = model;
    model2.mesh->skinning->weights[i + 0] = 0_r;
    model2.mesh->skinning->weights[i + 1] = 0_r;
    model2.mesh->skinning->weights[i + 2] = 0_r;
    model2.mesh->skinning->weights[i + 3] = 0_r;
    model::AutoCorrect(model2, test::ExpectNotOK{}); // AutoCorrect won't fix this
    model::Validate(model2, test::ExpectNotOK{});
  }
}

static ModelData GetTetMeshWithBlending() {
  ModelData model = model::LoadFromBytes(kTetMeshCubeJson, test::ExpectOK{});
  EXPECT_TRUE(model.mesh.has_value());
  model.blending = DynamicArray<BlendingData>{};

  BlendingData blending;
  blending.weights.resize(model.mesh->GetNumNodes() * 2);
  blending.indices.resize(model.mesh->GetNumNodes() * 2);

  blending.sourceShape = "one";
  model.blending->push_back(blending);

  blending.sourceShape = "two";
  model.blending->push_back(blending);

  return model;
}

TEST(ModelUtils, SaveLoad_BlendingData) {
  ModelData model = GetTetMeshWithBlending();
  EXPECT_TRUE(model.blending.has_value());
  model::Validate(model, test::ExpectOK{});
  TestSerializationRoundTrip(model);
}

TEST(ModelUtils, Validate_BlendingData) {
  ModelData model = GetTetMeshWithBlending();
  EXPECT_TRUE(model.blending.has_value());
  for (auto& blending : *model.blending) {
    // Source shape name can't be empty
    auto prevName = blending.sourceShape;
    blending.sourceShape.clear();
    model::Validate(model, test::ExpectNotOK{});
    blending.sourceShape = prevName;
    model::Validate(model, test::ExpectOK{}); // Valid again

    // weights.size() and indices.size() must be 2 * GetNumNodes()
    model.mesh->coordinates.push_back(0_r);
    model.mesh->coordinates.push_back(0_r);
    model.mesh->coordinates.push_back(0_r);
    model::Validate(model, test::ExpectNotOK{});
    model.mesh->coordinates.pop_back();
    model.mesh->coordinates.pop_back();
    model.mesh->coordinates.pop_back();
    model::Validate(model, test::ExpectOK{}); // Valid again
    blending.weights.push_back(0_r);
    model::Validate(model, test::ExpectNotOK{});
    blending.weights.pop_back();
    model::Validate(model, test::ExpectOK{}); // Valid again
    blending.indices.push_back(0);
    model::Validate(model, test::ExpectNotOK{});
    blending.indices.pop_back();
    model::Validate(model, test::ExpectOK{}); // Valid again

    // indices can't be negative
    for (int i = 0; i < isize(blending.indices); ++i) {
      int prev = blending.indices[i];
      blending.indices[i] = -1;
      model::Validate(model, test::ExpectNotOK{});
      blending.indices[i] = prev;
    }
    model::Validate(model, test::ExpectOK{}); // Valid again

    // weights must be finite
    for (auto badVal : kNonFiniteValues) {
      for (int i = 0; i < isize(blending.weights); ++i) {
        real prev = blending.weights[i];
        blending.weights[i] = badVal;
        model::Validate(model, test::ExpectNotOK{});
        blending.weights[i] = prev;
      }
    }
    model::Validate(model, test::ExpectOK{}); // Valid again
  }
}

TEST(ModelUtils, Validate_ConstrainedNodes) {
  // Load model with a mesh + constrained nodes
  ModelData model = model::LoadFromBytes(kTetMeshCubeJson, test::ExpectOK{});
  EXPECT_TRUE(model.mesh.has_value());
  EXPECT_TRUE(model.constrainedNodes.has_value());
  model::Validate(model, test::ExpectOK{}); // Starts valid
  auto& mesh = *model.mesh;
  auto& constrainedNodes = *model.constrainedNodes;

  // Node indices must be within range
  int const kTestValues[] = {-1, mesh.GetNumNodes()};
  for (int val : kTestValues) {
    for (int i = 0; i < isize(constrainedNodes); ++i) {
      int prev = constrainedNodes[i];
      constrainedNodes[i] = val;
      model::Validate(model, test::ExpectNotOK{});
      constrainedNodes[i] = prev; // restore
      model::Validate(model, test::ExpectOK{});
    }
  }

  // Can't have constrained nodes without a mesh
  model::Validate(model, test::ExpectOK{});
  model.mesh = std::nullopt;
  model::Validate(model, test::ExpectNotOK{});
}

TEST(ModelUtils, AutoCorrect_ConstrainedNodes) {
  // Load model with a mesh + constrained nodes
  ModelData model = model::LoadFromBytes(kTetMeshCubeJson, test::ExpectOK{});
  EXPECT_TRUE(model.mesh.has_value());
  EXPECT_TRUE(model.constrainedNodes.has_value());
  model::Validate(model, test::ExpectOK{}); // Starts valid
  auto& constrainedNodes = *model.constrainedNodes;

  // AutoCorrect should sort constrained node indices and prune duplicates
  constrainedNodes.clear();
  constrainedNodes = DynamicArray<int>{3, 2, 0, 2, 0, 4, 3, 3};
  model::AutoCorrect(model, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{});
  EXPECT_SPAN_EQ((DynamicArray<int>{0, 2, 3, 4}), constrainedNodes);
}

TEST(ModelUtils, Validate_ElementFrameAxes) {
  ModelData model = model::LoadFromBytes(kPolylineMeshJson, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{}); // Starts valid
  EXPECT_TRUE(model.elementFrameAxes.has_value());
  auto& axes = *model.elementFrameAxes;

  // Array length must be exactly 3 per element
  {
    axes.push_back(0_r);
    model::Validate(model, test::ExpectNotOK{});
    axes.pop_back();
    auto backup = axes.back();
    axes.pop_back();
    model::Validate(model, test::ExpectNotOK{});
    axes.push_back(backup);
    model::Validate(model, test::ExpectOK{}); // Valid again
  }

  // Values must be finite
  for (auto badVal : kNonFiniteValues) {
    for (int i = 0; i < isize(axes); ++i) {
      auto backup = axes[i];
      axes[i] = badVal;
      model::Validate(model, test::ExpectNotOK{});
      axes[i] = backup;
    }
  }
  model::Validate(model, test::ExpectOK{}); // Valid again

  // Axes must be unit length
  for (auto& val : axes) {
    if (val != 0_r) {
      auto backup = val;
      val *= 1.1;
      model::Validate(model, test::ExpectNotOK{});
      val = backup;
    }
  }
  model::Validate(model, test::ExpectOK{}); // Valid again

  // Axes must be orthogonal to the element direction
  for (int i = 0; i < isize(axes); i += 3) {
    auto backup = axes[i];
    axes[i] += 0.1_r; // Nudge in the X direction (same direction as the elements)
    model::Validate(model, test::ExpectNotOK{});
    axes[i] = backup;
  }
  model::Validate(model, test::ExpectOK{}); // Valid again
}

TEST(ModelUtils, AutoCorrect_ElementFrameAxes) {
  ModelData model = model::LoadFromBytes(kPolylineMeshJson, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{}); // Starts valid
  EXPECT_TRUE(model.elementFrameAxes.has_value());

  // Copy the valid axes
  DynamicArray<real> validAxes = *model.elementFrameAxes;

  // Modify the length to fail validation
  for (auto& val : *model.elementFrameAxes) {
    val *= 2_r;
  }
  model::Validate(model, test::ExpectNotOK{});

  // AutoCorrect to fix it
  model::AutoCorrect(model, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{});
  ASSERT_EQ(model.elementFrameAxes->size(), validAxes.size());
  for (int i = 0; i < isize(validAxes); ++i) {
    EXPECT_NEAR_EQ(validAxes[i], (*model.elementFrameAxes)[i]);
  }
}

TEST(ModelUtils, SaveLoad_Box) {
  // Load using LoadFromBytes
  ModelData model = model::LoadFromBytes(kBoxJson, test::ExpectOK{});
  EXPECT_TRUE(model.box.has_value());
  auto& box = *model.box;

  // Check it
  EXPECT_NEAR_EQ(Real3(1_r, 2_r, 3_r), box.center);
  EXPECT_NEAR_EQ(Real3(0.5_r, 1_r, 1.5_r), box.halfExtents);
  EXPECT_NEAR_EQ(Quaternion(0_r, 0_r, 0_r, 1_r), box.rotation);

  // Load using LoadFromFile to prove that it is equivalent.
  auto file = CreateTempFile("model_utils_test", ".mochi", test::ExpectOK{});
  WriteFile(file.Path(), kBoxJson, test::ExpectOK{});
  ModelData modelFromFile = model::LoadFromFile(file.Path().string(), test::ExpectOK{});
  EXPECT_EQ(model, modelFromFile);

  // AutoCorrect and Validate should succeed and change nothing.
  model::AutoCorrect(modelFromFile, test::ExpectOK());
  model::Validate(modelFromFile, test::ExpectOK{});
  EXPECT_EQ(model, modelFromFile); // No change

  // Round-trip serialization
  TestSerializationRoundTrip(model);
}

TEST(ModelUtils, Validate_Box) {
  // Load model with a box
  ModelData model = model::LoadFromBytes(kBoxJson, test::ExpectOK{});
  EXPECT_TRUE(model.box.has_value());
  model::Validate(model, test::ExpectOK{}); // Starts valid
  auto& box = *model.box;

  // Center must be finite
  for (real val : kNonFiniteValues) {
    for (int i = 0; i < 3; ++i) {
      real prev = box.center[i];
      box.center[i] = val;
      model::Validate(model, test::ExpectNotOK{});
      box.center[i] = prev; // restore
      model::Validate(model, test::ExpectOK{});
    }
  }

  // Half extents must be finite and non-negative
  for (real val : kNonFiniteValues) {
    for (int i = 0; i < 3; ++i) {
      real prev = box.halfExtents[i];
      box.halfExtents[i] = val;
      model::Validate(model, test::ExpectNotOK{});
      box.halfExtents[i] = -1_r;
      model::Validate(model, test::ExpectNotOK{});
      box.halfExtents[i] = prev; // restore
      model::Validate(model, test::ExpectOK{});
    }
  }

  // Rotation must be a unit quaternion
  box.rotation = Quaternion{Real4{1_r, 2_r, 3_r, 4_r}};
  model::Validate(model, test::ExpectNotOK{});
  box.rotation = Quaternion{Normalize(Real4{1_r, 2_r, 3_r, 4_r})};
  model::Validate(model, test::ExpectOK{});
}

TEST(ModelUtils, AutoCorrect_Box) {
  // Load model with a box
  ModelData model = model::LoadFromBytes(kBoxJson, test::ExpectOK{});
  EXPECT_TRUE(model.box.has_value());
  model::Validate(model, test::ExpectOK{}); // Starts valid
  auto& box = *model.box;

  // AutoCorrect should normalize the box rotation quaternion
  box.rotation = Quaternion{Real4{1_r, 2_r, 3_r, 4_r}}; // Length 2
  model::Validate(model, test::ExpectNotOK{}); // Invalid before AutoCorrect
  model::AutoCorrect(model, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{});
  EXPECT_NEAR_EQ(Normalize(Real4{1_r, 2_r, 3_r, 4_r}), box.rotation.ToReal4());
}

TEST(ModelUtils, SaveLoad_Plane) {
  // Load using LoadFromBytes
  ModelData model = model::LoadFromBytes(kPlaneJson, test::ExpectOK{});
  EXPECT_TRUE(model.plane.has_value());
  auto& plane = *model.plane;

  // Check it
  EXPECT_NEAR_EQ(Real3(0_r, 1_r, 0_r), plane.GetNormal());
  EXPECT_NEAR_EQ(0.5_r, plane.GetDistanceFromOrigin());

  // Load using LoadFromFile to prove that it is equivalent.
  auto file = CreateTempFile("model_utils_test", ".mochi", test::ExpectOK{});
  WriteFile(file.Path(), kPlaneJson, test::ExpectOK{});
  ModelData modelFromFile = model::LoadFromFile(file.Path().string(), test::ExpectOK{});
  EXPECT_EQ(model, modelFromFile);

  // AutoCorrect and Validate should succeed and change nothing.
  model::AutoCorrect(modelFromFile, test::ExpectOK());
  model::Validate(modelFromFile, test::ExpectOK{});
  EXPECT_EQ(model, modelFromFile); // No change

  // Round-trip serialization
  TestSerializationRoundTrip(model);
}

TEST(ModelUtils, Validate_Plane) {
  // Load model with a plane
  ModelData model = model::LoadFromBytes(kPlaneJson, test::ExpectOK{});
  EXPECT_TRUE(model.plane.has_value());
  model::Validate(model, test::ExpectOK{}); // Starts valid
  auto& plane = *model.plane;

  // Normal must be finite
  for (real val : kNonFiniteValues) {
    auto prev = plane.GetNormal();
    auto distance = plane.GetDistanceFromOrigin();
    for (int i = 0; i < 3; ++i) {
      auto normal = prev;
      normal[i] = val;
      plane = Plane(normal, distance);
      model::Validate(model, test::ExpectNotOK{});
    }
    plane = Plane(prev, plane.GetDistanceFromOrigin()); // restore
    model::Validate(model, test::ExpectOK{}); // Valid again
  }

  // Normal must be unit length
  plane = Plane{Real3{1_r, 2_r, 3_r}, 0.5_r};
  model::Validate(model, test::ExpectNotOK{});

  // Distance must be finite
  for (real val : kNonFiniteValues) {
    plane = Plane{Real3{0_r, 1_r, 0_r}, val};
    model::Validate(model, test::ExpectNotOK{});
  }

  // Zero and negative distance are OK though
  plane = Plane{Real3{0_r, 1_r, 0_r}, 0_r};
  model::Validate(model, test::ExpectOK{});
  plane = Plane{Real3{0_r, 1_r, 0_r}, -1_r};
  model::Validate(model, test::ExpectOK{});
}

TEST(ModelUtils, AutoCorrect_Plane) {
  // Load model with a plane
  ModelData model = model::LoadFromBytes(kPlaneJson, test::ExpectOK{});
  EXPECT_TRUE(model.plane.has_value());
  model::Validate(model, test::ExpectOK{}); // Starts valid
  auto& plane = *model.plane;

  // AutoCorrect should normalize the plane normal
  plane = Plane{Real3{1_r, 2_r, 3_r}, 0.5_r};
  model::AutoCorrect(model, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{});
  EXPECT_NEAR_EQ(Normalize(Real3{1_r, 2_r, 3_r}), model.plane->GetNormal());
  EXPECT_NEAR_EQ(0.5_r, model.plane->GetDistanceFromOrigin());
}

TEST(ModelUtils, SaveLoad_Sphere) {
  // Load using LoadFromBytes
  ModelData model = model::LoadFromBytes(kSphereJson, test::ExpectOK{});
  EXPECT_TRUE(model.sphere.has_value());
  auto& sphere = *model.sphere;

  // Check it
  EXPECT_NEAR_EQ(Real3(1_r, 2_r, 3_r), sphere.GetCenter());
  EXPECT_NEAR_EQ(0.5_r, sphere.GetRadius());

  // Load using LoadFromFile to prove that it is equivalent.
  auto file = CreateTempFile("model_utils_test", ".mochi", test::ExpectOK{});
  WriteFile(file.Path(), kSphereJson, test::ExpectOK{});
  ModelData modelFromFile = model::LoadFromFile(file.Path().string(), test::ExpectOK{});
  EXPECT_EQ(model, modelFromFile);

  // AutoCorrect and Validate should succeed and change nothing.
  model::AutoCorrect(modelFromFile, test::ExpectOK());
  model::Validate(modelFromFile, test::ExpectOK{});
  EXPECT_EQ(model, modelFromFile); // No change

  // Round-trip serialization
  TestSerializationRoundTrip(model);
}

TEST(ModelUtils, Validate_Sphere) {
  // Load model with a sphere
  ModelData model = model::LoadFromBytes(kSphereJson, test::ExpectOK{});
  EXPECT_TRUE(model.sphere.has_value());
  model::Validate(model, test::ExpectOK{}); // Starts valid
  auto& sphere = *model.sphere;

  // Radius must be positive finite.
  for (real val : kNonFiniteValues) {
    sphere = Sphere{Real3{1_r, 2_r, 3_r}, val};
    model::Validate(model, test::ExpectNotOK{});
  }
  sphere = Sphere{Real3{1_r, 2_r, 3_r}, 0_r};
  model::Validate(model, test::ExpectNotOK{});
  sphere = Sphere{Real3{1_r, 2_r, 3_r}, -1_r};
  model::Validate(model, test::ExpectNotOK{});
  model.sphere = Sphere{Real3{1_r, 2_r, 3_r}, 0.5_r};
  model::Validate(model, test::ExpectOK{}); // Valid again

  // Center must be finite
  for (real val : kNonFiniteValues) {
    model.sphere = Sphere{Real3{val, 2_r, 3_r}, 0.5_r};
    model::Validate(model, test::ExpectNotOK{});
    model.sphere = Sphere{Real3{1_r, val, 3_r}, 0.5_r};
    model::Validate(model, test::ExpectNotOK{});
    model.sphere = Sphere{Real3{1_r, 2_r, val}, 0.5_r};
    model::Validate(model, test::ExpectNotOK{});
  }
  model.sphere = Sphere{Real3{1_r, 2_r, 3_r}, 0.5_r};
  model::Validate(model, test::ExpectOK{}); // Valid again
}

TEST(ModelUtils, FlipWindingOrder_PolylineMesh) {
  ModelData model = model::LoadFromBytes(kPolylineMeshJson, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{});

  // Duplicate the model and flip winding order.
  ModelData model2 = model;
  model::FlipWindingOrder(model2, test::ExpectOK{});

  // Polylines have no winding order. FlipWindingOrder should be a no-op.
  EXPECT_EQ(model, model2);
  model::Validate(model2, test::ExpectOK{});
}

TEST(ModelUtils, FlipWindingOrder_TriMesh) {
  // Load a model with a triangle mesh
  ModelData model = model::LoadFromBytes(kTriMeshCubeJson, test::ExpectOK{});

  // Duplicate the model and flip winding order
  ModelData model2 = model;
  model::FlipWindingOrder(model2, test::ExpectOK{});

  // Invariants unaffected by winding flip.
  EXPECT_EQ(model.mesh->nodesPerElement, model2.mesh->nodesPerElement);
  EXPECT_EQ(model.mesh->coordinates, model2.mesh->coordinates);

  auto const nodeAt = [&](int n) {
    return Real3{
        model.mesh->coordinates[3 * n + 0],
        model.mesh->coordinates[3 * n + 1],
        model.mesh->coordinates[3 * n + 2]};
  };

  // Flipping must negate each triangle's signed area vector.
  auto const& c1 = model.mesh->connectivity;
  auto const& c2 = model2.mesh->connectivity;
  ASSERT_EQ(isize(c1), isize(c2));
  ASSERT_EQ(isize(c1) % 3, 0);
  for (int i = 0; i < isize(c1); i += 3) {
    Real3 const n1 =
        Cross(nodeAt(c1[i + 1]) - nodeAt(c1[i + 0]), nodeAt(c1[i + 2]) - nodeAt(c1[i + 0]));
    Real3 const n2 =
        Cross(nodeAt(c2[i + 1]) - nodeAt(c2[i + 0]), nodeAt(c2[i + 2]) - nodeAt(c2[i + 0]));
    EXPECT_GT(Norm(n1), 1e-6_r); // Zero-area triangle would make the sign-flip check below vacuous.
    EXPECT_NEAR_EQ(n2, -n1);
  }
}

TEST(ModelUtils, FlipWindingOrder_TetMesh) {
  // Load a model with a tetrahedral mesh
  ModelData model = model::LoadFromBytes(kTetMeshCubeJson, test::ExpectOK{});

  // Duplicate the model and flip winding order
  ModelData model2 = model;
  model::FlipWindingOrder(model2, test::ExpectOK{});

  // Invariants unaffected by winding flip.
  EXPECT_EQ(model.mesh->nodesPerElement, model2.mesh->nodesPerElement);
  EXPECT_EQ(model.mesh->coordinates, model2.mesh->coordinates);

  auto const nodeAt = [&](int n) {
    return Real3{
        model.mesh->coordinates[3 * n + 0],
        model.mesh->coordinates[3 * n + 1],
        model.mesh->coordinates[3 * n + 2]};
  };

  // Triple product of edge vectors from one vertex. Proportional to signed volume.
  auto const tetTripleProduct = [&](int i0, int i1, int i2, int i3) {
    Real3 const a = nodeAt(i0);
    return Dot(nodeAt(i1) - a, Cross(nodeAt(i2) - a, nodeAt(i3) - a));
  };

  // Flipping must negate each tetrahedron's signed volume.
  auto const& c1 = model.mesh->connectivity;
  auto const& c2 = model2.mesh->connectivity;
  ASSERT_EQ(isize(c1), isize(c2));
  ASSERT_EQ(isize(c1) % 4, 0);
  for (int i = 0; i < isize(c1); i += 4) {
    real const v1 = tetTripleProduct(c1[i + 0], c1[i + 1], c1[i + 2], c1[i + 3]);
    real const v2 = tetTripleProduct(c2[i + 0], c2[i + 1], c2[i + 2], c2[i + 3]);
    EXPECT_GT(
        Abs(v1), 1e-6_r); // Zero-volume tetrahedron would make the sign-flip check below vacuous.
    EXPECT_NEAR_EQ(v2, -v1);
  }
}

static void
TestBakeTransformMesh(ModelData const& model, Real3 const& scale, TransformRT const& rt) {
  EXPECT_TRUE(model.mesh.has_value());
  ModelData model2 = model;
  model::BakeTransform(model2, scale, rt, test::ExpectOK{});

  // These should be unchanged
  EXPECT_EQ(model.mesh->nodesPerElement, model2.mesh->nodesPerElement);
  EXPECT_EQ(model.mesh->skinning, model2.mesh->skinning);

  // Check coordinates
  auto const& nodes = Unflatten<Real3 const>(MakeConstSpan(model.mesh->coordinates));
  auto const& nodes2 = Unflatten<Real3 const>(MakeConstSpan(model2.mesh->coordinates));
  EXPECT_EQ(nodes.size(), nodes2.size());
  for (int i = 0; i < isize(nodes); ++i) {
    EXPECT_NEAR_EQ(rt.TransformPoint(scale * nodes[i]), nodes2[i]);
  }

  // FlipWindingOrder should have been called automatically if the scale performed mirroring on
  // an odd numer of axes. This happens when converting between left handed and right handed
  // coordinate systems.
  int numMirroredAxes = //
      static_cast<int>(scale[0] < 0_r) + //
      static_cast<int>(scale[1] < 0_r) + //
      static_cast<int>(scale[2] < 0_r);
  bool expectReverseWinding = (numMirroredAxes == 1) || (numMirroredAxes == 3);
  if (expectReverseWinding) {
    EXPECT_NE(model.mesh->connectivity, model2.mesh->connectivity);
    model::FlipWindingOrder(model2, test::ExpectOK{});
    EXPECT_EQ(model.mesh->connectivity, model2.mesh->connectivity);
  } else {
    EXPECT_EQ(model.mesh->connectivity, model2.mesh->connectivity);
  }
}

static void TestBakeTransformBox(Box const& original, Real3 const& scale, TransformRT const& rt) {
  ModelData model;
  model.box = original;
  model::BakeTransform(model, scale, rt, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{});

  ASSERT_TRUE(model.box.has_value());
  Box const& baked = *model.box;
  TransformRT const bakedTransform{baked.rotation, baked.center};

  real constexpr kSigns[] = {-1_r, 1_r};
  for (real sx : kSigns) {
    for (real sy : kSigns) {
      for (real sz : kSigns) {
        Real3 const local{
            sx * original.halfExtents[0],
            sy * original.halfExtents[1],
            sz * original.halfExtents[2]};
        Real3 const originalCorner = original.center + original.rotation * local;
        Real3 const expectedCorner = rt.TransformPoint(scale * originalCorner);
        Real3 const bakedLocalCorner = bakedTransform.TransformPointInverse(expectedCorner);
        EXPECT_NEAR_TOL(Abs(bakedLocalCorner), baked.halfExtents, 1e-5_r);
      }
    }
  }
}

static void TestBakeTransformElementFrameAxes(DynamicArray<int> const& connectivity) {
  ModelData model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 2;
  model.mesh->coordinates = DynamicArray<real>{0_r, 0_r, 0_r, 1_r, 1_r, 0_r};
  model.mesh->connectivity = connectivity;
  Real3 const axis = Normalize(Real3{1_r, -1_r, 0_r});
  model.elementFrameAxes = DynamicArray<real>{axis[0], axis[1], axis[2]};
  model::Validate(model, test::ExpectOK{});

  Real3 constexpr kScale = {2_r, 1_r, 3_r};
  TransformRT const rt{
      Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, kPI * 0.5_r), Real3{1_r, 2_r, 3_r}};
  Real3 const expectedAxis = {2_r / Sqrt(5_r), 1_r / Sqrt(5_r), 0_r};

  model::BakeTransform(model, kScale, rt, test::ExpectOK{});

  model::Validate(model, test::ExpectOK{});
  ASSERT_TRUE(model.elementFrameAxes.has_value());
  auto const axes = Unflatten<Real3 const>(MakeConstSpan(*model.elementFrameAxes));
  ASSERT_EQ(axes.size(), 1);
  EXPECT_NEAR_EQ(expectedAxis, axes[0]);

  auto const nodes = Unflatten<Real3 const>(MakeConstSpan(model.mesh->coordinates));
  ASSERT_EQ(nodes.size(), 2);
  Real3 const tangent = Normalize(nodes[1] - nodes[0]);
  EXPECT_NEAR_EQ(0_r, Dot(axes[0], tangent));
}

TEST(ModelUtils, BakeTransform_ElementFrameAxes) {
  TestBakeTransformElementFrameAxes(DynamicArray<int>{0, 1});
  TestBakeTransformElementFrameAxes(DynamicArray<int>{});
}

TEST(ModelUtils, BakeTransform_Box) {
  Box const axisAlignedBox{
      Real3{0.1_r, 0.2_r, 0.3_r}, Real3{0.5_r, 1.0_r, 1.5_r}, Quaternion::Identity()};
  Box const rotatedBox{
      axisAlignedBox.center,
      axisAlignedBox.halfExtents,
      Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, 30_r * kRadiansPerDegree)};

  TransformRT const transforms[] = {
      TransformRT::Identity(),
      TransformRT{
          Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, 20_r * kRadiansPerDegree),
          Real3{1_r, 2_r, 3_r}},
  };

  real constexpr kSigns[] = {-1_r, 1_r};
  for (auto const& rt : transforms) {
    for (real sx : kSigns) {
      for (real sy : kSigns) {
        for (real sz : kSigns) {
          TestBakeTransformBox(axisAlignedBox, Real3{sx * 1_r, sy * 2_r, sz * 3_r}, rt);
          TestBakeTransformBox(rotatedBox, Real3{sx * 2_r, sy * 2_r, sz * 2_r}, rt);
        }
      }
    }
  }
}

TEST(ModelUtils, BakeTransform) {
  ModelData models[2]{
      model::LoadFromBytes(kTriMeshCubeJson, test::ExpectOK{}),
      model::LoadFromBytes(kTriMeshCubeJson, test::ExpectOK{})};

  Real3 constexpr kScaleValues[] = {
      Real3{1_r, 1_r, 1_r},
      Real3{0.1_r, 0.2_r, 0.3_r},
      Real3{-1_r, 2_r, 3_r},
      Real3{1_r, -2_r, 3_r},
      Real3{1_r, 2_r, -3_r},
      Real3{-1_r, -2_r, 3_r},
      Real3{-1_r, 2_r, -3_r},
      Real3{-1_r, -2_r, -3_r},
  };

  Real3 constexpr kTranslationValues[] = {
      Real3{0_r, 0_r, 0_r},
      Real3{0.1_r, 0.2_r, 0.3_r},
      Real3{-0.1_r, 0.2_r, -0.3_r},
  };

  Quaternion const kRotationValues[] = {
      Quaternion::Identity(),
      Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, 0.5_r * kPI),
      Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 0.5_r * kPI),
      Quaternion::FromAxisAngle(Real3{0_r, 0_r, 1_r}, 0.5_r * kPI),
  };

  for (auto const& model : models) {
    for (auto const& scale : kScaleValues) {
      for (auto const& trans : kTranslationValues) {
        for (auto const& rot : kRotationValues) {
          TestBakeTransformMesh(model, scale, TransformRT(rot, trans));
        }
      }
    }
  }
}

static ModelData GetModelWithSdf() {
  // Start with a triangle mesh
  ModelData model = model::LoadFromBytes(kTetMeshCubeJson, test::ExpectOK{});

  // Compute an SDF grid
  GridSdfParams params;
  params.resolutionMode = GridSdfResolutionMode::Explicit;
  params.boundaryPaddingDist = 0.1_r;
  params.resolutionDelta = {0.1_r, 0.1_r, 0.1_r};
  model::BakeSdf(model, params, test::ExpectOK{});

  return model;
}

TEST(ModelUtils, BakeSdf) {
  // Start with a triangle mesh
  ModelData model = GetModelWithSdf();

  // Sanity Check (robust tests for SDF generation are elsewhere)
  ASSERT_TRUE(model.sdf.has_value());
  auto const& sdf = *model.sdf;
  EXPECT_LE(13, sdf.dims[0]); // At least 13 are needed for 0.1m resolution for a 1.2m padded cube.
  EXPECT_LE(13, sdf.dims[1]);
  EXPECT_LE(13, sdf.dims[2]);
  EXPECT_NEAR_EQ(Real3(-0.1_r, -0.1_r, -0.1_r), sdf.bounds.GetMin());
  EXPECT_NEAR_EQ(Real3(1.1_r, 1.1_r, 1.1_r), sdf.bounds.GetMax());
  EXPECT_NEAR_EQ(Real3(0_r, 0_r, 0_r), sdf.negativeValueBounds.GetMin());
  EXPECT_NEAR_EQ(Real3(1_r, 1_r, 1_r), sdf.negativeValueBounds.GetMax());
  EXPECT_EQ(sdf.dims[0] * sdf.dims[1] * sdf.dims[2], isize(sdf.values));
  EXPECT_FALSE(sdf.scale.has_value());
  EXPECT_FALSE(sdf.rotation.has_value());
  EXPECT_FALSE(sdf.translation.has_value());

  // AutoCorrect should not modify it.
  ModelData model2 = model;
  model::AutoCorrect(model2, test::ExpectOK{});
  model::Validate(model2, test::ExpectOK{});
  EXPECT_EQ(model, model2);

  // Round-trip serialization
  TestSerializationRoundTrip(model);
}

TEST(ModelUtils, BakeTransform_Sdf) {
  ModelData model = GetModelWithSdf();
  auto& sdf = *model.sdf;

  auto const uniformScale = Real3{0.1_r, 0.1_r, 0.1_r};
  auto const rotation = Quaternion::FromAxisAngle(Real3{1_r, 0_r, 0_r}, kPI * 0.5_r);
  auto const translation = Real3{1.23_r, 2.34_r, 3.45_r};

  // Bake just uniform scale
  model::BakeTransform(model, uniformScale, TransformRT::Identity(), test::ExpectOK{});
  model::Validate(model, test::ExpectOK{}); // Still valid
  EXPECT_TRUE(model.sdf.has_value());
  EXPECT_TRUE(sdf.scale.has_value());
  EXPECT_FALSE(sdf.rotation.has_value());
  EXPECT_FALSE(sdf.translation.has_value());
  EXPECT_EQ(uniformScale, *sdf.scale);
  sdf.scale = std::nullopt; // clear

  // Bake just rotation
  model::BakeTransform(
      model, Real3{1_r, 1_r, 1_r}, TransformRT{rotation, Real3{}}, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{}); // Still valid
  EXPECT_TRUE(model.sdf.has_value());
  EXPECT_FALSE(sdf.scale.has_value());
  EXPECT_TRUE(sdf.rotation.has_value());
  EXPECT_FALSE(sdf.translation.has_value());
  EXPECT_EQ(rotation, *sdf.rotation);
  sdf.rotation = std::nullopt; // clear

  // Bake just translation
  model::BakeTransform(
      model,
      Real3{1_r, 1_r, 1_r},
      TransformRT{Quaternion::Identity(), translation},
      test::ExpectOK{});
  model::Validate(model, test::ExpectOK{}); // Still valid
  EXPECT_TRUE(model.sdf.has_value());
  EXPECT_FALSE(sdf.scale.has_value());
  EXPECT_FALSE(sdf.rotation.has_value());
  EXPECT_TRUE(sdf.translation.has_value());
  EXPECT_EQ(translation, *sdf.translation);
  sdf.translation = std::nullopt; // clear

  // Bake uniform scale, rotation, translation
  model::BakeTransform(model, uniformScale, TransformRT{rotation, translation}, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{}); // Still valid
  EXPECT_TRUE(model.sdf.has_value());
  EXPECT_TRUE(sdf.scale.has_value());
  EXPECT_TRUE(sdf.rotation.has_value());
  EXPECT_TRUE(sdf.translation.has_value());
  EXPECT_EQ(uniformScale, *sdf.scale);
  EXPECT_EQ(rotation, *sdf.rotation);
  EXPECT_EQ(translation, *sdf.translation);

  // Round-trip serialization (with scale, rotation, and translation)
  TestSerializationRoundTrip(model);

  // Clear previous transforms
  sdf.scale = std::nullopt;
  sdf.rotation = std::nullopt;
  sdf.translation = std::nullopt;

  // Negative scale is OK as long as all axes have equal absolute value.
  Real3 constexpr kNegativeScaleCases[] = {
      Real3{1_r, 1_r, -1_r},
      Real3{1_r, -1_r, 1_r},
      Real3{1_r, -1_r, -1_r},
      Real3{-1_r, 1_r, 1_r},
      Real3{-1_r, 1_r, -1_r},
      Real3{-1_r, -1_r, 1_r},
      Real3{-1_r, -1_r, -1_r}};
  for (auto const& scale : kNegativeScaleCases) {
    model::BakeTransform(model, scale, TransformRT{rotation, translation}, test::ExpectOK{});
    model::Validate(model, test::ExpectOK{}); // Still valid
    EXPECT_TRUE(model.sdf.has_value());
    EXPECT_TRUE(sdf.scale.has_value());
    EXPECT_TRUE(sdf.rotation.has_value());
    EXPECT_TRUE(sdf.translation.has_value());
    EXPECT_EQ(scale, *sdf.scale);
    EXPECT_EQ(rotation, *sdf.rotation);
    EXPECT_EQ(translation, *sdf.translation);
    sdf.scale = std::nullopt;
    sdf.rotation = std::nullopt;
    sdf.translation = std::nullopt;

    // Repeat with 2x the scale
    model::BakeTransform(model, 2_r * scale, TransformRT{rotation, translation}, test::ExpectOK{});
    model::Validate(model, test::ExpectOK{}); // Still valid
    EXPECT_TRUE(model.sdf.has_value());
    EXPECT_TRUE(sdf.scale.has_value());
    EXPECT_TRUE(sdf.rotation.has_value());
    EXPECT_TRUE(sdf.translation.has_value());
    EXPECT_EQ(2_r * scale, *sdf.scale);
    EXPECT_EQ(rotation, *sdf.rotation);
    EXPECT_EQ(translation, *sdf.translation);
    TestSerializationRoundTrip(model);
    sdf.scale = std::nullopt;
    sdf.rotation = std::nullopt;
    sdf.translation = std::nullopt;
  }

  // If we call BakeTransform twice, then the SDF transformations will be concatenated. This is
  // consistent with the mesh data, which will reflect the sequence of transformations.
  model::BakeTransform(model, uniformScale, TransformRT{rotation, translation}, test::ExpectOK{});
  model::BakeTransform(model, uniformScale, TransformRT{rotation, translation}, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{}); // Still valid
  EXPECT_TRUE(model.sdf.has_value());
  EXPECT_TRUE(sdf.scale.has_value());
  EXPECT_TRUE(sdf.rotation.has_value());
  EXPECT_TRUE(sdf.translation.has_value());
  auto expectedTransform = TransformSRT(uniformScale[0], rotation, translation) *
      TransformSRT(uniformScale[0], rotation, translation);
  EXPECT_NEAR_EQ(
      Real3(
          expectedTransform.GetScale(), expectedTransform.GetScale(), expectedTransform.GetScale()),
      *sdf.scale);
  EXPECT_EQ(expectedTransform.GetRotation(), *sdf.rotation);
  EXPECT_EQ(expectedTransform.GetTranslation(), *sdf.translation);
  sdf.scale = std::nullopt;
  sdf.rotation = std::nullopt;
  sdf.translation = std::nullopt;

  // We can also apply mirroring after a previous transformation.
  model::BakeTransform(model, uniformScale, TransformRT{rotation, translation}, test::ExpectOK{});
  model::BakeTransform(
      model, Real3{2_r, -2_r, 2_r}, TransformRT{rotation, translation}, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{}); // Still valid
  EXPECT_TRUE(model.sdf.has_value());
  EXPECT_TRUE(sdf.scale.has_value());
  EXPECT_TRUE(sdf.rotation.has_value());
  EXPECT_TRUE(sdf.translation.has_value());

  // Calculate a matrix that would apply the above sequence of transformations.
  VMatrix4x4r transformMat1 = Dot4x4(
      ToVMatrix4x4(TransformRT{rotation, translation}),
      VDiagonalMatrix<4>(ToSimd(uniformScale, 1_r)));
  VMatrix4x4r transformMat2 = Dot4x4(
      ToVMatrix4x4(TransformRT{rotation, translation}),
      VDiagonalMatrix<4>(Vec4r{2_r, -2_r, 2_r, 1_r}));
  VMatrix4x4r expectedMat = Dot4x4(transformMat2, transformMat1);

  // We will consider the stored data to be correct as long as it results in the same matrix.
  VMatrix4x4r actualMat = Dot4x4(
      ToVMatrix4x4(TransformRT{*sdf.rotation, *sdf.translation}),
      VDiagonalMatrix<4>(ToSimd(*sdf.scale, 1_r)));
  EXPECT_NEAR_TOL(actualMat, expectedMat, Vec4r(1e-5_r));

  // Clear transform
  sdf.scale = std::nullopt;
  sdf.rotation = std::nullopt;
  sdf.translation = std::nullopt;

  // Repeat with a mirroring after a non-90-degree rotation.
  Quaternion xRot30 = Quaternion::FromAxisAngle(
      Real3{1_r, 0_r, 0_r}, 30_r * kRadiansPerDegree); // 30 deg rotation about X
  model::BakeTransform(
      model, Real3{1_r, 1_r, 1_r}, TransformRT{xRot30, translation}, test::ExpectOK{});
  model::BakeTransform(model, Real3{2_r, -2_r, 2_r}, TransformRT{translation}, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{}); // Still valid
  EXPECT_TRUE(model.sdf.has_value());
  EXPECT_TRUE(sdf.scale.has_value());
  EXPECT_TRUE(sdf.rotation.has_value());
  EXPECT_TRUE(sdf.translation.has_value());
  transformMat1 = ToVMatrix4x4(TransformRT{xRot30, translation});
  transformMat2 = Dot4x4(
      ToVMatrix4x4(TransformRT{translation}), VDiagonalMatrix<4>(Vec4r{2_r, -2_r, 2_r, 1_r}));
  expectedMat = Dot4x4(transformMat2, transformMat1);
  actualMat = Dot4x4(
      ToVMatrix4x4(TransformRT{*sdf.rotation, *sdf.translation}),
      VDiagonalMatrix<4>(ToSimd(*sdf.scale, 1_r)));
  EXPECT_NEAR_TOL(actualMat, expectedMat, Vec4r(1e-5_r));

  // If scale is not uniform by absolute value, then the SDF data will be discarded (not an error).
  for (int i = 0; i < 3; ++i) {
    ModelData model2 = model;
    Real3 badScale = {1_r, 1_r, 1_r};
    badScale[i] += 0.01_r;
    model::BakeTransform(model2, badScale, TransformRT{}, test::ExpectOK{});
    EXPECT_FALSE(model2.sdf.has_value());
  }
}

TEST(ModelUtils, BakeTransform_Plane) {
  Real3 const normal = Normalize(Real3{1_r, 2_r, 3_r});
  real constexpr kDistance = 0.5_r;
  Real3 const tangent = Normalize(Cross(normal, Real3{1_r, 0_r, 0_r}));
  TransformRT const rt{
      Quaternion::FromAxisAngle(Real3{0_r, 1_r, 0_r}, 30_r * kRadiansPerDegree),
      Real3{1_r, 2_r, 3_r}};

  Real3 constexpr kScaleValues[] = {
      Real3{1_r, 1_r, 1_r},
      Real3{2_r, 2_r, 2_r},
      Real3{-2_r, 2_r, 2_r},
      Real3{1_r, 2_r, 3_r},
      Real3{-1_r, 2_r, 3_r}};

  for (auto const& scale : kScaleValues) {
    ModelData model;
    model.plane = Plane{normal, kDistance};
    model::BakeTransform(model, scale, rt, test::ExpectOK{});
    model::Validate(model, test::ExpectOK{});
    Plane const& baked = *model.plane;

    // Points on the original plane must land on the transformed plane.
    for (Real3 const& point : {normal * kDistance, normal * kDistance + tangent}) {
      Real3 const moved = rt.TransformPoint(scale * point);
      EXPECT_NEAR_TOL(Dot(baked.GetNormal(), moved), baked.GetDistanceFromOrigin(), 1e-5_r);
    }

    // Mirroring must not turn the plane inside out: the positive side stays positive.
    Real3 const outside = rt.TransformPoint(scale * (normal * (kDistance + 1_r)));
    EXPECT_GT(Dot(baked.GetNormal(), outside), baked.GetDistanceFromOrigin());
  }
}

TEST(ModelUtils, BakeCoordinateSpaceTransform_Model) {
  // FDL keeps handedness and units. Unity (RUF) measured in centimeters flips handedness and
  // rescales, so between them every kind of model content is checked against mirroring and a
  // change of units. `toDir` converts a direction, which carries no units.
  auto const runCase = [](CoordinateSpace const& toSpace, auto const& toDir) {
    real const unitScale = toSpace.unitsPerMeter;
    bool const flipsHandedness =
        CoordinateSpaceConverter{CoordinateSpace::Default(), toSpace}.FlipsHandedness();
    auto const toPoint = [&](Real3 const& v) { return unitScale * toDir(v); };

    // A model holds only one kind of shape, so check each one with its own model.
    auto const convert = [&](ModelData& model) {
      model::Validate(model, test::ExpectOK{});
      model::BakeCoordinateSpaceTransform(
          model, CoordinateSpace::Default(), toSpace, test::ExpectOK{});
      model::Validate(model, test::ExpectOK{});
    };

    ModelData const original = model::LoadFromBytes(kTriMeshCubeJson, test::ExpectOK{});
    ModelData meshModel = original;
    convert(meshModel);
    auto const nodes = Unflatten<Real3 const>(MakeConstSpan(original.mesh->coordinates));
    auto const converted = Unflatten<Real3 const>(MakeConstSpan(meshModel.mesh->coordinates));
    ASSERT_EQ(nodes.size(), converted.size());
    for (int i = 0; i < isize(nodes); ++i) {
      EXPECT_EQ(toPoint(nodes[i]), converted[i]);
    }
    // BakeCoordinateSpaceTransform_FlipsHandedness covers the winding order exhaustively.
    EXPECT_EQ(flipsHandedness, meshModel.mesh->connectivity != original.mesh->connectivity);

    Real3 const center{1_r, 2_r, 3_r};
    ModelData sphereModel;
    sphereModel.sphere = Sphere{center, 0.5_r};
    convert(sphereModel);
    EXPECT_EQ(Sphere(toPoint(center), 0.5_r * unitScale), *sphereModel.sphere);

    Real3 const normal = Normalize(center);
    ModelData planeModel;
    planeModel.plane = Plane{normal, 0.5_r};
    convert(planeModel);
    EXPECT_EQ(toDir(normal), planeModel.plane->GetNormal());
    EXPECT_EQ(0.5_r * unitScale, planeModel.plane->GetDistanceFromOrigin());

    Real3 const halfExtents{0.5_r, 1_r, 1.5_r};
    ModelData boxModel;
    boxModel.box = Box{center, halfExtents, Quaternion::Identity()};
    convert(boxModel);
    EXPECT_EQ(toPoint(center), boxModel.box->center);
    EXPECT_EQ(halfExtents * unitScale, boxModel.box->halfExtents);

    // A box stores its orientation as a quaternion, which cannot represent 90 degrees exactly.
    // Negating one of its local axes describes the same box, so mirroring is free to flip one:
    // compare the axes up to sign, which still pins down the permutation.
    for (int axis = 0; axis < 3; ++axis) {
      Real3 const localAxis = BasisVector<real, 3>(axis);
      EXPECT_NEAR_EQ(Abs(toDir(localAxis)), Abs(boxModel.box->rotation * localAxis));
    }

    // A mirrored box that is already rotated cannot keep its local axes, so it takes the general
    // matrix decomposition path instead of composing quaternions.
    Quaternion const tilt = Quaternion::FromAxisAngle(Normalize(center), 0.7_r);
    ModelData tiltedBoxModel;
    tiltedBoxModel.box = Box{center, halfExtents, tilt};
    convert(tiltedBoxModel);
    EXPECT_EQ(toPoint(center), tiltedBoxModel.box->center);
    EXPECT_EQ(halfExtents * unitScale, tiltedBoxModel.box->halfExtents);
    for (int axis = 0; axis < 3; ++axis) {
      Real3 const localAxis = BasisVector<real, 3>(axis);
      EXPECT_NEAR_EQ(Abs(toDir(tilt * localAxis)), Abs(tiltedBoxModel.box->rotation * localAxis));
    }

    // The SDF grid is not resampled; the conversion is stored as the grid-to-model transform
    // instead. A point in grid space must therefore still land where the conversion puts it. This
    // is the only content type that pins down which axis carries the reflection, because every
    // other one is symmetric under negating an axis.
    ModelData sdfModel = GetModelWithSdf();
    convert(sdfModel);
    ASSERT_TRUE(sdfModel.sdf.has_value()); // It must not have been silently discarded.
    TransformRT const gridFromModel{
        sdfModel.sdf->rotation.value_or(Quaternion::Identity()),
        sdfModel.sdf->translation.value_or(Real3{})};
    Real3 const gridPoint{0.25_r, -0.5_r, 0.75_r};
    EXPECT_NEAR_TOL(
        toPoint(gridPoint),
        gridFromModel.TransformPoint(
            sdfModel.sdf->scale.value_or(Real3{1_r, 1_r, 1_r}) * gridPoint),
        1e-5_r * unitScale);
  };

  runCase(CoordinateSpace{CoordinateSpaceAxes::FDL, 1_r}, [](Real3 const& v) {
    return Real3{v[0], -v[2], v[1]};
  });
  runCase(CoordinateSpace{CoordinateSpaceAxes::RUF, 100_r}, [](Real3 const& v) {
    return Real3{-v[1], v[2], v[0]};
  });
}

TEST(ModelUtils, BakeCoordinateSpaceTransform_FlipsHandedness) {
  ModelData const original = model::LoadFromBytes(kTriMeshCubeJson, test::ExpectOK{});
  auto const& axesEnumInfo = SReflect::GetTypeInfo<CoordinateSpaceAxes>();

  for (auto const& from : axesEnumInfo._items) {
    for (auto const& to : axesEnumInfo._items) {
      CoordinateSpace const fromSpace{static_cast<CoordinateSpaceAxes>(from._value), 1_r};
      CoordinateSpace const toSpace{static_cast<CoordinateSpaceAxes>(to._value), 1_r};
      CoordinateSpaceConverter converter{fromSpace, toSpace};

      MeshData mesh = *original.mesh;
      model::BakeCoordinateSpaceTransform(mesh, fromSpace, toSpace, test::ExpectOK{});

      ModelData model = original;
      model::BakeCoordinateSpaceTransform(model, fromSpace, toSpace, test::ExpectOK{});
      EXPECT_EQ(mesh, *model.mesh); // Both overloads must agree.

      auto const originalCoords = Unflatten<Real3 const>(MakeConstSpan(original.mesh->coordinates));
      auto const convertedCoords = Unflatten<Real3 const>(MakeConstSpan(mesh.coordinates));
      ASSERT_EQ(originalCoords.size(), convertedCoords.size());
      for (int i = 0; i < isize(originalCoords); ++i) {
        EXPECT_EQ(converter.TranslationToOutput(originalCoords[i]), convertedCoords[i]);
      }

      if (converter.FlipsHandedness()) {
        model::FlipWindingOrder(mesh, test::ExpectOK{}); // Flip it back
      }
      EXPECT_EQ(original.mesh->connectivity, mesh.connectivity);
    }
  }
}

TEST(ModelUtils, BakeCoordinateSpaceTransform_LosslessRoundTrip) {
  std::string_view const models[] = {kTriMeshCubeJson, kTetMeshCubeJson, kPolylineMeshJson};
  auto const& axesEnumInfo = SReflect::GetTypeInfo<CoordinateSpaceAxes>();

  for (auto const& json : models) {
    ModelData const original = model::LoadFromBytes(json, test::ExpectOK{});
    for (auto const& from : axesEnumInfo._items) {
      for (auto const& to : axesEnumInfo._items) {
        CoordinateSpace const fromSpace{static_cast<CoordinateSpaceAxes>(from._value), 1_r};
        CoordinateSpace const toSpace{static_cast<CoordinateSpaceAxes>(to._value), 1_r};
        ModelData model = original;
        model::BakeCoordinateSpaceTransform(model, fromSpace, toSpace, test::ExpectOK{});
        model::BakeCoordinateSpaceTransform(model, toSpace, fromSpace, test::ExpectOK{});
        EXPECT_EQ(original, model);
        // Repeat with the MeshData overload
        model::BakeCoordinateSpaceTransform(*model.mesh, fromSpace, toSpace, test::ExpectOK{});
        model::BakeCoordinateSpaceTransform(*model.mesh, toSpace, fromSpace, test::ExpectOK{});
        EXPECT_EQ(original, model);
      }
    }
  }
}

TEST(ModelUtils, BakeCoordinateSpaceTransform_ElementFrameAxes) {
  Real3 const axis = Normalize(Real3{1_r, -1_r, -1_r}); // Perpendicular to the element below.
  ModelData srcModel;
  srcModel.mesh.emplace();
  srcModel.mesh->nodesPerElement = 2;
  srcModel.mesh->coordinates = DynamicArray<real>{0_r, 0_r, 0_r, 3_r, 2_r, 1_r};
  srcModel.mesh->connectivity = DynamicArray<int>{0, 1};
  srcModel.elementFrameAxes = DynamicArray<real>{axis[0], axis[1], axis[2]};
  model::Validate(srcModel, test::ExpectOK{});

  auto const& axesEnumInfo = SReflect::GetTypeInfo<CoordinateSpaceAxes>();
  for (auto const& from : axesEnumInfo._items) {
    for (auto const& to : axesEnumInfo._items) {
      CoordinateSpace const fromSpace{
          static_cast<CoordinateSpaceAxes>(from._value), 1_r}; // m scale
      CoordinateSpace const toSpace{static_cast<CoordinateSpaceAxes>(to._value), 100_r}; // cm scale
      CoordinateSpaceConverter converter(fromSpace, toSpace);
      auto model = srcModel; // copy

      // A change of units must neither rotate nor renormalize a direction.
      model::BakeCoordinateSpaceTransform(model, fromSpace, toSpace, test::ExpectOK{});
      auto const axes = Unflatten<Real3 const>(MakeConstSpan(*model.elementFrameAxes));
      EXPECT_EQ(converter.DirectionToOutput(axis), axes[0]);
    }
  }
}

TEST(ModelUtils, BakeCoordinateSpaceTransform_NoOpAndValidation) {
  ModelData const original = model::LoadFromBytes(kTriMeshCubeJson, test::ExpectOK{});
  ModelData model = original;
  MeshData mesh = *original.mesh;

  // Identical spaces return early without touching the data.
  auto const unity = CoordinateSpace::Unity();
  model::BakeCoordinateSpaceTransform(model, unity, unity, test::ExpectOK{});
  model::BakeCoordinateSpaceTransform(mesh, unity, unity, test::ExpectOK{});

  // An invalid space is rejected before anything is modified.
  CoordinateSpace const invalid{CoordinateSpaceAxes::Default, 0_r};
  model::BakeCoordinateSpaceTransform(model, invalid, unity, test::ExpectNotOK{});
  model::BakeCoordinateSpaceTransform(model, unity, invalid, test::ExpectNotOK{});
  model::BakeCoordinateSpaceTransform(mesh, invalid, unity, test::ExpectNotOK{});
  model::BakeCoordinateSpaceTransform(mesh, unity, invalid, test::ExpectNotOK{});

  EXPECT_EQ(original, model);
  EXPECT_EQ(*original.mesh, mesh);
}

TEST(ModelUtils, Validate_Sdf) {
  ModelData model = GetModelWithSdf();
  EXPECT_TRUE(model.sdf.has_value());
  model::Validate(model, test::ExpectOK{}); // Starts valid

  // Dimensions must be at least 2x2x2
  Int3 constexpr kInvalidDims[] = {
      Int3{0, 2, 2}, Int3{2, 0, 2}, Int3{2, 2, 0}, Int3{-1, 2, 2}, Int3{2, -1, 2}, Int3{2, 2, -1}};
  for (Int3 dims : kInvalidDims) {
    ModelData model2 = model;
    model2.sdf->dims = dims;
    model2.sdf->values.resize(Abs(dims[0] * dims[1] * dims[2]));
    model::Validate(model2, test::ExpectNotOK{});
  }
  {
    // Proof that 2x2x2 is valid
    ModelData model2 = model;
    model2.sdf->dims = Int3{2, 2, 2};
    model2.sdf->values.resize(8);
    model::Validate(model2, test::ExpectOK{});
  }

  // Values array size must match product of dimensions
  {
    ModelData model2 = model;
    model2.sdf->values.push_back(0_r);
    model::Validate(model2, test::ExpectNotOK{});
    model2.sdf->values.pop_back();
    real prev = model2.sdf->values.back();
    model2.sdf->values.pop_back();
    model::Validate(model2, test::ExpectNotOK{});
    model2.sdf->values.push_back(prev);
    model::Validate(model2, test::ExpectOK{}); // Valid again
  }

  // Values must be finite
  for (real val : kNonFiniteValues) {
    for (int i = 0; i < isize(model.sdf->values); ++i) {
      real prev = model.sdf->values[i];
      model.sdf->values[i] = val;
      model::Validate(model, test::ExpectNotOK{});
      model.sdf->values[i] = prev; // restore
    }
  }
  model::Validate(model, test::ExpectOK{}); // Valid again

  // Bounds must be valid (positive volume, finite)
  Aabb const kInvalidBounds[] = {
      Aabb{Real3{0_r, 0_r, 0_r}, Real3{0_r, 1_r, 1_r}},
      Aabb{Real3{0_r, 0_r, 0_r}, Real3{1_r, 0_r, 1_r}},
      Aabb{Real3{0_r, 0_r, 0_r}, Real3{1_r, 1_r, 0_r}},
      Aabb{Real3{0_r, 0_r, 0_r}, Real3{-1_r, 1_r, 1_r}},
      Aabb{Real3{0_r, 0_r, 0_r}, Real3{1_r, -1_r, 1_r}},
      Aabb{Real3{0_r, 0_r, 0_r}, Real3{1_r, 1_r, -1_r}},
      Aabb{Real3{0_r, 0_r, 0_r}, Real3{kInf, kInf, kInf}},
  };
  for (auto bounds : kInvalidBounds) {
    auto prev = model.sdf->bounds;
    model.sdf->bounds = bounds;
    model::Validate(model, test::ExpectNotOK{});
    model.sdf->bounds = prev; // restore
  }
  model::Validate(model, test::ExpectOK{}); // Valid again

  // negativeValueBounds must also be valid
  for (auto bounds : kInvalidBounds) {
    auto prev = model.sdf->negativeValueBounds;
    model.sdf->negativeValueBounds = bounds;
    model::Validate(model, test::ExpectNotOK{});
    model.sdf->negativeValueBounds = prev; // restore
  }
  model::Validate(model, test::ExpectOK{}); // Valid again

  // negativeValueBounds must fit within the grid bounds
  {
    auto prev = model.sdf->negativeValueBounds;
    model.sdf->negativeValueBounds = model.sdf->bounds;
    model::Validate(model, test::ExpectOK{}); // This is OK
    Real3 min = model.sdf->bounds.GetMin();
    Real3 max = model.sdf->bounds.GetMax();
    for (int i = 0; i < 3; ++i) {
      min[i] -= 1e-7_r; // Slightingly outside model.sdf->bounds
      model.sdf->negativeValueBounds = Aabb{min, max};
      model::Validate(model, test::ExpectNotOK{});
      min = model.sdf->bounds.GetMin();
      max[i] += 1e-7_r; // Slightingly outside model.sdf->bounds
      model.sdf->negativeValueBounds = Aabb{min, max};
      model::Validate(model, test::ExpectNotOK{});
    }
    model.sdf->negativeValueBounds = prev;
    model::Validate(model, test::ExpectOK{}); // Valid again
  }

  // Optional scale must be finite, non-zero, and uniform absolute value.
  for (int i = 0; i < 3; ++i) {
    model.sdf->scale = Real3{2_r, 2_r, 2_r};
    model::Validate(model, test::ExpectOK{});
    (*model.sdf->scale)[i] = std::numeric_limits<real>::infinity();
    model::Validate(model, test::ExpectNotOK{}); // not finite
    (*model.sdf->scale)[i] = 0_r;
    model::Validate(model, test::ExpectNotOK{}); // zero scale
    (*model.sdf->scale)[i] = 0.5_r;
    model::Validate(model, test::ExpectNotOK{}); // not uniform absolute value
    (*model.sdf->scale)[i] = -2_r;
    model::Validate(model, test::ExpectOK{}); // negative scale is OK if uniform absolute value
  }
  model.sdf->scale = std::nullopt; // no scale is OK.
  model::Validate(model, test::ExpectOK{});

  // Optional rotation must be finite and unit quaternion
  {
    model.sdf->rotation = Quaternion{Real4{kNonFiniteValues[0], 0_r, 0_r, 1_r}};
    model::Validate(model, test::ExpectNotOK{}); // not finite
    model.sdf->rotation = Quaternion{0_r, 0_r, 0_r, 0_r};
    model::Validate(model, test::ExpectNotOK{}); // zero magnitude
    model.sdf->rotation = Quaternion{Real4{1_r, 2_r, 3_r, 4_r}};
    model::Validate(model, test::ExpectNotOK{}); // Not unit length
    model.sdf->rotation = Quaternion{Normalize(Real4{1_r, 2_r, 3_r, 4_r})};
    model::Validate(model, test::ExpectOK{});
    model.sdf->rotation = std::nullopt;
    model::Validate(model, test::ExpectOK{});
  }

  // Optional translation must be finite
  {
    for (real val : kNonFiniteValues) {
      for (int i = 0; i < 3; ++i) {
        model.sdf->translation = Real3{};
        (*model.sdf->translation)[i] = val;
        model::Validate(model, test::ExpectNotOK{});
      }
    }
    model.sdf->translation = std::nullopt;
    model::Validate(model, test::ExpectOK{});
  }
}

TEST(ModelUtils, AutoCorrect_Sdf) {
  ModelData model = GetModelWithSdf();
  EXPECT_TRUE(model.sdf.has_value());
  model::AutoCorrect(model, test::ExpectOK{}); // Starts valid
  auto& sdf = *model.sdf;

  // AutoCorrect should normalize rotation if it is not unit length
  sdf.rotation = Quaternion{Real4{1_r, 2_r, 3_r, 4_r}};
  model::AutoCorrect(model, test::ExpectOK{});
  EXPECT_TRUE(sdf.rotation.has_value());
  EXPECT_NEAR_EQ(Normalize(Real4{1_r, 2_r, 3_r, 4_r}), sdf.rotation->ToReal4());
  sdf.rotation = std::nullopt;
}

TEST(ModelUtils, Serialize_MaterialData) {
  ModelData model;
  model.material.emplace(PerElementSoftMaterialData{});

  // This data would not pass validation, but it round-trip serialization should be lossless.
  model.material = PerElementSoftMaterialData{};
  model.material->youngsModulus = DynamicArray<real>{1_r, 2_r};
  model.material->poissonRatio = DynamicArray<real>{2_r, 3_r};
  model.material->anisoAlpha = DynamicArray<real>{3_r, 4_r};
  model.material->anisoLength = DynamicArray<real>{4_r, 5_r};
  model.material->anisoTheta = DynamicArray<real>{5_r, 6_r};
  model.material->anisoPhi = DynamicArray<real>{7_r, 8_r};
  model.material->arapStiffness = DynamicArray<real>{8_r, 9_r};
  model.material->shapeTargetTensor = DynamicArray<real>{
      9_r,
      10_r,
      11_r,
      12_r,
      13_r,
      14_r}; // Size must be a multiple of 6 to serialize as a 2D dataset in H5 format.
  TestSerializationRoundTrip(model, /*shouldValidate*/ false);
}

TEST(ModelUtils, Validate_MaterialData) {
  // Load model with a tetrahedral mesh
  ModelData model = model::LoadFromBytes(kTetMeshCubeJson, test::ExpectOK{});
  EXPECT_TRUE(model.mesh.has_value());
  int const numElements = model.mesh->GetNumElements();
  EXPECT_EQ(5, numElements);

  // Helper to create valid soft material data for a given material type
  auto createValidMaterial = [numElements](SoftMaterialType type) {
    PerElementSoftMaterialData material;
    material.type = type;
    if ((type == SoftMaterialType::NeoHookean) || (type == SoftMaterialType::StVenantKirchhoff) ||
        (type == SoftMaterialType::LinearElastic) || (type == SoftMaterialType::ActiveNeoHookean)) {
      material.youngsModulus = DynamicArray<real>(numElements, 1e5_r);
      material.poissonRatio = DynamicArray<real>(numElements, 0.45_r);
    }
    if (type == SoftMaterialType::ActiveNeoHookean) {
      material.anisoAlpha = DynamicArray<real>(numElements, 1000_r);
      material.anisoLength = DynamicArray<real>(numElements, 1_r);
      material.anisoTheta = DynamicArray<real>(numElements, 0_r);
      material.anisoPhi = DynamicArray<real>(numElements, 0_r);
    }
    if (type == SoftMaterialType::Arap || type == SoftMaterialType::ActiveShapeTargetingArap) {
      material.arapStiffness = DynamicArray<real>(numElements, 1000_r);
    }
    if (type == SoftMaterialType::ActiveShapeTargetingArap) {
      material.shapeTargetTensor = DynamicArray<real>(numElements * 6, 0_r);
    }
    return material;
  };

  // Test validation for each material type
  static_assert(
      static_cast<int>(SoftMaterialType::Count) == 6,
      "If you add a material type, then plese update this code.");
  for (int iMaterial = 0; iMaterial < static_cast<int>(SoftMaterialType::Count); ++iMaterial) {
    auto materialType = static_cast<SoftMaterialType>(iMaterial);

    model.material = createValidMaterial(materialType);
    model::Validate(model, test::ExpectOK{}); // Starts valid
    auto& material = *model.material;

    if ((materialType == SoftMaterialType::NeoHookean) ||
        (materialType == SoftMaterialType::StVenantKirchhoff) ||
        (materialType == SoftMaterialType::LinearElastic) ||
        (materialType == SoftMaterialType::ActiveNeoHookean)) {
      // youngsModulus must be finite and positive
      for (int i = 0; i < numElements; ++i) {
        for (real val : kNonFiniteValues) {
          real prev = material.youngsModulus[i];
          material.youngsModulus[i] = val;
          model::Validate(model, test::ExpectNotOK{});
          material.youngsModulus[i] = prev; // restore
        }
        // Zero or negative is invalid
        real prev = material.youngsModulus[i];
        material.youngsModulus[i] = 0_r;
        model::Validate(model, test::ExpectNotOK{});
        material.youngsModulus[i] = -1_r;
        model::Validate(model, test::ExpectNotOK{});
        material.youngsModulus[i] = prev; // restore
      }
      model::Validate(model, test::ExpectOK{}); // Valid again

      // Array size must match numElements
      material.youngsModulus.push_back(1e5_r);
      model::Validate(model, test::ExpectNotOK{});
      material.youngsModulus.pop_back();
      model::Validate(model, test::ExpectOK{}); // Valid again

      // poissonRatio must be in (-1, 0.5)
      for (int i = 0; i < numElements; ++i) {
        for (real val : kNonFiniteValues) {
          real prev = material.poissonRatio[i];
          material.poissonRatio[i] = val;
          model::Validate(model, test::ExpectNotOK{});
          material.poissonRatio[i] = prev; // restore
        }
        // Out of range values
        real prev = material.poissonRatio[i];
        material.poissonRatio[i] = -1_r;
        model::Validate(model, test::ExpectNotOK{});
        material.poissonRatio[i] = 0.5_r;
        model::Validate(model, test::ExpectNotOK{});
        material.poissonRatio[i] = prev; // restore
      }
      model::Validate(model, test::ExpectOK{}); // Valid again

      // Array size must match numElements
      material.poissonRatio.push_back(0.45_r);
      model::Validate(model, test::ExpectNotOK{});
      material.poissonRatio.pop_back();
      model::Validate(model, test::ExpectOK{}); // Valid again
    }

    // Test anisotropic parameters (only valid for ActiveNeoHookean)
    if (materialType == SoftMaterialType::ActiveNeoHookean) {
      // anisoAlpha must be finite
      for (int i = 0; i < numElements; ++i) {
        for (real val : kNonFiniteValues) {
          real prev = material.anisoAlpha[i];
          material.anisoAlpha[i] = val;
          model::Validate(model, test::ExpectNotOK{});
          material.anisoAlpha[i] = prev; // restore
        }
      }
      model::Validate(model, test::ExpectOK{}); // Valid again

      // anisoLength must be finite and non-negative
      for (int i = 0; i < numElements; ++i) {
        for (real val : kNonFiniteValues) {
          real prev = material.anisoLength[i];
          material.anisoLength[i] = val;
          model::Validate(model, test::ExpectNotOK{});
          material.anisoLength[i] = prev; // restore
        }
        // Negative is invalid
        real prev = material.anisoLength[i];
        material.anisoLength[i] = -1_r;
        model::Validate(model, test::ExpectNotOK{});
        material.anisoLength[i] = prev; // restore
      }
      model::Validate(model, test::ExpectOK{}); // Valid again

      // anisoTheta and anisoPhi must be finite
      for (int i = 0; i < numElements; ++i) {
        for (real val : kNonFiniteValues) {
          real prev = material.anisoTheta[i];
          material.anisoTheta[i] = val;
          model::Validate(model, test::ExpectNotOK{});
          material.anisoTheta[i] = prev;

          prev = material.anisoPhi[i];
          material.anisoPhi[i] = val;
          model::Validate(model, test::ExpectNotOK{});
          material.anisoPhi[i] = prev;
        }
      }
      model::Validate(model, test::ExpectOK{}); // Valid again
    }

    // Test mu (only valid for Arap and ActiveShapeTargetingArap)
    if (materialType == SoftMaterialType::Arap ||
        materialType == SoftMaterialType::ActiveShapeTargetingArap) {
      // ARAP stiffness must be finite
      for (int i = 0; i < numElements; ++i) {
        for (real val : kNonFiniteValues) {
          real prev = material.arapStiffness[i];
          material.arapStiffness[i] = val;
          model::Validate(model, test::ExpectNotOK{});
          material.arapStiffness[i] = prev; // restore
        }
      }
      model::Validate(model, test::ExpectOK{}); // Valid again

      // Array size must match numElements
      material.arapStiffness.push_back(1000_r);
      model::Validate(model, test::ExpectNotOK{});
      material.arapStiffness.pop_back();
      model::Validate(model, test::ExpectOK{}); // Valid again
    }

    // Test shapeTargetTensor (only valid for ActiveShapeTargetingArap)
    if (materialType == SoftMaterialType::ActiveShapeTargetingArap) {
      // shapeTargetTensor must be finite
      for (int i = 0; i < isize(material.shapeTargetTensor); ++i) {
        for (real val : kNonFiniteValues) {
          real prev = material.shapeTargetTensor[i];
          material.shapeTargetTensor[i] = val;
          model::Validate(model, test::ExpectNotOK{});
          material.shapeTargetTensor[i] = prev; // restore
        }
      }
      model::Validate(model, test::ExpectOK{}); // Valid again

      // Size must be numElements * 6
      material.shapeTargetTensor.push_back(0_r);
      model::Validate(model, test::ExpectNotOK{});
      material.shapeTargetTensor.pop_back();
      model::Validate(model, test::ExpectOK{}); // Valid again
    }
  }
}

static ModelData GetTetMeshModelWithVisualMesh(int weightsPerNode) {
  // Load a valid model with a tetrahedral mesh
  ModelData tetMeshModel = model::LoadFromBytes(kTetMeshCubeJson, test::ExpectOK{});
  EXPECT_TRUE(tetMeshModel.mesh.has_value());

  // Load a valid model with a triangular mesh
  ModelData triMeshModel = model::LoadFromBytes(kTriMeshCubeJson, test::ExpectOK{});
  EXPECT_TRUE(triMeshModel.mesh.has_value());
  auto const& triMesh = *triMeshModel.mesh;

  // Normally a visual mesh would be higher resolution than the surface of the simulation mesh, but
  // not in this case.
  EXPECT_EQ(isize(tetMeshModel.mesh->coordinates), isize(triMesh.coordinates));

  // Build a visual mesh. For simplicity, each visual node is weighted to one simulation node.
  MeshData visualMesh = triMesh;
  visualMesh.skinning = SkinningData{};
  visualMesh.skinning->weightsPerNode = weightsPerNode;
  visualMesh.skinning->indices.resize(weightsPerNode * triMesh.GetNumNodes());
  visualMesh.skinning->weights.resize(weightsPerNode * triMesh.GetNumNodes());
  for (int i = 0; i < triMesh.GetNumNodes(); ++i) {
    visualMesh.skinning->indices[weightsPerNode * i] = i;
    visualMesh.skinning->weights[weightsPerNode * i] = 1_r;
  }

  tetMeshModel.visualMesh = std::move(visualMesh);
  return tetMeshModel;
}

TEST(ModelUtils, SaveLoad_VisualMesh) {
  // Tetrahedral meshes with valid visual meshes
  for (int weightsPerNode = 1; weightsPerNode <= 4; ++weightsPerNode) {
    ModelData model = GetTetMeshModelWithVisualMesh(weightsPerNode);
    model::Validate(model, test::ExpectOK{});
    TestSerializationRoundTrip(model);
  }

  // It is less common for a model with a triangle mesh to have a visual mesh, but it is legal.
  // We can use the same triangle mesh for both because they have the same number of nodes
  for (int weightsPerNode = 1; weightsPerNode <= 4; ++weightsPerNode) {
    ModelData model = GetTetMeshModelWithVisualMesh(weightsPerNode);
    model.mesh = model.visualMesh;
    model.mesh->skinning = std::nullopt;
    model::Validate(model, test::ExpectOK{});
    TestSerializationRoundTrip(model);
  }
}

TEST(ModelUtils, AutoCorrect_VisualMesh) {
  // AutoCorrect should normalize the skinning weights
  int constexpr kWeightsPerNode = 3;
  ModelData model = GetTetMeshModelWithVisualMesh(kWeightsPerNode);
  auto& skinning = *model.visualMesh->skinning;
  for (int i = 0; i < model.visualMesh->GetNumNodes(); ++i) {
    // All the weight on one node
    skinning.weights[kWeightsPerNode * i + 0] = 0_r;
    skinning.weights[kWeightsPerNode * i + 1] = 2.34_r;
    skinning.weights[kWeightsPerNode * i + 2] = 0_r;
    model::AutoCorrect(model, test::ExpectOK{});
    EXPECT_NEAR_EQ(0_r, skinning.weights[kWeightsPerNode * i + 0]);
    EXPECT_NEAR_EQ(1_r, skinning.weights[kWeightsPerNode * i + 1]);
    EXPECT_NEAR_EQ(0_r, skinning.weights[kWeightsPerNode * i + 2]);

    // Weight split between multiple nodes
    skinning.weights[kWeightsPerNode * i + 0] = 0.5_r;
    skinning.weights[kWeightsPerNode * i + 1] = 0_r;
    skinning.weights[kWeightsPerNode * i + 2] = 1.5_r;
    model::AutoCorrect(model, test::ExpectOK{});
    EXPECT_NEAR_EQ(0.25_r, skinning.weights[kWeightsPerNode * i + 0]);
    EXPECT_NEAR_EQ(0_r, skinning.weights[kWeightsPerNode * i + 1]);
    EXPECT_NEAR_EQ(0.75_r, skinning.weights[kWeightsPerNode * i + 2]);
  }
  // If the visual mesh has no triangles, then AutoCorrect should remove it.
  EXPECT_TRUE(model.visualMesh.has_value());
  model.visualMesh->connectivity.clear();
  model::AutoCorrect(model, test::ExpectOK{});
  EXPECT_FALSE(model.visualMesh.has_value());
}

TEST(ModelUtils, Validate_VisualMesh) {
  int constexpr kWeightsPerNode = 4;
  ModelData srcModel = GetTetMeshModelWithVisualMesh(kWeightsPerNode);
  model::Validate(srcModel, test::ExpectOK{}); // Starts valid

  // Visual mesh must be a valid triangle mesh.
  {
    ModelData model = srcModel; // copy
    EXPECT_EQ(3, model.visualMesh->nodesPerElement);
    model.visualMesh->nodesPerElement = 4;
    model::Validate(model, test::ExpectNotOK{});
    model.visualMesh->nodesPerElement = 3;
    model::Validate(model, test::ExpectOK{}); // Valid again
    TestValidateMeshCoordinates(model, *model.visualMesh);
    TestValidateMeshConnectivity(model, *model.visualMesh);
  }

  // It is technically legal for a model to have a visual mesh without skinning data, however, it
  // won't be deformable.
  {
    ModelData model = srcModel; // copy
    model.visualMesh->skinning = std::nullopt;
    model::Validate(model, test::ExpectOK{});
  }

  // Visual mesh skinning data must be valid. Indices must be in the half open range [0, numNodes)
  {
    ModelData model = srcModel; // copy
    int const kInvalidSkinningIndices[] = {-1, model.mesh->GetNumNodes()};
    TestValidateSkinningData(model, *model.visualMesh->skinning, kInvalidSkinningIndices);
  }
}

// Build a ModelData whose simulation mesh is a unit-cube tetrahedralization (8 nodes, 5 tets).
static ModelData MakeUnitCubeTetModel() {
  auto [coords, conn] = test::CreateMinimalTetMeshUnitCube();
  ModelData model;
  model.mesh.emplace();
  model.mesh->nodesPerElement = 4;
  Span<real> flatCoords = Flatten(MakeSpan(coords));
  Span<int> flatConn = Flatten(MakeSpan(conn));
  model.mesh->coordinates.assign(flatCoords.begin(), flatCoords.end());
  model.mesh->connectivity.assign(flatConn.begin(), flatConn.end());
  return model;
}

// Set the visual mesh to a valid triangle mesh whose nodes are the given points, connected as a
// simple fan so it passes full mesh validation. Requires at least 3 points to form a triangle.
static void SetVisualNodes(ModelData& model, DynamicArray<Real3> const& pts) {
  model.visualMesh.emplace();
  model.visualMesh->nodesPerElement = 3;
  auto const flat = Flatten(MakeConstSpan(pts));
  model.visualMesh->coordinates.assign(flat.begin(), flat.end());
  DynamicArray<int>& conn = model.visualMesh->connectivity;
  for (int i = 1; i + 1 < isize(pts); ++i) {
    conn.push_back(0);
    conn.push_back(i);
    conn.push_back(i + 1);
  }
}

// Wrap a single point of interest in a valid triangle visual mesh (node 0 is the point; nodes 1-2
// are placeholders at the origin). Callers assert only on node 0.
static void SetVisualPointOfInterest(ModelData& model, Real3 const& p) {
  SetVisualNodes(model, {p, Real3{}, Real3{}});
}

// Reconstruct the i-th visual node position from its skinning weights and the simulation mesh.
static Real3 ReconstructVisualNode(ModelData const& model, int i) {
  auto const tetCoords = Unflatten<Real3 const>(MakeConstSpan(model.mesh->coordinates));
  auto const& skin = *model.visualMesh->skinning;
  Real3 p{};
  for (int j = 0; j < skin.weightsPerNode; ++j) {
    p = p +
        skin.weights[i * skin.weightsPerNode + j] *
            tetCoords[skin.indices[i * skin.weightsPerNode + j]];
  }
  return p;
}

TEST(ModelUtils, GenerateVisualMeshEmbedding_InteriorReconstruction) {
  ModelData model = MakeUnitCubeTetModel();
  Real3 const interior{0.3_r, 0.4_r, 0.5_r};
  SetVisualPointOfInterest(model, interior);
  model::GenerateVisualMeshEmbedding(model, test::ExpectOK{});

  ASSERT_TRUE(model.visualMesh->skinning.has_value());
  auto const& skin = *model.visualMesh->skinning;
  EXPECT_EQ(4, skin.weightsPerNode);

  // An interior point lies inside its containing tet, so all barycentric weights are non-negative,
  // sum to 1, and reconstruct the node position.
  real sum = 0_r;
  for (int j = 0; j < 4; ++j) {
    EXPECT_GE(skin.weights[j], -1e-5_r);
    sum += skin.weights[j];
  }
  EXPECT_NEAR_EQ(1_r, sum);
  EXPECT_NEAR_EQ(interior, ReconstructVisualNode(model, 0));
}

TEST(ModelUtils, GenerateVisualMeshEmbedding_VertexCoincidence) {
  ModelData model = MakeUnitCubeTetModel();
  Real3 const vertex{0_r, 0_r, 0_r}; // Node 0 of the unit cube.
  SetVisualPointOfInterest(model, vertex);
  model::GenerateVisualMeshEmbedding(model, test::ExpectOK{});

  auto const& skin = *model.visualMesh->skinning;
  auto const tetCoords = Unflatten<Real3 const>(MakeConstSpan(model.mesh->coordinates));

  // The weight on the coincident vertex is ~1; all others ~0.
  for (int j = 0; j < 4; ++j) {
    real const expected = NearEqual(tetCoords[skin.indices[j]], vertex) ? 1_r : 0_r;
    EXPECT_NEAR_EQ(expected, skin.weights[j]);
  }
  EXPECT_NEAR_EQ(vertex, ReconstructVisualNode(model, 0));
}

TEST(ModelUtils, GenerateVisualMeshEmbedding_ExteriorExtrapolation) {
  ModelData model = MakeUnitCubeTetModel();
  Real3 const exterior{1.2_r, 0.5_r, 0.5_r}; // Just outside the +x face.
  SetVisualPointOfInterest(model, exterior);
  auto noWarn = test::SuppressLogWarning(); // The exterior node may trip the embedding warning.
  model::GenerateVisualMeshEmbedding(model, test::ExpectOK{});

  auto const& skin = *model.visualMesh->skinning;

  // The nearest tet is chosen and the (unclamped) barycentric weights extrapolate affinely: they
  // still sum to 1 (with at least one negative) and reconstruct the original point.
  real sum = 0_r;
  real minWeight = kInf;
  for (int j = 0; j < 4; ++j) {
    sum += skin.weights[j];
    minWeight = Min(minWeight, skin.weights[j]);
  }
  EXPECT_NEAR_EQ(1_r, sum);
  EXPECT_LT(minWeight, 0_r);
  EXPECT_NEAR_EQ(exterior, ReconstructVisualNode(model, 0));
}

TEST(ModelUtils, GenerateVisualMeshEmbedding_Preconditions) {
  // A non-tetrahedral simulation mesh (triangles) is rejected.
  {
    ModelData model = model::LoadFromBytes(kTriMeshCubeJson, test::ExpectOK{});
    SetVisualPointOfInterest(model, Real3{0.5_r, 0.5_r, 0.5_r});
    model::GenerateVisualMeshEmbedding(model, test::ExpectNotOK{});
  }
  // A missing visual mesh is rejected.
  {
    ModelData model = MakeUnitCubeTetModel();
    model::GenerateVisualMeshEmbedding(model, test::ExpectNotOK{});
  }
  // A visual mesh without coordinates is rejected.
  {
    ModelData model = MakeUnitCubeTetModel();
    model.visualMesh.emplace();
    model.visualMesh->nodesPerElement = 3;
    model::GenerateVisualMeshEmbedding(model, test::ExpectNotOK{});
  }
  // An empty tetrahedral mesh (no elements) is rejected.
  {
    ModelData model = MakeUnitCubeTetModel();
    model.mesh->connectivity.clear();
    SetVisualPointOfInterest(model, Real3{0.5_r, 0.5_r, 0.5_r});
    model::GenerateVisualMeshEmbedding(model, test::ExpectNotOK{});
  }
}

TEST(ModelUtils, GenerateVisualMeshEmbedding_Shape) {
  ModelData model = MakeUnitCubeTetModel();
  DynamicArray<Real3> const pts = {
      Real3{0.1_r, 0.1_r, 0.1_r},
      Real3{0.5_r, 0.5_r, 0.5_r},
      Real3{0.9_r, 0.8_r, 0.7_r},
      Real3{0_r, 0_r, 0_r},
      Real3{1_r, 1_r, 1_r}};
  SetVisualNodes(model, pts);
  model::GenerateVisualMeshEmbedding(model, test::ExpectOK{});

  int const numVis = isize(pts);
  int const numTetNodes = model.mesh->GetNumNodes();
  auto const& skin = *model.visualMesh->skinning;
  EXPECT_EQ(4, skin.weightsPerNode);
  EXPECT_EQ(numVis * 4, isize(skin.indices));
  EXPECT_EQ(numVis * 4, isize(skin.weights));
  for (int idx : skin.indices) {
    EXPECT_GE(idx, 0);
    EXPECT_LT(idx, numTetNodes);
  }
}

TEST(ModelUtils, GenerateVisualMeshEmbedding_AutoCorrectValidate) {
  ModelData model = MakeUnitCubeTetModel();

  // A valid triangle visual mesh (unit-cube surface) sharing the tet mesh's coordinate frame.
  auto [visCoords, visConn] = test::CreateMinimalTriMeshUnitCube();
  model.visualMesh.emplace();
  model.visualMesh->nodesPerElement = 3;
  {
    Span<real> flatCoords = Flatten(MakeSpan(visCoords));
    Span<int> flatConn = Flatten(MakeSpan(visConn));
    model.visualMesh->coordinates.assign(flatCoords.begin(), flatCoords.end());
    model.visualMesh->connectivity.assign(flatConn.begin(), flatConn.end());
  }

  model::GenerateVisualMeshEmbedding(model, test::ExpectOK{});
  ASSERT_TRUE(model.visualMesh->skinning.has_value());

  // The produced sum-to-1 weights survive AutoCorrect (NormalizeWeights) unchanged, and the
  // resulting model validates.
  DynamicArray<real> const weightsBefore = model.visualMesh->skinning->weights;
  model::AutoCorrect(model, test::ExpectOK{});
  model::Validate(model, test::ExpectOK{});
  ASSERT_EQ(weightsBefore.size(), model.visualMesh->skinning->weights.size());
  for (int i = 0; i < isize(weightsBefore); ++i) {
    EXPECT_NEAR_EQ(weightsBefore[i], model.visualMesh->skinning->weights[i]);
  }
}

TEST(ModelUtils, GenerateVisualMeshEmbedding_ParallelReconstruction) {
  // Exercise the multi-threaded ParallelForN path (minPerTask = 256) with a visual mesh large
  // enough to split the work across tasks, and verify every node reconstructs from its weights.
  TaskScheduler scheduler;

  ModelData model = MakeUnitCubeTetModel();

  // 8x8x8 = 512 points strictly inside the unit cube, exceeding minPerTask so work spans tasks.
  DynamicArray<Real3> pts;
  pts.reserve(512);
  for (int x = 0; x < 8; ++x) {
    for (int y = 0; y < 8; ++y) {
      for (int z = 0; z < 8; ++z) {
        Real3 const p{
            (static_cast<real>(x) + 0.5_r) / 8_r,
            (static_cast<real>(y) + 0.5_r) / 8_r,
            (static_cast<real>(z) + 0.5_r) / 8_r};
        pts.push_back(p);
      }
    }
  }
  SetVisualNodes(model, pts);
  model::GenerateVisualMeshEmbedding(model, test::ExpectOK{});

  int const numVis = isize(pts);
  auto const& skin = *model.visualMesh->skinning;
  EXPECT_EQ(4, skin.weightsPerNode);
  EXPECT_EQ(numVis * 4, isize(skin.indices));
  EXPECT_EQ(numVis * 4, isize(skin.weights));

  // Every visual node must reconstruct to its original position from the parallel-computed
  // weights, exercising per-node indexing for all i (not just i == 0).
  for (int i = 0; i < numVis; ++i) {
    EXPECT_NEAR_EQ(pts[i], ReconstructVisualNode(model, i));
  }
}

// Build SkinningData holding the given flat weights (weightsPerNode is irrelevant to the metric).
static SkinningData MakeSkinning(DynamicArray<real> const& weights) {
  SkinningData skin;
  skin.weightsPerNode = 4;
  skin.weights.assign(weights.begin(), weights.end());
  return skin;
}

TEST(ModelUtils, MaxVisualMeshExtrapolation_Value) {
  // The result is the largest -weight over all nodes: 0 when every weight is non-negative, and the
  // magnitude of the most-negative weight otherwise (here node 2's -0.5 dominates node 0's -0.2).
  EXPECT_NEAR_EQ(
      0_r, model::MaxVisualMeshExtrapolation(MakeSkinning({0.25_r, 0.25_r, 0.25_r, 0.25_r})));
  EXPECT_NEAR_EQ(
      0.5_r,
      model::MaxVisualMeshExtrapolation(MakeSkinning(
          {1.2_r, -0.2_r, 0_r, 0_r, 0.25_r, 0.25_r, 0.25_r, 0.25_r, 1.5_r, -0.5_r, 0_r, 0_r})));
}

TEST(ModelUtils, MaxVisualMeshExtrapolation_Empty) {
  // Empty weights have no extrapolation.
  EXPECT_NEAR_EQ(0_r, model::MaxVisualMeshExtrapolation(SkinningData{}));
}

TEST(ModelUtils, MaxVisualMeshExtrapolation_FromEmbedding) {
  // Connection test: an exterior visual point embedded by GenerateVisualMeshEmbedding reports a
  // positive extrapolation.
  ModelData model = MakeUnitCubeTetModel();
  SetVisualPointOfInterest(model, Real3{1.2_r, 0.5_r, 0.5_r}); // Just outside the +x face.
  auto noWarn = test::SuppressLogWarning(); // The exterior node may trip the embedding warning.
  model::GenerateVisualMeshEmbedding(model, test::ExpectOK{});

  ASSERT_TRUE(model.visualMesh->skinning.has_value());
  EXPECT_GT(model::MaxVisualMeshExtrapolation(*model.visualMesh->skinning), 0_r);
}

TEST(ModelUtils, GenerateVisualMeshEmbedding_WarnsOnGrossExtrapolation) {
  // Run the embedding while capturing Warning-channel logs. Swapping the log callback also bypasses
  // the harness's fail-on-warning behavior.
  auto countWarnings = [](Real3 const& p) {
    ModelData model = MakeUnitCubeTetModel();
    SetVisualPointOfInterest(model, p);

    int warnCount = 0;
    auto const prevFn = GetLogCallback();
    SetLogCallback([&warnCount](LogChannel ch, char const*, char const*, int) {
      if (ch == LogChannel::Warning) {
        ++warnCount;
      }
    });
    MOCHI_DEFER(SetLogCallback(prevFn));
    model::GenerateVisualMeshEmbedding(model, test::ExpectOK{});
    return warnCount;
  };

  // A grossly misaligned node (far outside the mesh) warns; an interior node does not.
  EXPECT_EQ(1, countWarnings(Real3{5_r, 0.5_r, 0.5_r}));
  EXPECT_EQ(0, countWarnings(Real3{0.5_r, 0.5_r, 0.5_r}));
}

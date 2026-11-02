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

// Loading and saving pipeline files (meshing/processing_modifiers/processing_serialization.h) with
// the real modifiers.

#include "meshing/processing_modifiers/processing_serialization.h"
#include "meshing/processing_modifiers/processing_stack.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>

#include <picojson/picojson.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace superdex::studio;
namespace fs = std::filesystem;

namespace {

class ProcessingSerializationTest : public ::testing::Test {
 protected:
  // Writes @p document as a pipeline file and loads it.
  LoadedPipeline Load(std::string const& document) {
    fs::create_directories(_path.parent_path());
    std::ofstream(_path) << document;
    LoadedPipeline loaded;
    mochi::Error error;
    EXPECT_TRUE(LoadProcessingPipeline(_path.string(), _path, loaded, error))
        << error.GetDescription();
    return loaded;
  }

  // The document Studio saves for @p loaded.
  [[nodiscard]] picojson::object Save(LoadedPipeline const& loaded) const {
    std::string const text =
        SerializeProcessingPipeline(loaded.modifiers, loaded.editorState, {}, _path);
    picojson::value saved;
    std::string parseError;
    picojson::parse(saved, text.begin(), text.end(), &parseError);
    EXPECT_EQ(parseError, "");
    return saved.get<picojson::object>();
  }

  static picojson::object const& Modifier(picojson::object const& document, int modifier) {
    return document.at("modifiers").get<picojson::array>().at(modifier).get<picojson::object>();
  }

  static picojson::object const& Properties(picojson::object const& document, int modifier) {
    return Modifier(document, modifier).at("properties").get<picojson::object>();
  }

  mochi::TempDirCleanup _rootCleanup =
      mochi::CreateTempDirectory("processing_serialization_test", mochi::test::ExpectOK{});
  fs::path _path = _rootCleanup.Path() / "intermediates" / "part.StudioProcessing.json";
};

} // namespace

// Before the STEP mesher's rewrite the Isotropic backend was called CadMesher, and bodySelection
// chose the bodies to mesh, 0 for all of them, which is what combineTouchingSolids now defaults to.
TEST_F(ProcessingSerializationTest, ReadsCadSourcesSavedBeforeTheStepMesherRewrite) {
  LoadedPipeline const loaded = Load(R"({
  "version": 1,
  "modifiers": [
    {"modifier": "Source from CAD Model", "method": "From Model Viewer", "enabled": true,
     "collapsed": true,
     "properties": {"backend": "CadMesher", "bodySelection": 0, "linearDeflection": 0.02}},
    {"modifier": "Source from CAD Model", "method": "From File", "enabled": false,
     "collapsed": true,
     "properties": {"path": "../cad/part.step",
                    "step": {"backend": "CadMesher", "bodySelection": 0}}}
  ]
})");
  picojson::object const saved = Save(loaded);

  EXPECT_EQ(loaded.issues, std::vector<std::string>{});
  picojson::object const& viewer = Properties(saved, 0);
  EXPECT_EQ(viewer.at("backend").get<std::string>(), "Isotropic");
  EXPECT_TRUE(viewer.at("combineTouchingSolids").get<bool>());
  EXPECT_EQ(viewer.count("bodySelection"), 0u);
  EXPECT_DOUBLE_EQ(viewer.at("linearDeflection").get<double>(), 0.02);
  picojson::object const& step = Properties(saved, 1).at("step").get<picojson::object>();
  EXPECT_EQ(step.at("backend").get<std::string>(), "Isotropic");
  EXPECT_TRUE(step.at("combineTouchingSolids").get<bool>());
  EXPECT_EQ(step.count("bodySelection"), 0u);
}

TEST_F(ProcessingSerializationTest, TetrahedralizeAndNoSdfExportRoundTrip) {
  LoadedPipeline loaded = Load(R"({
  "version": 1,
  "modifiers": [
    {"modifier": "Tetrahedralize", "method": "Delaunay Refinement", "enabled": true,
     "collapsed": true,
     "properties": {"edgeSize": 0.25, "relativeToMeshSize": false,
                    "facetDistanceBound": 0.04, "facetAngleBound": 24,
                    "cellRadiusEdgeRatio": 2.5, "detectFeatures": false,
                    "sharpFeatureAngle": 45, "perturb": false, "exude": false, "odt": true}},
    {"modifier": "Embed Visual Mesh", "method": "Upstream Surface", "enabled": true,
     "collapsed": true, "properties": {"referenceIndex": 0}},
    {"modifier": "Export Mochi Model", "method": "SDF", "enabled": true, "collapsed": true,
     "properties": {"bakeSdf": false}}
  ]
})");
  fs::path const source = _rootCleanup.Path() / "cad" / "part.step";
  RefreshAutoExportPaths(loaded.modifiers, source.string());
  picojson::object const saved = Save(loaded);

  EXPECT_EQ(loaded.issues, std::vector<std::string>{});
  ASSERT_EQ(loaded.modifiers.size(), 3u);
  EXPECT_EQ(Modifier(saved, 0).at("modifier").get<std::string>(), "Tetrahedralize");
  EXPECT_EQ(Modifier(saved, 0).at("method").get<std::string>(), "Delaunay Refinement");
  picojson::object const& tet = Properties(saved, 0);
  EXPECT_DOUBLE_EQ(tet.at("edgeSize").get<double>(), 0.25);
  EXPECT_DOUBLE_EQ(tet.at("facetDistanceBound").get<double>(), 0.04);
  EXPECT_DOUBLE_EQ(tet.at("facetAngleBound").get<double>(), 24.0);
  EXPECT_DOUBLE_EQ(tet.at("cellRadiusEdgeRatio").get<double>(), 2.5);
  EXPECT_FALSE(tet.at("relativeToMeshSize").get<bool>());
  EXPECT_FALSE(tet.at("detectFeatures").get<bool>());
  EXPECT_DOUBLE_EQ(tet.at("sharpFeatureAngle").get<double>(), 45.0);
  EXPECT_FALSE(tet.at("perturb").get<bool>());
  EXPECT_FALSE(tet.at("exude").get<bool>());
  EXPECT_TRUE(tet.at("odt").get<bool>());
  EXPECT_EQ(Modifier(saved, 1).at("modifier").get<std::string>(), "Embed Visual Mesh");
  EXPECT_EQ(Modifier(saved, 1).at("method").get<std::string>(), "Upstream Surface");
  EXPECT_EQ(Properties(saved, 1).at("referenceIndex").get<double>(), 0.0);
  EXPECT_EQ(Modifier(saved, 2).at("modifier").get<std::string>(), "Export Mochi Model");
  EXPECT_EQ(Modifier(saved, 2).at("method").get<std::string>(), "SDF");
  EXPECT_FALSE(Properties(saved, 2).at("bakeSdf").get<bool>());
  EXPECT_EQ(
      loaded.modifiers[2]->ExportPath(),
      (_rootCleanup.Path() / "collision" / "part.mochi.h5").string());
  EXPECT_EQ(Properties(saved, 2).count("exportPath"), 0u);
}

TEST_F(ProcessingSerializationTest, ListsWhatItCannotReadAndLoadsTheRest) {
  LoadedPipeline const loaded = Load(R"({
  "version": 1,
  "colour": "red",
  "editorState": {"cadScale": [1, 1, 1], "cadTilt": 0},
  "modifiers": [
    {"modifier": "Source from CAD Model", "method": "From Model Viewer", "enabeld": true,
     "properties": {"backend": "Quadric", "linearDeflexion": 0.02}},
    {"modifier": "Refine Mesh", "method": "Make It Nice", "enabled": true, "properties": {}},
    {"modifier": "Sharpen Mesh", "method": "Unsharp", "enabled": true, "properties": {}}
  ]
})");

  std::vector<std::string> const expected{
      "unknown field 'colour'",
      "editorState: 1 unreadable field",
      "modifiers[0] (Source from CAD Model: From Model Viewer): unknown field 'enabeld'",
      "modifiers[0] (Source from CAD Model: From Model Viewer): 2 unreadable properties",
      "modifiers[1] (Refine Mesh: Make It Nice): unknown method 'Make It Nice'",
      "modifiers[2] (Sharpen Mesh: Unsharp): unknown modifier 'Sharpen Mesh'",
  };
  EXPECT_EQ(loaded.issues, expected);
  EXPECT_EQ(loaded.modifiers.size(), 3u);
}

TEST_F(ProcessingSerializationTest, ListsValuesOfTheWrongType) {
  LoadedPipeline const loaded = Load(R"({
  "version": 1,
  "editorState": {"cadScale": "big"},
  "modifiers": [
    {"modifier": "Source from CAD Model", "method": "From Model Viewer", "enabled": "yes",
     "collapsed": 3, "properties": {"linearDeflection": "fine"}}
  ]
})");

  std::vector<std::string> const expected{
      "editorState: 1 unreadable field",
      "modifiers[0] (Source from CAD Model: From Model Viewer): 2 unreadable fields",
      "modifiers[0] (Source from CAD Model: From Model Viewer): 1 unreadable property",
  };
  EXPECT_EQ(loaded.issues, expected);
}

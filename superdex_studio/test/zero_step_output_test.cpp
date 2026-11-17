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

#include "tools/prepare_urdf_command.h"
#include "tools/zero_step_output.h"

#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_physics/cpp_api/mochi_actor.h>
#include <mochi_physics/cpp_api/mochi_context.h>
#include <mochi_physics/cpp_api/mochi_scene.h>

#include <gtest/gtest.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

TEST(PrepareUrdfCommandTest, WriterExceptionUnwindsGeneratedOutput) {
  namespace fs = std::filesystem;
  mochi::Error error;
  auto directory = mochi::CreateTempDirectory("preparation_exception_test", error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  fs::path const output = directory.Path() / "output";
  bool wrotePartialFile = false;
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  int const result = superdex::studio::RunUrdfPreparation([&]() -> int {
    fs::create_directory(output);
    MOCHI_DEFER(fs::remove_all(output));
    mochi::WriteFile(output / "partial.glb", std::string_view{"partial writer output"}, error);
    if (!error.IsOK()) {
      return 2;
    }
    wrotePartialFile = fs::exists(output / "partial.glb");
    throw std::runtime_error("injected writer failure");
  });
  std::string const standardOutput = testing::internal::GetCapturedStdout();
  std::string const diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_EQ(result, 1);
  EXPECT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_TRUE(wrotePartialFile);
  EXPECT_FALSE(fs::exists(output));
  EXPECT_TRUE(standardOutput.empty());
  EXPECT_NE(diagnostic.find("injected writer failure"), std::string::npos);
}

TEST(ZeroStepOutputTest, GetterErrorDiscardsPartialAndCompleteJson) {
  auto* context = mochi::CreateContext(0);
  MOCHI_DEFER(mochi::DestroyContext(context));
  auto* scene = context->CreateScene("output_failure_test");
  mochi::ArticulatedActorParams params;
  params.joints.resize(1);
  params.joints[0].type = mochi::ArticulatedJointType::Free;
  params.links.resize(1);
  params.links[0].name = "base";
  params.links[0].colliderType = mochi::ColliderType::None;
  mochi::Error error;
  auto* actor = scene->CreateArticulatedActor(params, error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  ASSERT_NE(actor, nullptr);
  actor->GetArticulatedDofLimits({}, error);
  ASSERT_FALSE(error.IsOK());

  for (std::string const json : {R"({"articulations":[)", R"({"mass_kg":0})"}) {
    testing::internal::CaptureStdout();
    testing::internal::CaptureStderr();
    int const result = superdex::studio::PublishZeroStepJson(json, error);
    std::string const output = testing::internal::GetCapturedStdout();
    std::string const diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_EQ(result, 1);
    EXPECT_TRUE(output.empty());
    EXPECT_NE(diagnostic.find(error.GetDescription()), std::string::npos);
  }
}

TEST(ZeroStepOutputTest, SuccessfulReportIsPublishedUnchanged) {
  std::string const json = R"({"precision_bytes":4,"positive_steps":0,"articulations":[]})";
  mochi::Error error;
  testing::internal::CaptureStdout();
  testing::internal::CaptureStderr();
  int const result = superdex::studio::PublishZeroStepJson(json, error);
  std::string const output = testing::internal::GetCapturedStdout();
  std::string const diagnostic = testing::internal::GetCapturedStderr();
  EXPECT_EQ(result, 0);
  EXPECT_EQ(output, json);
  EXPECT_TRUE(diagnostic.empty());
}

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

#include <gtest/gtest.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <superdex_robotics/utils/file_utils.h>
#include "core/ik_design_session.h"

using namespace superdex::studio;
using namespace mochi;
using namespace mochi::test;

TEST(IKDesignSessionTest, NormalizedIdentityAndPerDocumentChoice) {
  IKRecordingMemory settings;
  auto const root = std::filesystem::temp_directory_path();
  Path const doc{root / "a.superdex_ik"}, other{root / "b.superdex_ik"};
  Path const clip{root / "pose.rrd"};
  EXPECT_TRUE(RememberIKRecording(settings, doc, clip));
  EXPECT_EQ(
      RememberedIKRecording(settings, Path{root / "nested" / ".." / "a.superdex_ik"}),
      IKDesignPathKey(clip));
  EXPECT_TRUE(RememberedIKRecording(settings, other).empty());
  EXPECT_FALSE(RememberIKRecording(settings, doc, clip));
  EXPECT_TRUE(RememberIKRecording(settings, other, Path{root / "second.rrd"}));
  EXPECT_TRUE(RememberIKRecording(settings, doc, {}));
  EXPECT_TRUE(RememberedIKRecording(settings, doc).empty());
  EXPECT_FALSE(RememberedIKRecording(settings, other).empty());
}

TEST(IKDesignSessionTest, SettingsMapRoundTrip) {
  IKRecordingMemory settings;
  Path const doc{std::filesystem::temp_directory_path() / "remembered.superdex_ik"};
  Path const clip{std::filesystem::temp_directory_path() / "remembered.rrd"};
  RememberIKRecording(settings, doc, clip);
  RememberIKRecording(
      settings, Path{std::filesystem::temp_directory_path() / "other.superdex_ik"}, clip);
  auto const json = SReflect::ToJsonString(settings, false);
  mochi::Error error;
  auto const loaded = superdex::robotics::LoadParamsFromPathOrJson<IKRecordingMemory>(json, error);
  ASSERT_TRUE(error.IsOK()) << error.GetDescription();
  EXPECT_EQ(RememberedIKRecording(loaded, doc), IKDesignPathKey(clip));
  EXPECT_EQ(loaded, settings);
}

TEST(IKDesignSessionTest, MovesRewriteDocumentAndRecordingWithoutAuthoringState) {
  IKRecordingMemory settings;
  auto const root = std::filesystem::temp_directory_path();
  Path const doc{root / "old.superdex_ik"}, newDoc{root / "new.superdex_ik"};
  Path const clip{root / "old.rrd"}, newClip{root / "new.rrd"};
  RememberIKRecording(settings, doc, clip);
  EXPECT_TRUE(RewriteIKDesignPath(settings, doc, newDoc));
  EXPECT_TRUE(RememberedIKRecording(settings, doc).empty());
  EXPECT_EQ(RememberedIKRecording(settings, newDoc), IKDesignPathKey(clip));
  EXPECT_TRUE(RewriteIKDesignPath(settings, clip, newClip));
  EXPECT_EQ(RememberedIKRecording(settings, newDoc), IKDesignPathKey(newClip));
  EXPECT_FALSE(RewriteIKDesignPath(settings, newClip, newClip));
  EXPECT_FALSE(RememberIKRecording(settings, {}, newClip));
}

#if MOCHI_PLATFORM_WINDOWS
TEST(IKDesignSessionTest, WindowsCaseAliasesUseOnePreference) {
  IKRecordingMemory settings;
  RememberIKRecording(
      settings, Path{"C:/OPEN/design/hand.superdex_ik"}, Path{"C:/OPEN/design/test.rrd"});
  EXPECT_EQ(
      RememberedIKRecording(settings, Path{"c:/open/design/HAND.superdex_ik"}),
      IKDesignPathKey(Path{"c:/open/design/test.rrd"}));
  EXPECT_EQ(settings.size(), 1);
}
#endif

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

#include "superdex_asset_tuner/json_path.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace superdex::tuner {
namespace {

// Each change as "path: before -> after".
std::vector<std::string> Diff(std::string_view before, std::string_view after) {
  picojson::value a;
  picojson::value b;
  std::string error;
  EXPECT_TRUE(ParseJson(before, a, error)) << error;
  EXPECT_TRUE(ParseJson(after, b, error)) << error;
  std::vector<std::string> lines;
  for (JsonChange const& change : DiffJson(a, b)) {
    lines.push_back(change.path + ": " + change.before + " -> " + change.after);
  }
  return lines;
}

TEST(JsonDiffTest, EqualDocumentsHaveNoChanges) {
  EXPECT_TRUE(Diff(R"({"a": [1, {"b": 2}]})", R"({"a": [1, {"b": 2}]})").empty());
}

TEST(JsonDiffTest, NamedElementsMatchByNameEvenWhenReordered) {
  std::vector<std::string> const expected{
      "loads[name=lift].profile.magnitude: 2 -> 3",
      R"(loads[name=push]:  -> {"name":"push"})",
  };

  EXPECT_EQ(
      Diff(
          R"({"loads": [{"name": "drag"}, {"name": "lift", "profile": {"magnitude": 2}}]})",
          R"({"loads": [{"name": "lift", "profile": {"magnitude": 3}}, {"name": "drag"},
                        {"name": "push"}]})"),
      expected);
}

TEST(JsonDiffTest, UnnamedArraysMatchByIndexOnlyAtTheSameLength) {
  std::vector<std::string> const expected{
      "gravity[2]: -9.5 -> -1",
      "keys: [1,2] -> [1,2,3]",
      "removed: true -> ",
  };

  EXPECT_EQ(
      Diff(
          R"({"gravity": [0, 0, -9.5], "keys": [1, 2], "removed": true})",
          R"({"gravity": [0, 0, -1], "keys": [1, 2, 3]})"),
      expected);
}

} // namespace
} // namespace superdex::tuner

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

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/json_utils.h>

#include <gtest/gtest.h>

using namespace mochi;

TEST(JsonUtilsTest, NumericOverflowReturnsErrorInsteadOfThrowing) {
  bool const warningEnabled = IsLogChannelEnabled(LogChannel::Warning);
  EnableLogChannel(LogChannel::Warning, false);
  MOCHI_DEFER(EnableLogChannel(LogChannel::Warning, warningEnabled));
  for (auto const* text : {"1e400", "-1e400", R"({"value":1e400})"}) {
    Error error;
    EXPECT_NO_THROW((void)ParseJsonFromString(std::string_view(text), error));
    EXPECT_NOT_OK(error);
  }
}

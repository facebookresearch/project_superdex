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
#include <mochi_core/utils/stream.h>

#include <string_view>

using namespace mochi;

TEST(SpanStreamReader, Read) {
  constexpr std::string_view kData = "Hello, world! This stream is working.";
  char buffer[kData.size()] = {};

  SpanStreamReader reader(
      Span<uint8_t const>(reinterpret_cast<uint8_t const*>(kData.data()), kData.size()));
  EXPECT_EQ(0, reader.GetPosition());

  // Read "Hello"
  reader.Read(buffer, 5, test::ExpectOK{});
  buffer[7] = '\0';
  EXPECT_STREQ("Hello", buffer);
  EXPECT_EQ(5, reader.GetPosition());

  // Read zero bytes
  reader.Read(nullptr, 0, test::ExpectOK{});
  EXPECT_EQ(5, reader.GetPosition()); // no change

  // Read ", world! "
  reader.Read(buffer, 9, test::ExpectOK{});
  buffer[9] = '\0';
  EXPECT_STREQ(", world! ", buffer);
  EXPECT_EQ(14, reader.GetPosition());

  // Read "This stream " via StreamRead template
  std::array<char, 12> obj;
  StreamRead(obj, reader, test::ExpectOK{});
  EXPECT_EQ(std::string_view("This stream "), std::string_view(obj.data(), obj.size()));
  EXPECT_EQ(26, reader.GetPosition());

  // Try to read too many bytes
  reader.Read(buffer, 12, test::ExpectNotOK{});
  EXPECT_STREQ(", world! ", buffer); // no change
  EXPECT_EQ(26, reader.GetPosition()); // no change

  // Read all remaining bytes exactly
  reader.Read(buffer, 11, test::ExpectOK{});
  buffer[12] = '\0';
  EXPECT_STREQ("is working.", buffer);
  EXPECT_EQ(37, reader.GetPosition());

  // End of stream
  reader.Read(buffer, 1, test::ExpectNotOK{});
  reader.Read(buffer, 0, test::ExpectOK{});
  EXPECT_EQ(37, reader.GetPosition()); // no change
}

TEST(SpanStreamReader, Advance) {
  constexpr std::string_view kData = "Hello, world! This stream is working.";
  char buffer[kData.size()] = {};

  SpanStreamReader reader(
      Span<uint8_t const>(reinterpret_cast<uint8_t const*>(kData.data()), kData.size()));
  EXPECT_EQ(0, reader.GetPosition());

  // Advance zero bytes.
  reader.Advance(0, test::ExpectOK{});
  EXPECT_EQ(0, reader.GetPosition()); // no change

  // Advance past "Hello, " without reading.
  reader.Advance(7, test::ExpectOK{});
  EXPECT_EQ(7, reader.GetPosition());

  // Verify the next read picks up at the correct position.
  reader.Read(buffer, 6, test::ExpectOK{});
  buffer[6] = '\0';
  EXPECT_STREQ("world!", buffer);
  EXPECT_EQ(13, reader.GetPosition());

  // Try to advance past the end of the stream.
  reader.Advance(kData.size(), test::ExpectNotOK{});
  EXPECT_EQ(13, reader.GetPosition()); // no change

  // Advance all remaining bytes exactly.
  reader.Advance(kData.size() - 13, test::ExpectOK{});
  EXPECT_EQ(kData.size(), reader.GetPosition());

  // End of stream: zero-byte advance is OK, non-zero is not.
  reader.Advance(0, test::ExpectOK{});
  reader.Advance(1, test::ExpectNotOK{});
  EXPECT_EQ(kData.size(), reader.GetPosition()); // no change
}

TEST(DynamicArrayStreamWriter, Write) {
  DynamicArray<uint8_t> dst;
  DynamicArrayStreamWriter writer(dst);

  auto expectData = [&](std::string const& expected) {
    ASSERT_EQ(dst.size(), expected.size());
    EXPECT_STREQ(
        expected.c_str(),
        std::string(reinterpret_cast<char const*>(dst.data()), dst.size()).c_str());
  };

  // Nothing written yet
  expectData("");
  EXPECT_EQ(0, writer.GetPosition());

  // Write bytes
  writer.Write("Hello, ", 7, test::ExpectOK{});
  expectData("Hello, ");
  EXPECT_EQ(7, writer.GetPosition());

  // Write via StreamWrite utility
  std::array<char, 6> obj = {'w', 'o', 'r', 'l', 'd', '!'};
  StreamWrite(obj, writer, test::ExpectOK{});
  expectData("Hello, world!");
  EXPECT_EQ(13, writer.GetPosition());

  // Write nothing
  writer.Write(nullptr, 0, test::ExpectOK{});
  expectData("Hello, world!"); // no change
  EXPECT_EQ(13, writer.GetPosition()); // no change

  writer.Write(" ", 1, test::ExpectOK{});
  expectData("Hello, world! ");
  EXPECT_EQ(14, writer.GetPosition());

  writer.WriteAt(14, nullptr, 0, test::ExpectOK{});
  writer.WriteAt(14, " ", 1, test::ExpectNotOK{}); // WriteAt can't extend the stream
  writer.WriteAt(13, "  ", 2, test::ExpectNotOK{}); // WriteAt can't extend the stream
  writer.WriteAt(7, "again", 5, test::ExpectOK{});
  expectData("Hello, again! ");
  EXPECT_EQ(14, writer.GetPosition()); // no change

  StreamWriteAt(7, obj, writer, test::ExpectOK{});
  expectData("Hello, world! ");
  EXPECT_EQ(14, writer.GetPosition()); // no change
}

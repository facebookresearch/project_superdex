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

#include <mochi_core/memory/filo_allocator.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/file_utils.h>
#include <mochi_core/utils/group_rw.h>
#include <mochi_core/utils/string_utils.h>

#include <filesystem>
#include <string>
#include <vector>

using namespace mochi;

// Write various datasets to the current group
static void TestDataSetWrite(GroupWriter& writer) {
  // Test data
  double constexpr kDoubleData[] = {1.1, 2.2, 3.3, 4.4};
  float constexpr kFloatData[] = {5.5f, 6.6f, 7.7f, 8.8f};
  int constexpr kIntData[] = {10, 20, 30, 40};
  uint8_t constexpr kByteData[] = {0, 1, 2, 3};
  std::string const kStringData[] = {"hello", "world", "test", "data"};

  // Write 1D datasets
  writer.AddDataSet("double_1d", MakeConstSpan(kDoubleData), test::ExpectOK{});
  writer.AddDataSet("float_1d", MakeConstSpan(kFloatData), test::ExpectOK{});
  writer.AddDataSet("int_1d", MakeConstSpan(kIntData), test::ExpectOK{});
  writer.AddDataSet("byte_1d", MakeConstSpan(kByteData), test::ExpectOK{});
  writer.AddDataSet("string_1d", MakeConstSpan(kStringData), test::ExpectOK{});

  // Write 2D datasets
  size_t constexpr kDims2x2[2] = {2, 2};
  writer.AddDataSet(
      "double_2d", MakeConstSpan(kDoubleData), MakeConstSpan(kDims2x2), test::ExpectOK{});
  writer.AddDataSet(
      "float_2d", MakeConstSpan(kFloatData), MakeConstSpan(kDims2x2), test::ExpectOK{});
  writer.AddDataSet("int_2d", MakeConstSpan(kIntData), MakeConstSpan(kDims2x2), test::ExpectOK{});
  writer.AddDataSet("byte_2d", MakeConstSpan(kByteData), MakeConstSpan(kDims2x2), test::ExpectOK{});
  writer.AddDataSet(
      "string_2d", MakeConstSpan(kStringData), MakeConstSpan(kDims2x2), test::ExpectOK{});
}

// Read back the data from TestDataSetWrite
static void TestDataSetRead([[maybe_unused]] GroupReader& reader) {
  // We should be able to enumerate the datasets in the order they were written
  auto dataSetNames = Join(reader.GetDataSetNames(test::ExpectOK()), ", ");
  ASSERT_STREQ(
      "double_1d, float_1d, int_1d, byte_1d, string_1d, double_2d, float_2d, int_2d, byte_2d, string_2d",
      dataSetNames.c_str());
  DynamicArray<size_t> dims;

  // HasDataSet
  EXPECT_TRUE(reader.HasDataSet("double_1d"));
  EXPECT_TRUE(reader.HasDataSet("float_2d"));
  EXPECT_FALSE(reader.HasDataSet("Double_1d")); // Case sensitive
  EXPECT_FALSE(reader.HasDataSet("No such thing"));

  // 1D DataSets

  DynamicArray<double> doubleData;
  dims = reader.GetDataSetDimensions("double_1d", test::ExpectOK{});
  ASSERT_EQ(1, dims.size());
  EXPECT_EQ(4, dims[0]);
  reader.ReadDataSet("double_1d", doubleData, test::ExpectOK{});
  EXPECT_SPAN_EQ((DynamicArray<double>{1.1, 2.2, 3.3, 4.4}), doubleData);

  DynamicArray<float> floatData;
  dims = reader.GetDataSetDimensions("float_1d", test::ExpectOK{});
  ASSERT_EQ(1, dims.size());
  EXPECT_EQ(4, dims[0]);
  reader.ReadDataSet("float_1d", floatData, test::ExpectOK{});
  EXPECT_SPAN_EQ((DynamicArray<float>{5.5f, 6.6f, 7.7f, 8.8f}), floatData);

  DynamicArray<int> intData;
  dims = reader.GetDataSetDimensions("int_1d", test::ExpectOK{});
  ASSERT_EQ(1, dims.size());
  EXPECT_EQ(4, dims[0]);
  reader.ReadDataSet("int_1d", intData, test::ExpectOK{});
  EXPECT_SPAN_EQ((DynamicArray<int>{10, 20, 30, 40}), intData);

  DynamicArray<uint8_t> byteData;
  dims = reader.GetDataSetDimensions("byte_1d", test::ExpectOK{});
  ASSERT_EQ(1, dims.size());
  EXPECT_EQ(4, dims[0]);
  reader.ReadDataSet("byte_1d", byteData, test::ExpectOK{});
  EXPECT_SPAN_EQ((DynamicArray<uint8_t>{0, 1, 2, 3}), byteData);

  DynamicArray<std::string> stringData;
  dims = reader.GetDataSetDimensions("string_1d", test::ExpectOK{});
  ASSERT_EQ(1, dims.size());
  EXPECT_EQ(4, dims[0]);
  reader.ReadDataSet("string_1d", stringData, test::ExpectOK{});
  EXPECT_SPAN_EQ((DynamicArray<std::string>{"hello", "world", "test", "data"}), stringData);

  // 2D DataSets

  dims = reader.GetDataSetDimensions("double_2d", test::ExpectOK{});
  ASSERT_EQ(2, dims.size());
  EXPECT_EQ(2, dims[0]);
  EXPECT_EQ(2, dims[1]);
  doubleData.clear();
  doubleData.resize(4);
  reader.ReadDataSet("double_2d", MakeSpan(doubleData), test::ExpectOK{});
  EXPECT_SPAN_EQ((DynamicArray<double>{1.1, 2.2, 3.3, 4.4}), doubleData);

  dims = reader.GetDataSetDimensions("float_2d", test::ExpectOK{});
  ASSERT_EQ(2, dims.size());
  EXPECT_EQ(2, dims[0]);
  EXPECT_EQ(2, dims[1]);
  floatData.clear();
  floatData.resize(4);
  reader.ReadDataSet("float_2d", MakeSpan(floatData), test::ExpectOK{});
  EXPECT_SPAN_EQ((DynamicArray<float>{5.5f, 6.6f, 7.7f, 8.8f}), floatData);

  dims = reader.GetDataSetDimensions("int_2d", test::ExpectOK{});
  ASSERT_EQ(2, dims.size());
  EXPECT_EQ(2, dims[0]);
  EXPECT_EQ(2, dims[1]);
  intData.clear();
  intData.resize(4);
  reader.ReadDataSet("int_2d", MakeSpan(intData), test::ExpectOK{});
  EXPECT_SPAN_EQ((DynamicArray<int>{10, 20, 30, 40}), intData);

  dims = reader.GetDataSetDimensions("byte_2d", test::ExpectOK{});
  ASSERT_EQ(2, dims.size());
  EXPECT_EQ(2, dims[0]);
  EXPECT_EQ(2, dims[1]);
  byteData.clear();
  byteData.resize(4);
  reader.ReadDataSet("byte_2d", MakeSpan(byteData), test::ExpectOK{});
  EXPECT_SPAN_EQ((DynamicArray<uint8_t>{0, 1, 2, 3}), byteData);

  dims = reader.GetDataSetDimensions("string_2d", test::ExpectOK{});
  ASSERT_EQ(2, dims.size());
  EXPECT_EQ(2, dims[0]);
  EXPECT_EQ(2, dims[1]);
  stringData.clear();
  stringData.resize(4);
  reader.ReadDataSet("string_2d", MakeSpan(stringData), test::ExpectOK{});
  EXPECT_SPAN_EQ((DynamicArray<std::string>{"hello", "world", "test", "data"}), stringData);
}

// Write various groups within the current group.
// Give each group an attribute so we can prove that we entered it correctly.
static void TestGroupWrite(GroupWriter& writer) {
  {
    auto group = writer.EnterGroup("one", test::ExpectOK{});
    writer.AddAttribute("info", std::string("aaa"), test::ExpectOK{});
  }
  {
    auto group = writer.EnterGroup("two", test::ExpectOK{});
    writer.AddAttribute("info", std::string("bbb"), test::ExpectOK{});
  }
  {
    auto group = writer.EnterGroup("Three", test::ExpectOK{});
    writer.AddAttribute("info", std::string("ccc"), test::ExpectOK{});
  }
  {
    auto group = writer.EnterGroup("FOUR", test::ExpectOK{});
    writer.AddAttribute("info", std::string("ddd"), test::ExpectOK{});
  }
}

// Read everything from TestGroupWrite
static void TestGroupRead(GroupReader& reader) {
  // We should be able to enumerate the groups in the order they were written.
  auto dataSetNames = Join(reader.GetGroupNames(test::ExpectOK()), ", ");
  if (dataSetNames.ends_with(", my_last_group")) {
    // Trim the special group added by FullGroupWriterTest
    dataSetNames.resize(dataSetNames.size() - strlen(", my_last_group"));
  }
  ASSERT_STREQ("one, two, Three, FOUR", dataSetNames.c_str());

  // HasGroup
  EXPECT_TRUE(reader.HasGroup("one"));
  EXPECT_TRUE(reader.HasGroup("two"));
  EXPECT_FALSE(reader.HasGroup("THREE")); // Case sensitive
  EXPECT_FALSE(reader.HasGroup("No such thing")); // Case sensitive

  std::string info;
  {
    auto group = reader.EnterGroup("one", test::ExpectOK{});
    reader.ReadAttribute("info", info, test::ExpectOK{});
    EXPECT_STREQ("aaa", info.c_str());
  }
  {
    auto group = reader.EnterGroup("two", test::ExpectOK{});
    reader.ReadAttribute("info", info, test::ExpectOK{});
    EXPECT_STREQ("bbb", info.c_str());
  }
  {
    auto group = reader.EnterGroup("Three", test::ExpectOK{});
    reader.ReadAttribute("info", info, test::ExpectOK{});
    EXPECT_STREQ("ccc", info.c_str());
  }
  {
    auto group = reader.EnterGroup("FOUR", test::ExpectOK{});
    reader.ReadAttribute("info", info, test::ExpectOK{});
    EXPECT_STREQ("ddd", info.c_str());
  }
}

static void TestAttributeWrite(GroupWriter& writer) {
  // Write various single-value attributes
  writer.AddAttribute("double_attr", 1.1, test::ExpectOK{});
  writer.AddAttribute("float_attr", 2.3f, test::ExpectOK{});
  writer.AddAttribute("int32_attr", int32_t(123), test::ExpectOK{});
  writer.AddAttribute("uint64_attr", 2 * uint64_t(INT_MAX), test::ExpectOK{});
  writer.AddAttribute("string_attr", std::string_view("woot"), test::ExpectOK{});

  // Write various array attributes
  double constexpr myDoubles[] = {1.1, 2.2};
  float constexpr myFloats[] = {2.3f, 3.4f};
  int32_t constexpr myInt32s[] = {123, 456};
  uint64_t constexpr myUint64s[] = {2 * uint64_t(INT_MAX), 3 * uint64_t(INT_MAX)};
  writer.AddAttribute("double_attr_list", MakeConstSpan(myDoubles), test::ExpectOK{});
  writer.AddAttribute("float_attr_list", MakeConstSpan(myFloats), test::ExpectOK{});
  writer.AddAttribute("int32_attr_list", MakeConstSpan(myInt32s), test::ExpectOK{});
  writer.AddAttribute("uint64_attr_list", MakeConstSpan(myUint64s), test::ExpectOK{});
  // NOTE: Array of strings is not currently supported for attributes
}

static void TestAttributeRead(GroupReader& reader) {
  // Read everything from TestAttributeWrite

  EXPECT_TRUE(reader.HasAttribute("double_attr"));
  EXPECT_TRUE(reader.HasAttribute("float_attr_list"));
  EXPECT_FALSE(reader.HasAttribute("Double_attr")); // Case sensitive
  EXPECT_FALSE(reader.HasAttribute("No such thing"));

  double myDouble{};
  reader.ReadAttribute("double_attr", myDouble, test::ExpectOK{});
  EXPECT_EQ(1.1, myDouble);

  float myFloat{};
  reader.ReadAttribute("float_attr", myFloat, test::ExpectOK{});
  EXPECT_EQ(2.3f, myFloat);

  int32_t myInt32{};
  reader.ReadAttribute("int32_attr", myInt32, test::ExpectOK{});
  EXPECT_EQ(123, myInt32);

  uint64_t myUint64{};
  reader.ReadAttribute("uint64_attr", myUint64, test::ExpectOK{});
  EXPECT_EQ(2 * uint64_t(INT_MAX), myUint64);

  std::string myStr;
  reader.ReadAttribute("string_attr", myStr, test::ExpectOK{});
  EXPECT_STREQ("woot", myStr.c_str());

  double myDoubles[2] = {};
  reader.ReadAttribute("double_attr_list", MakeSpan(myDoubles), test::ExpectOK{});
  EXPECT_EQ(1.1, myDoubles[0]);
  EXPECT_EQ(2.2, myDoubles[1]);

  float myFloats[2] = {};
  reader.ReadAttribute("float_attr_list", MakeSpan(myFloats), test::ExpectOK{});
  EXPECT_EQ(2.3f, myFloats[0]);
  EXPECT_EQ(3.4f, myFloats[1]);

  int32_t myInt32s[2] = {};
  reader.ReadAttribute("int32_attr_list", MakeSpan(myInt32s), test::ExpectOK{});
  EXPECT_EQ(123, myInt32s[0]);
  EXPECT_EQ(456, myInt32s[1]);

  uint64_t myUint64s[2] = {};
  reader.ReadAttribute("uint64_attr_list", MakeSpan(myUint64s), test::ExpectOK{});
  EXPECT_EQ(2 * uint64_t(INT_MAX), myUint64s[0]);
  EXPECT_EQ(3 * uint64_t(INT_MAX), myUint64s[1]);
}

static void FullGroupWriterTestImpl(GroupWriter& writer) {
  TestAttributeWrite(writer); // Add attributes to the current group
  TestDataSetWrite(writer); // Add datasets
  TestAttributeWrite(writer); // Add attributes to the last dataset
  TestGroupWrite(writer); // Add groups
}

static void FullGroupReaderTestImpl(GroupReader& reader) {
  // Read all the stuff from FullGroupWriterTestImpl
  TestAttributeRead(reader);
  TestDataSetRead(reader);
  TestAttributeRead(reader);
  TestGroupRead(reader);
}

[[maybe_unused]] static void FullGroupWriterTest(GroupWriter& writer) {
  // Write a bunch of stuff at the root
  FullGroupWriterTestImpl(writer);
  // Repeat within a nested group
  auto group = writer.EnterGroup("my_last_group", test::ExpectOK{});
  FullGroupWriterTestImpl(writer);
}

[[maybe_unused]] static void FullGroupReaderTest(GroupReader& reader) {
  // Read all the stuff from FullGroupWriterTest
  FullGroupReaderTestImpl(reader);
  auto group = reader.EnterGroup("my_last_group", test::ExpectOK{});
  FullGroupReaderTestImpl(reader);
}

static void FullGroupReaderWriterTest(bool readFromBytes, int compression = 0) {
  auto temp = CreateTempFile("group_reader_writer_test", ".h5", test::ExpectOK{});
  {
    Error error;
    auto writer = CreateGroupWriterHDF5(temp.Path().string(), error);
    if constexpr (MOCHI_USE_HDF5) {
      EXPECT_OK(error);
      writer->SetCompression(compression);
      FullGroupWriterTest(*writer);
    } else {
      EXPECT_NOT_OK(error);
    }
  }
  {
    Error error;
    std::unique_ptr<GroupReader> reader;
    DynamicArray<char> fileBytes;
    if (readFromBytes) {
      fileBytes = ReadFileBytes(temp.Path().string(), test::ExpectOK{});
      reader = CreateGroupReaderFromBytesHDF5(fileBytes, error);
    } else {
      reader = CreateGroupReaderHDF5(temp.Path().string(), error);
    }
    if constexpr (MOCHI_USE_HDF5) {
      EXPECT_OK(error);
      FullGroupReaderTest(*reader);
    } else {
      EXPECT_NOT_OK(error);
    }
  }
}

TEST(GroupReaderWriter, H5File) {
  FullGroupReaderWriterTest(/*readFromBytes*/ false);
}

TEST(GroupReaderWriter, H5FileCompression) {
  for (int compression = 1; compression <= GroupWriter::kMaxCompression; ++compression) {
    FullGroupReaderWriterTest(/*readFromBytes*/ false, compression);
  }
}

TEST(GroupReaderWriter, H5Bytes) {
  FullGroupReaderWriterTest(/*readFromBytes*/ true);
}

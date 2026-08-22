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

#include <mochi_core/linear_algebra/matrix.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/defer.h>

#include <picojson/picojson.h>

using namespace mochi;

TEST(MatrixReflection, TypeInfo) {
  // Matrix<float, 3, 3>
  {
    using MatrixT = Matrix<float, 3, 3>;
    auto const& ti = SReflect::GetTypeInfo<MatrixT>();
    // TypeInfo fields
    EXPECT_EQ(ti._coreType, SReflect::CoreType::CT_matrix);
    EXPECT_EQ(ti._alignment, alignof(MatrixT));
    EXPECT_EQ(ti._sizeInBytes, sizeof(MatrixT));
    EXPECT_STREQ("Matrix<float,3,3,Direction::ColMajor,Ownership::Owner,0>", ti._name);
    EXPECT_STREQ(
        "mochi::Matrix<float,3,3,mochi::krylov::Direction::ColMajor,mochi::krylov::Ownership::Owner,0>",
        ti._nameWithNamespace);
    EXPECT_EQ(ti._typeId, SReflect::ComputeTypeId(ti._nameWithNamespace));
    // MatrixTypeInfo fields
    EXPECT_EQ(ti._innerTypeInfo, &SReflect::GetTypeInfo<float>());
    EXPECT_EQ(ti._isRowMajor, false);
    EXPECT_EQ(ti._isNumRowsDynamic, false);
    EXPECT_EQ(ti._isNumColumnsDynamic, false);
    EXPECT_EQ(ti._fixedNumRows, 3);
    EXPECT_EQ(ti._fixedNumColumns, 3);
    EXPECT_EQ(true, ti.IsMemCopySafe());
  }

  // RowVector<float>
  {
    using MatrixT = RowVector<float>;
    auto const& ti = SReflect::GetTypeInfo<MatrixT>();
    // TypeInfo fields
    EXPECT_EQ(ti._coreType, SReflect::CoreType::CT_matrix);
    EXPECT_EQ(ti._alignment, alignof(MatrixT));
    EXPECT_EQ(ti._sizeInBytes, sizeof(MatrixT));
    EXPECT_STREQ("Matrix<float,1,-1,Direction::RowMajor,Ownership::Owner,0>", ti._name);
    EXPECT_STREQ(
        "mochi::Matrix<float,1,-1,mochi::krylov::Direction::RowMajor,mochi::krylov::Ownership::Owner,0>",
        ti._nameWithNamespace);
    EXPECT_EQ(ti._typeId, SReflect::ComputeTypeId(ti._nameWithNamespace));
    // MatrixTypeInfo fields
    EXPECT_EQ(ti._innerTypeInfo, &SReflect::GetTypeInfo<float>());
    EXPECT_EQ(ti._isRowMajor, true);
    EXPECT_EQ(ti._isNumRowsDynamic, false);
    EXPECT_EQ(ti._isNumColumnsDynamic, true);
    EXPECT_EQ(ti._fixedNumRows, 1);
    EXPECT_EQ(ti._fixedNumColumns, 0);
    EXPECT_EQ(false, ti.IsMemCopySafe()); // Can't memcpy a dynamic matirx
  }

  // RowMatrixView<double>
  {
    using MatrixT = RowMatrixView<double>;
    auto const& ti = SReflect::GetTypeInfo<MatrixT>();
    // TypeInfo fields
    EXPECT_EQ(ti._coreType, SReflect::CoreType::CT_matrix);
    EXPECT_EQ(ti._alignment, alignof(MatrixT));
    EXPECT_EQ(ti._sizeInBytes, sizeof(MatrixT));
    EXPECT_STREQ("Matrix<double,-1,-1,Direction::RowMajor,Ownership::View,0>", ti._name);
    EXPECT_STREQ(
        "mochi::Matrix<double,-1,-1,mochi::krylov::Direction::RowMajor,mochi::krylov::Ownership::View,0>",
        ti._nameWithNamespace);
    EXPECT_EQ(ti._typeId, SReflect::ComputeTypeId(ti._nameWithNamespace));
    // MatrixTypeInfo fields
    EXPECT_EQ(ti._innerTypeInfo, &SReflect::GetTypeInfo<double>());
    EXPECT_EQ(ti._isRowMajor, true);
    EXPECT_EQ(ti._isNumRowsDynamic, true);
    EXPECT_EQ(ti._isNumColumnsDynamic, true);
    EXPECT_EQ(ti._fixedNumRows, 0);
    EXPECT_EQ(ti._fixedNumColumns, 0);
    EXPECT_EQ(false, ti.IsMemCopySafe()); // Can't memcpy a view matrix
  }
}

TEST(MatrixReflection, TryResize) {
  // Matrix<float, 3, 3>
  {
    Matrix<float, 3, 3> m;
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    EXPECT_TRUE(ti.TryResize(&m, 3, 3)); // same size
    EXPECT_FALSE(ti.TryResize(&m, 4, 3)); // fixed rows
    EXPECT_FALSE(ti.TryResize(&m, 3, 4)); // fixed columns
  }

  // RowMatrixView<double>
  {
    RowMatrix<double> src(3, 3);
    RowMatrixView<double> m(src);
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    EXPECT_TRUE(ti.TryResize(&m, 3, 3)); // same size
    EXPECT_FALSE(ti.TryResize(&m, 4, 3)); // Not allowed to resize a view matrix
    EXPECT_FALSE(ti.TryResize(&m, 3, 4)); // Not allowed to resize a view matrix
  }

  // ColumnVector<float>
  {
    ColumnVector<float> m(3);
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    EXPECT_TRUE(ti.TryResize(&m, 3, 1)); // same size
    EXPECT_FALSE(ti.TryResize(&m, 3, 2)); // fixed columns
    EXPECT_TRUE(ti.TryResize(&m, 4, 1)); // successfully resize num rows
    EXPECT_EQ(4, m.Rows());
    EXPECT_EQ(1, m.Cols());
  }

  // RowVector<double>
  {
    RowVector<double> m(3);
    auto const& ti = SReflect::GetTypeInfo<decltype(m)>();
    EXPECT_TRUE(ti.TryResize(&m, 1, 3)); // same size
    EXPECT_FALSE(ti.TryResize(&m, 2, 3)); // fixed rows
    EXPECT_TRUE(ti.TryResize(&m, 1, 2)); // successfully resize num cols
    EXPECT_EQ(1, m.Rows());
    EXPECT_EQ(2, m.Cols());
  }
}

TEST(MatrixReflection, Serialization) {
  // NOTE: This function tests JSON serialization. We don't currently test binary serialization
  // directly. We rely on simple_reflection_test for that. The code in this file is already
  // sufficient to show that Simple Reflection can access every value of our Matrix class. Mochi
  // doesn't need to do anything special to support specific serialization formats.

  // RowVector<int, 3>
  {
    RowVector<int, 3> m{{1, 2, 3}};
    EXPECT_STREQ("[1,2,3]", SReflect::ToJsonString(m, /*pretty*/ false).c_str());
    EXPECT_TRUE(SReflect::FromJsonString(m, "[4,5,6]"));
    EXPECT_EQ(4, m(0, 0));
    EXPECT_EQ(5, m(0, 1));
    EXPECT_EQ(6, m(0, 2));
  }

  // ColumnVector<int>
  {
    ColumnVector<int> m{{1, 2, 3}}; // in col-major order
    EXPECT_STREQ("[1,2,3]", SReflect::ToJsonString(m, /*pretty*/ false).c_str());
    EXPECT_TRUE(SReflect::FromJsonString(m, "[4,5,6]"));
    EXPECT_EQ(4, m(0, 0));
    EXPECT_EQ(5, m(1, 0));
    EXPECT_EQ(6, m(2, 0));
  }

  // RowMatrix<int, 2, 3>
  {
    RowMatrix<int, 2, 3> m{{1, 2, 3}, {4, 5, 6}};
    EXPECT_STREQ("[[1,2,3],[4,5,6]]", SReflect::ToJsonString(m, /*pretty*/ false).c_str());
    EXPECT_TRUE(SReflect::FromJsonString(m, "[[4,5,6],[7,8,9]]"));
    EXPECT_EQ(4, m(0, 0));
    EXPECT_EQ(5, m(0, 1));
    EXPECT_EQ(6, m(0, 2));
    EXPECT_EQ(7, m(1, 0));
    EXPECT_EQ(8, m(1, 1));
    EXPECT_EQ(9, m(1, 2));
  }

  // RowMatrixView with dynamic lead dimension
  {
    RowMatrix<int> src{{1, 2, 3}, {4, 5, 6}};
    RowMatrixView<int, krylov::kDynamic, krylov::kDynamic, krylov::kDynamic> m{
        src.Block(0, 0, 2, 2)}; // Dynamic size and lead dimension
    EXPECT_STREQ("[[1,2],[4,5]]", SReflect::ToJsonString(m, /*pretty*/ false).c_str());
    EXPECT_TRUE(SReflect::FromJsonString(m, "[[6,7],[8,9]]"));
    EXPECT_EQ(6, m(0, 0));
    EXPECT_EQ(7, m(0, 1));
    EXPECT_EQ(8, m(1, 0));
    EXPECT_EQ(9, m(1, 1));
  }

  // Matrix<int, 2, 3>
  {
    Matrix<int, 2, 3> m{{1, 4}, {2, 5}, {3, 6}}; // Specified in col-major order
    EXPECT_STREQ("[[1,2,3],[4,5,6]]", SReflect::ToJsonString(m, /*pretty*/ false).c_str());
    EXPECT_TRUE(SReflect::FromJsonString(m, "[[4,5,6],[7,8,9]]"));
    EXPECT_EQ(4, m(0, 0));
    EXPECT_EQ(5, m(0, 1));
    EXPECT_EQ(6, m(0, 2));
    EXPECT_EQ(7, m(1, 0));
    EXPECT_EQ(8, m(1, 1));
    EXPECT_EQ(9, m(1, 2));
  }

  // MatrixView with compile-time lead dimension
  {
    Matrix<int, 2, 3> src{{1, 4}, {2, 5}, {3, 6}}; // Specified in col-major order
    MatrixView<int, 2, 2, krylov::Direction::ColMajor, 2> m{src.data(), 2, 2, 2};
    EXPECT_STREQ("[[1,2],[4,5]]", SReflect::ToJsonString(m, /*pretty*/ false).c_str());
    EXPECT_TRUE(SReflect::FromJsonString(m, "[[6,7],[8,9]]"));
    EXPECT_EQ(6, m(0, 0));
    EXPECT_EQ(7, m(0, 1));
    EXPECT_EQ(8, m(1, 0));
    EXPECT_EQ(9, m(1, 1));
  }
}

// Helper to serialize without templates
static std::string ToJsonStr(SReflect::TypeInfo const& ti, void const* src) {
  picojson::value json;
  ti.Serialize(src, json);
  return json.serialize(/*pretty*/ false);
}

// Helper to deserialize without templates
static void FromJsonStr(SReflect::TypeInfo const& ti, void* dst, std::string const& str) {
  std::istringstream stream{str};
  picojson::value jsonValue;
  picojson::parse(jsonValue, stream);
  int numIssues = 0;
  ti.Deserialize(jsonValue, dst, {}, numIssues);
  EXPECT_EQ(0, numIssues);
}

// Helper to serialize then deserialize without templates
static void TestJsonRoundTrip(SReflect::TypeInfo const& ti, void* obj, std::string const& str) {
  FromJsonStr(ti, obj, str);
  EXPECT_STREQ(str.c_str(), ToJsonStr(ti, obj).c_str());
}

TEST(MatrixReflection, Factory) {
  // RowMatrix<int>
  {
    SReflect::TypeInfo const& ti = SReflect::GetTypeInfo<RowMatrix<int>>();
    // Create a new matrix (using runtime type, not a template)
    void* ptr = ti.New();
    MOCHI_DEFER(ti.Delete(ptr));
    picojson::value json;
    ti.Serialize(ptr, json);
    EXPECT_STREQ("[]", ToJsonStr(ti, ptr).c_str()); // empty by default
    TestJsonRoundTrip(ti, ptr, "[[1,2,3],[4,5,6]]"); // Load and save this
    void* ptr2 = ti.Clone(ptr);
    MOCHI_DEFER(ti.Delete(ptr2));
    EXPECT_NE(ptr, ptr2);
    EXPECT_STREQ("[[1,2,3],[4,5,6]]", ToJsonStr(ti, ptr2).c_str()); // successfully copied
  }

  // ColumnVector<int, 3>
  {
    SReflect::TypeInfo const& ti = SReflect::GetTypeInfo<ColumnVector<int, 3>>();
    // Create a new matrix (using runtime type, not a template)
    void* ptr = ti.New();
    MOCHI_DEFER(ti.Delete(ptr));
    picojson::value json;
    ti.Serialize(ptr, json);
    TestJsonRoundTrip(ti, ptr, "[1,2,3]"); // Load and save this
    void* ptr2 = ti.Clone(ptr);
    MOCHI_DEFER(ti.Delete(ptr2));
    EXPECT_NE(ptr, ptr2);
    EXPECT_STREQ("[1,2,3]", ToJsonStr(ti, ptr2).c_str()); // successfully copied
  }

  // MatrixView<int>
  {
    SReflect::TypeInfo const& ti = SReflect::GetTypeInfo<MatrixView<int>>();
    EXPECT_EQ((void*)nullptr, ti.New()); // Can't create a view matrix this way
  }
}

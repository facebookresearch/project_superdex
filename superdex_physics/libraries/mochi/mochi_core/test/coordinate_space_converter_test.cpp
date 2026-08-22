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
#include <mochi_core/utils/coordinate_space_converter.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/math_utils.h>
#include <mochi_core/utils/matrix_utils.h>
#include <mochi_core/utils/quaternion_utils.h>

using namespace mochi;

// A coordinate convention paired with its three-letter name, eg. {CoordinateSpaceAxes::FLU, "FLU"}.
namespace {
struct AxesEntry {
  CoordinateSpaceAxes axes = {};
  char const* name = nullptr;
};
} // namespace

// Every valid axis convention, taken from the enum's reflection data.
static DynamicArray<AxesEntry> AllAxes() {
  DynamicArray<AxesEntry> entries = [] {
    auto const& enumInfo = SReflect::GetTypeInfo<CoordinateSpaceAxes>();
    DynamicArray<AxesEntry> result;
    result.reserve(enumInfo._items.size());
    for (auto const& item : enumInfo._items) {
      result.push_back({static_cast<CoordinateSpaceAxes>(item._value), item._name});
    }
    return result;
  }();
  return entries;
}

// The unit vector that an axis letter names, in a shared (right, up, forward) semantic basis.
static Real3 SemanticVector(char letter) {
  switch (letter) {
    case 'R':
      return {1_r, 0_r, 0_r};
    case 'L':
      return {-1_r, 0_r, 0_r};
    case 'U':
      return {0_r, 1_r, 0_r};
    case 'D':
      return {0_r, -1_r, 0_r};
    case 'F':
      return {0_r, 0_r, 1_r};
    case 'B':
      return {0_r, 0_r, -1_r};
  }
  ADD_FAILURE() << "Unknown axis letter: " << letter;
  return {};
}

// Express a semantic direction in the coordinates of the space named by @p name.
static Real3 InSpace(char const* name, Real3 const& semantic) {
  return {
      Dot(SemanticVector(name[0]), semantic),
      Dot(SemanticVector(name[1]), semantic),
      Dot(SemanticVector(name[2]), semantic)};
}

// A convention is right-handed when its axes satisfy X cross Y == Z.
static bool IsRightHanded(char const* name) {
  return Cross(SemanticVector(name[0]), SemanticVector(name[1])) == SemanticVector(name[2]);
}

// Transform a point by a homogeneous matrix.
static Real3 ApplyHomogeneous(Matrix4x4r const& m, Real3 const& point) {
  Real4 const result = DotMatVec(m, Real4{point[0], point[1], point[2], 1_r});
  return {result[0], result[1], result[2]};
}

// Rotation of 120 degrees about (1,1,1). Every component is exactly representable.
static Quaternion const kExactRotation{0.5_r, 0.5_r, 0.5_r, 0.5_r};

TEST(CoordinateSpaceConverter, DefaultConstructedConverterIsIdentity) {
  CoordinateSpaceConverter const converter;
  EXPECT_FALSE(converter.FlipsHandedness());
  EXPECT_EQ(1_r, converter.GetScale());
  EXPECT_EQ(Eye<3>(), converter.GetDirectionMatrix());
  EXPECT_EQ(Eye<4>(), converter.GetTransformMatrix());

  Real3 const v{1_r, 2_r, 3_r};
  EXPECT_EQ(v, converter.TranslationToOutput(v));
  EXPECT_EQ(v, converter.DirectionToOutput(v));
}

TEST(CoordinateSpaceConverter, MapsBasisVectorsAndHandednessForAllPairs) {
  for (AxesEntry const& from : AllAxes()) {
    for (AxesEntry const& to : AllAxes()) {
      CoordinateSpaceConverter const converter{{from.axes, 1_r}, {to.axes, 1_r}};

      // Each input axis must come out pointing in the same semantic direction.
      for (int axis = 0; axis < 3; ++axis) {
        EXPECT_EQ(
            InSpace(to.name, SemanticVector(from.name[axis])),
            converter.DirectionToOutput(BasisVector<real, 3>(axis)));
      }

      EXPECT_EQ(IsRightHanded(from.name) != IsRightHanded(to.name), converter.FlipsHandedness());

      if (from.axes == to.axes) {
        // Expect identity transformation
        EXPECT_EQ(Eye<3>(), converter.GetDirectionMatrix());
        EXPECT_EQ(Eye<4>(), converter.GetTransformMatrix());
        EXPECT_FALSE(converter.FlipsHandedness());
      }
    }
  }
}

TEST(CoordinateSpaceConverter, ScaleAppliesToTranslationsButNotDirections) {
  // Default is X-forward, Y-left, Z-up; Unreal is X-forward, Y-right, Z-up. Only Y flips.
  CoordinateSpaceConverter const converter{CoordinateSpace::Default(), CoordinateSpace::Unreal()};
  EXPECT_EQ(100_r, converter.GetScale());
  EXPECT_TRUE(converter.FlipsHandedness());

  Matrix3x3r expectedDirection{};
  expectedDirection[0][0] = 1_r;
  expectedDirection[1][1] = -1_r;
  expectedDirection[2][2] = 1_r;
  EXPECT_EQ(expectedDirection, converter.GetDirectionMatrix());

  Matrix4x4r expectedTransform{};
  expectedTransform[0][0] = 100_r;
  expectedTransform[1][1] = -100_r;
  expectedTransform[2][2] = 100_r;
  expectedTransform[3][3] = 1_r;
  EXPECT_EQ(expectedTransform, converter.GetTransformMatrix());

  Real3 const v{1_r, 2_r, 3_r};
  EXPECT_EQ(Real3(100_r, -200_r, 300_r), converter.TranslationToOutput(v));
  EXPECT_EQ(Real3(1_r, -2_r, 3_r), converter.DirectionToOutput(v));
}

template <typename T>
static void ExpectVectorConversion() {
  CoordinateSpaceConverter const converter{CoordinateSpace::Default(), CoordinateSpace::Unreal()};
  NdArray<T, 3> const v{T(1), T(2), T(3)};
  EXPECT_EQ((NdArray<T, 3>{T(100), T(-200), T(300)}), converter.TranslationToOutput(v));
  EXPECT_EQ((NdArray<T, 3>{T(1), T(-2), T(3)}), converter.DirectionToOutput(v));
}

TEST(CoordinateSpaceConverter, ConvertsVectorsInFloatAndDouble) {
  ExpectVectorConversion<float>();
  ExpectVectorConversion<double>();
}

template <typename T>
static void ExpectFlatSpanConversion() {
  CoordinateSpaceConverter const converter{CoordinateSpace::Default(), CoordinateSpace::Unreal()};

  DynamicArray<T> translations{T(1), T(2), T(3), T(4), T(5), T(6)};
  converter.TranslationsToOutput(MakeSpan(translations), test::ExpectOK{});
  DynamicArray<T> const expectedTranslations{T(100), T(-200), T(300), T(400), T(-500), T(600)};
  EXPECT_EQ(expectedTranslations, translations);

  DynamicArray<T> directions{T(1), T(2), T(3), T(4), T(5), T(6)};
  converter.DirectionsToOutput(MakeSpan(directions), test::ExpectOK{});
  DynamicArray<T> const expectedDirections{T(1), T(-2), T(3), T(4), T(-5), T(6)};
  EXPECT_EQ(expectedDirections, directions);
}

TEST(CoordinateSpaceConverter, ConvertsFlatSpansInFloatAndDouble) {
  ExpectFlatSpanConversion<float>();
  ExpectFlatSpanConversion<double>();
}

TEST(CoordinateSpaceConverter, FlatSpanConversionRejectsBadSizeWithoutMutating) {
  CoordinateSpaceConverter const converter{CoordinateSpace::Default(), CoordinateSpace::Unreal()};

  DynamicArray<real> values{1_r, 2_r, 3_r, 4_r};
  DynamicArray<real> const original = values;

  converter.TranslationsToOutput(MakeSpan(values), test::ExpectNotOK{});
  converter.DirectionsToOutput(MakeSpan(values), test::ExpectNotOK{});
  EXPECT_EQ(original, values);
}

TEST(CoordinateSpaceConverter, FlatSpanConversionHonorsPreexistingError) {
  CoordinateSpaceConverter const converter{CoordinateSpace::Default(), CoordinateSpace::Unreal()};

  DynamicArray<real> values{1_r, 2_r, 3_r};
  DynamicArray<real> const original = values;

  Error error;
  MOCHI_ERROR_SET(error, "pre-existing");
  converter.TranslationsToOutput(MakeSpan(values), error);
  converter.DirectionsToOutput(MakeSpan(values), error);
  EXPECT_EQ(original, values);
}

TEST(CoordinateSpaceConverter, RotationConversionPreservesTheRotationForAllPairs) {
  Real3 const v{1_r, 2_r, 4_r};
  for (AxesEntry const& from : AllAxes()) {
    for (AxesEntry const& to : AllAxes()) {
      CoordinateSpaceConverter const converter{{from.axes, 1_r}, {to.axes, 1_r}};
      Quaternion const converted = converter.RotationToOutput(kExactRotation);

      // Rotating a converted vector must agree with converting the rotated vector.
      EXPECT_EQ(
          converter.DirectionToOutput(kExactRotation * v),
          converted * converter.DirectionToOutput(v));
    }
  }
}

TEST(CoordinateSpaceConverter, TransformRTConversionPreservesTheTransform) {
  TransformRT const transform{kExactRotation, Real3{0.5_r, -1.5_r, 2_r}};
  Real3 const point{1.25_r, -0.5_r, 3_r};

  // One handedness-preserving pair and one handedness-flipping pair.
  for (CoordinateSpace const& to : {CoordinateSpace::Filament(), CoordinateSpace::Unreal()}) {
    CoordinateSpaceConverter const converter{CoordinateSpace::Default(), to};
    TransformRT const converted = converter.TransformToOutput(transform);
    EXPECT_EQ(
        converter.TranslationToOutput(transform.TransformPoint(point)),
        converted.TransformPoint(converter.TranslationToOutput(point)));
  }
}

TEST(CoordinateSpaceConverter, HomogeneousTransformMatchesPointConversion) {
  // A power-of-two scale keeps the inverse conversion exact.
  CoordinateSpaceConverter const converter{
      {CoordinateSpaceAxes::FLU, 1_r}, {CoordinateSpaceAxes::URF, 4_r}};

  // Non-uniform scale plus a translation, laid out row-major.
  Matrix4x4r value{};
  value[0][0] = 2_r;
  value[1][1] = -0.5_r;
  value[2][2] = 8_r;
  value[0][3] = 0.5_r;
  value[1][3] = -1.5_r;
  value[2][3] = 2.5_r;
  value[3][3] = 1_r;

  Real3 const point{1.25_r, -0.5_r, 3_r};
  EXPECT_EQ(
      converter.TranslationToOutput(ApplyHomogeneous(value, point)),
      ApplyHomogeneous(converter.TransformToOutput(value), converter.TranslationToOutput(point)));
}

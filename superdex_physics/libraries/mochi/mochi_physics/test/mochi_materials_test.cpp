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

#include <mochi_physics/src/mochi_materials.h>
#include <mochi_physics/src/mochi_soft.h>

#include <mochi_core/test/mochi_test_helpers.h>

#include <gtest/gtest.h>

#include <array>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <utility>

using namespace mochi;
using namespace mochi::materials;

template <std::size_t... Is>
constexpr bool MaterialVariantMatchesSoftMaterialType(std::index_sequence<Is...>) {
  return (
      (materials::utils::MaterialTraits<std::variant_alternative_t<Is, AnyMaterialParams>>::kType ==
       static_cast<SoftMaterialType>(Is)) &&
      ...);
}

static_assert(
    std::variant_size_v<AnyMaterialParams> == static_cast<std::size_t>(SoftMaterialType::Count));
static_assert(MaterialVariantMatchesSoftMaterialType(
    std::make_index_sequence<std::variant_size_v<AnyMaterialParams>>{}));

// The tests below exercise the per-element material machinery (homogeneous round-trip,
// heterogeneous field scatter, PSD-strategy consistency rules, single-element updates, and the
// compatibility predicate) uniformly across all material models. The per-material knowledge is
// isolated in a MaterialTraits<ParamsT> specialization so each behavior is written once as a
// TYPED_TEST.
namespace {

// Non-default sentinels (distinct from the library defaults so a missed copy is detectable).
constexpr real kTestDensity = 1234_r;
// Mutually distinct, non-default damping sentinels: distinct so the two material-wide fields
// cannot be transposed undetected; non-default (nonzero) so a dropped copy is detectable.
constexpr real kTestMassDamping = 0.25_r; ///< Mass damping coefficient `α` [1/s].
constexpr real kTestStiffnessDamping = 0.01_r; ///< Stiffness damping coefficient `β` [s].
// Non-default (defaults to `false`) so a dropped copy of the flag is detectable.
constexpr bool kTestStiffnessDampingIncludeGeometricTerm = true;
constexpr real kBaseValue = 1500_r; ///< Homogeneous base representative scalar [Pa].
constexpr real kBasePoisson = 0.3_r; ///< Poisson ratio of every homogeneous base.
constexpr real kElemValue0 = 2000_r; ///< Heterogeneous field value at element 0 [Pa].
constexpr real kElemValue1 = 3000_r; ///< Heterogeneous field value at element 1 [Pa].

// Sets the common (non-per-material) fields shared by every test material to their sentinels.
void SetCommonMaterialParams(SoftMaterialParams& p) {
  p.density = kTestDensity;
  p.massDampingCoefficient = kTestMassDamping;
  p.stiffnessDampingCoefficient = kTestStiffnessDamping;
  p.stiffnessDampingIncludeGeometricTerm = kTestStiffnessDampingIncludeGeometricTerm;
}

template <int kN>
std::array<MaterialPsdStrategy, kN> FilledPsd(MaterialPsdStrategy psd) {
  std::array<MaterialPsdStrategy, kN> out{};
  out.fill(psd);
  return out;
}

void ExpectCommonMaterialParams(
    SoftMaterialParams const& got,
    SoftMaterialType expectedType,
    real expectedDensity) {
  EXPECT_EQ(got.type, expectedType);
  EXPECT_EQ(got.density, expectedDensity);
  EXPECT_EQ(got.massDampingCoefficient, kTestMassDamping);
  EXPECT_EQ(got.stiffnessDampingCoefficient, kTestStiffnessDamping);
  EXPECT_EQ(got.stiffnessDampingIncludeGeometricTerm, kTestStiffnessDampingIncludeGeometricTerm);
}

void ExpectCommonMaterialParams(SoftMaterialParams const& got, SoftMaterialParams const& expected) {
  ExpectCommonMaterialParams(got, expected.type, expected.density);
}

template <typename ParamsT, int kSkip = 0>
constexpr MaterialPsdStrategy ExplicitPsdStrategy() {
  int skipped = 0;
  for (int i = 0; i < static_cast<int>(MaterialPsdStrategy::Count); ++i) {
    auto const psd = static_cast<MaterialPsdStrategy>(i);
    if (psd == MaterialPsdStrategy::MaterialDefault || psd == MaterialPsdStrategy::None ||
        !materials::IsPsdStrategySupported<ParamsT>(psd)) {
      continue;
    }
    if (skipped++ == kSkip) {
      return psd;
    }
  }
  return MaterialPsdStrategy::MaterialDefault;
}

/// @brief Per-material test trait. Each specialization provides the value/PSD factories
/// (@c MakeBase, @c MakeField), the stored-PSD accessor (@c StoredPsd), and complete
/// value-verification hooks (@c ExpectBase for a homogeneous base, @c ExpectElement for a
/// scattered field element).
/// @c kPsdA / @c kPsdB are production-supported explicit PSD strategies for the material. @c
/// kNumPsd is the number of independent PSD axes (2 only for Active Neo-Hookean: passive + active).
/// Specialized per material; the primary template is intentionally undefined.
template <typename ParamsT>
struct MaterialTraits;

// Shared trait for the Lamé-based materials (NeoHookean, StVenantKirchhoff, LinearElastic), which
// store per-element Young's modulus / Poisson ratio. Derived traits add the valid PSD strategies;
// LinearElastic adds none because it opts out of PSD-consistency rules.
template <typename ParamsT>
struct LameTraits {
  static constexpr int kNumPsd = 1;
  static constexpr bool kHasPsd = !std::is_same_v<ParamsT, LinearElasticMaterialParams>;
  static constexpr MaterialPsdStrategy kPsdA =
      kHasPsd ? ExplicitPsdStrategy<ParamsT>() : MaterialPsdStrategy::MaterialDefault;
  static constexpr MaterialPsdStrategy kPsdB =
      kHasPsd ? ExplicitPsdStrategy<ParamsT, 1>() : MaterialPsdStrategy::MaterialDefault;
  static constexpr real kRTol = 1e-4_r; // Young's modulus / Poisson ratio round-trip through λ, μ.

  static SoftMaterialParams MakeBase(
      std::array<MaterialPsdStrategy, kNumPsd> psd,
      real value = kBaseValue) {
    SoftMaterialParams p;
    p.type = materials::utils::MaterialTraits<ParamsT>::kType;
    // p is non-const, so casting away const on the production accessor's result is well-defined
    // (mirrors the non-const soft::details::GetMatchingPerElementParams overload).
    auto& sub = const_cast<ParamsT&>(soft::details::GetTypedMaterialParams<ParamsT>(p));
    sub.youngsModulus = value;
    sub.poissonRatio = kBasePoisson;
    if constexpr (!std::is_same_v<ParamsT, LinearElasticMaterialParams>) {
      sub.psdStrategy = psd[0];
    }
    SetCommonMaterialParams(p);
    return p;
  }

  static PerElementSoftMaterialData MakeField(
      MaterialPsdStrategy psd = MaterialPsdStrategy::MaterialDefault) {
    PerElementSoftMaterialData f;
    f.type = materials::utils::MaterialTraits<ParamsT>::kType;
    f.psdStrategy = psd;
    f.youngsModulus = {kElemValue0, kElemValue1};
    f.poissonRatio = {0.2_r, 0.3_r}; // Distinct per element to exercise the scatter.
    return f;
  }

  static std::array<MaterialPsdStrategy, kNumPsd> StoredPsd(CSoftMaterialParams const& m) {
    return {soft::details::GetMatchingPerElementParams<ParamsT>(m).psdStrategy};
  }

  static std::array<MaterialPsdStrategy, kNumPsd> DefaultPsd() {
    return {ParamsT{}.psdStrategy};
  }

  static void ExpectBase(SoftMaterialParams const& got, SoftMaterialParams const& expected) {
    ExpectCommonMaterialParams(got, expected);
    auto const& g = soft::details::GetTypedMaterialParams<ParamsT>(got);
    auto const& e = soft::details::GetTypedMaterialParams<ParamsT>(expected);
    EXPECT_NEAR_RTOL(g.youngsModulus, e.youngsModulus, kRTol);
    EXPECT_NEAR_RTOL(g.poissonRatio, e.poissonRatio, kRTol);
    if constexpr (!std::is_same_v<ParamsT, LinearElasticMaterialParams>) {
      EXPECT_EQ(g.psdStrategy, e.psdStrategy);
    }
  }

  static void ExpectElement(
      SoftMaterialParams const& got,
      PerElementSoftMaterialData const& field,
      int idx,
      std::array<MaterialPsdStrategy, kNumPsd> expectedPsd) {
    ExpectCommonMaterialParams(got, materials::utils::MaterialTraits<ParamsT>::kType, kTestDensity);
    auto const& sub = soft::details::GetTypedMaterialParams<ParamsT>(got);
    EXPECT_NEAR_RTOL(sub.youngsModulus, field.youngsModulus[idx], kRTol);
    EXPECT_NEAR_RTOL(sub.poissonRatio, field.poissonRatio[idx], kRTol);
    if constexpr (!std::is_same_v<ParamsT, LinearElasticMaterialParams>) {
      EXPECT_EQ(sub.psdStrategy, expectedPsd[0]);
    }
  }
};

template <>
struct MaterialTraits<NeoHookeanMaterialParams> : LameTraits<NeoHookeanMaterialParams> {
  static constexpr char const* kName = "NeoHookean";
};

template <>
struct MaterialTraits<StVenantKirchhoffMaterialParams>
    : LameTraits<StVenantKirchhoffMaterialParams> {
  static constexpr char const* kName = "StVenantKirchhoff";
};

// LinearElastic has no psdStrategy member and opts out of PSD-consistency rules (no kPsdA/kPsdB),
// so it only participates in the Homogeneous suite plus a dedicated standalone carve-out test.
template <>
struct MaterialTraits<LinearElasticMaterialParams> : LameTraits<LinearElasticMaterialParams> {
  static constexpr char const* kName = "LinearElastic";
};

template <>
struct MaterialTraits<ActiveNeoHookeanMaterialParams> {
  static constexpr char const* kName = "ActiveNeoHookean";
  static constexpr int kNumPsd = 2; // passive (Lamé) + active (anisotropic).
  static constexpr bool kHasPsd = true;
  static constexpr MaterialPsdStrategy kPsdA =
      ExplicitPsdStrategy<ActiveNeoHookeanMaterialParams>();
  static constexpr MaterialPsdStrategy kPsdB =
      ExplicitPsdStrategy<ActiveNeoHookeanMaterialParams, 1>();
  static constexpr real kRTol = 1e-4_r;

  static SoftMaterialParams MakeBase(
      std::array<MaterialPsdStrategy, kNumPsd> psd,
      real value = kBaseValue) {
    SoftMaterialParams p;
    p.type = SoftMaterialType::ActiveNeoHookean;
    p.activeNeoHookean.passiveIsotropic.youngsModulus = value;
    p.activeNeoHookean.passiveIsotropic.poissonRatio = kBasePoisson;
    p.activeNeoHookean.passiveIsotropic.psdStrategy = psd[0];
    p.activeNeoHookean.activeAnisotropic.alpha = value * (1_r / 100_r);
    p.activeNeoHookean.activeAnisotropic.length = value * (1_r / 10000_r);
    p.activeNeoHookean.activeAnisotropic.anisoDir =
        ActiveAnisoArapMaterialParams::ComputeFiberDirection(
            value * (1_r / 10000_r), value * (2_r / 10000_r));
    p.activeNeoHookean.activeAnisotropic.psdStrategy = psd[1];
    SetCommonMaterialParams(p);
    return p;
  }

  static PerElementSoftMaterialData MakeField(
      MaterialPsdStrategy psd = MaterialPsdStrategy::MaterialDefault) {
    PerElementSoftMaterialData f;
    f.type = SoftMaterialType::ActiveNeoHookean;
    f.psdStrategy = psd;
    f.youngsModulus = {kElemValue0, kElemValue1};
    f.poissonRatio = {0.2_r, 0.3_r};
    f.anisoAlpha = {40_r, 50_r};
    f.anisoLength = {0.4_r, 0.6_r};
    f.anisoTheta = {0.1_r, 0.2_r};
    f.anisoPhi = {0.3_r, 0.4_r};
    return f;
  }

  static std::array<MaterialPsdStrategy, kNumPsd> StoredPsd(CSoftMaterialParams const& m) {
    auto const& pe = soft::details::GetMatchingPerElementParams<ActiveNeoHookeanMaterialParams>(m);
    return {pe.lame.psdStrategy, pe.aniso.psdStrategy};
  }

  static std::array<MaterialPsdStrategy, kNumPsd> DefaultPsd() {
    return {NeoHookeanMaterialParams{}.psdStrategy, ActiveAnisoArapMaterialParams{}.psdStrategy};
  }

  static void ExpectBase(SoftMaterialParams const& got, SoftMaterialParams const& expected) {
    ExpectCommonMaterialParams(got, expected);
    auto const& g = got.activeNeoHookean;
    auto const& e = expected.activeNeoHookean;
    EXPECT_NEAR_RTOL(g.passiveIsotropic.youngsModulus, e.passiveIsotropic.youngsModulus, kRTol);
    EXPECT_NEAR_RTOL(g.passiveIsotropic.poissonRatio, e.passiveIsotropic.poissonRatio, kRTol);
    EXPECT_EQ(g.passiveIsotropic.psdStrategy, e.passiveIsotropic.psdStrategy);
    EXPECT_NEAR_RTOL(g.activeAnisotropic.alpha, e.activeAnisotropic.alpha, kRTol);
    EXPECT_NEAR_RTOL(g.activeAnisotropic.length, e.activeAnisotropic.length, kRTol);
    EXPECT_NEAR_EQ(g.activeAnisotropic.anisoDir, e.activeAnisotropic.anisoDir);
    EXPECT_EQ(g.activeAnisotropic.psdStrategy, e.activeAnisotropic.psdStrategy);
  }

  static void ExpectElement(
      SoftMaterialParams const& got,
      PerElementSoftMaterialData const& field,
      int idx,
      std::array<MaterialPsdStrategy, kNumPsd> expectedPsd) {
    ExpectCommonMaterialParams(got, SoftMaterialType::ActiveNeoHookean, kTestDensity);
    auto const& p = got.activeNeoHookean;
    EXPECT_NEAR_RTOL(p.passiveIsotropic.youngsModulus, field.youngsModulus[idx], kRTol);
    EXPECT_NEAR_RTOL(p.passiveIsotropic.poissonRatio, field.poissonRatio[idx], kRTol);
    EXPECT_EQ(p.passiveIsotropic.psdStrategy, expectedPsd[0]);
    EXPECT_NEAR_RTOL(p.activeAnisotropic.alpha, field.anisoAlpha[idx], kRTol);
    EXPECT_NEAR_RTOL(p.activeAnisotropic.length, field.anisoLength[idx], kRTol);
    // Fiber direction is stored (not re-derived), so it matches the input angles exactly.
    EXPECT_NEAR_EQ(
        p.activeAnisotropic.anisoDir,
        ActiveAnisoArapMaterialParams::ComputeFiberDirection(
            field.anisoTheta[idx], field.anisoPhi[idx]));
    EXPECT_EQ(p.activeAnisotropic.psdStrategy, expectedPsd[1]);
  }
};

template <>
struct MaterialTraits<ActiveShapeTargetingArapMaterialParams> {
  static constexpr char const* kName = "ActiveShapeTargetingArap";
  static constexpr int kNumPsd = 1;
  static constexpr bool kHasPsd = true;
  static constexpr MaterialPsdStrategy kPsdA =
      ExplicitPsdStrategy<ActiveShapeTargetingArapMaterialParams>();
  static constexpr MaterialPsdStrategy kPsdB =
      ExplicitPsdStrategy<ActiveShapeTargetingArapMaterialParams, 1>();
  static constexpr real kRTol = 1e-6_r; // Stiffness / tensor are stored directly (no conversion).

  static SoftMaterialParams MakeBase(
      std::array<MaterialPsdStrategy, kNumPsd> psd,
      real value = kBaseValue) {
    SoftMaterialParams p;
    p.type = SoftMaterialType::ActiveShapeTargetingArap;
    p.activeShapeTargetingArap.stiffness = value;
    for (int i = 0; i < 6; ++i) {
      p.activeShapeTargetingArap.shapeTargetTensor[i] = value + StaticCast<real>(i + 1);
    }
    p.activeShapeTargetingArap.psdStrategy = psd[0];
    SetCommonMaterialParams(p);
    return p;
  }

  static PerElementSoftMaterialData MakeField(
      MaterialPsdStrategy psd = MaterialPsdStrategy::MaterialDefault) {
    PerElementSoftMaterialData f;
    f.type = SoftMaterialType::ActiveShapeTargetingArap;
    f.psdStrategy = psd;
    f.arapStiffness = {kElemValue0, kElemValue1};
    // 6 distinct entries per element so the shape-target-tensor scatter is exercised.
    f.shapeTargetTensor = {
        0.1_r, 0.2_r, 0.3_r, 0.4_r, 0.5_r, 0.6_r, 0.7_r, 0.8_r, 0.9_r, 1.0_r, 1.1_r, 1.2_r};
    return f;
  }

  static std::array<MaterialPsdStrategy, kNumPsd> StoredPsd(CSoftMaterialParams const& m) {
    return {soft::details::GetMatchingPerElementParams<ActiveShapeTargetingArapMaterialParams>(m)
                .psdStrategy};
  }

  static std::array<MaterialPsdStrategy, kNumPsd> DefaultPsd() {
    return {ActiveShapeTargetingArapMaterialParams{}.psdStrategy};
  }

  static void ExpectBase(SoftMaterialParams const& got, SoftMaterialParams const& expected) {
    ExpectCommonMaterialParams(got, expected);
    EXPECT_NEAR_RTOL(
        got.activeShapeTargetingArap.stiffness, expected.activeShapeTargetingArap.stiffness, kRTol);
    for (int k = 0; k < 6; ++k) {
      EXPECT_NEAR_RTOL(
          got.activeShapeTargetingArap.shapeTargetTensor[k],
          expected.activeShapeTargetingArap.shapeTargetTensor[k],
          kRTol);
    }
    EXPECT_EQ(
        got.activeShapeTargetingArap.psdStrategy, expected.activeShapeTargetingArap.psdStrategy);
  }

  static void ExpectElement(
      SoftMaterialParams const& got,
      PerElementSoftMaterialData const& field,
      int idx,
      std::array<MaterialPsdStrategy, kNumPsd> expectedPsd) {
    ExpectCommonMaterialParams(got, SoftMaterialType::ActiveShapeTargetingArap, kTestDensity);
    EXPECT_NEAR_RTOL(got.activeShapeTargetingArap.stiffness, field.arapStiffness[idx], kRTol);
    for (int k = 0; k < 6; ++k) {
      EXPECT_NEAR_RTOL(
          got.activeShapeTargetingArap.shapeTargetTensor[k],
          field.shapeTargetTensor[idx * 6 + k],
          kRTol);
    }
    EXPECT_EQ(got.activeShapeTargetingArap.psdStrategy, expectedPsd[0]);
  }
};

template <>
struct MaterialTraits<ArapMaterialParams> {
  static constexpr char const* kName = "Arap";
  static constexpr int kNumPsd = 1;
  static constexpr bool kHasPsd = true;
  static constexpr MaterialPsdStrategy kPsdA = ExplicitPsdStrategy<ArapMaterialParams>();
  static constexpr MaterialPsdStrategy kPsdB = ExplicitPsdStrategy<ArapMaterialParams, 1>();
  static constexpr real kRTol = 1e-6_r;

  static SoftMaterialParams MakeBase(
      std::array<MaterialPsdStrategy, kNumPsd> psd,
      real value = kBaseValue) {
    SoftMaterialParams p;
    p.type = SoftMaterialType::Arap;
    p.arap.stiffness = value;
    p.arap.psdStrategy = psd[0];
    SetCommonMaterialParams(p);
    return p;
  }

  static PerElementSoftMaterialData MakeField(
      MaterialPsdStrategy psd = MaterialPsdStrategy::MaterialDefault) {
    PerElementSoftMaterialData f;
    f.type = SoftMaterialType::Arap;
    f.psdStrategy = psd;
    f.arapStiffness = {kElemValue0, kElemValue1};
    return f;
  }

  static std::array<MaterialPsdStrategy, kNumPsd> StoredPsd(CSoftMaterialParams const& m) {
    return {soft::details::GetMatchingPerElementParams<ArapMaterialParams>(m).psdStrategy};
  }

  static std::array<MaterialPsdStrategy, kNumPsd> DefaultPsd() {
    return {ArapMaterialParams{}.psdStrategy};
  }

  static void ExpectBase(SoftMaterialParams const& got, SoftMaterialParams const& expected) {
    ExpectCommonMaterialParams(got, expected);
    EXPECT_NEAR_RTOL(got.arap.stiffness, expected.arap.stiffness, kRTol);
    EXPECT_EQ(got.arap.psdStrategy, expected.arap.psdStrategy);
  }

  static void ExpectElement(
      SoftMaterialParams const& got,
      PerElementSoftMaterialData const& field,
      int idx,
      std::array<MaterialPsdStrategy, kNumPsd> expectedPsd) {
    ExpectCommonMaterialParams(got, SoftMaterialType::Arap, kTestDensity);
    EXPECT_NEAR_RTOL(got.arap.stiffness, field.arapStiffness[idx], kRTol);
    EXPECT_EQ(got.arap.psdStrategy, expectedPsd[0]);
  }
};

// Custom gtest type-name suffixes (e.g. "MaterialPsdConsistency/Arap.FieldSetRules").
struct MaterialTypeNames {
  template <typename ParamsT>
  static std::string GetName(int) {
    return MaterialTraits<ParamsT>::kName;
  }
};

CSoftMaterialParams MakeMaterial(SoftMaterialParams const& params) {
  CSoftMaterialParams material;
  soft::SetMaterialParams(params, material);
  return material;
}

template <typename Tr>
CSoftMaterialParams MakeMaterial(
    std::array<MaterialPsdStrategy, Tr::kNumPsd> psd,
    real value = kBaseValue) {
  return MakeMaterial(Tr::MakeBase(psd, value));
}

SoftMaterialParams GetMaterial(CSoftMaterialParams const& material) {
  SoftMaterialParams params;
  soft::GetMaterialParams(material, params);
  return params;
}

SoftMaterialParams GetElement(CSoftMaterialParams const& material, int elementIndex) {
  SoftMaterialParams params;
  soft::GetMaterialParamsField(material, elementIndex, params);
  return params;
}

// Corrupts the first value-bearing array of a field so it fails value (not size) validation.
void CorruptFirstValue(PerElementSoftMaterialData& field) {
  if (!field.youngsModulus.empty()) {
    field.youngsModulus[0] = -1_r; // Young's modulus must be positive.
  } else {
    field.arapStiffness[0] = -1_r; // ARAP stiffness must be positive.
  }
}

template <typename VariantT>
struct VariantTypes;

template <typename... Ts>
struct VariantTypes<std::variant<Ts...>> {
  using Type = ::testing::Types<Ts...>;
};

using AllMaterials = VariantTypes<AnyMaterialParams>::Type;

template <typename ParamsT>
class MaterialHomogeneous : public ::testing::Test {};
template <typename ParamsT>
class MaterialPsdConsistency : public ::testing::Test {};
template <typename ParamsT>
class MaterialElementUpdate : public ::testing::Test {};
template <typename ParamsT>
class MaterialCompatibility : public ::testing::Test {};

TYPED_TEST_SUITE(MaterialHomogeneous, AllMaterials, MaterialTypeNames);
TYPED_TEST_SUITE(MaterialPsdConsistency, AllMaterials, MaterialTypeNames);
TYPED_TEST_SUITE(MaterialElementUpdate, AllMaterials, MaterialTypeNames);
TYPED_TEST_SUITE(MaterialCompatibility, AllMaterials, MaterialTypeNames);

} // namespace

// Verifies homogeneous material params round-trip and reset per-element state for every material.
TYPED_TEST(MaterialHomogeneous, RoundTripsAndResetsFieldState) {
  using Tr = MaterialTraits<TypeParam>;
  constexpr int kN = Tr::kNumPsd;
  auto const homogeneousBase = Tr::MakeBase(FilledPsd<kN>(Tr::kPsdA));

  auto material = MakeMaterial(homogeneousBase);

  // GetMaterialParams returns the homogeneous base (type, density, value).
  Tr::ExpectBase(GetMaterial(material), homogeneousBase);

  // A homogeneous (size-1) field broadcasts to any element index.
  Tr::ExpectBase(GetElement(material, 7), homogeneousBase);

  // Re-setting the homogeneous base discards a prior per-element field.
  auto field = Tr::MakeField();
  auto const before = GetElement(material, 0);
  soft::SetMaterialParamsField(&field, /*numElements*/ 3, material, test::ExpectNotOK{});
  Tr::ExpectBase(GetElement(material, 0), before);

  // A null per-element field is likewise rejected without modifying state.
  soft::SetMaterialParamsField(
      /*materialField*/ nullptr, /*numElements*/ 2, material, test::ExpectNotOK{});
  Tr::ExpectBase(GetElement(material, 0), before);

  soft::SetMaterialParamsField(&field, /*numElements*/ 2, material, test::ExpectOK{});

  // Field took effect.
  Tr::ExpectBase(GetMaterial(material), homogeneousBase);
  Tr::ExpectElement(GetElement(material, 1), field, 1, Tr::StoredPsd(material));
  soft::SetMaterialParams(homogeneousBase, material);
  Tr::ExpectBase(GetElement(material, 1), homogeneousBase);
}

// Verifies the bulk per-element field setter rejects out-of-range values and leaves state
// unchanged.
TYPED_TEST(MaterialHomogeneous, BulkFieldRejectsInvalidValuesLeavingStateUnchanged) {
  using Tr = MaterialTraits<TypeParam>;
  constexpr int kN = Tr::kNumPsd;

  auto material = MakeMaterial<Tr>(FilledPsd<kN>(Tr::kPsdA));
  auto const before = GetElement(material, 0);

  // A negative value in any populated field array must be rejected by the shared validator,
  // leaving the previously valid material untouched.
  auto field = Tr::MakeField(Tr::kPsdA);
  CorruptFirstValue(field);
  soft::SetMaterialParamsField(&field, /*numElements*/ 2, material, test::ExpectNotOK{});
  Tr::ExpectBase(GetElement(material, 0), before);
}

// Verifies per-element field PSD inheritance, compatibility, rejection, and state preservation.
TYPED_TEST(MaterialPsdConsistency, FieldSetRules) {
  using Tr = MaterialTraits<TypeParam>;
  if constexpr (Tr::kHasPsd) {
    constexpr int kN = Tr::kNumPsd;

    // A MaterialDefault field inherits the base's (per-axis) explicit strategy, and per-element
    // values are scattered correctly.
    {
      auto basePsd = FilledPsd<kN>(Tr::kPsdA);
      if constexpr (kN >= 2) {
        basePsd[kN - 1] = Tr::kPsdB; // Distinct axes prove each is inherited independently.
      }
      auto material = MakeMaterial<Tr>(basePsd);
      auto field = Tr::MakeField(MaterialPsdStrategy::MaterialDefault);
      soft::SetMaterialParamsField(&field, /*numElements*/ 2, material, test::ExpectOK{});

      // GetElement returns the resolved/effective PSD strategy, not MaterialDefault.
      EXPECT_EQ(Tr::StoredPsd(material), basePsd);
      Tr::ExpectElement(GetElement(material, 0), field, 0, basePsd);
      Tr::ExpectElement(GetElement(material, 1), field, 1, basePsd);
    }

    // Every supported matching base/field strategy is accepted.
    for (int i = 0; i < static_cast<int>(MaterialPsdStrategy::Count); ++i) {
      auto const psd = static_cast<MaterialPsdStrategy>(i);
      if (!materials::IsPsdStrategySupported<TypeParam>(psd)) {
        continue;
      }
      auto material = MakeMaterial<Tr>(FilledPsd<kN>(psd));
      auto const expectedPsd = Tr::StoredPsd(material);
      auto field = Tr::MakeField(psd);
      soft::SetMaterialParamsField(&field, /*numElements*/ 2, material, test::ExpectOK{});
      EXPECT_EQ(Tr::StoredPsd(material), expectedPsd);
    }

    // An explicit field strategy conflicting with the explicit base is rejected (each axis
    // checked).
    for (int axis = 0; axis < kN; ++axis) {
      auto basePsd = FilledPsd<kN>(Tr::kPsdA);
      basePsd[axis] = Tr::kPsdB;
      auto material = MakeMaterial<Tr>(basePsd);
      auto const before = GetElement(material, 0);
      auto const storedPsdBefore = Tr::StoredPsd(material);

      auto field = Tr::MakeField(Tr::kPsdA);
      soft::SetMaterialParamsField(&field, /*numElements*/ 2, material, test::ExpectNotOK{});

      Tr::ExpectBase(GetElement(material, 0), before);
      EXPECT_EQ(Tr::StoredPsd(material), storedPsdBefore);
    }

    // An explicit field strategy is accepted when the base strategy is left at MaterialDefault.
    {
      auto material = MakeMaterial<Tr>(FilledPsd<kN>(MaterialPsdStrategy::MaterialDefault));

      // A homogeneous MaterialDefault base resolves to the model's concrete default on read.
      EXPECT_EQ(Tr::StoredPsd(material), Tr::DefaultPsd());

      auto field = Tr::MakeField(Tr::kPsdA);
      soft::SetMaterialParamsField(&field, /*numElements*/ 2, material, test::ExpectOK{});
      EXPECT_EQ(Tr::StoredPsd(material), FilledPsd<kN>(Tr::kPsdA));
    }
  }
}

// Verifies single-element updates grow homogeneous storage while preserving neighbor values.
TYPED_TEST(MaterialElementUpdate, GrowsFromHomogeneousPreservingNeighbor) {
  using Tr = MaterialTraits<TypeParam>;
  constexpr int kN = Tr::kNumPsd;

  constexpr MaterialPsdStrategy kPsd = Tr::kPsdA;

  auto const base = Tr::MakeBase(FilledPsd<kN>(kPsd));
  auto material = MakeMaterial(base); // Homogeneous (size 1).

  // Updating a single element grows the per-element arrays, fills untouched elements with the
  // homogeneous base value, and preserves the PSD strategy.
  auto const updated = Tr::MakeBase(FilledPsd<kN>(kPsd), 4000_r);
  soft::SetMaterialParamsField(updated, 1, 2, material, test::ExpectOK{});

  if constexpr (Tr::kHasPsd) {
    EXPECT_EQ(Tr::StoredPsd(material), FilledPsd<kN>(kPsd));
  }
  Tr::ExpectBase(GetElement(material, 0), base); // Grown-fill keeps the homogeneous base.
  Tr::ExpectBase(GetElement(material, 1), updated); // Updated element holds the new value.
}

// Verifies single-element updates reject out-of-range values and leave state unchanged.
TYPED_TEST(MaterialElementUpdate, RejectsInvalidValuesLeavingStateUnchanged) {
  using Tr = MaterialTraits<TypeParam>;
  constexpr int kN = Tr::kNumPsd;

  auto material = MakeMaterial<Tr>(FilledPsd<kN>(Tr::kPsdA));
  auto const before = GetElement(material, 0);

  // A negative scalar makes every model's value(s) invalid (Young's modulus / stiffness / alpha),
  // so the shared validator must reject the update and leave state untouched.
  auto const invalid = Tr::MakeBase(FilledPsd<kN>(Tr::kPsdA), -1_r);
  soft::SetMaterialParamsField(
      invalid, /*elementIndex*/ 0, /*numElements*/ 2, material, test::ExpectNotOK{});
  Tr::ExpectBase(GetElement(material, 0), before);
}

// Verifies single-element updates do not disturb pre-existing heterogeneous field values.
TYPED_TEST(MaterialElementUpdate, PreservesExistingHeterogeneousNeighbors) {
  using Tr = MaterialTraits<TypeParam>;
  constexpr int kN = Tr::kNumPsd;

  constexpr MaterialPsdStrategy kPsd = Tr::kPsdA;

  auto material = MakeMaterial<Tr>(FilledPsd<kN>(kPsd));
  auto field = Tr::MakeField(kPsd);
  soft::SetMaterialParamsField(&field, /*numElements*/ 2, material, test::ExpectOK{});

  auto const updated = Tr::MakeBase(FilledPsd<kN>(kPsd), 4000_r);
  soft::SetMaterialParamsField(updated, 1, 2, material, test::ExpectOK{});

  Tr::ExpectElement(GetElement(material, 0), field, 0, FilledPsd<kN>(kPsd));
  Tr::ExpectBase(GetElement(material, 1), updated);
}

// Verifies single-element updates remain compatible after a field establishes an explicit PSD.
TYPED_TEST(MaterialElementUpdate, UpdatesAfterExplicitFieldPsd) {
  using Tr = MaterialTraits<TypeParam>;
  if constexpr (Tr::kHasPsd) {
    constexpr int kN = Tr::kNumPsd;

    auto material = MakeMaterial<Tr>(FilledPsd<kN>(MaterialPsdStrategy::MaterialDefault));
    auto field = Tr::MakeField(Tr::kPsdA);
    soft::SetMaterialParamsField(&field, /*numElements*/ 2, material, test::ExpectOK{});

    auto const updated = Tr::MakeBase(FilledPsd<kN>(Tr::kPsdA), 4000_r);
    soft::SetMaterialParamsField(updated, 1, 2, material, test::ExpectOK{});

    EXPECT_EQ(Tr::StoredPsd(material), FilledPsd<kN>(Tr::kPsdA));
    Tr::ExpectElement(GetElement(material, 0), field, 0, FilledPsd<kN>(Tr::kPsdA));
    Tr::ExpectBase(GetElement(material, 1), updated);
  }
}

// Verifies the compatibility predicate matches the PSD rules enforced by field setters.
TYPED_TEST(MaterialCompatibility, MatchesFieldSetRules) {
  using Tr = MaterialTraits<TypeParam>;
  if constexpr (Tr::kHasPsd) {
    constexpr int kN = Tr::kNumPsd;
    auto const base = Tr::MakeBase(FilledPsd<kN>(Tr::kPsdA));

    auto material = MakeMaterial(base);
    auto const before = GetElement(material, 0);
    auto const storedPsdBefore = Tr::StoredPsd(material);

    EXPECT_TRUE(soft::IsMaterialParamsFieldCompatible(material, base));

    // Differing on any single axis makes the params incompatible.
    for (int axis = 0; axis < kN; ++axis) {
      auto psd = FilledPsd<kN>(Tr::kPsdA);
      psd[axis] = Tr::kPsdB;
      auto const incompatible = Tr::MakeBase(psd, 4000_r);
      EXPECT_FALSE(soft::IsMaterialParamsFieldCompatible(material, incompatible));
      soft::SetMaterialParamsField(incompatible, 0, 2, material, test::ExpectNotOK{});
      Tr::ExpectBase(GetElement(material, 0), before);
      EXPECT_EQ(Tr::StoredPsd(material), storedPsdBefore);
    }
  }
}

// Verifies the reference material stiffness store (consumed by stiffness damping) is identical
// whether grown via the single-element promotion path or built via the bulk rebuild path. Both
// construct the same heterogeneous 2-element field, so promotion must equal rebuild, and the
// grown-fill element 0 must equal the original homogeneous tensor. NeoHookean is a representative
// Lamé material; the store branch is only reachable with stiffness damping enabled.
TEST(MaterialReferenceStiffness, PromotionMatchesRebuild) {
  using Tr = MaterialTraits<NeoHookeanMaterialParams>;
  constexpr MaterialPsdStrategy kPsd = Tr::kPsdA;
  constexpr real kUpdatedValue = 4000_r;

  // Homogeneous base with stiffness damping enabled so the reference-stiffness store is built.
  auto base = Tr::MakeBase({kPsd});
  base.stiffnessDampingCoefficient = 1_r;

  // Promote path: start homogeneous (size-1 store), then update element 1 to grow to size 2.
  auto promoted = MakeMaterial(base);
  ASSERT_EQ(isize(promoted.referenceMaterialStiffness.data), 1);
  auto const homogeneousTensor = promoted.referenceMaterialStiffness.data[0];
  auto const updated = Tr::MakeBase({kPsd}, kUpdatedValue);
  soft::SetMaterialParamsField(
      updated, /*elementIndex*/ 1, /*numElements*/ 2, promoted, test::ExpectOK{});

  // Rebuild path: same base, then bulk-set the equivalent heterogeneous field. Element 0 mirrors
  // the homogeneous base; element 1 mirrors the promoted update.
  auto rebuilt = MakeMaterial(base);
  PerElementSoftMaterialData field;
  field.type = SoftMaterialType::NeoHookean;
  field.psdStrategy = MaterialPsdStrategy::MaterialDefault; // Inherit the base PSD strategy.
  field.youngsModulus = {kBaseValue, kUpdatedValue};
  field.poissonRatio = {kBasePoisson, kBasePoisson};
  soft::SetMaterialParamsField(&field, /*numElements*/ 2, rebuilt, test::ExpectOK{});

  auto const& promoteStore = promoted.referenceMaterialStiffness.data;
  auto const& rebuildStore = rebuilt.referenceMaterialStiffness.data;
  ASSERT_EQ(isize(promoteStore), 2);
  ASSERT_EQ(isize(rebuildStore), 2);

  for (int e = 0; e < 2; ++e) {
    for (int a = 0; a < 6; ++a) {
      for (int b = 0; b < 6; ++b) {
        EXPECT_NEAR_RTOL(promoteStore[e][a][b], rebuildStore[e][a][b], Tr::kRTol);
        if (e == 0) {
          EXPECT_NEAR_RTOL(promoteStore[0][a][b], homogeneousTensor[a][b], Tr::kRTol);
        }
      }
    }
  }
}

// Verifies LinearElastic ignores PSD strategy while still scattering field values correctly.
TEST(Materials, LinearElastic_IgnoresPsdStrategy) {
  using Tr = MaterialTraits<LinearElasticMaterialParams>;
  auto material = MakeMaterial<Tr>({MaterialPsdStrategy::MaterialDefault});

  auto field = Tr::MakeField(MaterialPsdStrategy::AbsEigenProjection);
  soft::SetMaterialParamsField(&field, /*numElements*/ 2, material, test::ExpectOK{});
  Tr::ExpectElement(GetElement(material, 0), field, 0, Tr::StoredPsd(material));
  Tr::ExpectElement(GetElement(material, 1), field, 1, Tr::StoredPsd(material));

  EXPECT_TRUE(
      soft::IsMaterialParamsFieldCompatible(material, Tr::MakeBase({MaterialPsdStrategy::None})));
}

// Verifies materials of different types are rejected by the compatibility predicate and setters.
TEST(Materials, DifferentMaterialType_IsIncompatibleAndSettersReject) {
  auto material =
      MakeMaterial<MaterialTraits<NeoHookeanMaterialParams>>({MaterialPsdStrategy::Projection});
  auto const before = GetElement(material, 0);

  auto const other =
      MaterialTraits<ArapMaterialParams>::MakeBase({MaterialPsdStrategy::Projection});
  EXPECT_FALSE(soft::IsMaterialParamsFieldCompatible(material, other));
  soft::SetMaterialParamsField(other, 0, 2, material, test::ExpectNotOK{});
  MaterialTraits<NeoHookeanMaterialParams>::ExpectBase(GetElement(material, 0), before);

  auto field = MaterialTraits<ArapMaterialParams>::MakeField();
  soft::SetMaterialParamsField(&field, /*numElements*/ 2, material, test::ExpectNotOK{});
  MaterialTraits<NeoHookeanMaterialParams>::ExpectBase(GetElement(material, 0), before);
}

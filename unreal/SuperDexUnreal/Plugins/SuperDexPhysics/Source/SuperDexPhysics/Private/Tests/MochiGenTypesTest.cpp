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

#include "MochiGenTypes.h"
#include "MochiGenSupport.h"
#include "MochiTypes.h"

#include "Misc/AutomationTest.h"
#include "UObject/UnrealType.h"

#include <type_traits>

static_assert(EMochiActorSegmentElementType::Default == EMochiActorSegmentElementType::P1Q3);
static_assert(EMochiActorBoundaryElementType::Default == EMochiActorBoundaryElementType::P1Q3);
static_assert(std::is_same_v<decltype(FMochiBox::Center), FVector3f>);
static_assert(std::is_same_v<decltype(FMochiBox::HalfExtents), FVector3f>);
static_assert(std::is_same_v<decltype(FMochiBox::Rotation), FQuat4f>);
static_assert(std::is_same_v<decltype(FMochiGridSdfParams::ResolutionDelta), FVector3f>);
static_assert(std::is_same_v<decltype(FMochiGridSdfParams::MinGridResolution), FIntVector>);
static_assert(std::is_same_v<decltype(FMochiGridSdfData::Bounds), FBox3f>);
static_assert(std::is_same_v<decltype(FMochiModelData::Box), TOptional<FMochiBox>>);
static_assert(
    std::is_same_v<decltype(FMochiModelData::ConstrainedNodes), TOptional<FMochiInt32Array>>);
static_assert(std::is_same_v<decltype(FMochiIKSolverParams::RelTol), float>);
static_assert(
    std::is_same_v<decltype(FMochiNonLinearSolverParams::PsdProjMode), EMochiPsdProjectionMode>);

#if WITH_DEV_AUTOMATION_TESTS

#if ENGINE_MAJOR_VERSION >= 5
#define MOCHI_GEN_TYPES_TEST_FLAGS \
  (EAutomationTestFlags_ApplicationContextMask | EAutomationTestFlags::SmokeFilter)
#else
#define MOCHI_GEN_TYPES_TEST_FLAGS \
  (EAutomationTestFlags::ApplicationContextMask | EAutomationTestFlags::SmokeFilter)
#endif

static_assert(std::is_same_v<
              decltype(ToMochi(EMochiActorSegmentElementType::P1Q1)),
              mochi::ActorSegmentElementType>);
static_assert(std::is_same_v<
              decltype(ToMochi(EMochiActorBoundaryElementType::P1Q1)),
              mochi::ActorBoundaryElementType>);
static_assert(
    std::is_same_v<decltype(ToMochi(EMochiPoseConstraintType::Joint)), mochi::PoseConstraintType>);

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiGenTypesEnumsTest,
    "SuperDexPhysics.GenTypes.Enums",
    MOCHI_GEN_TYPES_TEST_FLAGS)

bool FMochiGenTypesEnumsTest::RunTest(const FString& Parameters) {
  TestEqual(
      TEXT("Collider converts to native"),
      static_cast<int>(ToMochi(EMochiColliderType::Sdf)),
      static_cast<int>(mochi::ColliderType::Sdf));
  TestEqual(
      TEXT("Collider converts to Unreal"),
      ToUnreal(mochi::ColliderType::Sdf),
      EMochiColliderType::Sdf);

  TestEqual(
      TEXT("GMRES keeps its native ordinal"),
      static_cast<int>(ToMochi(EMochiLinearSolverType::GMRES)),
      static_cast<int>(mochi::LinearSolverType::GMRES));
  TestEqual(
      TEXT("LDLT converts to Unreal"),
      ToUnreal(mochi::LinearSolverType::LDLT),
      EMochiLinearSolverType::LDLT);
  TestEqual(
      TEXT("Auto converts to the native Auto alias"),
      ToUnreal(mochi::LinearSolverType::Auto),
      EMochiLinearSolverType::Auto);
  TestEqual(
      TEXT("LineSearchType uses its canonical type name"),
      ToUnreal(mochi::LineSearchType::WolfeStrong),
      EMochiLineSearchType::WolfeStrong);
  TestEqual(
      TEXT("IC0 converts to Unreal"),
      ToUnreal(mochi::PreconditionerType::IC0),
      EMochiPreconditionerType::IC0);

  TestEqual(
      TEXT("Residual norm keeps its canonical native name"),
      ToUnreal(mochi::LinearSolverConvergenceNorm::ResidualL2),
      EMochiLinearSolverConvergenceNorm::ResidualL2);
  TestEqual(
      TEXT("Preconditioned residual norm keeps its canonical native name"),
      ToUnreal(mochi::LinearSolverConvergenceNorm::PreconditionedResidualL2),
      EMochiLinearSolverConvergenceNorm::PreconditionedResidualL2);

  TestEqual(
      TEXT("Segment P1Q2 converts directly to native"),
      static_cast<int>(ToMochi(EMochiActorSegmentElementType::P1Q2)),
      static_cast<int>(mochi::ActorSegmentElementType::P1Q2));
  TestEqual(
      TEXT("Boundary P1Q3 converts directly to native"),
      static_cast<int>(ToMochi(EMochiActorBoundaryElementType::P1Q3)),
      static_cast<int>(mochi::ActorBoundaryElementType::P1Q3));
  TestEqual(
      TEXT("Native segment Default is retained"),
      ToUnreal(mochi::ActorSegmentElementType::Default),
      EMochiActorSegmentElementType::Default);
  TestEqual(
      TEXT("Experimental boundary values are not discarded"),
      ToUnreal(mochi::ActorBoundaryElementType::ExperimentalP1Q16),
      EMochiActorBoundaryElementType::ExperimentalP1Q16);
  TestEqual(
      TEXT("Pose conversion keeps native-domain link terminology"),
      ToUnreal(mochi::PoseConstraintType::LinkRotation),
      EMochiPoseConstraintType::LinkRotation);
  const UEnum* LinearSolverEnum = StaticEnum<EMochiLinearSolverType>();
  TestNotNull(TEXT("Linear solver reflection"), LinearSolverEnum);
  if (!LinearSolverEnum) {
    return false;
  }
  for (const auto& Entry : TArray<TPair<FString, mochi::LinearSolverType>>{
           {TEXT("GMRES"), mochi::LinearSolverType::GMRES},
           {TEXT("CudaGMRES"), mochi::LinearSolverType::CudaGMRES},
           {TEXT("AugmentedCG"), mochi::LinearSolverType::AugmentedCG},
           {TEXT("LDLT"), mochi::LinearSolverType::LDLT},
           {TEXT("Auto"), mochi::LinearSolverType::Auto},
           {TEXT("Default"), mochi::LinearSolverType::Default}}) {
    TestEqual(
        *FString::Printf(TEXT("%s has its native ordinal"), *Entry.Key),
        LinearSolverEnum->GetValueByNameString(Entry.Key),
        static_cast<int64>(Entry.Value));
  }
  const UEnum* SegmentEnum = StaticEnum<EMochiActorSegmentElementType>();
  const UEnum* BoundaryEnum = StaticEnum<EMochiActorBoundaryElementType>();
  const UEnum* PoseEnum = StaticEnum<EMochiPoseConstraintType>();
  TestEqual(
      TEXT("Segment P1Q1 starts at native zero"),
      SegmentEnum->GetValueByNameString(TEXT("P1Q1")),
      int64{0});
  TestEqual(
      TEXT("Boundary P1Q1 starts at native zero"),
      BoundaryEnum->GetValueByNameString(TEXT("P1Q1")),
      int64{0});
  TArray<FString> SegmentNames;
  TArray<FString> BoundaryNames;
  for (int32 Index = 0; Index < SegmentEnum->NumEnums(); ++Index) {
    SegmentNames.Add(SegmentEnum->GetNameStringByIndex(Index));
  }
  for (int32 Index = 0; Index < BoundaryEnum->NumEnums(); ++Index) {
    BoundaryNames.Add(BoundaryEnum->GetNameStringByIndex(Index));
  }
  TestFalse(TEXT("No segment Nodal enumerator remains"), SegmentNames.Contains(TEXT("Nodal")));
  TestFalse(TEXT("No boundary Nodal enumerator remains"), BoundaryNames.Contains(TEXT("Nodal")));
  TestEqual(
      TEXT("Native pose enum has no UI Any"),
      PoseEnum->GetValueByNameString(TEXT("Any")),
      int64{INDEX_NONE});
#if WITH_EDITOR
  TestTrue(
      TEXT("Segment Count is hidden"),
      SegmentEnum->HasMetaData(TEXT("Hidden"), SegmentEnum->GetIndexByNameString(TEXT("Count"))));
  TestTrue(
      TEXT("Experimental boundary entries are hidden"),
      BoundaryEnum->HasMetaData(
          TEXT("Hidden"), BoundaryEnum->GetIndexByNameString(TEXT("ExperimentalP1Q7"))));
  TestTrue(
      TEXT("Pose Invalid is hidden"),
      PoseEnum->HasMetaData(TEXT("Hidden"), PoseEnum->GetIndexByNameString(TEXT("Invalid"))));
#endif
  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiGenTypesStructsTest,
    "SuperDexPhysics.GenTypes.Structs",
    MOCHI_GEN_TYPES_TEST_FLAGS)

bool FMochiGenTypesStructsTest::RunTest(const FString& Parameters) {
  mochi::BoundarySubsamplingParams BoundarySubsampling;
  BoundarySubsampling.subsamplingDensity = 0.25;
  BoundarySubsampling.strategy = mochi::BoundarySubsamplingStrategy::AreaProportional;
  const FMochiBoundarySubsamplingParams ConvertedBoundarySubsampling =
      ::ToUnreal(BoundarySubsampling);
  TestEqual(
      TEXT("Generated boundary subsampling converts density"),
      ConvertedBoundarySubsampling.SubsamplingDensity,
      0.25f);
  TestEqual(
      TEXT("Generated boundary subsampling converts strategy"),
      ConvertedBoundarySubsampling.Strategy,
      EMochiBoundarySubsamplingStrategy::AreaProportional);

  FMochiNonLinearSolverParams NonLinearSolver;
  const mochi::NonLinearSolverParams NativeNonLinearDefaults;
  TestEqual(
      TEXT("PSD mode uses the native default"),
      NonLinearSolver.PsdProjMode,
      ::ToUnreal(NativeNonLinearDefaults.psdProjMode));
  TestEqual(
      TEXT("Line search iterations use the native default"),
      NonLinearSolver.LineSearchMaxIter,
      NativeNonLinearDefaults.lineSearchMaxIter);
  TestEqual(
      TEXT("Line search alpha uses the native default"),
      NonLinearSolver.LineSearchAlpha,
      NativeNonLinearDefaults.lineSearchAlpha);
  NonLinearSolver.LineSearchMaxIter = 13;
  NonLinearSolver.LineSearchAlpha = 0.25f;
  const mochi::NonLinearSolverParams NativeLineSearch = ::ToMochi(NonLinearSolver);
  TestEqual(
      TEXT("Canonical line search iterations convert"), NativeLineSearch.lineSearchMaxIter, 13);
  TestEqual(TEXT("Canonical line search alpha converts"), NativeLineSearch.lineSearchAlpha, 0.25f);
  const FMochiNonLinearSolverParams ReloadedLineSearch = ::ToUnreal(NativeLineSearch);
  TestEqual(
      TEXT("Canonical line search iterations round trip"),
      ReloadedLineSearch.LineSearchMaxIter,
      13);
  TestEqual(
      TEXT("Canonical line search alpha round trips"), ReloadedLineSearch.LineSearchAlpha, 0.25f);
  for (mochi::PsdProjectionMode Mode :
       {mochi::PsdProjectionMode::Never,
        mochi::PsdProjectionMode::Always,
        mochi::PsdProjectionMode::IfFailRetry,
        mochi::PsdProjectionMode::IfFailAlways,
        mochi::PsdProjectionMode::Count,
        mochi::PsdProjectionMode::Default}) {
    NonLinearSolver.PsdProjMode = ::ToUnreal(Mode);
    const mochi::NonLinearSolverParams Native = ::ToMochi(NonLinearSolver);
    TestEqual(TEXT("Typed PSD field converts to native"), Native.psdProjMode, Mode);
    const FMochiNonLinearSolverParams Reloaded = ::ToUnreal(Native);
    TestEqual(TEXT("Typed PSD field round trips"), Reloaded.PsdProjMode, ::ToUnreal(Mode));
  }

  FMochiAsyncStepParams AsyncStep;
  const mochi::AsyncStepParams NativeAsyncDefaults;
  TestEqual(
      TEXT("Async fixed-time flag uses the native default"),
      AsyncStep.bUseFixedTimeStep,
      NativeAsyncDefaults.useFixedTimeStep);
  TestEqual(
      TEXT("Async fixed time uses the native default"),
      AsyncStep.FixedTimeStepSeconds,
      NativeAsyncDefaults.fixedTimeStepSeconds);
  TestFalse(
      TEXT("Async callback defaults to empty"), static_cast<bool>(AsyncStep.TimeStepCallback));
  AsyncStep.bUseFixedTimeStep = true;
  AsyncStep.FixedTimeStepSeconds = 0.01;
  AsyncStep.DynamicTimeStepMinSeconds = 0.002;
  AsyncStep.DynamicTimeStepMaxSeconds = 0.02;
  int32 CallbackCalls = 0;
  AsyncStep.TimeStepCallback = [&CallbackCalls]() {
    ++CallbackCalls;
    return 0.003;
  };
  const mochi::AsyncStepParams NativeAsyncStep = ::ToMochi(AsyncStep);
  TestTrue(TEXT("Async step converts fixed-time flag"), NativeAsyncStep.useFixedTimeStep);
  TestEqual(TEXT("Async step converts fixed time"), NativeAsyncStep.fixedTimeStepSeconds, 0.01);
  const FMochiAsyncStepParams RoundTrippedAsyncStep = ::ToUnreal(NativeAsyncStep);
  TestEqual(
      TEXT("Async step round trips minimum dynamic time"),
      RoundTrippedAsyncStep.DynamicTimeStepMinSeconds,
      0.002);
  TestEqual(
      TEXT("Async step round trips maximum dynamic time"),
      RoundTrippedAsyncStep.DynamicTimeStepMaxSeconds,
      0.02);
  TestEqual(TEXT("Conversions do not invoke the callback"), CallbackCalls, 0);
  if (TestTrue(
          TEXT("Native callback is copied"), static_cast<bool>(NativeAsyncStep.timeStepCallback))) {
    TestEqual(
        TEXT("Native callback returns the configured time"),
        NativeAsyncStep.timeStepCallback(),
        0.003);
  }
  if (TestTrue(
          TEXT("Unreal callback is copied"),
          static_cast<bool>(RoundTrippedAsyncStep.TimeStepCallback))) {
    TestEqual(
        TEXT("Unreal callback returns the configured time"),
        RoundTrippedAsyncStep.TimeStepCallback(),
        0.003);
  }
  TestEqual(TEXT("Copied callbacks retain their captured state"), CallbackCalls, 2);
  TestNull(
      TEXT("C++ callback is not reflected"),
      FMochiAsyncStepParams::StaticStruct()->FindPropertyByName(TEXT("TimeStepCallback")));
  TestFalse(
      TEXT("Empty native callback remains empty"),
      static_cast<bool>(::ToUnreal(NativeAsyncDefaults).TimeStepCallback));
  TestFalse(
      TEXT("Empty Unreal callback remains empty"),
      static_cast<bool>(::ToMochi(FMochiAsyncStepParams{}).timeStepCallback));

  FMochiGridSdfParams GridSdf;
  GridSdf.ResolutionDelta = FVector3f{0.25f, 0.5f, 0.75f};
  GridSdf.MinGridResolution = FIntVector{2, 3, 4};
  GridSdf.BoundaryPaddingDist = 0.125f;
  const mochi::GridSdfParams NativeGridSdf = ::ToMochi(GridSdf);
  TestTrue(
      TEXT("Grid SDF converts each resolution delta component"),
      NativeGridSdf.resolutionDelta == mochi::Real3{0.25f, 0.5f, 0.75f});
  TestTrue(TEXT("Grid SDF converts Int3"), NativeGridSdf.minGridResolution == mochi::Int3{2, 3, 4});
  TestEqual(
      TEXT("Grid SDF padding converts without a margin"),
      NativeGridSdf.boundaryPaddingDist,
      0.125f);
  const FMochiGridSdfParams RoundTrippedGridSdf = ::ToUnreal(NativeGridSdf);
  TestEqual(
      TEXT("Grid SDF round trips FVector3f storage"),
      RoundTrippedGridSdf.ResolutionDelta,
      GridSdf.ResolutionDelta);
  TestEqual(
      TEXT("Grid SDF round trips Int3"),
      RoundTrippedGridSdf.MinGridResolution,
      GridSdf.MinGridResolution);

  mochi::GridSdfData NativeSdfData;
  NativeSdfData.bounds =
      mochi::Aabb(mochi::Real3{-1.0f, -2.0f, -3.0f}, mochi::Real3{4.0f, 5.0f, 6.0f});
  NativeSdfData.negativeValueBounds =
      mochi::Aabb(mochi::Real3{-0.5f, -1.0f, -1.5f}, mochi::Real3{2.0f, 2.5f, 3.0f});
  const FMochiGridSdfData SdfData = ::ToUnreal(NativeSdfData);
  TestEqual(TEXT("AABB minimum uses FBox3f"), SdfData.Bounds.Min, FVector3f(-1.0f, -2.0f, -3.0f));
  TestEqual(TEXT("AABB maximum uses FBox3f"), SdfData.Bounds.Max, FVector3f(4.0f, 5.0f, 6.0f));
  const mochi::GridSdfData RoundTrippedSdfData = ::ToMochi(SdfData);
  TestTrue(TEXT("SDF bounds round trip"), RoundTrippedSdfData.bounds == NativeSdfData.bounds);
  TestTrue(
      TEXT("SDF negative-value bounds round trip"),
      RoundTrippedSdfData.negativeValueBounds == NativeSdfData.negativeValueBounds);
  const mochi::Box NativeBoxDefaults;
  const mochi::Box ConvertedBoxDefaults = ::ToMochi(FMochiBox{});
  TestTrue(
      TEXT("Box center uses the native default"),
      ConvertedBoxDefaults.center == NativeBoxDefaults.center);
  TestTrue(
      TEXT("Box extents use the native default"),
      ConvertedBoxDefaults.halfExtents == NativeBoxDefaults.halfExtents);
  TestTrue(
      TEXT("Box rotation uses the native default"),
      ConvertedBoxDefaults.rotation == NativeBoxDefaults.rotation);

  mochi::ModelData NativeModel;
  FMochiModelData Model = ::ToUnreal(NativeModel);
  TestFalse(TEXT("Absent optional array remains absent"), Model.ConstrainedNodes.IsSet());

  NativeModel.constrainedNodes.emplace();
  Model = ::ToUnreal(NativeModel);
  TestTrue(TEXT("Present empty optional array remains present"), Model.ConstrainedNodes.IsSet());
  if (Model.ConstrainedNodes.IsSet()) {
    TestEqual(
        TEXT("Present empty optional array remains empty"),
        Model.ConstrainedNodes.GetValue().Values.Num(),
        0);
  }

  NativeModel.constrainedNodes->push_back(7);
  NativeModel.constrainedNodes->push_back(11);
  NativeModel.experimentalDataDetected = true;
  Model = ::ToUnreal(NativeModel);
  TestTrue(TEXT("Experimental-data flag converts"), Model.bExperimentalDataDetected);
  TestTrue(
      TEXT("Present populated optional array remains present"), Model.ConstrainedNodes.IsSet());
  if (Model.ConstrainedNodes.IsSet()) {
    TestTrue(
        TEXT("Present populated optional array preserves values"),
        Model.ConstrainedNodes.GetValue().Values == TArray<int32>{7, 11});
  }

  const mochi::ModelData RoundTrippedModel = ::ToMochi(Model);
  TestTrue(TEXT("Experimental-data flag round trips"), RoundTrippedModel.experimentalDataDetected);
  TestTrue(
      TEXT("Optional array round trip remains present"),
      RoundTrippedModel.constrainedNodes.has_value());
  if (RoundTrippedModel.constrainedNodes.has_value()) {
    TestTrue(
        TEXT("Optional array round trip preserves values"),
        *RoundTrippedModel.constrainedNodes == mochi::DynamicArray<int>{7, 11});
  }

  const mochi::LinearSolverParams NativeLinearDefaults;
  FMochiLinearSolverParams LinearSolver;
  TestEqual(
      TEXT("Linear solver constructor preserves native solver default"),
      LinearSolver.SolverType,
      ::ToUnreal(NativeLinearDefaults.solverType));
  TestEqual(
      TEXT("Linear solver constructor preserves native iteration default"),
      LinearSolver.MaxIter,
      NativeLinearDefaults.maxIter);
  TestEqual(
      TEXT("Linear absolute tolerance uses the native default"),
      LinearSolver.AbsTol,
      NativeLinearDefaults.absTol,
      0.0f);
  TestEqual(
      TEXT("Linear relative tolerance uses the native default"),
      LinearSolver.RelTol,
      NativeLinearDefaults.relTol,
      0.0f);
  LinearSolver.AbsTol = 0.125f;
  LinearSolver.RelTol = 0.25f;
  const mochi::LinearSolverParams NativeLinearSolver = ::ToMochi(LinearSolver);
  TestEqual(
      TEXT("Canonical linear absolute tolerance converts"), NativeLinearSolver.absTol, 0.125f);
  TestEqual(TEXT("Canonical linear relative tolerance converts"), NativeLinearSolver.relTol, 0.25f);
  const FMochiLinearSolverParams ReloadedLinearSolver = ::ToUnreal(NativeLinearSolver);
  TestEqual(
      TEXT("Canonical linear absolute tolerance round trips"), ReloadedLinearSolver.AbsTol, 0.125f);
  TestEqual(
      TEXT("Canonical linear relative tolerance round trips"), ReloadedLinearSolver.RelTol, 0.25f);

  const mochi::experimental::IKSolverParams NativeIkDefaults;
  const FMochiIKSolverParams IkDefaults;
  TestEqual(
      TEXT("IK relative tolerance uses the native default"),
      IkDefaults.RelTol,
      NativeIkDefaults.relTol,
      0.0f);
  mochi::experimental::IKSolverParams NativeIkSolver;
  NativeIkSolver.relTol = 2e-5f;
  const FMochiIKSolverParams IkSolver = ::ToUnreal(NativeIkSolver);
  TestEqual(
      TEXT("IK conversion keeps unscaled relative tolerance"),
      IkSolver.RelTol,
      NativeIkSolver.relTol,
      0.0f);
  TestEqual(
      TEXT("IK unscaled relative tolerance round trips"),
      ::ToMochi(IkSolver).relTol,
      NativeIkSolver.relTol,
      0.0f);

  mochi::PerElementSoftMaterialData NativeMaterial;
  NativeMaterial.type = mochi::SoftMaterialType::ActiveShapeTargetingArap;
  NativeMaterial.psdStrategy = mochi::MaterialPsdStrategy::AbsEigenProjection;
  NativeMaterial.youngsModulus = {1200.0f};
  NativeMaterial.poissonRatio = {0.25f};
  NativeMaterial.anisoAlpha = {100.0f};
  NativeMaterial.anisoLength = {1.5f};
  NativeMaterial.anisoTheta = {0.5f};
  NativeMaterial.anisoPhi = {0.75f};
  NativeMaterial.arapStiffness = {900.0f};
  NativeMaterial.shapeTargetTensor = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
  const mochi::PerElementSoftMaterialData RoundTrippedMaterial =
      ::ToMochi(::ToUnreal(NativeMaterial));
  TestEqual(
      TEXT("Per-element material type round trips"),
      RoundTrippedMaterial.type,
      NativeMaterial.type);
  TestEqual(
      TEXT("Per-element PSD strategy round trips"),
      RoundTrippedMaterial.psdStrategy,
      NativeMaterial.psdStrategy);
  for (auto Field :
       {&mochi::PerElementSoftMaterialData::youngsModulus,
        &mochi::PerElementSoftMaterialData::poissonRatio,
        &mochi::PerElementSoftMaterialData::anisoAlpha,
        &mochi::PerElementSoftMaterialData::anisoLength,
        &mochi::PerElementSoftMaterialData::anisoTheta,
        &mochi::PerElementSoftMaterialData::anisoPhi,
        &mochi::PerElementSoftMaterialData::arapStiffness,
        &mochi::PerElementSoftMaterialData::shapeTargetTensor}) {
    TestTrue(
        TEXT("All per-element material arrays round trip"),
        RoundTrippedMaterial.*Field == NativeMaterial.*Field);
  }

  return true;
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FMochiGenTypesSupportTest,
    "SuperDexPhysics.GenTypes.Support",
    MOCHI_GEN_TYPES_TEST_FLAGS)

bool FMochiGenTypesSupportTest::RunTest(const FString& Parameters) {
  const FVector3f Position(1.25f, -2.0f, 3.5f);
  const mochi::Real3 NativePosition = MochiGenSupport::ToMochiReal3(Position);
  TestEqual(TEXT("Representation keeps native coordinates"), NativePosition[0], 1.25f);
  TestEqual(
      TEXT("Vector representation round trips"),
      MochiGenSupport::ToUnrealReal3<FVector3f>(NativePosition),
      Position);

  const FQuat4f Rotation(0.1f, 0.2f, 0.3f, 0.4f);
  TestEqual(
      TEXT("Quaternion representation round trips"),
      MochiGenSupport::ToUnrealQuaternion(MochiGenSupport::ToMochiQuaternion(Rotation)),
      Rotation);
  const FBox3f Bounds(FVector3f(-1, -2, -3), FVector3f(4, 5, 6));
  TestEqual(
      TEXT("Bounds endpoints round trip"),
      MochiGenSupport::ToUnrealAabb(MochiGenSupport::ToMochiAabb(Bounds)),
      Bounds);

  const FString Text(5, TEXT("A\0\u03bb\U0001f600"));
  const mochi::DynamicString NativeText = MochiGenSupport::ToMochiString(Text);
  const mochi::DynamicString ExpectedText("A\0\xCE\xBB\xF0\x9F\x98\x80", 8);
  TestTrue(TEXT("UTF8 keeps embedded NUL and multibyte text"), NativeText == ExpectedText);
  TestEqual(
      TEXT("UTF8 round trips through FString"), MochiGenSupport::ToUnrealString(NativeText), Text);

  const TArray<int32> Values{1, -2, 3};
  TestTrue(
      TEXT("Array representations round trip"),
      MochiGenSupport::ToUnrealArray<int32>(MochiGenSupport::ToMochiArray<int>(Values)) == Values);
  const TOptional<TArray<int32>> Unset;
  const TOptional<TArray<int32>> PresentEmpty(TArray<int32>{});
  const auto Convert = [](const TArray<int32>& Value) {
    return MochiGenSupport::ToMochiArray<int>(Value);
  };
  TestFalse(
      TEXT("Unset optional stays unset"),
      MochiGenSupport::ToMochiOptional<mochi::DynamicArray<int>>(Unset, Convert).has_value());
  const auto NativeEmpty =
      MochiGenSupport::ToMochiOptional<mochi::DynamicArray<int>>(PresentEmpty, Convert);
  TestTrue(TEXT("Present-empty optional stays set"), NativeEmpty.has_value());
  TestTrue(
      TEXT("Present-empty array stays empty"), NativeEmpty.has_value() && NativeEmpty->empty());
  return true;
}

#undef MOCHI_GEN_TYPES_TEST_FLAGS

#endif // WITH_DEV_AUTOMATION_TESTS

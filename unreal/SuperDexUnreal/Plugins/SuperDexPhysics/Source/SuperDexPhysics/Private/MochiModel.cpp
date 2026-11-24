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

#include "MochiModel.h"

#include "MochiActorObject.h"
#include "MochiConversions.h"
#include "MochiLegacyBounds.h"
#include "MochiPhysicsModule.h"
#include "Serialization/ArchiveProxy.h"
#include "Serialization/StructuredArchiveAdapters.h"
#include "UObject/ObjectVersion.h"
#include "UObject/UnrealType.h"

#include <mochi_core/geometry/mesh_data_utils.h>
#include <mochi_core/geometry/model_data.h>
#include <mochi_physics/utils/mochi_model_utils.h>
#include "PhysicsEngine/BodySetup.h"

#if WITH_EDITOR
#include <string>
#include <utility>
#include "DesktopPlatformModule.h"
#include "EditorDirectories.h"
#include "Framework/Application/SlateApplication.h"
#include "IDesktopPlatform.h"
#include "Misc/FileHelper.h"
#endif

//------------------------------------------------------------------------------
// MOCHI MODEL
//------------------------------------------------------------------------------

namespace {

inline mochi::Quaternion ToMochiType(FQuat R) {
  using Mr = mochi::real;
  return mochi::Quaternion((Mr)R.X, (Mr)R.Y, (Mr)R.Z, (Mr)R.W);
}

inline mochi::Real3 ToMochiType(FVector T) {
  using Mr = mochi::real;
  return mochi::Real3((Mr)T.X, (Mr)T.Y, (Mr)T.Z);
}

// Input: LH, Z up, centimeters (Unreal)
// Output: LH, Z up, meters (Unreal)
inline mochi::VMatrix4x4r UnrealToMatrix(const FVector& S, const FTransform& RT) {
  mochi::VMatrix4x4r ScaleMatrix = mochi::VDiagonalMatrix<4>(ToSimd(ToMochiType(S), 1_r));
  mochi::Quaternion R = ToMochiType(RT.GetRotation());
  mochi::Real3 T = ToMochiType(RT.GetTranslation() / UnrealCentimetersSpace.UnitsPerMeter);
  mochi::VMatrix4x4r Rt = ToVMatrix4x4(mochi::TransformRT(R, T));
  mochi::VMatrix4x4r Srt = Dot4x4(Rt, ScaleMatrix);
  return Srt;
}

bool HasModelGeometry(const FMochiModelData& Data) {
  return Data.Mesh.IsSet() || Data.VisualMesh.IsSet() || Data.ContactSkinMesh.IsSet() ||
      Data.Box.IsSet() || Data.Plane.IsSet() || Data.Sphere.IsSet() || Data.Sdf.IsSet();
}

class FLegacyBoundsPayloadArchive final : public FArchiveProxy {
 public:
  FLegacyBoundsPayloadArchive(FArchive& Archive, int64 Begin, int64 End)
      : FArchiveProxy(Archive), Begin(Begin), End(End) {}

  using FArchiveProxy::operator<<;

  virtual void Serialize(void* Data, int64 NumBytes) override {
    if (!CanRead(NumBytes)) {
      Fail();
      return;
    }
    InnerArchive.Serialize(Data, NumBytes);
  }

  virtual FArchive& operator<<(FName& Name) override {
    // Only indexed package names are admitted by the enclosing tag observer.
    if (!CanRead(2 * sizeof(int32))) {
      Fail();
      return *this;
    }
    const int64 Before = Tell();
    InnerArchive << Name;
    if (Tell() > End || Tell() - Before != 2 * sizeof(int32)) {
      Fail();
    }
    return *this;
  }

  virtual void Seek(int64 Position) override {
    if (Position < Begin || Position > End) {
      Fail();
      return;
    }
    InnerArchive.Seek(Position);
  }

 private:
  bool CanRead(int64 NumBytes) {
    const int64 Position = Tell();
    return NumBytes >= 0 && Position >= Begin && Position <= End && NumBytes <= End - Position;
  }

  void Fail() {
    SetError();
    InnerArchive.SetError();
  }

  const int64 Begin;
  const int64 End;
};

// TODO: After affected UMochiModel packages are resaved in the native-generated
// layout and validated without this reader, remove it, the private legacy schema,
// and the legacy-load branch in UMochiModel::Serialize.
class FLegacyModelBoundsArchive final : public FArchiveProxy {
 public:
  explicit FLegacyModelBoundsArchive(FArchive& Archive) : FArchiveProxy(Archive) {}

  using FArchiveProxy::operator<<;

  virtual void Serialize(void* Data, int64 NumBytes) override {
    const int64 Before = Tell();
    InnerArchive.Serialize(Data, NumBytes);
    if (NumBytes != sizeof(int32) || HeaderSizePosition >= 0 || !IsModelSdfScope()) {
      return;
    }
    if (InnerArchive.UEVer() < EUnrealEngineObjectUE5Version::PROPERTY_TAG_COMPLETE_TYPE_NAME) {
      if (State == ETagState::StructType) {
        HeaderSizePosition = Before;
      }
    } else if (State == ETagState::LegacyOwner) {
      if (!bReadLeafInnerCount) {
        int32 InnerCount;
        FMemory::Memcpy(&InnerCount, Data, sizeof(InnerCount));
        if (InnerCount != 0) {
          State = ETagState::None;
          return;
        }
        bReadLeafInnerCount = true;
      } else {
        HeaderSizePosition = Before;
      }
    }
  }

  virtual bool ShouldSkipProperty(const FProperty* InProperty) const override {
    const bool bSkip = InnerArchive.ShouldSkipProperty(InProperty);
    if (!bSkip) {
      const_cast<FLegacyModelBoundsArchive*>(this)->RecoverMatchedBounds(InProperty);
    }
    return bSkip;
  }

  virtual FArchive& operator<<(FName& Name) override {
    const int64 Before = Tell();
    InnerArchive << Name;
    if (!IsModelSdfScope() || Tell() - Before != 2 * sizeof(int32)) {
      State = ETagState::None;
      return *this;
    }
    if (Name == TEXT("Bounds") || Name == TEXT("NegativeValueBounds")) {
      bNegativeBounds = Name == TEXT("NegativeValueBounds");
      HeaderSizePosition = INDEX_NONE;
      bReadLeafInnerCount = false;
      State = ETagState::BoundsName;
    } else if (State == ETagState::BoundsName && Name == NAME_StructProperty) {
      State = ETagState::StructType;
    } else if (State == ETagState::StructType && Name == TEXT("MochiAabb")) {
      State = ETagState::LegacyType;
    } else if (State == ETagState::LegacyType && Name == TEXT("/Script/SuperDexPhysics")) {
      State = ETagState::LegacyOwner;
    } else {
      State = ETagState::None;
    }
    return *this;
  }

  void Apply(FMochiModelData& Data) const {
    if (!Data.Sdf.IsSet()) {
      return;
    }
    if (Bounds.IsSet()) {
      Data.Sdf->Bounds = Bounds.GetValue();
    }
    if (NegativeValueBounds.IsSet()) {
      Data.Sdf->NegativeValueBounds = NegativeValueBounds.GetValue();
    }
  }

 private:
  enum class ETagState : uint8 { None, BoundsName, StructType, LegacyType, LegacyOwner };

  bool IsModelSdfScope() {
    PropertyChain.Reset();
    GetSerializedPropertyChain(PropertyChain);
    return PropertyChain.Num() == 2 && PropertyChain[0]->GetFName() == TEXT("Sdf") &&
        PropertyChain[0]->GetOwnerStruct() == FMochiModelData::StaticStruct() &&
        PropertyChain[1]->GetFName() == TEXT("ModelData") &&
        PropertyChain[1]->GetOwnerStruct() == UMochiModel::StaticClass();
  }

  void RecoverMatchedBounds(const FProperty* Property) {
    const bool bLegacyShortType =
        InnerArchive.UEVer() < EUnrealEngineObjectUE5Version::PROPERTY_TAG_COMPLETE_TYPE_NAME &&
        State == ETagState::LegacyType;
    if ((State != ETagState::LegacyOwner && !bLegacyShortType) || HeaderSizePosition < 0 ||
        !Property || Property->GetOwnerStruct() != FMochiGridSdfData::StaticStruct() ||
        Property->GetFName() != (bNegativeBounds ? TEXT("NegativeValueBounds") : TEXT("Bounds")) ||
        !IsModelSdfScope()) {
      return;
    }

    // This matched-property callback runs after the tag and before conversion or
    // unknown-property tracking consumes its body. A peek must restore that cursor.
    const int64 Begin = Tell();
    const int64 SizePosition = HeaderSizePosition;
    const bool bRecoverNegativeBounds = bNegativeBounds;
    State = ETagState::None;
    HeaderSizePosition = INDEX_NONE;
    constexpr int64 SizeFieldBytes = sizeof(int32);
    if (Begin < SizeFieldBytes || SizePosition > Begin - SizeFieldBytes) {
      Fail();
      return;
    }
    InnerArchive.Seek(SizePosition);
    FLegacyBoundsPayloadArchive SizeArchive(
        InnerArchive, SizePosition, SizePosition + SizeFieldBytes);
    int32 PayloadSize = 0;
    SizeArchive << PayloadSize;
    const bool bBadSize =
        SizeArchive.IsError() || SizeArchive.Tell() != SizePosition + SizeFieldBytes;
    InnerArchive.Seek(Begin);
    constexpr int32 MaxLegacyBoundsPayloadBytes = 4096;
    const int64 TotalSize = InnerArchive.TotalSize();
    if (bBadSize || PayloadSize <= 0 || PayloadSize > MaxLegacyBoundsPayloadBytes ||
        TotalSize < Begin || PayloadSize > TotalSize - Begin) {
      Fail();
      return;
    }
    const int64 End = Begin + PayloadSize;
    FMochiLegacyBounds Old;
    FLegacyBoundsPayloadArchive Payload(InnerArchive, Begin, End);
    FMochiLegacyBounds::StaticStruct()->SerializeTaggedProperties(
        FStructuredArchiveFromArchive(Payload).GetSlot(),
        reinterpret_cast<uint8*>(&Old),
        FMochiLegacyBounds::StaticStruct(),
        nullptr);
    const bool bBadPayload = Payload.IsError() || Payload.Tell() != End;
    InnerArchive.Seek(Begin);
    if (bBadPayload) {
      Fail();
    } else if (bRecoverNegativeBounds) {
      NegativeValueBounds = FBox3f(Old.Min, Old.Max);
    } else {
      Bounds = FBox3f(Old.Min, Old.Max);
    }
  }

  void Fail() {
    SetError();
    InnerArchive.SetError();
  }

  ETagState State = ETagState::None;
  bool bNegativeBounds = false;
  bool bReadLeafInnerCount = false;
  int64 HeaderSizePosition = INDEX_NONE;
  TArray<FProperty*> PropertyChain;
  TOptional<FBox3f> Bounds;
  TOptional<FBox3f> NegativeValueBounds;
};

constexpr int64 BytesPerFloat = sizeof(float);

// Regroups a float array so that every first byte precedes every second byte, and so on.
// Across a distance field the exponent and high mantissa bytes change slowly while the
// low ones look like noise, so interleaved floats give a general-purpose compressor
// almost nothing to work with; separating the planes exposes the runs. This is what
// HDF5's shuffle filter does, and it accounts for most of the ratio the H5 binary
// achieves over plain zlib.
void ShuffleFloatBytes(const float* Src, uint8* Dst, int64 NumElements) {
  const uint8* SrcBytes = reinterpret_cast<const uint8*>(Src);
  for (int64 i = 0; i < NumElements; ++i) {
    for (int64 B = 0; B < BytesPerFloat; ++B) {
      Dst[B * NumElements + i] = SrcBytes[i * BytesPerFloat + B];
    }
  }
}

void UnshuffleFloatBytes(const uint8* Src, float* Dst, int64 NumElements) {
  uint8* DstBytes = reinterpret_cast<uint8*>(Dst);
  for (int64 i = 0; i < NumElements; ++i) {
    for (int64 B = 0; B < BytesPerFloat; ++B) {
      DstBytes[i * BytesPerFloat + B] = Src[B * NumElements + i];
    }
  }
}

} // namespace

void UMochiModel::Serialize(FArchive& Ar) {
  Ar.UsingCustomVersion(FMochiObjectVersion::GUID);
  const int32 ModelVersion = Ar.CustomVer(FMochiObjectVersion::GUID);
  const bool bLegacyBounds = Ar.IsLoading() && Ar.IsPersistent() && !Ar.IsTransacting() &&
      !Ar.UseUnversionedPropertySerialization() &&
      ModelVersion >= FMochiObjectVersion::TypedModelData &&
      ModelVersion < FMochiObjectVersion::NativeGeneratedStructs;
  if (bLegacyBounds) {
    FLegacyModelBoundsArchive Legacy(Ar);
    Super::Serialize(Legacy);
    if (Ar.IsError()) {
      return;
    }
    Legacy.Apply(ModelData);
  } else if (Ar.IsSaving() && ModelData.Sdf.IsSet()) {
    // The reflected array must stay out of the int32-sized tagged property payload.
    // Its contents are restored before the 64-bit bulk serializer reads them.
    TArray<float> Values = MoveTemp(ModelData.Sdf->Values);
    Super::Serialize(Ar);
    ModelData.Sdf->Values = MoveTemp(Values);
  } else {
    Super::Serialize(Ar);
  }

  if (Ar.CustomVer(FMochiObjectVersion::GUID) >= FMochiObjectVersion::TypedModelData) {
    SerializeSdfValues(Ar);
  }
}

void UMochiModel::SerializeSdfValues(FArchive& Ar) {
  if (Ar.IsSaving()) {
    // A signed distance field is smooth, so it compresses well; without that an asset
    // grows several times over the H5 binary it replaces. Size64Bit is what moves the
    // element count and on-disk size in the bulk header off int32; the payload being
    // 64-bit is not enough on its own.
    SdfValues.SetBulkDataFlags(BULKDATA_SerializeCompressed | BULKDATA_Size64Bit);
    const TArray<float>* Values = ModelData.Sdf.IsSet() ? &ModelData.Sdf->Values : nullptr;
    const int64 NumElements = Values ? Values->Num() : 0;
    // Shuffling straight into the bulk buffer keeps this from needing a second copy of a
    // grid that can run to gigabytes.
    SdfValues.Lock(LOCK_READ_WRITE);
    ShuffleFloatBytes(
        Values ? Values->GetData() : nullptr,
        static_cast<uint8*>(SdfValues.Realloc(NumElements * BytesPerFloat)),
        NumElements);
    SdfValues.Unlock();
  }

  SdfValues.Serialize(Ar, this);

  if (Ar.IsLoading()) {
    const int64 NumElements = SdfValues.GetBulkDataSize() / BytesPerFloat;
    if (NumElements == 0) {
      if (ModelData.Sdf.IsSet()) {
        ModelData.Sdf->Values.Reset();
      }
      return;
    }
    TArray<float>& Values =
        ModelData.Sdf.IsSet() ? ModelData.Sdf->Values : ModelData.Sdf.Emplace().Values;
    Values.SetNumUninitialized(static_cast<int32>(NumElements));
    UnshuffleFloatBytes(
        static_cast<const uint8*>(SdfValues.LockReadOnly()), Values.GetData(), NumElements);
    SdfValues.Unlock();
    // ModelData owns the values from here; keeping the bulk copy would double the
    // resident cost of a large grid.
    SdfValues.RemoveBulkData();
  }
}

void UMochiModel::PostLoad() {
  Super::PostLoad();

#if WITH_EDITORONLY_DATA
  if (!AssetImportData) {
    AssetImportData = NewObject<UAssetImportData>(this, TEXT("AssetImportData"));
  }
  // Models saved before AssetImportData kept the absolute source path in Filename. Carry it over
  // as is; the next import or source file change stores it the way AssetImportData does.
  if (!Filename_DEPRECATED.IsEmpty()) {
    if (AssetImportData->GetSourceData().SourceFiles.IsEmpty()) {
      FAssetImportInfo Info;
      Info.Insert(FAssetImportInfo::FSourceFile(Filename_DEPRECATED));
      AssetImportData->SourceData = MoveTemp(Info);
    }
    Filename_DEPRECATED.Empty();
  }
#endif

  if (!HasModelGeometry(ModelData)) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiModel] %s has no model data. If it was saved with only an H5 binary, "
            "resave it with a build that still reads one."),
        *GetPathName());
  }
}

// NB: Geometry flows through the following order of transformations:
// 1. H5 geometry is stored either LH, Z up, meters if the MochiModel was generated
//    in Unreal, or RH, Y up, meters if the MochiModel was imported by dragging an
//    external .mochi.h5 into Unreal (e.g. from mochi_samples_app).
// 2. Apply ImportTransform (SRT). This serves to apply a *permanent*, editor-time
//    transform to the geometry. It is used to apply the necessary basis change to
//    MochiModels imported from Mochi via drag/drop (see MochiFactory.cpp). Since
//    it is exposed to the editor, the user may also choose to use it to apply
//    additional translation, rotation, and non-uniform scale. ImportTransform is
//    LH, Z up, cm. Since H5 geometry is stored in meters, we need convert translation
//    to meters (see UnrealToMatrix above).
// 3. Apply BakeScale. This runtime adjustment typically comes from the component
//    transform scale of components loading Mochi shapes. It allows the user to
//    non-uniformly scale meshes from the component details. It is LH, Z up.
// 4. Apply BakeTransformRT. Similar to BakeScale, this allows for a runtime
//    adjustment. It is currently only used by UMochiArticulation to
//    translate and rotate link shapes via ShapeRotation and ShapeTranslation.
//    BakeTransformRT is LH, Z up, cm, so we need convert translation to meters.
// 5. Apply the basis change matrix to transform geometry into the simulation space.
//    This takes us from LH, Z up, m (Unreal) to whatever simulation space is
//    configured in the project settings or overridden in a MochiSceneSettingsActor.
//    Since we are converting the geometry from meters, we also need to scale by the
//    output space's units per meters. This can be assumed 1.0 as long as
//    FMochiSpaceDefinition::UnitsPerMeter remains VisibleAnywhere, but the code as
//    written supports the field if/when we are able to guarantee all other parameters
//    are converted to the target simulation space (e.g. MaterialParams, ContactParams,
//    gravity vector, etc.)
//
// Final Order (Right to Left):
//
// FinalMatrix = (basis change * units scale) * (BakTransformRT * BakeScale) * ImportTransform
//             = (SpaceMatrix)                * (BakeMatrix)                 * ImportMatrix

mochi::ShapeHandle UMochiModel::GetMochiShape(
    const FVector& BakeScale,
    const FTransform& BakeTransformRT) const {
  FUnrealToMochiConverter U2m;
  mochi::VMatrix4x4r ImportMatrix = UnrealToMatrix(ImportTransform.GetScale3D(), ImportTransform);
  mochi::VMatrix4x4r BakeMatrix = UnrealToMatrix(BakeScale, BakeTransformRT);
  mochi::VMatrix4x4r SpaceMatrix = Dot4x4(
      U2m.GetBasisChangeMatrix(),
      mochi::VDiagonalMatrix<4>(static_cast<mochi::real>(U2m.OutputSpace.UnitsPerMeter)));
  mochi::VMatrix4x4r FinalMatrix = Dot4x4(SpaceMatrix, Dot4x4(BakeMatrix, ImportMatrix));
  return GetMochiShape_Raw(FinalMatrix);
}

mochi::ShapeHandle UMochiModel::GetMochiShape(
    const FVector& BakeScale,
    const mochi::VMatrix4x4r& BakeTransformRT) const {
  MOCHI_PROFILE_SCOPE();
  FUnrealToMochiConverter U2m;
  mochi::VMatrix4x4r ImportMatrix = UnrealToMatrix(ImportTransform.GetScale3D(), ImportTransform);
  mochi::VMatrix4x4r BakeMatrix =
      Dot4x4(BakeTransformRT, mochi::VDiagonalMatrix<4>(ToSimd(ToMochiType(BakeScale), 1_r)));
  mochi::VMatrix4x4r SpaceMatrix = Dot4x4(
      U2m.GetBasisChangeMatrix(),
      mochi::VDiagonalMatrix<4>(static_cast<mochi::real>(U2m.OutputSpace.UnitsPerMeter)));
  mochi::VMatrix4x4r FinalMatrix = Dot4x4(SpaceMatrix, Dot4x4(BakeMatrix, ImportMatrix));
  return GetMochiShape_Raw(FinalMatrix);
}

mochi::ShapeHandle UMochiModel::GetMochiShape_Raw(const mochi::VMatrix4x4r& BakeMatrix) const {
  MOCHI_PROFILE_SCOPE();
  if (!IsInGameThread()) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "[UMochiModel] Attempting to load UMochiModel %s shape on thread that is not the UE game thread. "
            "It is highly recommended that all shape loading be performed on the UE game thread."),
        *GetName());
  }
  mochi::ErrorLog E{};
  mochi::Context* const MochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();
  std::pair<mochi::Real3, mochi::TransformRT> BakeTransform =
      mochi::DecomposeMatrixTransform(BakeMatrix);
  mochi::Real3& BakeScale = BakeTransform.first;
  mochi::TransformRT& BakeTransformRT = BakeTransform.second;

  if (!HasModelGeometry(ModelData)) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiModel] Asked for the shape of %s, which holds no model data."),
        *GetPathName());
    return mochi::ShapeHandle();
  }

  // CreateModelShape takes no bake parameters, unlike LoadShapeFromBytes, so the transform
  // has to be baked into an owning copy first and Mochi then copies that again. A
  // bake-aware overload upstream would remove one of the two. It also runs AutoCorrect and
  // Validate, which LoadShapeFromBytes did not, so a model Mochi considers malformed now
  // fails here instead of loading and simulating badly.
  mochi::ModelData Baked = ::ToMochi(ModelData);
  mochi::model_utils::BakeTransform(Baked, BakeScale, BakeTransformRT, E);
  MOCHI_ERROR_RETURN(E, {});
  const mochi::ShapeHandle Shape = MochiWorld->CreateModelShape(Baked, E);
  MOCHI_ERROR_RETURN(E, {});
  return Shape;
}

bool UMochiModel::IsValid() const {
  return HasModelGeometry(ModelData);
}

mochi::ModelData UMochiModel::GetMochiModelData(mochi::Error& Error) const {
  MOCHI_ERROR_RETURN(Error, {});
  if (!HasModelGeometry(ModelData)) {
    MOCHI_ERROR_SET(Error, "UMochiModel::GetMochiModelData called on asset with no data");
    return {};
  }
  mochi::ModelData MochiModelData = ::ToMochi(ModelData);

  // Bake the same import/space transform the shape-loading path applies (GetMochiShape ->
  // GetMochiShape_Raw, which bakes before CreateModelShape). With no explicit bake,
  // BakeScale is 1 and BakeTransformRT is identity, so FinalMatrix = SpaceMatrix * ImportMatrix:
  // the asset's editor-time ImportTransform followed by the Unreal->mochi basis change and units
  // scale. This puts ModelData in the same native mochi space the shape path produces, keeping the
  // two loaders consistent for any ImportTransform / simulation space. For the SuperDex bots
  // configuration the net transform is identity, so this is a no-op there.
  FUnrealToMochiConverter U2m;
  mochi::VMatrix4x4r ImportMatrix = UnrealToMatrix(ImportTransform.GetScale3D(), ImportTransform);
  mochi::VMatrix4x4r SpaceMatrix = Dot4x4(
      U2m.GetBasisChangeMatrix(),
      mochi::VDiagonalMatrix<4>(static_cast<mochi::real>(U2m.OutputSpace.UnitsPerMeter)));
  mochi::VMatrix4x4r FinalMatrix = Dot4x4(SpaceMatrix, ImportMatrix);
  std::pair<mochi::Real3, mochi::TransformRT> BakeTransform =
      mochi::DecomposeMatrixTransform(FinalMatrix);
  mochi::Real3& BakeScale = BakeTransform.first;
  mochi::TransformRT& BakeTransformRT = BakeTransform.second;
  mochi::model_utils::BakeTransform(MochiModelData, BakeScale, BakeTransformRT, Error);
  MOCHI_ERROR_RETURN(Error, {});

  return MochiModelData;
}

mochi::ShapeHandle UMochiModel::GetTriangularMochiShapeFromStaticMesh(
    const UStaticMesh* Mesh,
    const ETriangularMeshCollisionSource CollisionSource,
    FVector Scale3D) {
  mochi::Context* const MochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();

  if (!Mesh) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to create triangular mesh because static mesh is invalid!"))
    return mochi::ShapeHandle();
  }
  if (!Mesh->bAllowCPUAccess) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "[UMochiModel] Static mesh does not have bAllowCPUAccess enabled; undefined behavior may occur, especially in packaged builds!"))
  }

  FUnrealToMochiConverter U2m;
  Scale3D = U2m.ScaleToOutput(Scale3D);
  // complex collision and simple collision seem to have different winding order
  bool bReverseWindingOrder = CollisionSource == ETriangularMeshCollisionSource::Simple;
  // if handedness changes we need to rever winding order again
  bReverseWindingOrder = U2m.FlipsHandedness() != bReverseWindingOrder;

  TArray<mochi::real> Coordinates;
  TArray<int> Connectivity;

  UBodySetup* BodySetup = Mesh->GetBodySetup();

  switch (CollisionSource) {
    case ETriangularMeshCollisionSource::Simple:
      if (!BodySetup || BodySetup->AggGeom.ConvexElems.IsEmpty()) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT(
                "[UMochiModel] Failed to create triangular mesh because simple collider is not present! [%s]"),
            *Mesh->GetName())
        return mochi::ShapeHandle();
      }

      for (const FKConvexElem& ConvexElem : BodySetup->AggGeom.ConvexElems) {
        const TArray<FVector>& ElemVertices = ConvexElem.VertexData;
        const TArray<int32> ElemIndices = ConvexElem.IndexData;

        if (ElemIndices.Num() % 3 != 0) {
          UE_LOG(
              LogMochiPhysics,
              Error,
              TEXT(
                  "[UMochiModel] Failed to create triangular mesh because simple collider is invalid! [%s]"),
              *Mesh->GetName())
          return mochi::ShapeHandle();
        }

        const int32 VertexOffset = Coordinates.Num() / 3;
        for (const FVector& Vertex : ConvexElem.VertexData) {
          mochi::Real3 V = U2m.TranslationToMochi(Vertex);
          Coordinates.Add(Scale3D.X * V[0]);
          Coordinates.Add(Scale3D.Y * V[1]);
          Coordinates.Add(Scale3D.Z * V[2]);
        }

        for (int32 i = 0; i < ElemIndices.Num(); i += 3) {
          if (bReverseWindingOrder) {
            // Inverted triangle winding
            Connectivity.Add(VertexOffset + ElemIndices[i]);
            Connectivity.Add(VertexOffset + ElemIndices[i + 1]);
            Connectivity.Add(VertexOffset + ElemIndices[i + 2]);
          } else {
            // Standard triangle winding
            Connectivity.Add(VertexOffset + ElemIndices[i]);
            Connectivity.Add(VertexOffset + ElemIndices[i + 2]);
            Connectivity.Add(VertexOffset + ElemIndices[i + 1]);
          }
        }
      }
      UE_LOG(
          LogMochiPhysics,
          Display,
          TEXT(
              "[UMochiModel] Created triangular mesh using simple colliders for [%s] (%d coordinates, %d connectivity)"),
          *Mesh->GetName(),
          Coordinates.Num(),
          Connectivity.Num());
      break;

    default:
    case ETriangularMeshCollisionSource::Complex:
      if (!Mesh->GetRenderData()) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT(
                "[UMochiModel] Failed to create triangular mesh because complex collider render data is invalid! [%s]"),
            *Mesh->GetName())
        return mochi::ShapeHandle();
      }
      const FStaticMeshLODResources* LOD = Mesh->GetRenderData()->GetCurrentFirstLOD(0);

      if (!LOD) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT(
                "[UMochiModel] Failed to create triangular mesh because unable to retrieve LOD data! [%s]"),
            *Mesh->GetName())
        return mochi::ShapeHandle();
      }

      uint32 VertexCount = LOD->VertexBuffers.PositionVertexBuffer.GetNumVertices();
      uint32 IndexCount = LOD->IndexBuffer.GetArrayView().Num();

      if (IndexCount % 3 != 0 || VertexCount == 0) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT(
                "[UMochiModel] Failed to create triangular mesh because complex collider mesh is invalid! [%s]"),
            *Mesh->GetName())
        return mochi::ShapeHandle();
      }

      if (IndexCount == 0) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT("[UMochiModel] Failed to create triangular mesh. Index buffer is empty! [%s]"),
            *Mesh->GetName())
        return mochi::ShapeHandle();
      }

      // 3 coords per vertex
      Coordinates.SetNum(VertexCount * 3);
      Connectivity.SetNum(IndexCount);

      for (uint32 i = 0; i < VertexCount; i++) {
        FVector3f Position = LOD->VertexBuffers.PositionVertexBuffer.VertexPosition(i);
        mochi::Real3 P = U2m.TranslationToMochi(static_cast<FVector>(Position));
        int Idx = i * 3;
        Coordinates[Idx] = Scale3D.X * P[0];
        Coordinates[Idx + 1] = Scale3D.Y * P[1];
        Coordinates[Idx + 2] = Scale3D.Z * P[2];
      }

      for (uint32 i = 0; i < IndexCount; i += 3) {
        if (bReverseWindingOrder) {
          // Inverted triangle winding
          Connectivity[i] = LOD->IndexBuffer.GetArrayView()[i];
          Connectivity[i + 1] = LOD->IndexBuffer.GetArrayView()[i + 1];
          Connectivity[i + 2] = LOD->IndexBuffer.GetArrayView()[i + 2];
        } else {
          // Standard triangle winding
          Connectivity[i] = LOD->IndexBuffer.GetArrayView()[i];
          Connectivity[i + 2] = LOD->IndexBuffer.GetArrayView()[i + 1];
          Connectivity[i + 1] = LOD->IndexBuffer.GetArrayView()[i + 2];
        }
      }
      UE_LOG(
          LogMochiPhysics,
          Display,
          TEXT(
              "[UMochiModel] Created triangular mesh using complex colliders for [%s] (%d coordinates, %d connectivity)"),
          *Mesh->GetName(),
          Coordinates.Num(),
          Connectivity.Num())
      break;
  }

  if (Coordinates.Num() == 0 || Connectivity.Num() == 0) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiModel] Failed to create mochi::ShapeHandle from triangular mesh because coordinates and/or connectivity were empty; did you enable bAllowCPUAccess? [%s]"),
        *Mesh->GetName());
    return mochi::ShapeHandle();
  }

  mochi::Error E;
  mochi::ShapeHandle ShapeHandle =
      MochiWorld->CreateTriMeshShape(ToMochiSpan(Coordinates), ToMochiSpan(Connectivity), E);

  if (!ShapeHandle.IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to create mochi::ShapeHandle from triangular mesh [%s]"),
        *Mesh->GetName());
    return mochi::ShapeHandle();
  }

  if (!E.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to initialize triangular mesh [%s] : %hs"),
        *Mesh->GetName(),
        E.GetDescription())
    return mochi::ShapeHandle();
  }

  return ShapeHandle;
}

#if WITH_EDITOR
void UMochiModel::PostInitProperties() {
  Super::PostInitProperties();
  if (!HasAnyFlags(RF_ClassDefaultObject | RF_NeedLoad)) {
    AssetImportData = NewObject<UAssetImportData>(this, TEXT("AssetImportData"));
  }
}

void UMochiModel::GetAssetRegistryTags(FAssetRegistryTagsContext Context) const {
  if (AssetImportData) {
    Context.AddTag(FAssetRegistryTag(
        SourceFileTagName(),
        AssetImportData->GetSourceData().ToJson(),
        FAssetRegistryTag::TT_Hidden));
  }
  Super::GetAssetRegistryTags(Context);
}

FString UMochiModel::GetSourceFilePath() const {
  return AssetImportData ? AssetImportData->GetFirstFilename() : FString();
}

void UMochiModel::Import() {
  MOCHI_PROFILE_SCOPE();
  IDesktopPlatform* DesktopPlatform = FDesktopPlatformModule::Get();
  if (!DesktopPlatform) {
    return;
  }
  const void* ParentWindowWindowHandle =
      FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr);
  const FString FileTypes = TEXT("HDF5 files (*.h5)|*.h5|All files (*.*)|*.*");
  const FString SourcePath = GetSourceFilePath();
  const FString DefaultPath = SourcePath.Len()
      ? SourcePath
      : FEditorDirectories::Get().GetLastDirectory(ELastDirectory::GENERIC_IMPORT);
  TArray<FString> OpenFilenames;
  int32 FilterIndex = -1;
  const bool bOpened = DesktopPlatform->OpenFileDialog(
      ParentWindowWindowHandle,
      TEXT("Open Mochi H5 File"),
      DefaultPath,
      TEXT(""),
      FileTypes,
      EFileDialogFlags::None,
      OpenFilenames,
      FilterIndex);
  if (bOpened) {
    // Did not set flag for multiple selections, so should be just one
    check(OpenFilenames.Num() == 1);
    ImportFromFilename(OpenFilenames[0]);
  }
}

void UMochiModel::Reimport() {
  const FString SourcePath = GetSourceFilePath();
  if (!SourcePath.IsEmpty()) {
    ImportFromFilename(SourcePath);
  }
}

bool UMochiModel::ImportFromFilename(const FString& InFilename) {
  MOCHI_PROFILE_SCOPE();
  TArray<uint8> Bytes;
  if (!FFileHelper::LoadFileToArray(Bytes, *InFilename)) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] Failed to read file: [%s]"), *InFilename);
    return false;
  }
  if (Bytes.Num() == 0) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] File is empty: [%s]"), *InFilename);
    return false;
  }

  mochi::Error Error;
  const mochi::ModelData MochiModelData =
      mochi::model_utils::LoadFromBytes(ToMochiBinarySpan(Bytes), Error);
  if (!Error.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to read the model in [%s]. Reason: %hs"),
        *InFilename,
        Error.ToString().c_str());
    return false;
  }
  if (MochiModelData.experimentalDataDetected) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiModel] [%s] holds experimental data (rom, sample_meshes or bsh) that "
            "FMochiModelData cannot represent."),
        *InFilename);
    return false;
  }

  if (!ApplyModelData(MochiModelData)) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] Failed to import: [%s]"), *InFilename);
    return false;
  }
  if (AssetImportData) {
    AssetImportData->Update(InFilename);
  }
  return true;
}

bool UMochiModel::ExportToFilename(const FString& InFilename) const {
  if (!HasModelGeometry(ModelData)) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] %s has no model data to export."),
        *GetPathName());
    return false;
  }
  mochi::Error Error;
  mochi::model_utils::SaveToFile(
      ::ToMochi(ModelData), std::string{TCHAR_TO_ANSI(*InFilename)}, mochi::FileFormat::H5, Error);
  if (!Error.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to export %s. Reason: %hs"),
        *GetPathName(),
        Error.ToString().c_str());
    return false;
  }
  return true;
}

bool UMochiModel::ApplyModelData(const mochi::ModelData& Data) {
  Modify(true);
  // RefreshMeshCounts is where a model Mochi cannot build a shape from gets caught, and it
  // needs ModelData already set to do that. Hold onto the old contents until it passes so
  // that a failed import or bake leaves the asset as it was rather than half-replaced.
  // Moving is cheap, and RefreshMeshCounts writes no counts unless it succeeds.
  FMochiModelData Previous = MoveTemp(ModelData);
  ModelData = ::ToUnreal(Data);
  if (!RefreshMeshCounts()) {
    ModelData = MoveTemp(Previous);
    return false;
  }
  MarkPackageDirty();
  return true;
}

bool UMochiModel::RefreshMeshCounts() {
  MOCHI_PROFILE_SCOPE();
  mochi::Context* const MochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();
  const mochi::ShapeHandle Shape = GetMochiShape(FVector::OneVector, FTransform::Identity);
  if (!Shape.IsValid()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] Shape is invalid"));
    return false;
  }
  mochi::Error E{};
  const mochi::MeshDataView ShapeMesh = MochiWorld->GetShapeMesh(Shape, E);
  MOCHI_ERROR_RETURN(E, false);
  if (ShapeMesh.IsEmpty()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] Shape does not have mesh data"));
    return false;
  }

  // Polyline mesh (rod) - no surface mesh or volume elements
  if (ShapeMesh.nodesPerElement == 2) {
    NumNodes = ShapeMesh.GetNumNodes();
    NumEdges = ShapeMesh.GetNumElements();
    NumPolyhedrons = 0;
    NumSurfaceNodes = 0;
    NumSurfaceEdges = 0;
    NumSurfaceFaces = 0;
    return true;
  }

  const mochi::MeshDataView SurfaceMesh = MochiWorld->GetShapeSurfaceMesh(Shape, E);
  MOCHI_ERROR_RETURN(E, false);
  if (SurfaceMesh.IsEmpty()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] Shape does not have surface mesh data"));
    return false;
  }
  // Tetrahedral mesh
  if (ShapeMesh.nodesPerElement == 4) {
    NumNodes = ShapeMesh.GetNumNodes();
    NumEdges = static_cast<int>(mochi::ComputeEdgeIndices(ShapeMesh).size() / 2);
    NumPolyhedrons = ShapeMesh.GetNumElements();
  } else {
    NumNodes = 0;
    NumEdges = 0;
    NumPolyhedrons = 0;
  }
  // Surface mesh
  NumSurfaceNodes = SurfaceMesh.GetNumNodes();
  NumSurfaceEdges = static_cast<int>(mochi::ComputeEdgeIndices(SurfaceMesh).size() / 2);
  NumSurfaceFaces = SurfaceMesh.GetNumElements();
  return true;
}

bool UMochiModel::ConvertTetMeshToSurfaceMesh() {
  if (!HasModelGeometry(ModelData)) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] No model data to process."));
    return false;
  }

  mochi::Error Error;
  mochi::Context* const MochiContext = IMochiPhysicsModule::Get()->GetMochiContext();

  mochi::ModelData Model = ::ToMochi(ModelData);

  // early-out if no mesh or not a tetrahedral mesh
  if (!Model.mesh.has_value() || Model.mesh->nodesPerElement != 4) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiModel] %s does not contain a tetrahedral mesh. Aborting."),
        *GetName());
    return false;
  }

  // The surface of a tet mesh is derived by Mochi, so go through a shape to read it back.
  const mochi::ShapeHandle Shape = MochiContext->CreateModelShape(Model, Error);
  if (!Error.IsOK() || !Shape.IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to create shape from model data. Reason: %hs"),
        Error.ToString().c_str());
    return false;
  }

  const mochi::MeshDataView SurfaceMeshView = MochiContext->GetShapeSurfaceMesh(Shape, Error);
  if (!Error.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to get surface mesh from shape. Reason: %hs"),
        Error.ToString().c_str());
    return false;
  }
  if (SurfaceMeshView.IsEmpty()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiModel] Failed to get surface mesh from shape: Surface mesh data is unavailable."));
    return false;
  }

  // Replace existing mesh with surface mesh
  Model.mesh.emplace();
  Model.mesh->nodesPerElement = SurfaceMeshView.nodesPerElement;
  // assign to copy from temporary SurfaceMeshView
  Model.mesh->coordinates.assign(
      SurfaceMeshView.coordinates.begin(), SurfaceMeshView.coordinates.end());
  Model.mesh->connectivity.assign(
      SurfaceMeshView.connectivity.begin(), SurfaceMeshView.connectivity.end());

  // Validate the new model data
  mochi::model_utils::AutoCorrect(Model, Error);
  mochi::model_utils::Validate(Model, Error);
  if (!Error.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Surface mesh data is invalid. Reason: %hs"),
        Error.ToString().c_str());
    return false;
  }

  if (!ApplyModelData(Model)) {
    return false;
  }

  UE_LOG(
      LogMochiPhysics,
      Display,
      TEXT("[UMochiModel] Successfully converted to surface mesh with %d nodes and %d triangles."),
      NumSurfaceNodes,
      NumSurfaceFaces);

  return true;
}

bool UMochiModel::BakeSdf(const FMochiGridSdfParams* SdfParams) {
  if (!HasModelGeometry(ModelData)) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] No model data to process."));
    return false;
  }

  mochi::Error Error;
  mochi::ModelData MochiModelData = ::ToMochi(ModelData);

  // Check if this has a mesh
  if (!MochiModelData.mesh.has_value()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] Cannot bake SDF: no mesh data."));
    return false;
  }

  // Use provided params or default
  mochi::GridSdfParams MochiSdfParams = SdfParams ? ::ToMochi(*SdfParams) : mochi::GridSdfParams{};

  // Bake the SDF
  mochi::model_utils::BakeSdf(MochiModelData, MochiSdfParams, Error);
  if (!Error.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to bake SDF. Reason: %hs"),
        Error.ToString().c_str());
    return false;
  }

  if (!ApplyModelData(MochiModelData)) {
    return false;
  }

  UE_LOG(LogMochiPhysics, Display, TEXT("[UMochiModel] Successfully baked SDF into model."));
  return true;
}
#endif // WITH_EDITOR

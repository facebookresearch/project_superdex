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
#include "MochiPhysicsModule.h"

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

inline mochi::Quaternion ToMochiType(FQuat r) {
  using mr = mochi::real;
  return mochi::Quaternion((mr)r.X, (mr)r.Y, (mr)r.Z, (mr)r.W);
}

inline mochi::Real3 ToMochiType(FVector t) {
  using mr = mochi::real;
  return mochi::Real3((mr)t.X, (mr)t.Y, (mr)t.Z);
}

// Input: LH, Z up, centimeters (Unreal)
// Output: LH, Z up, meters (Unreal)
inline mochi::VMatrix4x4r UnrealToMatrix(const FVector& S, const FTransform& RT) {
  mochi::VMatrix4x4r s = mochi::VDiagonalMatrix<4>(ToSimd(ToMochiType(S), 1_r));
  mochi::Quaternion r = ToMochiType(RT.GetRotation());
  mochi::Real3 t = ToMochiType(RT.GetTranslation() / kUnrealCentimetersSpace.UnitsPerMeter);
  mochi::VMatrix4x4r rt = ToVMatrix4x4(mochi::TransformRT(r, t));
  mochi::VMatrix4x4r srt = Dot4x4(rt, s);
  return srt;
}

constexpr int64 kBytesPerFloat = sizeof(float);

// Regroups a float array so that every first byte precedes every second byte, and so on.
// Across a distance field the exponent and high mantissa bytes change slowly while the
// low ones look like noise, so interleaved floats give a general-purpose compressor
// almost nothing to work with; separating the planes exposes the runs. This is what
// HDF5's shuffle filter does, and it accounts for most of the ratio the H5 binary
// achieves over plain zlib.
void ShuffleFloatBytes(const float* Src, uint8* Dst, int64 NumElements) {
  const uint8* srcBytes = reinterpret_cast<const uint8*>(Src);
  for (int64 i = 0; i < NumElements; ++i) {
    for (int64 b = 0; b < kBytesPerFloat; ++b) {
      Dst[b * NumElements + i] = srcBytes[i * kBytesPerFloat + b];
    }
  }
}

void UnshuffleFloatBytes(const uint8* Src, float* Dst, int64 NumElements) {
  uint8* dstBytes = reinterpret_cast<uint8*>(Dst);
  for (int64 i = 0; i < NumElements; ++i) {
    for (int64 b = 0; b < kBytesPerFloat; ++b) {
      dstBytes[i * kBytesPerFloat + b] = Src[b * NumElements + i];
    }
  }
}

} // namespace

void UMochiModel::Serialize(FArchive& Ar) {
  Ar.UsingCustomVersion(FMochiObjectVersion::GUID);
  Super::Serialize(Ar);

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
    const TArray<float>* values = ModelData.Sdf.IsSet() ? &ModelData.Sdf->Values : nullptr;
    const int64 numElements = values ? values->Num() : 0;
    // Shuffling straight into the bulk buffer keeps this from needing a second copy of a
    // grid that can run to gigabytes.
    SdfValues.Lock(LOCK_READ_WRITE);
    ShuffleFloatBytes(
        values ? values->GetData() : nullptr,
        static_cast<uint8*>(SdfValues.Realloc(numElements * kBytesPerFloat)),
        numElements);
    SdfValues.Unlock();
  }

  SdfValues.Serialize(Ar, this);

  if (Ar.IsLoading()) {
    const int64 numElements = SdfValues.GetBulkDataSize() / kBytesPerFloat;
    if (numElements == 0) {
      if (ModelData.Sdf.IsSet()) {
        ModelData.Sdf->Values.Reset();
      }
      return;
    }
    TArray<float>& values =
        ModelData.Sdf.IsSet() ? ModelData.Sdf->Values : ModelData.Sdf.Emplace().Values;
    values.SetNumUninitialized(static_cast<int32>(numElements));
    UnshuffleFloatBytes(
        static_cast<const uint8*>(SdfValues.LockReadOnly()), values.GetData(), numElements);
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

  if (ModelData.IsEmpty()) {
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
// finalMatrix = (basis change * units scale) * (BakTransformRT * BakeScale) * ImportTransform
//             = (spaceMatrix)                * (bakeMatrix)                 * importMatrix

mochi::ShapeHandle UMochiModel::GetMochiShape(
    const FVector& BakeScale,
    const FTransform& BakeTransformRT) const {
  FUnrealToMochiConverter u2m;
  auto importMatrix = UnrealToMatrix(ImportTransform.GetScale3D(), ImportTransform);
  auto bakeMatrix = UnrealToMatrix(BakeScale, BakeTransformRT);
  auto spaceMatrix = Dot4x4(
      u2m.GetBasisChangeMatrix(),
      mochi::VDiagonalMatrix<4>(static_cast<mochi::real>(u2m.OutputSpace.UnitsPerMeter)));
  auto finalMatrix = Dot4x4(spaceMatrix, Dot4x4(bakeMatrix, importMatrix));
  return GetMochiShape_Raw(finalMatrix);
}

mochi::ShapeHandle UMochiModel::GetMochiShape(
    const FVector& BakeScale,
    const mochi::VMatrix4x4r& BakeTransformRT) const {
  MOCHI_PROFILE_SCOPE();
  FUnrealToMochiConverter u2m;
  auto importMatrix = UnrealToMatrix(ImportTransform.GetScale3D(), ImportTransform);
  auto bakeMatrix =
      Dot4x4(BakeTransformRT, mochi::VDiagonalMatrix<4>(ToSimd(ToMochiType(BakeScale), 1_r)));
  auto spaceMatrix = Dot4x4(
      u2m.GetBasisChangeMatrix(),
      mochi::VDiagonalMatrix<4>(static_cast<mochi::real>(u2m.OutputSpace.UnitsPerMeter)));
  auto finalMatrix = Dot4x4(spaceMatrix, Dot4x4(bakeMatrix, importMatrix));
  return GetMochiShape_Raw(finalMatrix);
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
  mochi::ErrorLog e{};
  const auto mochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();
  auto [bakeScale, bakeTransformRT] = mochi::DecomposeMatrixTransform(BakeMatrix);

  if (ModelData.IsEmpty()) {
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
  mochi::ModelData baked = ModelData.ToMochi();
  mochi::model_utils::BakeTransform(baked, bakeScale, bakeTransformRT, e);
  MOCHI_ERROR_RETURN(e, {});
  const auto shape = mochiWorld->CreateModelShape(baked, e);
  MOCHI_ERROR_RETURN(e, {});
  return shape;
}

bool UMochiModel::IsValid() const {
  return !ModelData.IsEmpty();
}

mochi::ModelData UMochiModel::GetMochiModelData(mochi::Error& error) const {
  MOCHI_ERROR_RETURN(error, {});
  if (ModelData.IsEmpty()) {
    MOCHI_ERROR_SET(error, "UMochiModel::GetMochiModelData called on asset with no data");
    return {};
  }
  mochi::ModelData modelData = ModelData.ToMochi();

  // Bake the same import/space transform the shape-loading path applies (GetMochiShape ->
  // GetMochiShape_Raw, which bakes before CreateModelShape). With no explicit bake,
  // BakeScale is 1 and BakeTransformRT is identity, so finalMatrix = spaceMatrix * importMatrix:
  // the asset's editor-time ImportTransform followed by the Unreal->mochi basis change and units
  // scale. This puts ModelData in the same native mochi space the shape path produces, keeping the
  // two loaders consistent for any ImportTransform / simulation space. For the SuperDex bots
  // configuration the net transform is identity, so this is a no-op there.
  FUnrealToMochiConverter u2m;
  auto importMatrix = UnrealToMatrix(ImportTransform.GetScale3D(), ImportTransform);
  auto spaceMatrix = Dot4x4(
      u2m.GetBasisChangeMatrix(),
      mochi::VDiagonalMatrix<4>(static_cast<mochi::real>(u2m.OutputSpace.UnitsPerMeter)));
  auto finalMatrix = Dot4x4(spaceMatrix, importMatrix);
  auto [bakeScale, bakeTransformRT] = mochi::DecomposeMatrixTransform(finalMatrix);
  mochi::model_utils::BakeTransform(modelData, bakeScale, bakeTransformRT, error);
  MOCHI_ERROR_RETURN(error, {});

  return modelData;
}

mochi::ShapeHandle UMochiModel::GetTriangularMochiShapeFromStaticMesh(
    const UStaticMesh* mesh,
    const ETriangularMeshCollisionSource collisionSource,
    FVector scale3D) {
  const auto mochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();

  if (!mesh) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to create triangular mesh because static mesh is invalid!"))
    return mochi::ShapeHandle();
  }
  if (!mesh->bAllowCPUAccess) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT(
            "[UMochiModel] Static mesh does not have bAllowCPUAccess enabled; undefined behavior may occur, especially in packaged builds!"))
  }

  FUnrealToMochiConverter u2m;
  scale3D = u2m.ScaleToOutput(scale3D);
  // complex collision and simple collision seem to have different winding order
  bool bReverseWindingOrder = collisionSource == ETriangularMeshCollisionSource::Simple;
  // if handedness changes we need to rever winding order again
  bReverseWindingOrder = u2m.FlipsHandedness() != bReverseWindingOrder;

  TArray<mochi::real> Coordinates;
  TArray<int> Connectivity;

  UBodySetup* bodySetup = mesh->GetBodySetup();

  switch (collisionSource) {
    case ETriangularMeshCollisionSource::Simple:
      if (!bodySetup || bodySetup->AggGeom.ConvexElems.IsEmpty()) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT(
                "[UMochiModel] Failed to create triangular mesh because simple collider is not present! [%s]"),
            *mesh->GetName())
        return mochi::ShapeHandle();
      }

      for (const auto& convexElem : bodySetup->AggGeom.ConvexElems) {
        const TArray<FVector>& ElemVertices = convexElem.VertexData;
        const auto ElemIndices = convexElem.IndexData;

        if (ElemIndices.Num() % 3 != 0) {
          UE_LOG(
              LogMochiPhysics,
              Error,
              TEXT(
                  "[UMochiModel] Failed to create triangular mesh because simple collider is invalid! [%s]"),
              *mesh->GetName())
          return mochi::ShapeHandle();
        }

        const int32 vertexOffset = Coordinates.Num() / 3;
        for (const FVector& Vertex : convexElem.VertexData) {
          auto v = u2m.TranslationToMochi(Vertex);
          Coordinates.Add(scale3D.X * v[0]);
          Coordinates.Add(scale3D.Y * v[1]);
          Coordinates.Add(scale3D.Z * v[2]);
        }

        for (int32 i = 0; i < ElemIndices.Num(); i += 3) {
          if (bReverseWindingOrder) {
            // Inverted triangle winding
            Connectivity.Add(vertexOffset + ElemIndices[i]);
            Connectivity.Add(vertexOffset + ElemIndices[i + 1]);
            Connectivity.Add(vertexOffset + ElemIndices[i + 2]);
          } else {
            // Standard triangle winding
            Connectivity.Add(vertexOffset + ElemIndices[i]);
            Connectivity.Add(vertexOffset + ElemIndices[i + 2]);
            Connectivity.Add(vertexOffset + ElemIndices[i + 1]);
          }
        }
      }
      UE_LOG(
          LogMochiPhysics,
          Display,
          TEXT(
              "[UMochiModel] Created triangular mesh using simple colliders for [%s] (%d coordinates, %d connectivity)"),
          *mesh->GetName(),
          Coordinates.Num(),
          Connectivity.Num());
      break;

    default:
    case ETriangularMeshCollisionSource::Complex:
      if (!mesh->GetRenderData()) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT(
                "[UMochiModel] Failed to create triangular mesh because complex collider render data is invalid! [%s]"),
            *mesh->GetName())
        return mochi::ShapeHandle();
      }
      const FStaticMeshLODResources* LOD = mesh->GetRenderData()->GetCurrentFirstLOD(0);

      if (!LOD) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT(
                "[UMochiModel] Failed to create triangular mesh because unable to retrieve LOD data! [%s]"),
            *mesh->GetName())
        return mochi::ShapeHandle();
      }

      uint32 vertexCount = LOD->VertexBuffers.PositionVertexBuffer.GetNumVertices();
      uint32 indexCount = LOD->IndexBuffer.GetArrayView().Num();

      if (indexCount % 3 != 0 || vertexCount == 0) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT(
                "[UMochiModel] Failed to create triangular mesh because complex collider mesh is invalid! [%s]"),
            *mesh->GetName())
        return mochi::ShapeHandle();
      }

      if (indexCount == 0) {
        UE_LOG(
            LogMochiPhysics,
            Error,
            TEXT("[UMochiModel] Failed to create triangular mesh. Index buffer is empty! [%s]"),
            *mesh->GetName())
        return mochi::ShapeHandle();
      }

      // 3 coords per vertex
      Coordinates.SetNum(vertexCount * 3);
      Connectivity.SetNum(indexCount);

      for (uint32 i = 0; i < vertexCount; i++) {
        FVector3f position = LOD->VertexBuffers.PositionVertexBuffer.VertexPosition(i);
        auto p = u2m.TranslationToMochi(static_cast<FVector>(position));
        int idx = i * 3;
        Coordinates[idx] = scale3D.X * p[0];
        Coordinates[idx + 1] = scale3D.Y * p[1];
        Coordinates[idx + 2] = scale3D.Z * p[2];
      }

      for (uint32 i = 0; i < indexCount; i += 3) {
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
          *mesh->GetName(),
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
        *mesh->GetName());
    return mochi::ShapeHandle();
  }

  mochi::Error e;
  mochi::ShapeHandle shapeHandle =
      mochiWorld->CreateTriMeshShape(ToMochiSpan(Coordinates), ToMochiSpan(Connectivity), e);

  if (!shapeHandle.IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to create mochi::ShapeHandle from triangular mesh [%s]"),
        *mesh->GetName());
    return mochi::ShapeHandle();
  }

  if (!e.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to initialize triangular mesh [%s] : %hs"),
        *mesh->GetName(),
        e.GetDescription())
    return mochi::ShapeHandle();
  }

  return shapeHandle;
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
  IDesktopPlatform* desktopPlatform = FDesktopPlatformModule::Get();
  if (!desktopPlatform) {
    return;
  }
  const void* parentWindowWindowHandle =
      FSlateApplication::Get().FindBestParentWindowHandleForDialogs(nullptr);
  const FString fileTypes = TEXT("HDF5 files (*.h5)|*.h5|All files (*.*)|*.*");
  const FString sourcePath = GetSourceFilePath();
  const FString defaultPath = sourcePath.Len()
      ? sourcePath
      : FEditorDirectories::Get().GetLastDirectory(ELastDirectory::GENERIC_IMPORT);
  TArray<FString> openFilenames;
  int32 filterIndex = -1;
  const bool bOpened = desktopPlatform->OpenFileDialog(
      parentWindowWindowHandle,
      TEXT("Open Mochi H5 File"),
      defaultPath,
      TEXT(""),
      fileTypes,
      EFileDialogFlags::None,
      openFilenames,
      filterIndex);
  if (bOpened) {
    // Did not set flag for multiple selections, so should be just one
    check(openFilenames.Num() == 1);
    ImportFromFilename(openFilenames[0]);
  }
}

void UMochiModel::Reimport() {
  const FString sourcePath = GetSourceFilePath();
  if (!sourcePath.IsEmpty()) {
    ImportFromFilename(sourcePath);
  }
}

bool UMochiModel::ImportFromFilename(const FString& InFilename) {
  MOCHI_PROFILE_SCOPE();
  TArray<uint8> bytes;
  if (!FFileHelper::LoadFileToArray(bytes, *InFilename)) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] Failed to read file: [%s]"), *InFilename);
    return false;
  }
  if (bytes.Num() == 0) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] File is empty: [%s]"), *InFilename);
    return false;
  }

  mochi::Error error;
  const mochi::ModelData modelData =
      mochi::model_utils::LoadFromBytes(ToMochiBinarySpan(bytes), error);
  if (!error.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to read the model in [%s]. Reason: %hs"),
        *InFilename,
        error.ToString().c_str());
    return false;
  }
  if (modelData.experimentalDataDetected) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiModel] [%s] holds experimental data (rom, sample_meshes or bsh) that "
            "FMochiModelData cannot represent."),
        *InFilename);
    return false;
  }

  if (!ApplyModelData(modelData)) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] Failed to import: [%s]"), *InFilename);
    return false;
  }
  if (AssetImportData) {
    AssetImportData->Update(InFilename);
  }
  return true;
}

bool UMochiModel::ExportToFilename(const FString& InFilename) const {
  if (ModelData.IsEmpty()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] %s has no model data to export."),
        *GetPathName());
    return false;
  }
  mochi::Error error;
  mochi::model_utils::SaveToFile(
      ModelData.ToMochi(), std::string{TCHAR_TO_ANSI(*InFilename)}, mochi::FileFormat::H5, error);
  if (!error.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to export %s. Reason: %hs"),
        *GetPathName(),
        error.ToString().c_str());
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
  FMochiModelData previous = MoveTemp(ModelData);
  ModelData.FromMochi(Data);
  if (!RefreshMeshCounts()) {
    ModelData = MoveTemp(previous);
    return false;
  }
  MarkPackageDirty();
  return true;
}

bool UMochiModel::RefreshMeshCounts() {
  MOCHI_PROFILE_SCOPE();
  const auto mochiWorld = IMochiPhysicsModule::Get()->GetMochiContext();
  const auto shape = GetMochiShape(FVector::OneVector, FTransform::Identity);
  if (!shape.IsValid()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] Shape is invalid"));
    return false;
  }
  mochi::Error e{};
  auto const shapeMesh = mochiWorld->GetShapeMesh(shape, e);
  MOCHI_ERROR_RETURN(e, false);
  if (shapeMesh.IsEmpty()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] Shape does not have mesh data"));
    return false;
  }

  // Polyline mesh (rod) - no surface mesh or volume elements
  if (shapeMesh.nodesPerElement == 2) {
    NumNodes = shapeMesh.GetNumNodes();
    NumEdges = shapeMesh.GetNumElements();
    NumPolyhedrons = 0;
    NumSurfaceNodes = 0;
    NumSurfaceEdges = 0;
    NumSurfaceFaces = 0;
    return true;
  }

  auto const surfaceMesh = mochiWorld->GetShapeSurfaceMesh(shape, e);
  MOCHI_ERROR_RETURN(e, false);
  if (surfaceMesh.IsEmpty()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] Shape does not have surface mesh data"));
    return false;
  }
  // Tetrahedral mesh
  if (shapeMesh.nodesPerElement == 4) {
    NumNodes = shapeMesh.GetNumNodes();
    NumEdges = static_cast<int>(mochi::ComputeEdgeIndices(shapeMesh).size() / 2);
    NumPolyhedrons = shapeMesh.GetNumElements();
  } else {
    NumNodes = 0;
    NumEdges = 0;
    NumPolyhedrons = 0;
  }
  // Surface mesh
  NumSurfaceNodes = surfaceMesh.GetNumNodes();
  NumSurfaceEdges = static_cast<int>(mochi::ComputeEdgeIndices(surfaceMesh).size() / 2);
  NumSurfaceFaces = surfaceMesh.GetNumElements();
  return true;
}

bool UMochiModel::ConvertTetMeshToSurfaceMesh() {
  if (ModelData.IsEmpty()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] No model data to process."));
    return false;
  }

  mochi::Error error;
  auto const mochiContext = IMochiPhysicsModule::Get()->GetMochiContext();

  auto model = ModelData.ToMochi();

  // early-out if no mesh or not a tetrahedral mesh
  if (!model.mesh.has_value() || model.mesh->nodesPerElement != 4) {
    UE_LOG(
        LogMochiPhysics,
        Warning,
        TEXT("[UMochiModel] %s does not contain a tetrahedral mesh. Aborting."),
        *GetName());
    return false;
  }

  // The surface of a tet mesh is derived by Mochi, so go through a shape to read it back.
  auto const shape = mochiContext->CreateModelShape(model, error);
  if (!error.IsOK() || !shape.IsValid()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to create shape from model data. Reason: %hs"),
        error.ToString().c_str());
    return false;
  }

  auto const surfaceMeshView = mochiContext->GetShapeSurfaceMesh(shape, error);
  if (!error.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to get surface mesh from shape. Reason: %hs"),
        error.ToString().c_str());
    return false;
  }
  if (surfaceMeshView.IsEmpty()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT(
            "[UMochiModel] Failed to get surface mesh from shape: Surface mesh data is unavailable."));
    return false;
  }

  // Replace existing mesh with surface mesh
  model.mesh.emplace();
  model.mesh->nodesPerElement = surfaceMeshView.nodesPerElement;
  // assign to copy from temporary surfaceMeshView
  model.mesh->coordinates.assign(
      surfaceMeshView.coordinates.begin(), surfaceMeshView.coordinates.end());
  model.mesh->connectivity.assign(
      surfaceMeshView.connectivity.begin(), surfaceMeshView.connectivity.end());

  // Validate the new model data
  mochi::model_utils::AutoCorrect(model, error);
  mochi::model_utils::Validate(model, error);
  if (!error.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Surface mesh data is invalid. Reason: %hs"),
        error.ToString().c_str());
    return false;
  }

  if (!ApplyModelData(model)) {
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
  if (ModelData.IsEmpty()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] No model data to process."));
    return false;
  }

  mochi::Error error;
  mochi::ModelData modelData = ModelData.ToMochi();

  // Check if this has a mesh
  if (!modelData.mesh.has_value()) {
    UE_LOG(LogMochiPhysics, Error, TEXT("[UMochiModel] Cannot bake SDF: no mesh data."));
    return false;
  }

  // Use provided params or default
  mochi::GridSdfParams sdfParams = SdfParams ? SdfParams->ToMochi() : mochi::GridSdfParams{};

  // Bake the SDF
  mochi::model_utils::BakeSdf(modelData, sdfParams, error);
  if (!error.IsOK()) {
    UE_LOG(
        LogMochiPhysics,
        Error,
        TEXT("[UMochiModel] Failed to bake SDF. Reason: %hs"),
        error.ToString().c_str());
    return false;
  }

  if (!ApplyModelData(modelData)) {
    return false;
  }

  UE_LOG(LogMochiPhysics, Display, TEXT("[UMochiModel] Successfully baked SDF into model."));
  return true;
}
#endif // WITH_EDITOR

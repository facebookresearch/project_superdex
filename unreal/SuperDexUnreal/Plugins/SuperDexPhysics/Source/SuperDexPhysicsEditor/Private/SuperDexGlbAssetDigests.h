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

#pragma once

#include "CoreMinimal.h"

// Content digests of the materials and images of a SuperDex render model (.glb), used to tell
// whether two GLBs' same-named materials or images would import as the same Unreal asset.
namespace SuperDexGlbAssetDigests {

// Name of the asset Interchange imports a glTF material or image named GltfName as.
FString AssetName(const FString& GltfName);

// Object path of the asset named AssetName in the package folder PackagePath.
FString AssetObjectPath(const FString& PackagePath, const FString& AssetName);

// Digests of the named materials and images of a .glb, keyed by AssetName. An image's digest is
// that of its bytes; a material's covers its properties and its textures' images, whatever their
// order in the file. Unnamed materials and images get none. Returns false, with no digests, if the
// file is not a readable .glb.
bool Read(const FString& GlbPath, TMap<FString, FString>& OutDigests);

} // namespace SuperDexGlbAssetDigests

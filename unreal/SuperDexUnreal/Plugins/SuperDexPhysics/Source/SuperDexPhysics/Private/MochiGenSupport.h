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

#include <mochi_core/geometry/aabb.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/dynamic_string.h>
#include <mochi_core/utils/quaternion.h>

#include <optional>

/** Representation conversions shared by generated Unreal types; no unit scaling. */
namespace MochiGenSupport {

template <typename T>
TArray<T> ToUnrealArray(mochi::DynamicArray<T> const& Source) {
  if (Source.size() > static_cast<size_t>(MAX_int32)) {
    UE_LOG(LogTemp, Fatal, TEXT("Mochi array is too large for TArray."));
    return {};
  }
  return TArray<T>(Source.data(), static_cast<int32>(Source.size()));
}

template <typename T>
mochi::DynamicArray<T> ToMochiArray(TArray<T> const& Source) {
  mochi::DynamicArray<T> Result;
  if (Source.Num() != 0) {
    Result.assign(Source.GetData(), Source.GetData() + Source.Num());
  }
  return Result;
}

template <typename TVector>
mochi::Real3 ToMochiReal3(TVector const& Value) {
  return mochi::Real3{mochi::real(Value.X), mochi::real(Value.Y), mochi::real(Value.Z)};
}

template <typename TVector>
TVector ToUnrealReal3(mochi::Real3 const& Value) {
  return TVector(Value[0], Value[1], Value[2]);
}

inline mochi::Quaternion ToMochiQuaternion(FQuat4f const& Value) {
  return mochi::Quaternion(
      mochi::real(Value.X), mochi::real(Value.Y), mochi::real(Value.Z), mochi::real(Value.W));
}

inline FQuat4f ToUnrealQuaternion(mochi::Quaternion const& Value) {
  const mochi::Real4 Components = Value.ToReal4();
  return FQuat4f(Components[0], Components[1], Components[2], Components[3]);
}

template <typename TUnreal, typename TNativeArray, typename TConvert>
TArray<TUnreal> ToUnrealArray(TNativeArray const& Source, TConvert Convert) {
  TArray<TUnreal> Result;
  if (Source.size() > static_cast<size_t>(MAX_int32)) {
    UE_LOG(LogTemp, Fatal, TEXT("Mochi array is too large for TArray."));
    return Result;
  }
  Result.Reserve(static_cast<int32>(Source.size()));
  for (auto const& Value : Source) {
    Result.Add(Convert(Value));
  }
  return Result;
}

template <typename TNative, typename TUnreal, typename TConvert>
mochi::DynamicArray<TNative> ToMochiArray(TArray<TUnreal> const& Source, TConvert Convert) {
  mochi::DynamicArray<TNative> Result;
  Result.reserve(static_cast<size_t>(Source.Num()));
  for (const TUnreal& Value : Source) {
    Result.push_back(Convert(Value));
  }
  return Result;
}

inline mochi::DynamicString ToMochiString(FString const& Value) {
  const FTCHARToUTF8 Converted(*Value, Value.Len());
  return mochi::DynamicString(Converted.Get(), static_cast<size_t>(Converted.Length()));
}

inline FString ToUnrealString(mochi::DynamicString const& Value) {
  if (Value.size() > static_cast<size_t>(MAX_int32)) {
    UE_LOG(LogTemp, Error, TEXT("Mochi string is too large for FString."));
    return {};
  }
  const FUTF8ToTCHAR Converted(Value.data(), static_cast<int32>(Value.size()));
  return FString(Converted.Length(), Converted.Get());
}

template <typename TNative, typename TUnreal, typename TConvert>
std::optional<TNative> ToMochiOptional(TOptional<TUnreal> const& Source, TConvert Convert) {
  std::optional<TNative> Result{};
  if (Source.IsSet()) {
    Result = Convert(Source.GetValue());
  }
  return Result;
}

template <typename TUnreal, typename TNative, typename TConvert>
TOptional<TUnreal> ToUnrealOptional(std::optional<TNative> const& Source, TConvert Convert) {
  TOptional<TUnreal> Result{};
  if (Source.has_value()) {
    Result = Convert(*Source);
  }
  return Result;
}

inline mochi::Aabb ToMochiAabb(FBox3f const& Value) {
  return mochi::Aabb(ToMochiReal3(Value.Min), ToMochiReal3(Value.Max));
}

inline FBox3f ToUnrealAabb(mochi::Aabb const& Value) {
  return FBox3f(ToUnrealReal3<FVector3f>(Value.GetMin()), ToUnrealReal3<FVector3f>(Value.GetMax()));
}

} // namespace MochiGenSupport

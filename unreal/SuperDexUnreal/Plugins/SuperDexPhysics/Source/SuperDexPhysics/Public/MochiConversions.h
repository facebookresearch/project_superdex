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

#include <mochi_core/mochi_config.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/span.h>
#include <mochi_core/utils/vmatrix.h>

#include <mochi_core/utils/quaternion.h>
#include <mochi_core/utils/transform_rt.h>
#include <mochi_core/utils/transform_srt.h>

#include <MochiPhysicsSettings.h>
#include <MochiSpaceConverter.h>
#include <MochiTypes.h>

using mochi::operator""_r;

//------------------------------------------------------------------------------
// CONTAINER CONVERSIONS
//------------------------------------------------------------------------------

template <typename T>
mochi::Span<T const> ToMochiSpan(TArray<T> const& values) {
  return {values.GetData(), static_cast<size_t>(values.Num())};
}

template <typename ContainerT>
inline MOCHI_ANY auto ToMochiSpan(ContainerT& arr) {
  auto* ptr = arr.GetData();
  size_t len = (size_t)arr.Num();
  return mochi::Span<std::remove_reference_t<decltype(*ptr)>>{ptr, len};
}

inline mochi::Span<char const> ToMochiBinarySpan(TArray<uint8> const& values) {
  return {(char*)(values.GetData()), static_cast<size_t>(values.Num())};
}

template <typename T>
TArray<T> FromMochiSpan(mochi::Span<T const> Span) {
  TArray<T> array;
  array.SetNum(Span.size());
  auto srcBegin = Span.data();
  auto srcEnd = srcBegin + Span.size();
  auto dstBegin = array.GetData();
  std::copy(srcBegin, srcEnd, dstBegin);
  return array;
}

//------------------------------------------------------------------------------
// SPACE CONVERSIONS
//------------------------------------------------------------------------------

template <typename T>
struct TMochiToUnrealConverter : protected TMochiSpaceConverter<T> {
  using TMochiSpaceConverter<T>::FlipsHandedness;
  using TMochiSpaceConverter<T>::InputSpace;
  using TMochiSpaceConverter<T>::OutputSpace;

  using TVector = typename TMochiSpaceConverter<T>::TVector;
  using TQuat = typename TMochiSpaceConverter<T>::TQuat;
  using TTransform = typename TMochiSpaceConverter<T>::TTransform;
  using TMatrix3 = typename TMochiSpaceConverter<T>::TMatrix3;

  FORCEINLINE TMochiToUnrealConverter()
      : TMochiSpaceConverter<T>(
            UMochiPhysicsSettings::GetActiveSimulationSpace(),
            kUnrealCentimetersSpace) {}

  FORCEINLINE explicit TMochiToUnrealConverter(
      const FMochiSpaceDefinition& InInputSpace,
      const FMochiSpaceDefinition& InOutputSpace = kUnrealCentimetersSpace)
      : TMochiSpaceConverter<T>(InInputSpace, InOutputSpace) {}

  FORCEINLINE T MeasureToUnreal(mochi::real S, int N = 1) const {
    return this->TMochiSpaceConverter<T>::MeasureToOutput(S, N);
  }

  FORCEINLINE TVector TranslationToUnreal(const mochi::Real3& V) const {
    return this->TMochiSpaceConverter<T>::TranslationToOutput(
        TVector{static_cast<T>(V[0]), static_cast<T>(V[1]), static_cast<T>(V[2])});
  }

  FORCEINLINE TVector DirectionToUnreal(const mochi::Real3& V) const {
    return this->TMochiSpaceConverter<T>::DirectionToOutput(
        TVector{static_cast<T>(V[0]), static_cast<T>(V[1]), static_cast<T>(V[2])});
  }

  FORCEINLINE TQuat RotationToUnreal(const mochi::Quaternion& Q) const {
    const mochi::Real4 v = Q.ToReal4();
    return this->TMochiSpaceConverter<T>::RotationToOutput(TQuat{v[0], v[1], v[2], v[3]});
  }

  FORCEINLINE TTransform TransformToUnreal(const mochi::TransformRT& Tt) const {
    const TQuat r = RotationToUnreal(Tt.GetRotation());
    const TVector p = TranslationToUnreal(Tt.GetTranslation());
    return TTransform(r, p);
  }

  FORCEINLINE TTransform TransformToUnreal(const mochi::TransformSRT& Tt) const {
    const TQuat r = RotationToUnreal(Tt.GetRotation());
    const TVector p = TranslationToUnreal(Tt.GetTranslation());
    const auto s = Tt.GetScale();
    return TTransform(r, p, TVector(s));
  }

  FORCEINLINE FMochiMomentOfInertia InertiaToUnreal(const mochi::Real6& I) const {
    TMatrix3 i(
        TVector(I[0], I[1], I[2]), TVector(I[1], I[3], I[4]), TVector(I[2], I[4], I[5]), true);
    i = TMochiSpaceConverter<T>::InertiaToOutput(i);
    FMochiMomentOfInertia moi;
    moi.Ixx = i(0, 0), moi.Ixy = i(0, 1), moi.Ixz = i(0, 2), moi.Iyy = i(1, 1), moi.Iyz = i(1, 2),
    moi.Izz = i(2, 2);
    return moi;
  }
};

template <typename T>
struct TUnrealToMochiConverter : protected TMochiSpaceConverter<T> {
  using TMochiSpaceConverter<T>::FlipsHandedness;
  using TMochiSpaceConverter<T>::InputSpace;
  using TMochiSpaceConverter<T>::OutputSpace;
  using TMochiSpaceConverter<T>::ScaleToOutput;

  using TVector = typename TMochiSpaceConverter<T>::TVector;
  using TQuat = typename TMochiSpaceConverter<T>::TQuat;
  using TTransform = typename TMochiSpaceConverter<T>::TTransform;
  using TMatrix3 = typename TMochiSpaceConverter<T>::TMatrix3;

  using mr = mochi::real;

  FORCEINLINE TUnrealToMochiConverter()
      : TMochiSpaceConverter<T>(
            kUnrealCentimetersSpace,
            UMochiPhysicsSettings::GetActiveSimulationSpace()) {}

  FORCEINLINE explicit TUnrealToMochiConverter(
      const FMochiSpaceDefinition& InOutputSpace,
      const FMochiSpaceDefinition& InInputSpace = kUnrealCentimetersSpace)
      : TMochiSpaceConverter<T>(InInputSpace, InOutputSpace) {}

  FORCEINLINE mochi::real MeasureToMochi(T S, int N = 1) const {
    return this->TMochiSpaceConverter<T>::MeasureToOutput(static_cast<mr>(S), N);
  }

  FORCEINLINE mochi::Real3 TranslationToMochi(const TVector& V) const {
    const TVector v = TMochiSpaceConverter<T>::TranslationToOutput(V);
    return mochi::Real3(static_cast<mr>(v.X), static_cast<mr>(v.Y), static_cast<mr>(v.Z));
  }

  FORCEINLINE mochi::Real3 DirectionToMochi(const TVector& V) const {
    const TVector v = TMochiSpaceConverter<T>::DirectionToOutput(V);
    return mochi::Real3(static_cast<mr>(v.X), static_cast<mr>(v.Y), static_cast<mr>(v.Z));
  }

  FORCEINLINE mochi::Quaternion RotationToMochi(const TQuat& Q) const {
    const TQuat q = TMochiSpaceConverter<T>::RotationToOutput(Q);
    return mochi::Quaternion(
        static_cast<mr>(q.X), static_cast<mr>(q.Y), static_cast<mr>(q.Z), static_cast<mr>(q.W));
  }

  FORCEINLINE mochi::TransformRT TransformToMochi_RT(const TTransform& Tt) const {
    return mochi::TransformRT(
        RotationToMochi(Tt.GetRotation()), TranslationToMochi(Tt.GetTranslation()));
  }

  FORCEINLINE mochi::TransformSRT TransformToMochi_SRT(const TTransform& Tt) const {
    const TVector s = Tt.GetScale3D();
    // Mochi does not support non-uniform scale
    ensure(s.X == s.Y && s.X == s.Z);
    return mochi::TransformSRT(s.X, TransformToMochi_RT(Tt));
  }

  FORCEINLINE mochi::Real6 InertiaToMochi(const FMochiMomentOfInertia& I) const {
    TMatrix3 i(
        TVector(I.Ixx, I.Ixy, I.Ixz),
        TVector(I.Ixy, I.Iyy, I.Iyz),
        TVector(I.Ixz, I.Iyz, I.Izz),
        true);
    i = TMochiSpaceConverter<T>::InertiaToOutput(i);
    return mochi::Real6{
        static_cast<mr>(i(0, 0)),
        static_cast<mr>(i(0, 1)),
        static_cast<mr>(i(0, 2)),
        static_cast<mr>(i(1, 1)),
        static_cast<mr>(i(1, 2)),
        static_cast<mr>(i(2, 2))};
  }

  FORCEINLINE mochi::VMatrix4x4r GetBasisChangeMatrix() const {
    const auto& M = TMochiSpaceConverter<T>::InputToOutputMatrix;
    return mochi::VMatrix4x4r{
        mochi::Vec4r{
            static_cast<mr>(M(0, 0)), static_cast<mr>(M(0, 1)), static_cast<mr>(M(0, 2)), 0},
        mochi::Vec4r{
            static_cast<mr>(M(1, 0)), static_cast<mr>(M(1, 1)), static_cast<mr>(M(1, 2)), 0},
        mochi::Vec4r{
            static_cast<mr>(M(2, 0)), static_cast<mr>(M(2, 1)), static_cast<mr>(M(2, 2)), 0},
        mochi::Vec4r{0, 0, 0, 1}};
  }
};

typedef TMochiToUnrealConverter<double> FMochiToUnrealConverter;
typedef TUnrealToMochiConverter<double> FUnrealToMochiConverter;
typedef TMochiToUnrealConverter<float> FMochiToUnrealConverter_float;
typedef TUnrealToMochiConverter<float> FUnrealToMochiConverter_float;

//------------------------------------------------------------------------------
// BESPOKE TYPE CONVERSIONS
//------------------------------------------------------------------------------

FORCEINLINE FColor ColorToUnreal(mochi::Color C) {
  return FColor(C[0], C[1], C[2], C[3]);
}

FORCEINLINE void ContactPointToUnreal(
    const mochi::ContactPoint& MochiContact,
    mochi::Scene* Scene,
    FName NestedName,
    const FMochiToUnrealConverter& M2U,
    FMochiContactPoint& OutUnrealContact) {
  const auto* mochiActorA = Scene->GetActor(MochiContact.actorA);
  const auto* mochiActorB = Scene->GetActor(MochiContact.actorB);
  USceneComponent* mochiCompA = reinterpret_cast<USceneComponent*>(mochiActorA->GetUserData());
  USceneComponent* mochiCompB = reinterpret_cast<USceneComponent*>(mochiActorB->GetUserData());
  OutUnrealContact.InternalMochiActor = FName(mochiActorA->GetName());
  OutUnrealContact.OtherInternalMochiActor = FName(mochiActorB->GetName());
  OutUnrealContact.MochiComponent = mochiCompA;
  OutUnrealContact.OtherMochiComponent = mochiCompB;
  OutUnrealContact.NestedName = NestedName;
  OutUnrealContact.Location = M2U.TranslationToUnreal(MochiContact.posA);
  OutUnrealContact.OtherLocation = M2U.TranslationToUnreal(MochiContact.posB);
  OutUnrealContact.Velocity = M2U.TranslationToUnreal(MochiContact.pointVelocityA);
  OutUnrealContact.OtherVelocity = M2U.TranslationToUnreal(MochiContact.pointVelocityB);
  OutUnrealContact.Normal = M2U.DirectionToUnreal(MochiContact.normal);
  OutUnrealContact.Force = M2U.DirectionToUnreal(MochiContact.force);
  OutUnrealContact.Distance = M2U.MeasureToUnreal(MochiContact.distance);
  OutUnrealContact.TriangleIndex = MochiContact.elementIndex;
  // TODO: When Mochi mirrors a mesh collider (negative scale on an odd number of axes), it
  // reverses the triangle winding order. Reverse the parametric coordinates here too, so they
  // match the triangle as the UE mesh defines it.
  const auto& pc = MochiContact.parametricCoords;
  OutUnrealContact.ParametricCoords = FVector{pc[0], pc[1], pc[2]};
  OutUnrealContact.Area = M2U.MeasureToUnreal(MochiContact.intWeight, 2);
}

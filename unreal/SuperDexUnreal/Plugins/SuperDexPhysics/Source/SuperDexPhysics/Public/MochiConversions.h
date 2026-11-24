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
mochi::Span<T const> ToMochiSpan(TArray<T> const& Values) {
  return {Values.GetData(), static_cast<size_t>(Values.Num())};
}

template <typename ContainerT>
inline MOCHI_ANY auto ToMochiSpan(ContainerT& Arr) {
  auto* Ptr = Arr.GetData();
  size_t Len = (size_t)Arr.Num();
  return mochi::Span<std::remove_reference_t<decltype(*Ptr)>>{Ptr, Len};
}

inline mochi::Span<char const> ToMochiBinarySpan(TArray<uint8> const& Values) {
  return {(char*)(Values.GetData()), static_cast<size_t>(Values.Num())};
}

template <typename T>
TArray<T> FromMochiSpan(mochi::Span<T const> Span) {
  TArray<T> Array;
  Array.SetNum(Span.size());
  const T* SrcBegin = Span.data();
  const T* SrcEnd = SrcBegin + Span.size();
  T* DstBegin = Array.GetData();
  std::copy(SrcBegin, SrcEnd, DstBegin);
  return Array;
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
    const mochi::Real4 V = Q.ToReal4();
    return this->TMochiSpaceConverter<T>::RotationToOutput(TQuat{V[0], V[1], V[2], V[3]});
  }

  FORCEINLINE TTransform TransformToUnreal(const mochi::TransformRT& Tt) const {
    const TQuat R = RotationToUnreal(Tt.GetRotation());
    const TVector P = TranslationToUnreal(Tt.GetTranslation());
    return TTransform(R, P);
  }

  FORCEINLINE TTransform TransformToUnreal(const mochi::TransformSRT& Tt) const {
    const TQuat R = RotationToUnreal(Tt.GetRotation());
    const TVector P = TranslationToUnreal(Tt.GetTranslation());
    const mochi::real S = Tt.GetScale();
    return TTransform(R, P, TVector(S));
  }

  FORCEINLINE FMochiMomentOfInertia InertiaToUnreal(const mochi::Real6& I) const {
    TMatrix3 InertiaMatrix(
        TVector(I[0], I[1], I[2]), TVector(I[1], I[3], I[4]), TVector(I[2], I[4], I[5]), true);
    InertiaMatrix = TMochiSpaceConverter<T>::InertiaToOutput(InertiaMatrix);
    FMochiMomentOfInertia Moi;
    Moi.Ixx = InertiaMatrix(0, 0), Moi.Ixy = InertiaMatrix(0, 1), Moi.Ixz = InertiaMatrix(0, 2),
    Moi.Iyy = InertiaMatrix(1, 1), Moi.Iyz = InertiaMatrix(1, 2), Moi.Izz = InertiaMatrix(2, 2);
    return Moi;
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

  using Mr = mochi::real;

  FORCEINLINE TUnrealToMochiConverter()
      : TMochiSpaceConverter<T>(
            kUnrealCentimetersSpace,
            UMochiPhysicsSettings::GetActiveSimulationSpace()) {}

  FORCEINLINE explicit TUnrealToMochiConverter(
      const FMochiSpaceDefinition& InOutputSpace,
      const FMochiSpaceDefinition& InInputSpace = kUnrealCentimetersSpace)
      : TMochiSpaceConverter<T>(InInputSpace, InOutputSpace) {}

  FORCEINLINE mochi::real MeasureToMochi(T S, int N = 1) const {
    return this->TMochiSpaceConverter<T>::MeasureToOutput(static_cast<Mr>(S), N);
  }

  FORCEINLINE mochi::Real3 TranslationToMochi(const TVector& V) const {
    const TVector TranslationVector = TMochiSpaceConverter<T>::TranslationToOutput(V);
    return mochi::Real3(
        static_cast<Mr>(TranslationVector.X),
        static_cast<Mr>(TranslationVector.Y),
        static_cast<Mr>(TranslationVector.Z));
  }

  FORCEINLINE mochi::Real3 DirectionToMochi(const TVector& V) const {
    const TVector DirectionVector = TMochiSpaceConverter<T>::DirectionToOutput(V);
    return mochi::Real3(
        static_cast<Mr>(DirectionVector.X),
        static_cast<Mr>(DirectionVector.Y),
        static_cast<Mr>(DirectionVector.Z));
  }

  FORCEINLINE mochi::Quaternion RotationToMochi(const TQuat& Q) const {
    const TQuat OutputRotation = TMochiSpaceConverter<T>::RotationToOutput(Q);
    return mochi::Quaternion(
        static_cast<Mr>(OutputRotation.X),
        static_cast<Mr>(OutputRotation.Y),
        static_cast<Mr>(OutputRotation.Z),
        static_cast<Mr>(OutputRotation.W));
  }

  FORCEINLINE mochi::TransformRT TransformToMochi_RT(const TTransform& Tt) const {
    return mochi::TransformRT(
        RotationToMochi(Tt.GetRotation()), TranslationToMochi(Tt.GetTranslation()));
  }

  FORCEINLINE mochi::TransformSRT TransformToMochi_SRT(const TTransform& Tt) const {
    const TVector S = Tt.GetScale3D();
    // Mochi does not support non-uniform scale
    ensure(S.X == S.Y && S.X == S.Z);
    return mochi::TransformSRT(S.X, TransformToMochi_RT(Tt));
  }

  FORCEINLINE mochi::Real6 InertiaToMochi(const FMochiMomentOfInertia& I) const {
    TMatrix3 InertiaMatrix(
        TVector(I.Ixx, I.Ixy, I.Ixz),
        TVector(I.Ixy, I.Iyy, I.Iyz),
        TVector(I.Ixz, I.Iyz, I.Izz),
        true);
    InertiaMatrix = TMochiSpaceConverter<T>::InertiaToOutput(InertiaMatrix);
    return mochi::Real6{
        static_cast<Mr>(InertiaMatrix(0, 0)),
        static_cast<Mr>(InertiaMatrix(0, 1)),
        static_cast<Mr>(InertiaMatrix(0, 2)),
        static_cast<Mr>(InertiaMatrix(1, 1)),
        static_cast<Mr>(InertiaMatrix(1, 2)),
        static_cast<Mr>(InertiaMatrix(2, 2))};
  }

  FORCEINLINE mochi::VMatrix4x4r GetBasisChangeMatrix() const {
    const TMatrix3& M = TMochiSpaceConverter<T>::InputToOutputMatrix;
    return mochi::VMatrix4x4r{
        mochi::Vec4r{
            static_cast<Mr>(M(0, 0)), static_cast<Mr>(M(0, 1)), static_cast<Mr>(M(0, 2)), 0},
        mochi::Vec4r{
            static_cast<Mr>(M(1, 0)), static_cast<Mr>(M(1, 1)), static_cast<Mr>(M(1, 2)), 0},
        mochi::Vec4r{
            static_cast<Mr>(M(2, 0)), static_cast<Mr>(M(2, 1)), static_cast<Mr>(M(2, 2)), 0},
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
  const mochi::Actor* MochiActorA = Scene->GetActor(MochiContact.actorA);
  const mochi::Actor* MochiActorB = Scene->GetActor(MochiContact.actorB);
  USceneComponent* MochiCompA = reinterpret_cast<USceneComponent*>(MochiActorA->GetUserData());
  USceneComponent* MochiCompB = reinterpret_cast<USceneComponent*>(MochiActorB->GetUserData());
  OutUnrealContact.InternalMochiActor = FName(MochiActorA->GetName());
  OutUnrealContact.OtherInternalMochiActor = FName(MochiActorB->GetName());
  OutUnrealContact.MochiComponent = MochiCompA;
  OutUnrealContact.OtherMochiComponent = MochiCompB;
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
  const mochi::Real3& Pc = MochiContact.parametricCoords;
  OutUnrealContact.ParametricCoords = FVector{Pc[0], Pc[1], Pc[2]};
  OutUnrealContact.Area = M2U.MeasureToUnreal(MochiContact.intWeight, 2);
}

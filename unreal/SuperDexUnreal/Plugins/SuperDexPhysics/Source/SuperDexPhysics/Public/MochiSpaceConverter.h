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
#include "MatrixTypes.h"

#include "MochiTypes.h"

#include <mochi_core/utils/coordinate_space.h>

//------------------------------------------------------------------------------
// BASE SPACE CONVERTER
//------------------------------------------------------------------------------

template <typename T>
struct TMochiSpaceConverter {
 public:
  typedef UE::Geometry::TMatrix3<T> TMatrix3;
  typedef UE::Math::TMatrix<T> TMatrix4;
  typedef UE::Math::TVector<T> TVector;
  typedef UE::Math::TQuat<T> TQuat;
  typedef UE::Math::TPlane<T> TPlane;
  typedef UE::Math::TTransform<T> TTransform;

 public:
  FORCEINLINE TMochiSpaceConverter() : TMochiSpaceConverter(kUnrealCentimetersSpace, kMochiSpace) {}

  TMochiSpaceConverter(
      const FMochiSpaceDefinition& InInputSpace,
      const FMochiSpaceDefinition& InOutputSpace) {
    InputSpace = InInputSpace;
    OutputSpace = InOutputSpace;
    const TMatrix3 input = BuildBasis(InputSpace);
    const TMatrix3 inputInv = input.Transpose(); // orthonormal -> transpose=inverse
    const TMatrix3 output = BuildBasis(OutputSpace);
    InputToOutputMatrix = output * inputInv;
    InputToOutputScale = OutputSpace.UnitsPerMeter / InputSpace.UnitsPerMeter;
    bFlipsHandedness = InputToOutputMatrix.Determinant() < 0;
  }

  // Returns the inverse converter.
  FORCEINLINE TMochiSpaceConverter Inverse() const {
    return TMochiSpaceConverter(OutputSpace, InputSpace);
  }

  // True if the converter flips handedness (i.e. input and output handedness not equal.)
  FORCEINLINE bool FlipsHandedness() const {
    return bFlipsHandedness;
  }

  // Converts a scalar measure to output. Use N=1 for distance, N=2 for area, N=3 for volume,
  // etc.
  FORCEINLINE T MeasureToOutput(T S, int N = 1) const {
    for (int i = 0; i < N; ++i) {
      S *= InputToOutputScale;
    }
    return S;
  }

  // Converts a position, point, translation, etc. from the input to output space (i.e. applies
  // scaling.)
  FORCEINLINE TVector TranslationToOutput(const TVector& V) const {
    return InputToOutputMatrix * V * InputToOutputScale;
  }

  // Converts a direction, vector, normal, etc. from the input to output space (i.e. no scaling.)
  FORCEINLINE TVector DirectionToOutput(const TVector& V) const {
    return InputToOutputMatrix * V;
  }

  // Converts a rotation from the input to output space.
  FORCEINLINE TQuat RotationToOutput(const TQuat& Q) const {
    TVector qAxis = bFlipsHandedness ? TVector(-Q.X, -Q.Y, -Q.Z) : TVector(Q.X, Q.Y, Q.Z);
    TVector qAxisOut = DirectionToOutput(qAxis);
    return TQuat(qAxisOut.X, qAxisOut.Y, qAxisOut.Z, Q.W);
  }

  // Converts a transform from the input to output space. The input transform scale is preserved.
  FORCEINLINE TTransform TransformToOutput(const TTransform& X) const {
    return TTransform(
        RotationToOutput(X.GetRotation()), TranslationToOutput(X.GetTranslation()), X.GetScale3D());
  }

  // Converts a scale vector from input space to output space.
  // Unlike DirectionToOutput which applies the full transformation matrix (including sign
  // changes from axis flips), this function only permutes components based on which input
  // axis maps to which output axis.
  FORCEINLINE TVector ScaleToOutput(const TVector& S) const {
    const TMatrix3 InputToOutputMatrixAbs(
        InputToOutputMatrix.Row0.GetAbs(),
        InputToOutputMatrix.Row1.GetAbs(),
        InputToOutputMatrix.Row2.GetAbs(),
        true);
    return InputToOutputMatrixAbs * S;
  }

  // Converts an inertia tensor from the input to output space.
  FORCEINLINE TMatrix3 InertiaToOutput(const TMatrix3& I) const {
    return InputToOutputMatrix * I * InputToOutputMatrix.Transpose() * InputToOutputScale *
        InputToOutputScale;
  }

 public:
  // Builds the FRU basis from Mochi's authoritative coordinate-space implementation.
  FORCEINLINE static TMatrix3 BuildBasis(const FMochiSpaceDefinition& SpaceDef) {
    const mochi::CoordinateSpace Space = ToMochiCoordinateSpace(SpaceDef);
    auto ToVector = [](const mochi::Real3& V) {
      return TVector(static_cast<T>(V[0]), static_cast<T>(V[1]), static_cast<T>(V[2]));
    };
    return TMatrix3(
        ToVector(Space.GetForward()),
        ToVector(Space.GetRight()),
        ToVector(Space.GetUp()),
        false); // bRows = false
  }

  // Utility to convert a 3x3 matrix to a 4x4 matrix.
  FORCEINLINE static TMatrix4 ToMatrix4(const TMatrix3& M) {
    return TMatrix4(
        TPlane(M(0, 0), M(0, 1), M(0, 2), 0), // row 0
        TPlane(M(1, 0), M(1, 1), M(1, 2), 0), // row 1
        TPlane(M(2, 0), M(2, 1), M(2, 2), 0), // row 2
        TPlane(0, 0, 0, 1)); // row 3
  }

 public:
  FMochiSpaceDefinition InputSpace;
  FMochiSpaceDefinition OutputSpace;
  TMatrix3 InputToOutputMatrix;
  T InputToOutputScale = 0.0f;
  bool bFlipsHandedness = false;
};

typedef TMochiSpaceConverter<double> FMochiSpaceConverter;

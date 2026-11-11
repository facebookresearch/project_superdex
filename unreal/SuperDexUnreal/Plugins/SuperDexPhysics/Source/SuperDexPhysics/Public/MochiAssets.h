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
#include "MochiTypes.h"
#include "UObject/Object.h"

#include "MochiAssets.generated.h"

//------------------------------------------------------------------------------
// MOCHI MATERIAL PARAMS
//------------------------------------------------------------------------------

// Parameters that control the material of a soft actor.
UCLASS(BlueprintType, HideCategories = Object, DisplayName = "Mochi Material Params")
class SUPERDEXPHYSICS_API UMochiMaterialParams : public UObject {
  GENERATED_BODY()

 public:
  // Material type.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Mochi")
  EMochiSoftMaterialType MaterialType;

  // Material Young's modulus
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      DisplayName = "Young's Modulus",
      meta =
          (ForceUnits = "MPa",
           EditCondition =
               "MaterialType != EMochiSoftMaterialType::Arap && MaterialType != EMochiSoftMaterialType::ActiveShapeTargetingArap",
           EditConditionHides),
      Category = "Mochi")
  float YoungsModulus;

  // Material poisson ratio.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta =
          (UIMin = 0.0,
           UIMax = 0.49999,
           EditCondition =
               "MaterialType != EMochiSoftMaterialType::Arap && MaterialType != EMochiSoftMaterialType::ActiveShapeTargetingArap",
           EditConditionHides),
      Category = "Mochi")
  float PoissonRatio;

  // Density in the undeformed configuration.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ForceUnits = "kg/m3"), Category = "Mochi")
  float Density;

  // Mass damping coefficient.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (DisplayName = "Mass Damping Coefficient (1/s)"),
      Category = "Shell Material")
  float MassDampingCoefficient;

  // Stiffness damping coefficient.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (ForceUnits = "s"),
      Category = "Shell Material")
  float StiffnessDampingCoefficient;

  // Strategy for keeping the material's Hessian positive semi-definite.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      Category = "Mochi",
      AdvancedDisplay,
      meta =
          (EditCondition = "MaterialType != EMochiSoftMaterialType::LinearElastic",
           EditConditionHides))
  EMochiMaterialPsdStrategy PsdStrategy;

  // ARAP stiffness. Young's modulus and Poisson's ratio do not apply to these materials.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta =
          (ForceUnits = "Pa",
           ClampMin = 0,
           EditCondition =
               "MaterialType == EMochiSoftMaterialType::Arap || MaterialType == EMochiSoftMaterialType::ActiveShapeTargetingArap",
           EditConditionHides),
      Category = "ARAP Material")
  float ArapStiffness;

  // Local shape the material is driven towards, as an offset from the identity.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta =
          (EditCondition = "MaterialType == EMochiSoftMaterialType::ActiveShapeTargetingArap",
           EditConditionHides),
      Category = "ARAP Material")
  FMochiShapeTargetTensor ShapeTargetTensor;

  // Anisotropic stiffness along the fiber direction.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta =
          (ForceUnits = "Pa",
           ClampMin = 0,
           EditCondition = "MaterialType == EMochiSoftMaterialType::ActiveNeoHookean",
           EditConditionHides),
      Category = "Active Material")
  float AnisoAlpha;

  // Anisotropic reference length (dimensionless). Drives activation along the fiber.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta =
          (ClampMin = 0,
           EditCondition = "MaterialType == EMochiSoftMaterialType::ActiveNeoHookean",
           EditConditionHides),
      Category = "Active Material")
  float AnisoLength;

  // Fiber direction, as a unit vector in Mochi space (RH, Y up) rather than Unreal space.
  // The model geometry it applies to is stored in that space too, and this struct has no
  // access to the simulation space conversion that shape creation performs.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta =
          (EditCondition = "MaterialType == EMochiSoftMaterialType::ActiveNeoHookean",
           EditConditionHides),
      Category = "Active Material")
  FVector3f AnisoDirection;

 public:
  UMochiMaterialParams();
  // Convert to native Mochi type.
  mochi::SoftMaterialParams ToMochi() const;
  // Convert from native Mochi type.
  void FromMochi(const mochi::SoftMaterialParams& Mochi);
};

//------------------------------------------------------------------------------
// MOCHI SHELL MATERIAL PARAMS
//------------------------------------------------------------------------------

// Parameters that control the material of a shell actor.
UCLASS(BlueprintType, HideCategories = Object, DisplayName = "Mochi Shell Material Params")
class SUPERDEXPHYSICS_API UMochiShellMaterialParams : public UObject {
  GENERATED_BODY()

 public:
#if WITH_EDITORONLY_DATA
  // Young's modulus of the 3D solid material.
  UPROPERTY(EditAnywhere, meta = (ForceUnits = "Pa"), Category = "Estimate from 3D Parameters")
  float YoungModulus3D = 100000.0f;

  // Poisson's ratio of the 3D solid material.
  UPROPERTY(EditAnywhere, Category = "Estimate from 3D Parameters")
  float PoissonsRatio3D = 0.3f;

  // Volumetric density of the 3D solid material.
  UPROPERTY(EditAnywhere, meta = (ForceUnits = "kg/m^3"), Category = "Estimate from 3D Parameters")
  float Density3D = 1000.0f;

  // Shell thickness.
  UPROPERTY(EditAnywhere, meta = (ForceUnits = "m"), Category = "Estimate from 3D Parameters")
  float Thickness = 0.001f;
#endif

  // First membrane Lamé parameter (force per unit length).
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (ForceUnits = "Pa*m"),
      Category = "Shell Material")
  float MembraneLambda;

  // Second membrane Lamé parameter (shear modulus, force per unit length).
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (ForceUnits = "Pa*m"),
      Category = "Shell Material")
  float MembraneMu;

  // First bending stiffness parameter.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (ForceUnits = "Pa*m^3"),
      Category = "Shell Material")
  float BendingAlpha;

  // Second bending stiffness parameter.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (ForceUnits = "Pa*m^3"),
      Category = "Shell Material")
  float BendingBeta;

  // Area density (mass per unit area).
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (ForceUnits = "kg/m^2"),
      Category = "Shell Material")
  float Density;

  // Mass damping coefficient.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (DisplayName = "Mass Damping Coefficient (1/s)"),
      Category = "Shell Material")
  float MassDampingCoefficient;

  // Stiffness damping coefficient.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (ForceUnits = "s"),
      Category = "Shell Material")
  float StiffnessDampingCoefficient;

  // Dimensionless scale of the membrane collapse barrier, which resists element collapse under
  // large in-plane compression. Zero disables it.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = 0), Category = "Shell Material")
  float MembraneCollapseBarrierScale;

 public:
  UMochiShellMaterialParams();
  // Convert to native Mochi type.
  mochi::experimental::ShellMaterialParams ToMochi() const;
  // Convert from native Mochi type.
  void FromMochi(const mochi::experimental::ShellMaterialParams& Mochi);

#if WITH_EDITOR
  UFUNCTION(
      CallInEditor,
      Category = "Estimate from 3D Parameters",
      DisplayName = "Apply Estimate to Shell Material")
  void Apply3DParametersTo2D();
#endif

 private:
#if WITH_EDITOR
  bool TryDeriveFrom3DMaterial(mochi::experimental::ShellMaterialParams& Result) const;
#endif
  void SetElasticAndDensityParameters(const mochi::experimental::ShellMaterialParams& Mochi);
};

//------------------------------------------------------------------------------
// MOCHI ROD MATERIAL PARAMS
//------------------------------------------------------------------------------

// Parameters that control the material of a rod actor.
UCLASS(BlueprintType, HideCategories = Object, DisplayName = "Mochi Rod Material Params")
class SUPERDEXPHYSICS_API UMochiRodMaterialParams : public UObject {
  GENERATED_BODY()

 public:
  // Mass per unit undeformed length [kg/m].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ForceUnits = "kg/m"), Category = "Mochi")
  float LinearDensity;

  // Rotational inertia about the rod axis, per unit undeformed length [kg*m].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ForceUnits = "kg*m"), Category = "Mochi")
  float LinearRotationalInertia;

  // Axial stiffness (Young's modulus x cross-section area) [N].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ForceUnits = "N"), Category = "Mochi")
  float AxialStiffness;

  // Torsional stiffness (shear modulus x torsion constant) [N*m^2].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ForceUnits = "N*m^2"), Category = "Mochi")
  float TorsionalStiffness;

  // Flexural stiffness about two principal axes (E*I1, E*I2) [N*m^2].
  UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ForceUnits = "N*m^2"), Category = "Mochi")
  FVector2D FlexuralStiffness;

  // Mass damping coefficient.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadWrite,
      meta = (DisplayName = "Mass Damping Coefficient (1/s)"),
      Category = "Mochi")
  float MassDampingCoefficient;

  // Stiffness damping coefficient.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ForceUnits = "s"), Category = "Mochi")
  float StiffnessDampingCoefficient;

  // Dimensionless scale of the axial collapse barrier, which resists element collapse under large
  // axial compression. Zero disables it.
  UPROPERTY(EditAnywhere, BlueprintReadWrite, meta = (ClampMin = 0), Category = "Mochi")
  float AxialCollapseBarrierScale;

 public:
  UMochiRodMaterialParams();
  // Convert to native Mochi type.
  mochi::experimental::RodMaterialParams ToMochi() const;
  // Convert from native Mochi type.
  void FromMochi(const mochi::experimental::RodMaterialParams& Mochi);
};

//------------------------------------------------------------------------------
// MOCHI CONTACT PARAMS
//------------------------------------------------------------------------------

UCLASS(BlueprintType, HideCategories = Object, DisplayName = "Mochi Contact Params")
class SUPERDEXPHYSICS_API UMochiContactParams : public UObject {
  GENERATED_BODY()

 public:
  // Stiffness of the contact penalty force, in pressure per unit penetration distance.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Contact",
      meta = (DisplayName = "Penalty Coefficient (MPa/m)", ForceUnits = "MPa"))
  float PenaltyCoefficient;

  // Default detection threshold, if no extra padding is added. Determines the distance to the
  // object at which contact forces will begin to be applied.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Contact", meta = (Units = "m"))
  float PenaltyThresholdDist;

  // Extra padding for penalty detection. This is helpful when this collider actor is thin and the
  // colliding actor has no collider (i.e. contact cannot be sync).
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Contact", meta = (Units = "m"))
  float PenaltyExtraPadding;

  // [Experimental] Padding added to the collider's detection threshold when this actor is the
  // colliding actor. This is helpful to avoid tunneling of thin or coarsely sampled colliding
  // actors. Point-cloud colliders and SDF colliders of soft actors ignore it.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Contact",
      AdvancedDisplay,
      meta = (Units = "m"))
  float CollidingPenaltyThresholdPadding;

  // PolyReLU falloff: Half-distance over which the ReLU function is smoothed to allow
  // differentiation. Should typically be set to the same as PenaltyThresholdDist.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Contact", meta = (Units = "m"))
  float PenaltySmoothingHalfDistance;

  // True = Use the collider's normal (SDF gradient) for friction, False = use the normal at the
  // collision sample.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Friction", AdvancedDisplay)
  bool bFrictionWithColliderNormal;

  // Maximum alignment of colliding normals (0 means 90 deg, -1 means perfectly opposing, 1 means
  // all normals are accepted). Nominally contact normals are opposing; this determines the
  // threshold of disagreement (alignment) at which contacts are rejected for friction computation.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Friction",
      AdvancedDisplay,
      meta = (UIMin = -1.0, UIMax = 1.0))
  float MaxAlignmentNormals;

  // Viscous friction coefficient.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Friction")
  float ViscousFrictionCoefficient;

  // Coulomb friction coefficient.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Friction")
  float CoulombFrictionCoefficient;

  // Velocity threshold for Coulomb friction.
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Friction", meta = (Units = "m/s"))
  float FrictionFalloffVel;

  // [Experimental] Tangential displacement from the friction anchor over which elasto-plastic
  // friction reaches the Coulomb limit. Used instead of FrictionFalloffVel with elasto-plastic
  // friction (see FMochiExperimentalEvalParams::bElastoPlasticFriction). Must be finite and
  // strictly positive.
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Friction",
      AdvancedDisplay,
      meta = (Units = "m"))
  float PreslidingDisplacement;

  // Normal viscous damping coefficient [s/m].
  UPROPERTY(
      EditAnywhere,
      BlueprintReadOnly,
      Category = "Friction",
      meta = (DisplayName = "Normal Viscous Damping Coefficient (s/m)"))
  float NormalViscousDampingCoefficient;

  // Safety distance necessary for approximate SDFs (e.g., deep flow map).
  UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Deep Flow", AdvancedDisplay)
  float DistanceErrorBound;

 public:
  UMochiContactParams();
  // Convert to native Mochi type.
  mochi::ContactParams ToMochi() const;
  // Convert from native Mochi type.
  void FromMochi(const mochi::ContactParams& Mochi);
};

//------------------------------------------------------------------------------
// MOCHI LINEAR SOLVER PARAMS
//------------------------------------------------------------------------------

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiLinearSolverParams {
  GENERATED_BODY()
  // Type of linear solver to use.
  UPROPERTY(EditAnywhere, Category = "Linear Solver")
  EMochiLinearSolverType KspType;
  // Type of preconditioner to use.
  UPROPERTY(EditAnywhere, Category = "Linear Solver")
  EMochiPreconditionerType PcType;
  // Norm for stop criteria.
  UPROPERTY(EditAnywhere, Category = "Linear Solver")
  EMochiLinearSolverConvergenceNorm NormType;
  // Absolute tolerance. A negative value lets Mochi select it.
  UPROPERTY(EditAnywhere, Category = "Linear Solver")
  float ATol;
  // Relative tolerance. Ignored when using Eisenstat-Walker strategies no. 1 and 2 in the
  // non-linear solver.
  UPROPERTY(EditAnywhere, Category = "Linear Solver")
  float RTol;
  // Maximum number of iterations.
  UPROPERTY(EditAnywhere, Category = "Linear Solver")
  int MaxIter;

  FMochiLinearSolverParams();
  // Convert to native Mochi type.
  mochi::LinearSolverParams ToMochi() const;
  // Convert from native Mochi type.
  void FromMochi(const mochi::LinearSolverParams& Mochi);
};

//------------------------------------------------------------------------------
// MOCHI NON LINEAR SOLVER PARAMS
//------------------------------------------------------------------------------

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiNonLinearSolverParams {
  GENERATED_BODY()
  // Solver type
  UPROPERTY(EditAnywhere, Category = "Nonlinear Solver")
  EMochiNonLinearSolverType SolverType;

  // Maximum number of iterations.
  UPROPERTY(EditAnywhere, Category = "Nonlinear Solver")
  int MaxIter;
  // Absolute tolerance for non-linear convergence criterion.
  UPROPERTY(EditAnywhere, Category = "Nonlinear Solver")
  float AbsTol;
  // Relative tolerance for non-linear convergence criterion.
  UPROPERTY(EditAnywhere, Category = "Nonlinear Solver")
  float RelTol;

  // PSD projection mode (def: Always).
  UPROPERTY(EditAnywhere, Category = "Nonlinear Solver")
  int PsdProjMode;

  // Heuristically prevents explosions.
  UPROPERTY(EditAnywhere, Category = "Nonlinear Solver")
  bool bExplosionControl;
  // Absolute divergence tolerance.
  UPROPERTY(EditAnywhere, Category = "Nonlinear Solver")
  float AbsDivTol;
  // Relative (wrt initial iters Res) divergence tolerance.
  UPROPERTY(EditAnywhere, Category = "Nonlinear Solver")
  float RelDivTol;

  // Maximum number of line search iterations.
  UPROPERTY(EditAnywhere, Category = "Nonlinear Solver")
  int LsMaxIter;
  // Step reduction factor for line search.
  UPROPERTY(EditAnywhere, Category = "Nonlinear Solver")
  float LsAlpha;
  // Type of line-search to use.
  UPROPERTY(EditAnywhere, Category = "Nonlinear Solver")
  EMochiLsType LsType;

  // Verbosity level for solver output.
  UPROPERTY(EditAnywhere, Category = "Nonlinear Solver")
  EMochiVerbosityLevel Verbosity;

  FMochiNonLinearSolverParams();
  // Convert to native Mochi type.
  mochi::NonLinearSolverParams ToMochi() const;
  // Convert from native Mochi type.
  void FromMochi(const mochi::NonLinearSolverParams& Mochi);
};

//------------------------------------------------------------------------------
// MOCHI SATURATION HESSIAN PARAMS
//------------------------------------------------------------------------------

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiSaturationHessianParams {
  GENERATED_BODY()
  // Use fitted quadratic potential Hessian for contact friction saturation.
  UPROPERTY(EditAnywhere, Category = "Saturation Hessian")
  bool bContactFriction;
  // Use fitted quadratic potential Hessian for joint friction saturation.
  UPROPERTY(EditAnywhere, Category = "Saturation Hessian")
  bool bJointFriction;
  // Use fitted quadratic potential Hessian for constraint saturation.
  UPROPERTY(EditAnywhere, Category = "Saturation Hessian")
  bool bConstraintSaturation;

  FMochiSaturationHessianParams();
  mochi::SaturationHessianParams ToMochi() const;
  void FromMochi(const mochi::SaturationHessianParams& Mochi);
};

//------------------------------------------------------------------------------
// MOCHI EXPERIMENTAL EVAL PARAMS
//------------------------------------------------------------------------------

USTRUCT(BlueprintType)
struct SUPERDEXPHYSICS_API FMochiExperimentalEvalParams {
  GENERATED_BODY()
  // Controls whether force-saturation terms use a fitted quadratic Hessian per pathway.
  UPROPERTY(EditAnywhere, Category = "Experimental Eval")
  FMochiSaturationHessianParams FittedSaturationHessian;

  // Treat the normal contact force implicitly (if true) or explicitly (if false) for
  // dissipative contact terms.
  UPROPERTY(EditAnywhere, Category = "Experimental Eval")
  bool bImplicitNormalForceForDissipation = false;

  // Use elasto-plastic Coulomb friction, which holds a tangential load below the Coulomb limit
  // without slip (see UMochiContactParams::PreslidingDisplacement). Requires an integration method
  // whose step ends at its last stage; every EMochiIntegrationMethod currently qualifies.
  UPROPERTY(EditAnywhere, Category = "Experimental Eval")
  bool bElastoPlasticFriction = false;

  FMochiExperimentalEvalParams();
  // Convert to native Mochi type.
  mochi::ExperimentalEvalParams ToMochi() const;
  // Convert from native Mochi type.
  void FromMochi(const mochi::ExperimentalEvalParams& Mochi);
};

//------------------------------------------------------------------------------
// MOCHI SOLVER PARAMS
//------------------------------------------------------------------------------

UCLASS(BlueprintType, HideCategories = Object, DisplayName = "Mochi Solver Params")
class SUPERDEXPHYSICS_API UMochiSolverParams : public UObject {
  GENERATED_BODY()
 public:
  UPROPERTY(EditAnywhere, Category = "Mochi")
  FMochiNonLinearSolverParams NonLinearSolverParams;
  UPROPERTY(EditAnywhere, Category = "Mochi")
  FMochiLinearSolverParams LinearSolverParams;
  UPROPERTY(EditAnywhere, Category = "Mochi")
  EMochiIntegrationMethod IntegrationMethod;
  UPROPERTY(EditAnywhere, Category = "Mochi")
  FMochiExperimentalEvalParams ExperimentalEvalParams;

 public:
  UMochiSolverParams();
  // Convert to native Mochi type.
  mochi::SolverParams ToMochi() const;
  // Convert from native Mochi type.
  void FromMochi(const mochi::SolverParams& Mochi);
};

//------------------------------------------------------------------------------
// MOCHI SOLVER PARAMS
//------------------------------------------------------------------------------

UCLASS(BlueprintType, HideCategories = Object, DisplayName = "Mochi Contact Matrix")
class SUPERDEXPHYSICS_API UMochiContactMatrix : public UObject {
  GENERATED_BODY()
 public:
  UPROPERTY(EditAnywhere, meta = (ShowOnlyInnerProperties), Category = "Mochi")
  TArray<FMochiContactMatrixEntry> Matrix;
};

//------------------------------------------------------------------------------
// MOCHI MODEL MATERIAL MODIFIER
//------------------------------------------------------------------------------

// This asset can be used to modify the material properties of a mochi model.
// In its current implementation, this is done at the element level.
// In future versions, this process should be constructive using volumes (CSG)
// or defined as SDFs or something smarter.
UCLASS(BlueprintType, HideCategories = Object, DisplayName = "Mochi Model Material Modifier")
class SUPERDEXPHYSICS_API UMochiModelMaterialModifiers : public UObject {
  GENERATED_BODY()
 public:
  // Map of <ElementNumber, MaterialProperties>
  UPROPERTY(EditAnywhere, Category = "Mochi")
  TMap<int, FMochiMaterialModifier> MaterialModifiers;
};

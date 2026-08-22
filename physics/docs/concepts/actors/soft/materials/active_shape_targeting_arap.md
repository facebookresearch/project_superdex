---
title: Active Shape Targeting ARAP
sidebar_position: 8
---

# Active Shape Targeting ARAP

Active Shape Targeting ARAP extends [ARAP](./arap.md) with a user-controlled local shape target. It follows the shape-targeting model of [Klár et al. (2020)](#references) and can represent contraction, expansion, or shear while allowing rigid rotation.

**Enum value:** `SoftMaterialType::ActiveShapeTargetingArap`

## Formulation

The six [`shapeTargetTensor`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ActiveShapeTargetingArapMaterialParams.html) values define the symmetric target tensor

$$
\mathbf{S}_t
= \mathbf{1}
+ \begin{bmatrix}
  s_0 & s_1 & s_2 \\
  s_1 & s_3 & s_4 \\
  s_2 & s_4 & s_5
\end{bmatrix}~.
$$

SuperDex Physics forms the modified deformation gradient

$$
\mathbf{F}_t = \mathbf{F}\mathbf{S}_t
$$

and computes its rotation-variant SVD, $\mathbf{F}_t = \mathbf{U}_t\boldsymbol{\Sigma}_t\mathbf{V}_t^T$. The closest proper rotation is

$$
\mathbf{R}_t = \mathbf{U}_t\mathbf{V}_t^T~.
$$

The strain energy density and first Piola–Kirchhoff stress are

$$
\Psi(\mathbf{F})
= \frac{\mu}{2}\|\mathbf{F} - \mathbf{R}_t\mathbf{S}_t\|^2~,
\qquad
\mathbf{P}
= \mu(\mathbf{F} - \mathbf{R}_t\mathbf{S}_t)~.
$$

When all target parameters are zero, $\mathbf{S}_t=\mathbf{1}$ and the model reduces to standard [ARAP](./arap.md).

## Parameters

The public parameter type is [`ActiveShapeTargetingArapMaterialParams`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ActiveShapeTargetingArapMaterialParams.html).

| Parameter | Default | Units | Description |
|---|---|---|---|
| `stiffness` | 1,000 | Pa | Resistance to deviation from the target shape |
| `shapeTargetTensor` | `[0, 0, 0, 0, 0, 0]` | -- | Values $[s_0,s_1,s_2,s_3,s_4,s_5]$ defining $\mathbf{S}_t$ |
| `psdStrategy` | `Projection` | -- | [PSD enforcement strategy](./overview.md#psd-enforcement) |

Supported concrete PSD strategies: `None`, `Projection`, `PerTermProjection`, `AbsEigenProjection`. `MaterialDefault` resolves to the model's default, `Projection`. `PerTermProjection` is specific to this material and projects the tangent summands independently.

## Controlling Actuation

Actuation is application-driven: supply a nonzero [`shapeTargetTensor`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ActiveShapeTargetingArapMaterialParams.html) at creation time, or vary it at runtime to change the desired local contraction, expansion, or shear. To update a homogeneous target, call [`Actor::GetSoftMaterialParams()`](pathname:///generated/api/v1.0/cpp/classsuperdex_1_1Actor.html), modify `activeShapeTargetingArap.shapeTargetTensor`, and pass the result to [`Actor::SetSoftMaterialParams()`](pathname:///generated/api/v1.0/cpp/classsuperdex_1_1Actor.html). For actors with [per-element material data](../overview.mdx#softmaterialparams), use `experimental::GetSoftMaterialParamsField()` and `experimental::SetSoftMaterialParamsField()` instead.

## When to Use

- Muscle contraction simulation.
- Soft robotics with programmable local shape targets.
- Any scenario requiring active internal actuation with a controllable target tensor.

## References

- G. Klár et al., [Shape Targeting: A Versatile Active Elasticity Constitutive Model](https://par.nsf.gov/servlets/purl/10230451), 2020.

---
title: ARAP
sidebar_position: 6
---

# As-Rigid-As-Possible (ARAP)

ARAP penalizes deformation that differs from a rigid rotation, preserving local shape rather than matching a conventional isotropic elastic law. SuperDex Physics follows the rotation-variant SVD treatment described by [Kim and Eberle (2022)](#references).

**Enum value:** `SoftMaterialType::Arap`

## Formulation

SuperDex Physics computes a rotation-variant singular value decomposition

$$
\mathbf{F} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T~,
$$

in which $\mathbf{U}$ and $\mathbf{V}$ are proper rotations, so an inversion appears as a negative final signed principal stretch. The matrix

$$
\mathbf{R} = \mathbf{U}\mathbf{V}^T
$$

is therefore the closest proper rotation to $\mathbf{F}$, including for inverted elements. Let $\boldsymbol{\sigma} = (\sigma_1, \sigma_2, \sigma_3)$ contain the diagonal entries of $\boldsymbol{\Sigma}$ and let $\mathbf{1}_3 = (1, 1, 1)$. The strain energy density is

$$
\Psi(\mathbf{F})
= \frac{\mu}{2}\Vert\boldsymbol{\sigma} - \mathbf{1}_3\Vert_2^2
= \frac{\mu}{2}\|\mathbf{F} - \mathbf{R}\|^2~,
$$

and the first Piola–Kirchhoff stress is

$$
\mathbf{P} = \mu(\mathbf{F} - \mathbf{R})~.
$$

Compression and inversion can make the material tangent indefinite. The public [PSD strategies](./overview.md#psd-enforcement) regularize this tangent when requested by the nonlinear solver.

## Parameters

The public parameter type is [`ArapMaterialParams`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ArapMaterialParams.html).

| Parameter | Default | Description |
|---|---|---|
| `stiffness` | 1,000 Pa | Resistance to non-rigid deformation |
| `psdStrategy` | `Projection` | [PSD enforcement strategy](./overview.md#psd-enforcement) |

The default [`stiffness`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ArapMaterialParams.html) is numerically 100 times smaller than the default Young's modulus $E=100{,}000$ Pa used by the isotropic models. These parameters belong to different constitutive models and are not directly equivalent.

Supported concrete PSD strategies: `None`, `Projection`, `AbsEigenProjection`. `MaterialDefault` resolves to the model's default, `Projection`.

## When to Use

- Shape-preserving deformation where local rigidity matters more than conventional material calibration.
- As the passive basis for [Active Shape Targeting ARAP](./active_shape_targeting_arap.md).

## References

- T. Kim and D. Eberle, [Dynamic Deformables](https://www.tkim.graphics/DYNAMIC_DEFORMABLES/DynamicDeformables.pdf), 2022.

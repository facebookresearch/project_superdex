---
title: Saint Venant–Kirchhoff
sidebar_position: 5
---

# Saint Venant–Kirchhoff (StVK)

Saint Venant–Kirchhoff extends linear elasticity to finite deformation using the Green–Lagrange strain. The constitutive relation is linear in that nonlinear strain measure, making the model frame-indifferent but unreliable under large compression.

**Enum value:** `SoftMaterialType::StVenantKirchhoff`

## Formulation

The Green–Lagrange strain is

$$
\mathbf{E} = \frac{1}{2}\left(\mathbf{F}^T\mathbf{F} - \mathbf{1}\right)~.
$$

The strain energy density is

$$
\Psi(\mathbf{F})
= \mu\|\mathbf{E}\|^2
+ \frac{\lambda}{2}\left(\operatorname{tr}\mathbf{E}\right)^2~,
$$

and the first Piola–Kirchhoff stress is

$$
\mathbf{P}
= \mathbf{F}\left[2\mu\mathbf{E}
+ \lambda\left(\operatorname{tr}\mathbf{E}\right)\mathbf{1}\right]~.
$$

## Parameters

The public parameter type is [`StVenantKirchhoffMaterialParams`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1StVenantKirchhoffMaterialParams.html).

| Parameter | Default | Description |
|---|---|---|
| `youngsModulus` | 100,000 Pa | Stiffness |
| `poissonRatio` | 0.45 | Compressibility |
| `psdStrategy` | `Projection` | [PSD enforcement strategy](./overview.md#psd-enforcement) |

Supported concrete PSD strategies: `None`, `Projection`, `Fast`, `AbsEigenProjection`. `MaterialDefault` resolves to the model's default, `Projection`.

:::warning
For auxetic materials ($\nu < 0$), `Fast` does not guarantee a positive-semidefinite tangent. Use `Projection` or `AbsEigenProjection` when that guarantee is required.
:::

## When to Use

- Problems with large rigid rotations but only moderate elastic strain.
- Educational or comparative work requiring a simple frame-indifferent hyperelastic model.

:::caution
StVK has no infinite compression barrier. Its energy remains finite at $\mathbf{F}=\mathbf{0}$, and the first Piola–Kirchhoff stress vanishes there, so complete collapse is a pathological stationary state. For robust large-deformation simulation, use [Neo-Hookean](./neo_hookean.md) instead.
:::

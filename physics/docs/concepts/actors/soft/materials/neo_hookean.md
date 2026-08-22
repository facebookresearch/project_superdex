---
title: Neo-Hookean
sidebar_position: 3
---

# Neo-Hookean

The Neo-Hookean model is the **recommended default** for general-purpose deformable simulation. SuperDex Physics uses the inversion-robust formulation of [Smith et al. (2018)](#references).

**Enum value:** `SoftMaterialType::NeoHookean`

## Formulation

Following [Smith et al. (2018)](#references), SuperDex Physics implements the strain energy density

$$
\Psi(\mathbf{F})
= \frac{\widehat{\mu}}{2}(I_C - 3)
+ \frac{\widehat{\lambda}}{2}(J - \alpha)^2
- \frac{\widehat{\mu}}{2}\ln(I_C + 1)~,
$$

where

$$
I_C = \operatorname{tr}(\mathbf{F}^T\mathbf{F})~, \qquad
J = \det(\mathbf{F})~,
$$

and, in terms of the [standard Lamé parameters](./overview.md#common-parameters) $\mu$ and $\lambda$,

$$
\widehat{\mu} = \frac{4}{3}\mu~, \qquad
\widehat{\lambda} = \lambda + \frac{5}{6}\mu~, \qquad
\alpha = 1 + \frac{3\widehat{\mu}}{4\widehat{\lambda}}~.
$$

This reparameterization ensures that the model's small-strain response corresponds to linear elasticity with the standard Lamé parameters. The first Piola–Kirchhoff stress is

$$
\mathbf{P}
= \widehat{\mu}\left(1 - \frac{1}{I_C + 1}\right)\mathbf{F}
+ \widehat{\lambda}(J - \alpha)\operatorname{cof}(\mathbf{F})~.
$$

Because $I_C + 1$ is positive for every real deformation gradient, the logarithmic term remains defined through element inversion and regularizes the response as $I_C$ approaches zero. It is not a divergent barrier: the energy remains finite at complete collapse.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `youngsModulus` | 100,000 Pa | Stiffness |
| `poissonRatio` | 0.45 | Compressibility |
| `psdStrategy` | `Projection` | [PSD enforcement strategy](./overview.md#psd-enforcement) |

Supported concrete PSD strategies: `None`, `Projection`, `Fast`, `AbsEigenProjection`. `MaterialDefault` resolves to the model's default, `Projection`.

The public parameter type is `NeoHookeanMaterialParams`, an alias for [`SmithNeoHookeanMaterialParams`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1SmithNeoHookeanMaterialParams.html).

## When to Use

- **Recommended for most simulations.** The [Smith et al. (2018)](#references) formulation supports large deformation, extreme compression, and element inversion.
- It is somewhat more expensive than Linear Elastic, but substantially more appropriate for finite rotations and strains.

## References

- B. Smith, F. de Goes, and T. Kim, [Stable Neo-Hookean Flesh Simulation](https://doi.org/10.1145/3180491), 2018.

---
title: Active Neo-Hookean
sidebar_position: 7
---

# Active Neo-Hookean

Active Neo-Hookean is a composite material that combines passive Neo-Hookean elasticity with active anisotropic fiber contraction. It is intended for directionally actuated materials such as muscle tissue.

**Enum value:** `SoftMaterialType::ActiveNeoHookean`

## Formulation

The strain energy density is the sum of two components:

$$
\Psi(\mathbf{F})
= \Psi_{\mathrm{neo}}(\mathbf{F})
+ \Psi_{\mathrm{aniso}}(\mathbf{F})~.
$$

### Passive Isotropic Component

The [`passiveIsotropic`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ActiveNeoHookeanMaterialParams.html) component uses the inversion-robust [Neo-Hookean formulation](./neo_hookean.md) of [Smith et al. (2018)](#references).

### Active Anisotropic Component

The [`activeAnisotropic`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ActiveNeoHookeanMaterialParams.html) component follows [Kim et al. (2019)](#references) and penalizes deviation of the fiber stretch from a target length:

- `anisoDir` is the unit fiber direction.
- `alpha` is the anisotropic stiffness.
- `length` is the target fiber length; values below 1 produce contraction relative to the reference configuration.

The stress and tangent are sums of the passive and active component contributions.

## Parameters

| Parameter | Default | Units | Description |
|---|---|---|---|
| `passiveIsotropic.youngsModulus` | 100,000 | Pa | Passive isotropic stiffness |
| `passiveIsotropic.poissonRatio` | 0.45 | -- | Passive isotropic compressibility |
| `passiveIsotropic.psdStrategy` | `Projection` | -- | Passive component [PSD strategy](./overview.md#psd-enforcement) |
| `activeAnisotropic.alpha` | 1,000 | Pa | Fiber stiffness |
| `activeAnisotropic.length` | 1.0 | -- | Target fiber length |
| `activeAnisotropic.anisoDir` | `(1, 0, 0)` | -- | Unit fiber direction |
| `activeAnisotropic.psdStrategy` | `Projection` | -- | Active component [PSD strategy](./overview.md#psd-enforcement) |

PSD strategies are configured independently for the two subcomponents. The passive Neo-Hookean component supports the concrete strategies `None`, `Projection`, `Fast`, and `AbsEigenProjection`; the active anisotropic component supports `None`, `Projection`, and `AbsEigenProjection`. For either component, `MaterialDefault` resolves to `Projection`.

## When to Use

- Muscle tissue simulation where fibers contract along a preferred direction.
- Soft actuators with embedded directional stiffness.
- Any scenario requiring combined passive elasticity and active contraction.

## References

- T. Kim et al., [Anisotropic Elasticity for Inversion-Safety and Element Rehabilitation](http://tkim.graphics/ANISOTROPY/AnisotropyAndRehab.pdf), 2019.
- B. Smith, F. de Goes, and T. Kim, [Stable Neo-Hookean Flesh Simulation](https://doi.org/10.1145/3180491), 2018.

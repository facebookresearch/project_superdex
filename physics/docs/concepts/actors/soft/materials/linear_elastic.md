---
title: Linear Elastic
sidebar_position: 2
---

# Linear Elastic

Small-strain linear elasticity is the simplest and fastest material model, but has an extremely limited range of validity, requiring both small strains and small rotations of the material.
This model is widely used in engineering mechanics, for stress analysis of structural components with microscopically-small deformations (e.g., calculating the failure load of a concrete column).
However, in most human-scale manipulation scenarios, such objects are more likely to be modeled by rigid actors or static geometry.
This model is not recommended for use outside of artificial benchmarks or highly-constrained scenarios (e.g., a cushion fixed to a static piece of furniture by boundary conditions and subjected only to light loading).

**Enum value:** `SoftMaterialType::LinearElastic`

:::warning
Do not use this model when appreciable rotations or strains are expected. Use [Neo-Hookean](./neo_hookean.md) for general finite-deformation simulation.
:::

## Formulation

Let $\nabla\mathbf{u} = \mathbf{F} - \mathbf{1}$ be the displacement gradient. The infinitesimal strain is

$$
\boldsymbol{\varepsilon}
= \frac{1}{2}\left(\nabla\mathbf{u} + \nabla\mathbf{u}^T\right)~,
$$

which approximates the Green–Lagrange strain only when $\|\nabla\mathbf{u}\| \ll 1$. The strain energy density is

$$
\Psi(\mathbf{F})
= \mu \|\boldsymbol{\varepsilon}\|^2
+ \frac{\lambda}{2}\left(\operatorname{tr}\boldsymbol{\varepsilon}\right)^2~,
$$

and the first Piola–Kirchhoff stress is

$$
\mathbf{P}
= \lambda\left(\operatorname{tr}\boldsymbol{\varepsilon}\right)\mathbf{1}
+ 2\mu\boldsymbol{\varepsilon}~.
$$

The constant material tangent is

$$
\frac{\partial P_{ij}}{\partial F_{kl}}
= \lambda\,\delta_{ij}\delta_{kl}
+ \mu\left(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk}\right)~.
$$

This tangent is positive semidefinite, so [`LinearElasticMaterialParams`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1LinearElasticMaterialParams.html) has no material-level `psdStrategy` field.

The linear-elastic potential $\Psi$ cannot be expressed as a function of the nonlinear Green–Lagrange strain, but, motivated by the fact that $\boldsymbol{\varepsilon} \approx \mathbf{E}$ in the limit of $\Vert\nabla\mathbf{u}\Vert \ll 1$ where this model is appropriate, viscoelastic stiffness damping uses the above tangent as $\mathbb{C}_0$.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `youngsModulus` | 100,000 Pa | Stiffness |
| `poissonRatio` | 0.45 | Compressibility |

## Interaction With Recentering

By default, soft actors update the rigid local frame in which their displacements are defined, to prevent finite precision effects from polluting the strain. This is benign for most supported materials, since their energy densities are invariant under rigid transformations. However, the `LinearElastic` potential is not, because the linearized strain $\boldsymbol{\varepsilon}$ may be nonzero for rigid motions (unlike the nonlinear Green–Lagrange strain, which remains exactly zero). Recentering therefore alters the continuous problem being solved in an ad hoc way. To run a pure simulation of classical linear elasticity, disable recentering using [`Actor::SetRecenteringParams()`](pathname:///generated/api/v1.0/cpp/classsuperdex_1_1Actor.html) or set `ExperimentalSoftActorParams::useRecentering` to `false` during creation with `experimental::CreateSoftActor()`.

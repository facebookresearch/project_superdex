---
title: Overview
sidebar_position: 1
---

# Materials

SuperDex Physics supports several material models for soft body simulation. Each model is defined by a strain energy density $\Psi(\mathbf{F})$, where $\mathbf{F}$ is the deformation gradient. The first Piola–Kirchhoff stress $\mathbf{P} = \partial\Psi/\partial\mathbf{F}$ and material tangent $\partial\mathbf{P}/\partial\mathbf{F}$ follow from this energy.

## Common Parameters

Most isotropic material models use the following constitutive parameters:

| Parameter | Symbol | Units | Description |
|---|---|---|---|
| Young's modulus | $E$ | Pa | Overall stiffness. Higher values produce stiffer objects. Must be finite and positive. |
| Poisson's ratio | $\nu$ | -- | Compressibility. Must be finite and satisfy $-1 < \nu < 0.5$. |

[`SoftMaterialParams::density`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1SoftMaterialParams.html) is an actor-wide parameter shared by every material model, rather than a field of a model-specific `*MaterialParams` struct. It specifies mass per unit reference volume in kg/m$^3$ and defaults to 1000.

Young's modulus and Poisson's ratio define the Lamé parameters
$$
\lambda = \frac{E \nu}{(1 + \nu)(1 - 2\nu)}~, \qquad
\mu = \frac{E}{2(1 + \nu)}~,
$$
which are used in the definitions of several different material formulations.

:::warning Nearly incompressible materials
Notice that the first Lamé parameter, $\lambda$, diverges when $\nu$ approaches the ends of its allowed range.
The limit of $\nu \to -1$ is of little practical interest, but the limit of $\nu\to 0.5$ corresponds to near-incompressibility, and is common in soft materials like rubber and biological tissues.
The deformation of these materials occurs almost entirely in shearing modes, whose stiffness is controlled by $\mu$, also known as the shear modulus.
The stiffness of volume changes is controlled by the bulk modulus, $\kappa = \lambda + 2\mu/3$, which diverges alongside $\lambda$.
Approximating non-trivial shearing deformations on a discrete mesh often still requires some small volume changes of elements, which are penalized by $\kappa$.
This causes artificial stiffening known as volumetric locking in the finite element literature.
For general simulations of human-scale manipulation problems, it is recommended to avoid this limit and regularize handbook bulk modulus values of soft materials to keep $\nu\lesssim 0.49$.
:::

## Available Models

| Model | Use Case | Large Deformation | Inversion-Safe | Cost |
|---|---|---|---|---|
| [Linear Elastic](./linear_elastic.md) | Small displacement gradients only | No | N/A | Lowest |
| [Neo-Hookean](./neo_hookean.md) | General-purpose (recommended) | Yes | Yes | Medium |
| [Saint Venant–Kirchhoff](./stvk.md) | Large rotation, moderate strain | Yes | No | Medium |
| [ARAP](./arap.md) | Shape-preserving deformation | Yes | Yes | Medium |
| [Active Neo-Hookean](./active_stable_neo_hookean.md) | Muscle tissue with fiber actuation | Yes | Yes | High |
| [Active Shape Targeting ARAP](./active_shape_targeting_arap.md) | Programmable local shape targets | Yes | Yes | High |

For most simulations, **Neo-Hookean** is the recommended default.

## PSD Enforcement

The default choices of algebraic [Solvers](../../../solvers.md) used to solve implicit stages during time integration assume that energy contributions from soft materials have positive-semidefinite (PSD) Hessians.
The exact nonlinear material tangents can become indefinite and break this assumption.
The default behavior is therefore to replace them with PSD approximations, to improve solver performance.

A full-eigensystem `MaterialPsdStrategy::Projection` writes a symmetric material tangent as
$$
\frac{\partial\mathbf{P}}{\partial\mathbf{F}}
= \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T~,
$$
where the columns of $\mathbf{Q}$ are orthonormal eigenvectors and $\boldsymbol{\Lambda}$ is the diagonal matrix of eigenvalues, then replaces it with
$$
\mathbf{Q}\max(\boldsymbol{\Lambda}, \varepsilon)\mathbf{Q}^T~.
$$
Here $\varepsilon > 0$ is a small eigenvalue floor, and each diagonal entry $\lambda_i$ is replaced by $\max(\lambda_i, \varepsilon)$. Some materials use equivalent analytic projections of only their indefinite modes and may preserve structural zero modes.

PSD projection is controlled by two parameters:

- [`NonLinearSolverParams::psdProjMode`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1NonLinearSolverParams.html) decides when the nonlinear solve requests PSD projection. Its default is `PsdProjectionMode::Always`; retry modes request projection only after an iteration of [Newton's method](../../../solvers.md#newtons-method) fails, while `PsdProjectionMode::Never` never requests it.
- Each material's `MaterialPsdStrategy` decides how that material responds to a request. `MaterialPsdStrategy::None` ignores the request and leaves the material tangent unmodified.

The available material-level PSD strategies are summarized in the following table:

| Strategy | Description | Availability |
|---|---|---|
| `MaterialDefault` | Resolve to the selected material's concrete default strategy. | All nonlinear models |
| `Projection` | Project the tangent to be PSD, commonly by replacing $\lambda_i$ with $\max(\lambda_i, \varepsilon)$. | All nonlinear models |
| `Fast` | Drop or filter problematic terms. Faster, but may degrade nonlinear convergence. | StVK, Neo-Hookean |
| `AbsEigenProjection` | Use absolute eigenvalue filtering, commonly replacing $\lambda_i$ with $\max(\lvert\lambda_i\rvert, \varepsilon)$. | All nonlinear models |
| `PerTermProjection` | Project tangent summands independently. | Active Shape Targeting ARAP only |
| `None` | Do not modify the material tangent. | All nonlinear models |

The Linear Elastic model has no material-level `psdStrategy` field because its tangent is unconditionally PSD.

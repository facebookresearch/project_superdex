---
title: Contact
sidebar_position: 8
---

# Contact

SuperDex Physics uses a compliant contact model, which can be interpreted as either a penalty regularization of an inequality constraint or as a reduced model of local surface deformation.  This approach provides smooth, differentiable contact forces suitable for implicit time integration and optimization-based solvers, and provides spatial distributions of contact traction fields acting on the surfaces of bodies, rather than just point-contact force resultants.  Compliant contact allows for some interpenetration of colliding bodies, which is evaluated using signed distance fields (SDFs) and generalizations thereof defined on their geometries.

## Contact Roles

Every contact interaction involves two actors playing complementary roles:

1. **Collider** – The actor that provides a queryable **signed distance field (SDF)** used to determine penetration depth and contact normals.
2. **Colliding** – The actor whose surface is discretized with **quadrature sample points**. Contact traction is computed at these sample locations using the formulation detailed below by querying the collider's SDF.


The separation of roles allows SuperDex Physics to handle **asymmetric contact configurations**.  For example, a deformable body (colliding) can be pressed against a rigid obstacle (collider) without requiring the costlier evaluation of an SDF on deforming geometry.  However, a single actor can serve both roles simultaneously.  If two actors have colliders, and no [Contact Filtering](#contact-filtering) is applied in either direction, this results in two passes of contact force computation, whose results are summed.  However, even a single pass results in balanced forces applied to both bodies.

### Collider Representations

The `colliderType` / `collider_type` setting selects how an actor supplies distance fields. Its values are defined by the [`ColliderType`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ColliderType) enum:

| Setting | Representation | Notes |
|---|---|---|
| `Auto` | Actor- and shape-dependent | Selects analytic fields for supported shapes, `Sdf` for rigid mesh and soft actors, and `PointCloud` for shell and rod actors. |
| `None` | No collider | Disables only the collider role; the actor may still provide colliding samples. |
| `Sphere` | Analytic sphere SDF | Exact and inexpensive. |
| `Box` | Analytic box SDF | Exact for box geometry. |
| `Plane` | Analytic halfspace SDF | Infinite plane, commonly used for ground and boundaries. |
| `Mesh` | Triangle-mesh distance queries | Supports non-convex geometry, but is experimental, relatively slow, and limited to rigid actors and articulated links. |
| `Sdf` | Precomputed grid SDF | Supports complex geometry; approximates the exact SDF using trilinear interpolation, with a resolution-memory trade-off. |
| `PointCloud` | Spherical SDFs about material points | Quadrature discretization of the double-integral generalization below; interacts only with other point-cloud actors. |

Grid SDF construction is controlled by `GridSdfParams` ([C++](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1GridSdfParams.html), [Python](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.GridSdfParams)): [`resolutionMode`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1GridSdfParams.html) / [`resolution_mode`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.GridSdfParams.resolution_mode) selects the reference length used to size voxels, [`resolutionDelta`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1GridSdfParams.html) / [`resolution_delta`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.GridSdfParams.resolution_delta) scales that length per axis, [`minGridResolution`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1GridSdfParams.html) / [`min_grid_resolution`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.GridSdfParams.min_grid_resolution) sets the minimum voxel count per axis, and [`boundaryPaddingDist`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1GridSdfParams.html) / [`boundary_padding_dist`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.GridSdfParams.boundary_padding_dist) extends the grid beyond the shape bounds.

### Default Roles by Actor Type

| Actor Type | Colliding (surface samples) | Collider |
|---|---|---|
| Rigid (dynamic) | Yes | Yes (`colliderType = Auto`, which resolves to `Sdf` for mesh shapes) |
| Rigid (static) | No | Yes (`colliderType = Auto`, which resolves to `Sdf` for mesh shapes) |
| Soft | Yes | No (set `colliderType` to `Sdf` or `Auto` to enable a grid SDF mapped by the deformation; this is experimental and may be slow) |
| Articulated links | Yes (dynamic links only) | Yes by default; each link has its own [`colliderType`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ArticulatedLinkParams.html), which defaults to `Auto` |
| Shell | Yes | Yes (`colliderType = PointCloud` by default) |
| Rod | Yes | No (set `colliderType` to `PointCloud` or `Auto` to enable) |

## Contact Filtering

Filtering controls which ordered actor interactions are eligible. Each actor has an arbitrary string contact layer, assigned at creation or changed with [`Actor::SetContactLayer`](pathname:///generated/api/v1.0/cpp/classsuperdex_1_1Actor.html) ([`set_contact_layer`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Actor.set_contact_layer) in Python).

The C++ [`Scene`](pathname:///generated/api/v1.0/cpp/classsuperdex_1_1Scene.html) API and corresponding Python methods provide four controls:

- `Scene::EnableLayerContactAsymmetric(layerA, layerB, enable, error)` (`enable_layer_contact_asymmetric` in Python) controls the ordered interaction with `layerA` colliding against `layerB` as collider. It does not change the reverse direction.
- `Scene::EnableLayerContactSymmetric(layerA, layerB, enable, error)` (`enable_layer_contact_symmetric` in Python) applies the setting to both directions.
- `Scene::EnableActorContactAsymmetric(A, B, enable, includeNestedActors, error)` (`enable_actor_contact_asymmetric` in Python) controls an additional gate for the ordered actor pair.
- `Scene::EnableActorContactSymmetric(A, B, enable, includeNestedActors, error)` (`enable_actor_contact_symmetric` in Python) applies that actor-pair gate in both directions.

Both layer-level and actor-pair contact must be enabled for an interaction to occur; actor-pair settings cannot re-enable a layer-disabled interaction. Actor-pair filtering is useful when constrained actors overlap and would otherwise generate contact forces that oppose the constraint. Asymmetric filtering specifies which actor supplies samples and which supplies the collider field.

When `includeNestedActors` is `IncludeNestedActors::No`, these APIs affect only the exact handles passed. With `IncludeNestedActors::Yes`, a parent actor resolves to the parent plus its nested actors, and the setting is applied to every ordered pair in the cross-product of the two resolved handle sets. No pair outside that cross-product is affected. If the sets overlap, overlap pairs, including self-pairs, are affected. Python exposes the containing [`IncludeNestedActors`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.IncludeNestedActors) enum.

SuperDex Physics automatically disables contact between adjacent links when articulated or soft-skinned actors are created. A later actor-contact setting can override that automatic disable for any pair in the resolved sets. In particular, enabling contact between a parent and itself with `IncludeNestedActors::Yes` enables contact between its nested actors, including adjacent links, unless a later setting disables those pairs again.

## Formulation

We present the compliant contact formulation in the continuous setting first, before elaborating on the discretization used in simulations.

### Continuous Formulation

The continuous contact model combines a conservative normal penalty with dissipative friction and damping. The pairwise penalty energies contribute to the total conservative potential $U$ in the [system dynamics](./dynamics.md#system-dynamics), while the dissipative tractions derive from a state-dependent dissipation potential $R$.

#### Contact Kinematics

Let $\mathbf{x}_A(\mathbf{X}_A,t)$ denote the motion, or current position, of material point $\mathbf{X}_A$ on colliding actor $A$. Similarly, let $\mathbf{x}_B(\mathbf{X}_B,t)$ denote the collider motion and
$$
\mathbf{F}_B=\frac{\partial\mathbf{x}_B}{\partial\mathbf{X}_B}
$$
its deformation gradient. Collider $B$ supplies a (possibly approximate) SDF $\Phi_B(\mathbf{X}_B)$ in its reference coordinates and, where the motion is locally invertible, the current-to-reference map $\mathbf{X}_B(\mathbf{x},t)=\mathbf{x}_B^{-1}(\mathbf{x},t)$. Its spatial SDF is
$$
\phi_B(\mathbf{x},t)=\Phi_B\!\left(\mathbf{X}_B(\mathbf{x},t)\right)~.
$$
This construction covers the standard `Plane`, `Sphere`, `Box`, `Mesh`, and `Sdf` collider types. The nonlocal `PointCloud` model is the exception described under [double-integral generalization](#double-integral-generalization).

The trajectory of the sample on $A$, expressed in the reference coordinates of $B$, is
$$
\mathbf{Y}_{AB}(\mathbf{X}_A,t)
=\mathbf{X}_B\!\left(\mathbf{x}_A(\mathbf{X}_A,t),t\right)~.
$$
The collider-space relative contact velocity is its total derivative holding the colliding material point fixed:
$$
\mathbf{V}_{\mathrm{rel}}
=\left.\frac{d}{dt}\right|_{\mathbf{X}_A}\mathbf{Y}_{AB}(\mathbf{X}_A,t)
=\mathbf{F}_B^{-1}(\mathbf{v}_A - \mathbf{v}_B)~,
$$
where $\mathbf{v}_A=\left.\partial_t\mathbf{x}_A\right|_{\mathbf{X}_A}$ and $\mathbf{v}_B=\left.\partial_t\mathbf{x}_B\right|_{\mathbf{X}_B}$ are the world-space velocities of the spatially-coincident material points $\mathbf{X}_A$ and $\mathbf{X}_B$ on the two bodies.

For rigid motions, $\mathbf{F}_B$ is a rotation, and the collider motion $\mathbf{x}_B$ exactly maps an SDF $\Phi_B$ defined on the reference configuration to another SDF $\phi_B$ on the current configuration. However, there is usually still some approximation if the reference SDF is precomputed on a grid and interpolated.  For a soft actor with an `Sdf` collider, $\mathbf{X}_B$ is a non-rigid current-to-reference map. It may change lengths and angles, and it may be undefined at some query points, which are then rejected. The spatial field $\phi_B$ is generally not an exact SDF (even if $\Phi_B$ is), and the magnitude of $\mathbf{V}_{\mathrm{rel}}$ can be affected by local stretching of the collider.

#### Frictionless Normal Penalty

For colliding actor $A$ and collider $B$, the conservative contact potential is a surface integral over the reference boundary of $A$:
$$
\Pi_{A,B}(t) = \int_{\partial \Omega_A} W_{A,B}(\mathbf{X}_A,t) \, d\mathbf{X}_A~,
$$
where the energy density per unit reference area is
$$
W_{A,B}(\mathbf{X}_A,t) = \frac{k}{2} \, h_{\epsilon,\delta}\!\left(-\phi_B(\mathbf{x}_A(\mathbf{X}_A,t),t)\right)^2
=\frac{k}{2} \, h_{\epsilon,\delta}\!\left(-\Phi_B(\mathbf{Y}_{AB}(\mathbf{X}_A,t))\right)^2~.
$$
Here, $k > 0$ is the penalty coefficient. The ramp activation $h_{\epsilon,\delta}$ has smoothing half-width $\epsilon$ and contact threshold $\delta$. Summing $\Pi_{A,B}$ over active ordered actor pairs gives the contact contribution to $U$.

Let $z = -\phi_B(\mathbf{x}_A,t) - (\epsilon - \delta)$. SuperDex Physics uses the following $C^2$-continuous smoothed ramp:
$$
h_{\epsilon,\delta}\!\left(-\phi_B\right) =
\begin{cases}
0, & z \leq -\epsilon~, \\
\displaystyle \frac{3\epsilon}{16} + \left(\frac{1}{2} + \frac{3z}{8\epsilon} - \frac{z^3}{16\epsilon^3}\right)z,
& \lvert z\rvert < \epsilon~, \\
z, & z \geq \epsilon~.
\end{cases}
$$
The penalty is therefore zero when $\phi_B \geq \delta$, transitions smoothly over $\delta - 2\epsilon < \phi_B < \delta$, and is linear in penetration below that interval. A positive $\delta$ activates contact slightly outside the collider's zero level set.

#### Friction and Damping

##### Normal/Tangential Split

Dissipative contact tractions depend on the conservative normal response. The world-frame nominal normal traction on $A$, per unit reference area of $A$, is
$$
\mathbf{t}_n=-\frac{\partial W_{A,B}}{\partial\mathbf{x}_A}
=\mathbf{F}_B^{-T}\mathbf{T}_n~,
$$
where the uppercase $\mathbf{T}_n$ denotes the traction conjugate to the collider-reference position $\mathbf{Y}_{AB}$:
$$
\mathbf{T}_n=-\frac{\partial W_{A,B}}{\partial\mathbf{Y}_{AB}}
=\lambda_n\mathbf{G}_B~,
\qquad
\mathbf{G}_B=\nabla_{\mathbf{X}_B}\Phi_B(\mathbf{Y}_{AB})~.
$$
Here, $\mathbf{G}_B$ is the raw reference-field gradient and $\lambda_n\geq 0$ is its scalar coefficient in $\mathbf{T}_n$. Defining $\mathbf{N}_B=\mathbf{G}_B/\Vert\mathbf{G}_B\Vert_2$, we have $\lambda_n=\Vert\mathbf{T}_n\Vert_2/\Vert\mathbf{G}_B\Vert_2$. Grid interpolation can make $\mathbf{G}_B$ nonunit, so $\lambda_n$ is not generally the Euclidean magnitude of $\mathbf{T}_n$. For an exact reference SDF, $\Vert\mathbf{G}_B\Vert_2=1$ and $\lambda_n=\Vert\mathbf{T}_n\Vert_2$. The physical spatial gradient is likewise $\nabla_{\mathbf{x}}\phi_B=\mathbf{F}_B^{-T}\mathbf{G}_B$, which may be nonunit under a non-rigid pullback.

The relative velocity $\mathbf{V}_{\mathrm{rel}}$ from [contact kinematics](#contact-kinematics) is expressed in the same coordinates. Using the default collider-field direction $\mathbf{N}_B$, define
$$
V_n = \mathbf{V}_{\mathrm{rel}} \cdot \mathbf{N}_B~,
\qquad
\mathbf{V}_t = \left(\mathbf{1} - \mathbf{N}_B \otimes \mathbf{N}_B\right)\mathbf{V}_{\mathrm{rel}}~,
\qquad
V_t=\Vert\mathbf{V}_t\Vert_2~.
$$
The non-default setting `frictionWithColliderNormal = false` replaces $\mathbf{N}_B$ in the above with $-\mathbf{N}_A = -\mathbf{F}_B^{-1}\mathbf{n}_A$, where $\mathbf{n}_A$ is the colliding actor's outward-facing unit normal.
For exact contact between smooth surfaces, we expect $\mathbf{N}_A = -\mathbf{N}_B$.  However, for compliant contact with SDFs, this relation is only approximate.  Large deviations from this may lead to spurious contact tractions, so significantly misaligned contacts can be rejected entirely, and the dissipative response of accepted contacts can be attenuated as the rejection threshold is approached.  When the colliding-surface normal is available, define its alignment with the collider field as $s=\mathbf{G}_B\cdot\mathbf{N}_A$. A contact is rejected when $s>m$, where $m\in[-1,1]$ is the collider's maximum-normal-alignment parameter. For an accepted contact, the dissipative alignment factor is
$$
a=\frac{m-s}{m+1}
$$
when alignment fading is enabled and $m>-1$; otherwise, $a=1$. When both $\mathbf{G}_B$ and $\mathbf{N}_A$ are unit, $-1\leq s\leq m$ and therefore $a\in[0,1]$, fading dissipation from full strength for opposing normals ($s=-1$) to zero at the rejection threshold ($s=m$). The raw vectors are used, so grid interpolation or a non-rigid pullback can produce a factor above one. When the colliding normal is unavailable, as for shells and rods, alignment rejection and fading are disabled.

##### Traction Model

The dissipative traction contributions consist of Coulomb and viscous friction in the tangential direction, and separate normal viscous damping that affects restitution behavior in impacts. Coulomb friction is modeled using a viscous regularization of the static regime, similar to that of [Li et al. (2020)](#references), which allows the full dissipative response to be expressed in terms of the dissipation potential
$$
R=\widehat\lambda_n\left(\mu\rho(V_t)+\frac{c_t}{2}V_t^2+\frac{c_n}{2}V_n^2\right)~,
$$
where $\widehat\lambda_n=a\lambda_n$ is the effective dissipative normal-load coefficient, $\mu$ is the Coulomb friction coefficient, $c_t$ is the viscous friction coefficient, $c_n$ is the normal viscous damping coefficient, and $\rho$ is a regularized ramp function.  The default choice of regularization is
$$
\rho(V_t)=
\begin{cases}
\displaystyle \frac{V_t^2}{V_f}-\frac{V_t^3}{3V_f^2}, & 0\leq V_t<V_f~, \\
\displaystyle V_t-\frac{V_f}{3}, & V_t\geq V_f~,
\end{cases}
$$
in which $V_f > 0$ is the falloff velocity parameter determining the maximum slipping speed allowed within the nominally static friction regime.
With $\mathbf{T}_{\mathrm{diss}}=-\partial R/\partial\mathbf{V}_{\mathrm{rel}}$, the collider-reference contributions are
$$
\mathbf{T}_{v,n} = -c_n \widehat\lambda_n V_n \mathbf{N}_B~,
\qquad
\mathbf{T}_{v,t} = -c_t \widehat\lambda_n \mathbf{V}_t~,
$$
and
$$
\mathbf{T}_C =
\begin{cases}
\mathbf{0}, & V_t=0~, \\
\displaystyle -\mu \widehat\lambda_n \rho'(V_t)\frac{\mathbf{V}_t}{V_t}, & V_t>0~.
\end{cases}
$$
The total collider-reference traction and corresponding world-frame nominal traction on $A$ are
$$
\mathbf{T}_{\mathrm{diss}}
=\mathbf{T}_{v,n}+\mathbf{T}_{v,t}+\mathbf{T}_C~,
\qquad
\mathbf{t}_{\mathrm{diss}}
=\mathbf{F}_B^{-T}\mathbf{T}_{\mathrm{diss}}~.
$$
Although written simply as $R$, the dissipation potential depends on position state as well as relative velocity. Whether its stage-discrete force derives from an incremental potential therefore depends on how its position-dependence is evaluated, as discussed under [stage-discrete dissipation](#stage-discrete-dissipation).

#### Restitution Behavior of Rigid Collisions

The normal viscous damping corresponds to the impact model of [Hunt and Crossley (1975)](#references), applying a viscous force that is proportional to the elastic normal force from the penalty stiffness.
For rigid bodies colliding at a point aligned with their centers of mass, this leads to a velocity-dependent coefficient of restitution (CoR), $e$, defined as the ratio of post-impact speed $ev$ to pre-impact relative speed $v$. Hunt and Crossley's original analysis provided the formula $c_n \approx \frac{3}{2}(1 - e) / v$ in the nearly-elastic limit of $e\to 1$.
In the opposite limit of $e\to 0$, $ev \approx 1/c_n$, i.e., the rebound velocity is approximately the inverse of $c_n$.
The damping coefficient for general $e\in (0,1\rbrack$ is identified as the solution to an implicit relation by [Alaci et al. (2021)](#references), but it has no closed-form expression.
The rational approximation
$$
c_n \approx \frac{(1-e)\left(1 + \frac{9}{2}e\right)}{ev\left(1 + \frac{8}{3}e\right)}
$$
provides an explicit formula that holds to high accuracy and captures the two limits mentioned above. It is recommended for calibrating $c_n$ in practice, where approximation error from this formula will typically be smaller than discretization error in the simulation. This formula can also be inverted in closed form to estimate the effective CoR corresponding to a given damping coefficient and impact velocity.
For deformable bodies, restitution behavior is expected to depend mainly on the bulk elasticity and stiffness damping properties, not damping of the contact formulation.

This restitution behavior is a property of the continuous-time [dynamics](./dynamics.md#system-dynamics) of the formulation and is not guaranteed to be reproduced with discrete time-stepping. Observing it in practice typically requires at least second-order time integration (e.g., the C++ `IntegrationMethod::BDF2` or corresponding Python [`IntegrationMethod`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.IntegrationMethod) value). The time-step size must also be sufficiently small to resolve the contact event over several steps.

#### Contact Parameter Combination

For a contact pair, friction and damping coefficients are the geometric means of the actors' values. The penalty coefficient and friction falloff velocity are also geometric means when both actors are dynamic, but are taken from the colliding actor when the collider is static. Other contact parameters, including the smoothing distance and contact threshold, are taken from the collider.

#### Double-Integral Generalization

The `PointCloud` collider has no single field $\phi_B$. Instead, each material point $\mathbf{X}_B$ acts as a moving-sphere collider $B_{\mathbf{X}_B}$ with center $\mathbf{x}_B(\mathbf{X}_B,t)$, radius $L_B$, and SDF
$$
\phi_{\mathbf{X}_B}(\mathbf{x},t)
=\Vert\mathbf{x}-\mathbf{x}_B\Vert_2-L_B~,
$$
where arguments to $\mathbf{x}_B$ are dropped for brevity. Each sphere field follows only the translation of its interpolated center, with no pointwise deformation gradient, local material rotation, stretch, or rod twist.

The conservative contact potential of one such sphere is
$$
\Pi_{A,B_{\mathbf{X}_B}}(t)
= \int_{\partial\Omega_A} W_{A,B_{\mathbf{X}_B}}(\mathbf{X}_A,t) \, d\mathbf{X}_A~,
\qquad
W_{A,B_{\mathbf{X}_B}}(\mathbf{X}_A,t)
= \frac{k}{2} \, h_{\epsilon,\delta}\!\left(-\phi_{\mathbf{X}_B}(\mathbf{x}_A(\mathbf{X}_A,t),t)\right)^2~.
$$
The point-cloud potential integrates these single-collider potentials over the collider geometry:
$$
\Pi_{A,B}(t)
= \frac{1}{L_B^d}\int_{\Omega_B}\Pi_{A,B_{\mathbf{X}_B}}(t) \, d\mathbf{X}_B
= \frac{1}{L_B^d}\int_{\Omega_B}\int_{\partial\Omega_A}
W_{A,B_{\mathbf{X}_B}}(\mathbf{X}_A,t) \, d\mathbf{X}_A \, d\mathbf{X}_B~.
$$
Here, $L_B$ is the nonlocal interaction range and $d$ is the dimension of $\Omega_B$. The factor $L_B^{-d}$ compensates for the units of $d\mathbf{X}_B$, so the point-cloud potential has the same units as each $\Pi_{A,B_{\mathbf{X}_B}}$.

`PointCloud` colliders discretize the outer integral with quadrature over $B$. For each $\mathbf{X}_B$, the conservative normal traction from that moving-sphere collider supplies the normal-load coefficient used to compute its friction and damping. The factor $L_B^{-d}$ and the $B$-quadrature weight multiply this complete conservative and dissipative contribution before summation; dissipation is not computed from a normal traction already integrated over $B$.

The conservative normal force pair is central and therefore conserves angular momentum exactly. The tangential friction force pair instead has an $O(L_B)$ moment arm, so angular momentum is not conserved exactly at finite $L_B$. Under convergent refinement, this torque defect vanishes as the interaction range $L_B\to0$.

Point-cloud contact is evaluated only between actors that both use `PointCloud`. This is the only collider model that supports actor self-contact ($A=B$). For self-contact, admissible pairs additionally satisfy $\Vert\mathbf{X}_B-\mathbf{X}_A\Vert_2 \geq rL_B$ in the reference configuration, where $r > 1$ is a self-contact exclusion ratio parameter. See [`PointCloudColliderParams`](#pointcloudcolliderparams) for these settings.

### Discretization

#### Surface Quadrature

For the standard surface-contact case, the integral over $\partial\Omega_A$ is discretized using surface quadrature points $\mathbf{X}_{A,q}$ and reference-area weights $w_q$:
$$
\Pi_{A,B} \approx \sum_q w_q W_{A,B}(\mathbf{X}_{A,q})~.
$$
This discretization explains the asymmetric actor roles introduced above: actor $A$ must provide surface quadrature samples, while actor $B$ must provide a distance field that can be evaluated at their current positions. An actor may support either capability without supporting the other; for example, an infinite plane has an analytic SDF but no finite surface to sample. The asymmetry also makes [contact filtering](#contact-filtering) directional.

For rigid, soft, and articulated actors, the `boundaryElementType` / `boundary_element_type` parameter controls the quadrature rule on each surface face. Its values are defined by the [`ActorBoundaryElementType`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ActorBoundaryElementType) enum. Shells use the equivalent `contactElementType` parameter. Rod centerline contact also uses `contactElementType`, but with segment rules rather than the surface rules below:

| Setting | Quadrature Points per Face | Use Case |
|---|---|---|
| `P1Q1` | 1 | Fast, low-resolution contact |
| `P1Q3` (default) | 3 | Good balance of accuracy and performance |
| `P1Q6` | 6 | Higher accuracy for detailed contact |

Higher quadrature densities better resolve fine contact geometry and generally produce smoother forces, at the cost of additional distance-field evaluations.

:::note Experimental Quadrature Rules
Additional experimental quadrature rules (`ExperimentalP1Q7`, `ExperimentalP1Q12`, `ExperimentalP1Q16`) are available for advanced use cases.
:::

#### Stage-Discrete Dissipation

At implicit stage $i$, let $\mathbf{x}_{A,i}$ and $\mathbf{x}_{A,i}^0$ be the current and stage-start spatial positions of a sample on colliding actor $A$. Let $\mathbf{X}_{B,i}$ and $\mathbf{X}_{B,i}^0$ be the current and stage-start pullback maps described under [contact kinematics](#contact-kinematics). The relative contact velocity, expressed in collider-reference coordinates, is
$$
\mathbf{V}_{\mathrm{rel},i}
=\frac{\mathbf{X}_{B,i}(\mathbf{x}_{A,i})-\mathbf{X}_{B,i}^0(\mathbf{x}_{A,i}^0)}{\Delta t_i}~.
$$
This is the endpoint finite-difference counterpart of the continuum total derivative at fixed $\mathbf{X}_A$. Both the spatial sample position and the collider pullback map are evaluated independently at the current and stage-start configurations. This uses the same [stage-local state](./dynamics.md#mechanical-variables) as the rest of the mechanical system. $R$ contributes $\Delta t_iR$ to the [incremental potential $\Phi_i$](./dynamics.md#incremental-potential-form). However, the normal/tangential split makes $R$ position-state-dependent, so its potential contribution may be inconsistent with its contribution to the stage residual. Writing $q$ for the position state and $v_i(q)$ for its reconstructed velocity,
$$
\nabla_q\left[\Delta t_i R\right]
=\Delta t_i\frac{\partial R}{\partial q}+\frac{\partial R}{\partial v}~.
$$
Only the second term, $\partial R/\partial v$, contributes to the residual, but $\partial R/\partial q$ may be nonzero.  It is possible to recover consistency between the potential and residual by using explicit stage-start predictors to eliminate certain state-dependencies, similar to the treatment of [Li et al. (2020)](#references).  This may improve the performance of the nonlinear algebraic solver used for the implicit stage, but it can decrease accuracy and stability of the time integration at large time steps.

Two Boolean fields in `ExperimentalEvalParams` ([C++](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ExperimentalEvalParams.html), [Python](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ExperimentalEvalParams)) control the time levels used to evaluate the dissipative terms:

- `explicitNormals` selects the geometric data used for dissipation. When `true`, SuperDex Physics uses stage-start normals, alignment, and friction plane, and retains stage-start contacts alongside contacts detected in the current configuration. When `false` (default), it uses current geometry and reconstructs the stage-start distance to first order with the current field gradient.
- `implicitNormalForceForDissipation` selects the normal load used to scale dissipation. When `false` (default), SuperDex Physics uses the true or reconstructed stage-start normal-load scalar. When `true`, it uses the current normal-load scalar.

Separately, [`fadeFriction`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ExperimentalEvalParams.html) enables alignment-based attenuation of dissipative terms and defaults to `true`. Its state dependence is frozen when `explicitNormals = true`; with current normals it is another reason the default residual is not the gradient of the assembled scalar dissipation energy.

Their supported combinations are:

| [`explicitNormals`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ExperimentalEvalParams.html) | [`implicitNormalForceForDissipation`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ExperimentalEvalParams.html) | Evaluation | Incremental-potential status |
|---|---|---|---|
| `true` | `false` | Stage-start normals, alignment, friction plane, and true stage-start normal load. | The dissipative residual is exactly the gradient of $\Delta t_iR_i$. |
| `false` (default) | `false` (default) | Current geometry and reconstructed stage-start normal load. | Does not derive from an incremental potential. |
| `false` | `true` | Current geometry and current load. | Does not derive from an incremental potential. |
| `true` | `true` | Unsupported. | Unsupported. |

For `PointCloud` colliders discretizing the [double-integral generalization](#double-integral-generalization), each collider-point sphere's normal direction varies on the scale of the nonlocal interaction range $L_B$, which is typically small. Consequently, stage-start normals and normal loads can be poor predictors. For frictional point-cloud contact, current-state evaluation of both quantities (`explicitNormals = false` and `implicitNormalForceForDissipation = true`) is therefore recommended for improved accuracy and stability. As indicated in the table above, this combination does not produce a dissipative residual that is the exact gradient of $\Delta t_iR_i$.

## Contact Parameter Reference

### `ContactParams`

`ContactParams` ([C++](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1ContactParams.html), [Python](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ContactParams)) configures contact response per actor. The following table provides a correspondence between the C++ API names of parameters and the mathematical notation used in this document. Python uses the corresponding `snake_case` property names.

| Parameter | C++ Type | Default | Units | Description |
|---|---|---|---|---|
| `penaltyCoefficient` | `real` | `1e9` | Pa/m | Penalty coefficient $k$. Higher values reduce penetration but may degrade stability. |
| `penaltySmoothingHalfDistance` | `real` | `0.005` | m | Smoothing half-width $\epsilon$ of the ramp $h_{\epsilon,\delta}$. Larger values improve stability but may increase penetration. |
| `penaltyThresholdDefault` | `real` | `0.001` | m | Default contact threshold $\delta$ at which the penalty activates. |
| `penaltyThresholdExtraPadding` | `real` | `0` | m | Additional contribution to $\delta$ when the colliding actor uses `None` or `PointCloud`; helps prevent tunneling through thin actors. |
| `frictionWithColliderNormal` | `bool` | `true` | -- | Use the collider-field direction $\mathbf{N}_B$ for the normal/tangential split; when `false`, use $-\mathbf{N}_A$ where available. |
| `maxAlignmentNormals` | `real` | `0` | -- | Maximum-normal-alignment parameter $m$; contact is rejected when $s > m$. Valid range: [-1, 1]. |
| `viscousFrictionCoefficient` | `real` | `0` | s/m | Tangential viscous friction coefficient $c_t$. |
| `coulombFrictionCoefficient` | `real` | `0.5` | -- | Coulomb friction coefficient $\mu$. |
| `frictionFalloffVel` | `real` | `0.01` | m/s | Falloff velocity $V_f$ in the regularized Coulomb-friction ramp $\rho(V_t)$. |
| `normalViscousDampingCoefficient` | `real` | `0` | s/m | Normal viscous damping coefficient $c_n$, which controls velocity-dependent restitution. |
| `distanceErrorBound` | `real` | `0` | m | **Experimental.** Extra collider bounding-volume padding for approximate distance fields. |
| `collidingPenaltyLengthScale` | `real` | `1` | m | **Experimental.** Penalty correction scale for contact integrated over lower-dimensional colliding manifolds; always taken from the colliding actor. |

### `PointCloudColliderParams`

Point-cloud colliders are configured with additional parameters in `PointCloudColliderParams`.

| Parameter | Type | Default | Units | Description |
|---|---|---|---|---|
| `radius` | `real` | `0.01` | m | Interaction range $L_B$ of the point-cloud contact potential. |
| `selfContactExclusionRatio` | `real` | `1.5` | -- | Self-contact exclusion ratio $r$. |
| `spatialHashLoadFactor` | `real` | `0.0625` | -- | Load factor for the collision-detection spatial hash. |
| `selfContact` | `bool` | `false` | -- | Whether to enable self-contact. |
| `colliderTriangleElementType` | `Optional<ActorBoundaryElementType>` | `nullopt` | -- | Optional quadrature-based collider discretization for surfaces such as shells; otherwise uses mesh nodes. |
| `colliderSegmentElementType` | `Optional<ActorSegmentElementType>` | `nullopt` | -- | Optional quadrature-based collider discretization for curves such as rods; otherwise uses mesh nodes. |

## Contact Pipeline

The contact computation proceeds in three stages:

1. **Broad-phase**: axis-aligned bounding box (AABB) overlap tests rapidly cull actor pairs that cannot possibly be in contact, reducing the number of expensive narrow-phase evaluations.

2. **Narrow-phase**: Quadrature samples on the colliding actor are tested against the collider's distance field. Samples below the effective penalty threshold generate contact contributions.

3. **Response**: Contact forces and their Jacobians are computed from the penalty potential and assembled into the relevant actor or coupled solve system. Friction forces are combined with normal contact forces at this stage.

## Examples

- **[Damping Parameter Sweep](../examples/advanced/damping_sweep.md)**: demonstrates use of normal viscous damping. Python example: `examples/example_damping_sweep.py`.

## References

- S. Alaci, C. Filote, F.-C. Ciornei, O. V. Grosu, and M. S. Raboaca, [An Analytical Solution for Non-Linear Viscoelastic Impact](https://doi.org/10.3390/math9161849), *Mathematics*, 9(16), 1849, 2021.
- K. H. Hunt and F. R. E. Crossley, [Coefficient of Restitution Interpreted as Damping in Vibroimpact](https://doi.org/10.1115/1.3423596), *Journal of Applied Mechanics*, 42(2), 440-445, 1975.
- M. Li, Z. Ferguson, T. Schneider, T. Langlois, D. Zorin, D. Panozzo, C. Jiang, and D. M. Kaufman, [Incremental Potential Contact: Intersection- and Inversion-Free, Large-Deformation Dynamics](https://doi.org/10.1145/3386569.3392425), *ACM Transactions on Graphics (SIGGRAPH)*, 39(4), Article 49, 2020.

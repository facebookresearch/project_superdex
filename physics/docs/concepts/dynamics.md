---
sidebar_position: 3
title: Dynamics
---

# Dynamics

SuperDex Physics advances mechanical systems by stable implicit time integration of their equations of motion. This page introduces the continuous system dynamics, describes SuperDex Physics' unified implicit time integration, and discusses how problem structure can often be exploited to formulate the implicit equations as an optimization problem within each step.

## System Dynamics

Much of the dynamics simulated by SuperDex Physics can be expressed in the framework of classical Lagrangian mechanics, although it needs to be augmented with additional non-conservative forces to include dissipative mechanisms.

Let $q\in\mathcal Q$ be the generalized configuration and $v=\dot q\in T_q\mathcal Q$ its generalized velocity. When $\mathcal Q$ is a vector space, tangent-space increments can be added directly. Rotations, articulated poses, and rod frames instead lie on nonlinear configuration manifolds, so SuperDex Physics applies increments in a tangent space and maps them back to $\mathcal Q$ with type-specific updates.

Define the Lagrangian

$$
L(q,v,t)=T(q,v,t)-U(q,t)~,
$$

where $T$ is kinetic energy and $U$ is the total conservative potential. The Euler–Lagrange equations are

$$
\frac{d}{dt}\frac{\partial T}{\partial v}
-\frac{\partial T}{\partial q}
+\frac{\partial U}{\partial q}
=Q_{\mathrm{nc}}~,
$$

where $Q_{\mathrm{nc}}$ contains nonconservative generalized forces. When dissipative forces derive from a dissipation potential $R$,

$$
Q_{\mathrm{diss}}=-\frac{\partial R}{\partial v}~,
$$

and the equations can be written as

$$
\frac{d}{dt}\frac{\partial T}{\partial v}
-\frac{\partial T}{\partial q}
+\frac{\partial U}{\partial q}
+\frac{\partial R}{\partial v}
=Q_{\mathrm{other}}~.
$$

Here, $Q_{\mathrm{other}}$ collects forces not represented by $U$ or $R$. Depending on the system, the total conservative potential $U$ can include elastic, gravitational, contact, constraint, and applied-load terms. The actor pages define the forms of $T$, $U$, and $R$ used by each model.

## Time Integration

SuperDex Physics advances differential variables governed by

$$
\dot y = f(t,y)
$$

with a unified family of implicit multistep and Runge–Kutta methods. Each method first combines completed-step states, then solves one or more implicit stages. Backward Euler is the default. [Kennedy and Carpenter (2016)](#references) provide a detailed review of diagonally implicit Runge–Kutta methods, which the reader may find helpful for additional background information.

### Unified Formulation

A multistep method uses states from multiple completed time steps, whereas a multistage method evaluates one or more intermediate stage states within the current time step. The formulation below combines both structures: the coefficients $\alpha\in\mathbb R^l$ and $\beta\in\mathbb R$ define the $l$-step structure, while the coefficients defining the $s$-stage structure are canonically expressed as a so-called Butcher tableau $(A \in\mathbb R^{s\times s},b\in\mathbb{R}^s,c\in\mathbb{R}^s)$, to facilitate the algebraic definitions below. Readers interested in a high-level conceptual overview of implicit integration may consider substituting $l=s=1$ and $\alpha_1=\beta=A_{11}=b_1=c_1=1$ into these formulas to recover the simplest case of backward Euler. The full list of supported methods is given in the [Available Methods](#available-methods) section below.

Define the multistep base state
$$
y_n^\star = \sum_{r=1}^{l} \alpha_r y_{n+1-r}~,
$$
where the first coefficient multiplies the most recent completed state $y_n$. Let $\Delta t=t_{n+1}-t_n$ denote the full time-step size. Stage $i$ is
$$
\begin{aligned}
t_i &= t_n + (1-\beta+\beta c_i)\Delta t~, \\
k_i &= f(t_i, Y_i)~, \\
Y_i^0 &= y_n^\star + \beta \Delta t \sum_{j=1}^{i-1} A_{ij} k_j~, \\
\Delta t_i &= \beta A_{ii} \Delta t~, \\
Y_i &= Y_i^0 + \Delta t_i k_i~,
\end{aligned}
$$
and the completed step is
$$
y_{n+1} = y_n^\star + \beta \Delta t \sum_{i=1}^{s} b_i k_i~.
$$
Lowercase $y_n$ denotes the ODE state at a completed time step, whereas uppercase $Y_i^0$ and $Y_i$ denote the stage-start and stage-end states within the current step. SuperDex Physics reconstructs $y_n^\star$ from completed-step history, solves the stage states sequentially, and combines them to produce $y_{n+1}$. Although these states may coincide for particular methods, they have distinct roles in the general formulation.

For a mechanical system, the first-order ODE state is $y=(q,v)$, where $q$ is generalized configuration and $v$ is generalized velocity. Correspondingly, $Y_i^0=(q_i^0,v_i^0)$ and $Y_i=(q_i,v_i)$ contain the stage-start and stage-end mechanical states. Thus, SuperDex Physics solves stage $i$ as a backward-Euler-like problem over the stage duration $\Delta t_i$.

Stages are solved sequentially. Accordingly, SuperDex Physics supports lower-triangular $A$ with positive diagonal entries. Explicit stages and fully coupled implicit Runge–Kutta methods are not supported.

### Stage-State Reconstruction

SuperDex Physics stores stage-end states rather than the slopes $k_i$. Define

$$
\widetilde a_i
= A_{i,1:i-1}\left(A_{1:i-1,1:i-1}\right)^{-1}~,
\qquad
\widetilde b^T = b^T A^{-1}~.
$$

The states needed by later stages and by the completed step can then be reconstructed as

$$
Y_i^0
= y_n^\star
+ \sum_{j=1}^{i-1} \widetilde a_{ij}\left(Y_j-y_n^\star\right)~,
$$

$$
y_{n+1}
= y_n^\star
+ \sum_{i=1}^{s} \widetilde b_i\left(Y_i-y_n^\star\right)~.
$$

Here, $\widetilde a_{ij}$ is entry $j$ of row $\widetilde a_i$. For variables in vector spaces, these are ordinary weighted differences. For rotations and other manifold-valued configurations, SuperDex Physics applies the analogous type-specific difference and update operations.

### Mechanical Variables

Although the general stage state $Y_i=(q_i,v_i)$ contains both configuration and velocity, SuperDex Physics can use $q_i$ alone as the nonlinear unknown. Given the stage-start state $Y_i^0=(q_i^0,v_i^0)$, it reconstructs the stage-end velocity and acceleration as

$$
v_i = \frac{q_i-q_i^0}{\Delta t_i}~,
\qquad
a_i = \frac{v_i-v_i^0}{\Delta t_i}~.
$$

These are stage-local differences: the reference is the reconstructed stage-start state, not necessarily the state at the beginning of the full time step. For manifold-valued configurations, SuperDex Physics replaces the subtractions above with the analogous type-specific difference and tangent-space operations. Other rates used in dissipative terms (e.g., strain-rates for viscoelasticity, rates of constraint residuals for constraint damping, etc.) are also formulated as stage-local differences, to facilitate the use of optimization techniques to solve the implicit stage problem, as discussed further in the section on [Incremental Potential Form](#incremental-potential-form) below.

## Implicit Stage Problem

At each implicit stage $i$, SuperDex Physics treats the stage-start state $Y_i^0$ as fixed and solves for the configuration component $q_i$ of the stage-end state $Y_i$. The velocity and acceleration components are reconstructed from $q_i$ using the stage-local relations above. Substituting these relations into the discretized equations of motion produces the nonlinear stage residual

$$
r_i(q_i;Y_i^0,\Delta t_i)=0~.
$$

The arguments after the semicolon are fixed data for the stage. SuperDex Physics solves this equation with a Newton-like method, using the residual derivative

$$
K_i(q_i)=\frac{\partial r_i}{\partial q_i}~.
$$

This residual equation is the general form of the abstract implicit stage solve represented earlier by $Y_i=Y_i^0+\Delta t_i k_i$ together with $k_i=f(t_i,Y_i)$.

### Incremental Potential Form

The implicit stage problem to solve at each step is a system of nonlinear algebraic equations. The solution of arbitrary nonlinear systems is challenging, and typically relies on iterative algorithms like Newton's method, which are not guaranteed to converge unless specific conditions are met. However, additional techniques are available for solving optimization problems, where the residual is the gradient of some scalar potential. This is not guaranteed to be the case for arbitrary choices of configuration-dependent $R$ or $Q_{\mathrm{other}}$, but it does hold for many nontrivial systems. When the stage residual is the gradient of a potential, that potential is referred to as the "incremental potential", to distinguish it clearly from the physical potential energy $U$ of the continuous problem. [Gast et al. (2015)](#references) provide background on the general concept of incremental potential integration, using backward Euler as a representative implicit integrator. We now provide a brief introduction in the notation of the current page.

Consider a mechanical system whose configuration space is a vector space and for which $Q_{\mathrm{other}}=0$. Suppose its kinetic energy and dissipation potential have the forms

$$
T(v)=\frac12 v^TMv~,
\qquad
R=R(v)~.
$$

Here, the symmetric positive-definite, constant matrix $M$ maps generalized velocity to generalized momentum and is called the mass matrix; the velocity-only dependence of $R$ will ensure that its discrete contribution has the intended gradient. Using $q$ for the stage-end configuration unknown, define

$$
\widetilde q_i=q_i^0+\Delta t_i v_i^0~,
\qquad
v_i(q)=\frac{q-q_i^0}{\Delta t_i}~,
\qquad
a_i(q)=\frac{v_i(q)-v_i^0}{\Delta t_i}~.
$$

The incremental potential is

$$
\Phi_i(q)
=
\frac{1}{2\Delta t_i^2}\left\lVert q-\widetilde q_i\right\rVert_M^2
+U(q,t_i)
+\Delta t_iR\!\left(\frac{q-q_i^0}{\Delta t_i}\right)~,
$$

where $\lVert x\rVert_M^2=x^TMx$. Its gradient is

$$
r_i(q)=\nabla_q\Phi_i(q)
=Ma_i(q)+\frac{\partial U}{\partial q}
+\frac{\partial R}{\partial v}~.
$$

The implicit stage equation $r_i(q)=0$ is therefore the stationarity condition for $\Phi_i$. The factor $\Delta t_i$ multiplying $R$ is essential: the chain rule converts its configuration gradient into $\partial R/\partial v$ without an extra time-step factor. In this exact case,

$$
K_i(q)=\nabla_q^2\Phi_i(q)~.
$$

The assumptions above can be relaxed only when the resulting discrete terms remain integrable. A configuration-dependent kinetic energy, such as $T(q,v)=\tfrac12v^TM(q)v$, produces additional inertial terms through the full Euler–Lagrange operator and does not reduce to the quadratic inertial term shown above.

Similarly, a continuous dissipation potential can depend on configuration as well as velocity, $R(q,v,t)$, with dissipative force $-\partial R/\partial v$ at fixed $q$ and $t$. Directly inserting such an $R$ into the discrete objective generally gives

$$
\nabla_q\left[\Delta t_i R(q,v_i(q),t_i)\right]
=\Delta t_i\frac{\partial R}{\partial q}+\frac{\partial R}{\partial v}~.
$$

The additional configuration derivative is not part of the original dissipative force. An equivalent incremental potential therefore exists only if this extra term vanishes, belongs to the intended discrete model, or is avoided by explicitly evaluating the configuration-dependent coefficients and holding them fixed during the stage. Any force in $Q_{\mathrm{other}}$ must likewise derive from a potential that can be absorbed into $U$; otherwise it precludes an exact incremental-potential form.

When these conditions do not hold, the residual remains the authoritative discrete equation. SuperDex Physics may still assemble a scalar merit for line search, or use a symmetric, fitted, or positive-semidefinite approximation to the exact residual derivative. Important departures from this simple vector-space setting include:

- State-dependent dissipation can be potential-derived when its coefficients are fixed during the stage. Contact friction, for example, can evaluate selected normals or normal-force magnitudes explicitly to recover an integrable stage model.
- Some formulations, such as Newton–Euler rigid inertia, provide a residual without an exact incremental potential of this form.

Actor pages specialize this structure by defining their kinetic, conservative, and dissipative terms and documenting any problem-specific explicit evaluations or approximations.

## Available Methods

The Coefficients column lists $(\alpha,\beta)$ and the Butcher tableau $(A,b,c)$.

| Method | Steps | Stages | Order | Stability or structure | Coefficients |
| --- | ---: | ---: | ---: | --- | --- |
| `BackwardEuler` (BDF1, DIRK11) | 1 | 1 | 1 | L-stable | $((1),1)$; $\left[1\mid1\right]$; $b^T=(1)$ |
| `BDF2` | 2 | 1 | 2 | A-stable | $((4/3,-1/3),2/3)$; $\left[1\mid1\right]$; $b^T=(1)$ |
| `BDF3` | 3 | 1 | 3 | Not A-stable | $((18/11,-9/11,2/11),6/11)$; $\left[1\mid1\right]$; $b^T=(1)$ |
| `DIRK22` | 1 | 2 | 2 | L-stable | $((1),1)$; $\begin{bmatrix}\gamma_2&0&\gamma_2\\1-\gamma_2&\gamma_2&1\end{bmatrix}$; $b^T=(1-\gamma_2,\gamma_2)$ |
| `DIRK23` | 1 | 2 | 3 | A-stable, not L-stable | $((1),1)$; $\begin{bmatrix}\gamma_3&0&\gamma_3\\-1/\sqrt3&\gamma_3&1-\gamma_3\end{bmatrix}$; $b^T=(1/2,1/2)$ |
| `DIRK33` | 1 | 3 | 3 | L-stable | $((1),1)$; $\begin{bmatrix}\gamma&0&0&\gamma\\c_2-\gamma&\gamma&0&c_2\\b_1&b_2&\gamma&1\end{bmatrix}$; $b^T=(b_1,b_2,\gamma)$ |
| `SymplecticDIRK12` (implicit midpoint) | 1 | 1 | 2 | A-stable, symplectic | $((1),1)$; $\left[1/2\mid1/2\right]$; $b^T=(1)$ |
| `SymplecticDIRK22` | 1 | 2 | 2 | A-stable, symplectic | $((1),1)$; $\begin{bmatrix}1/4&0&1/4\\1/2&1/4&3/4\end{bmatrix}$; $b^T=(1/2,1/2)$ |

The DIRK constants are

$$
\begin{aligned}
\gamma_2 &= 1-\frac{\sqrt2}{2}~, \\
\gamma_3 &= \frac12+\frac{1}{2\sqrt3}~, \\
6\gamma^3-18\gamma^2+9\gamma-1 &= 0~,
\qquad \gamma \approx 0.435866521508459~, \\
b_1 &= -\frac32\gamma^2+4\gamma-\frac14~, \\
b_2 &= \frac32\gamma^2-5\gamma+\frac54~, \\
c_2 &= \frac{1+\gamma}{2}~.
\end{aligned}
$$

BDF2 starts with backward Euler until two completed states are available. BDF3 starts with backward Euler, switches to BDF2 when two completed states are available, and switches to BDF3 when three are available.

### Comparison and Recommendations

The default integration method is `BackwardEuler`. This provides maximum stability and is often useful in cases such as:

- Setting up new scenes without fully calibrating all parameters. The heavy numerical dissipation from backward Euler can compensate for incomplete specification of physical dissipation mechanisms.
- Running interactive teleoperation simulations, where only qualitative accuracy is required, but maximum robustness is needed with large, variable time-step sizes and noisy forces coming from user inputs.

**For scenarios where quantitative accuracy is desired, `BDF2` is recommended.** In most human-scale robotics applications, this should be sufficiently accurate to render time-discretization error secondary to errors coming from spatial discretization of deformable bodies or contact surfaces, approximate algebraic solution of the implicit step problem, or inherent modeling error in the continuous problem statement. However, one should keep the following considerations in mind when using `BDF2`:

- The higher accuracy of this time integrator will often highlight under-specification of physical damping mechanisms. Any stiffness-like term contributing to $U$ should have a corresponding dissipative contribution to $R$ to control oscillation. Specifically:
  - Objects may bounce excessively without setting [normal contact damping](./contact.md#friction-and-damping) in rigid collisions or including [viscoelastic stiffness damping](./actors/soft/overview.mdx#continuous-model) in deformable actors' material parameters.
  - [Compliant constraints](./constraints.mdx#parameters) and [articulated joint limits](./actors/articulated_actors.mdx#joint-limits) should have nonzero damping to control oscillation and bouncing.
  - [Pose controllers](./pose_controller.mdx#posetrackingparams) should have nonzero derivative control.
- Formal second-order accuracy depends on a constant time-step size. The method remains stable with arbitrary dynamic time-stepping, but is no longer expected to converge at its full order of accuracy.

Integrators other than `BackwardEuler` and `BDF2` are recommended only for special situations that are uncommon in typical robotics applications. E.g., a symplectic integrator may be preferred if long-time energy conservation is important, but most real object-manipulation scenarios include too much physical dissipation for the benefits of symplectic integration to be evident.

## Choosing a Time-Step Size

Choose a time-step size that resolves the dynamics of interest. For A-stable integration methods, including L-stable methods, the time-step size is not limited by the stiffness-driven linear-stability restrictions of explicit or semi-implicit methods. Consequently, SuperDex Physics can use substantially larger stable time steps than physics engines based on explicit or semi-implicit integration. This is particularly valuable for stiff systems, such as stiff or nearly incompressible soft bodies, thin shells and slender rods with widely separated deformation scales, and scenes with stiff contact or constraints. This ability to take fewer steps per simulated second often improves simulation performance and real-time factor.

:::note[Practical time steps]

Time steps of 10–25 ms (40–100 physics steps per simulated second) run robustly in most scenes, including complex contact-rich and deformable simulations. This is a practical starting range, not a guarantee. Smaller steps may still be required to resolve fast motion, accurately capture short-duration contact dynamics without excessive numerical dissipation, resolve dynamics associated with small geometric or discretization length scales, or improve nonlinear-solver convergence.

:::

## Examples

- **[Damping Parameter Sweep](../examples/advanced/damping_sweep.md)**: demonstrates use of a higher-order time integrator with physical dissipation mechanisms active. Python example: `examples/example_damping_sweep.py`.

## Related Concepts

- [Rigid Actors](./actors/rigid_actors.mdx)
- [Soft Actors](./actors/soft/overview.mdx)
- [Shell Actors](./actors/shell.mdx)
- [Rod Actors](./actors/rod.mdx)
- [Articulated Actors](./actors/articulated_actors.mdx)
- [Solvers](./solvers.md)

## References

- T. F. Gast, C. Schroeder, A. Stomakhin, C. Jiang, and J. M. Teran, [Optimization Integrator for Large Time Steps](https://doi.org/10.1109/TVCG.2015.2459687), IEEE Transactions on Visualization and Computer Graphics, 21(10):1103–1115, 2015.
- C. A. Kennedy and M. H. Carpenter, [Diagonally Implicit Runge-Kutta Methods for Ordinary Differential Equations: A Review](https://ntrs.nasa.gov/citations/20160005923), NASA/TM-2016-219173, 2016.

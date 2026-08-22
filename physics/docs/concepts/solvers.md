---
sidebar_position: 12
title: Solvers
---

# Solvers

The general form of the implicit time integrator defined on the [Dynamics](./dynamics.md#implicit-stage-problem) page requires solving a system of nonlinear equations (SNLE) in each implicit stage. In the notation established on that page, this problem can be written
$$
r_i(q_i;Y_i^0,\Delta t_i)=0~,
$$
where $q_i$ is the unknown stage-end configuration, $Y_i^0$ is the fixed stage-start state, and $\Delta t_i$ is the stage duration.
SuperDex Physics uses a quasi-Newton method with line search to solve the stage SNLE approximately. This method can be customized through a variety of user-facing options. The remainder of this page provides additional details on the SNLE solution methods, which may be useful for tuning performance in different applications.

In practice, the engine decomposes a scene into one or more "islands" of actors that can potentially interact with each other in a given step, to take advantage of multi-threaded parallelism and limit the cost of solution methods that scale super-linearly with the total number of unknown configuration DoFs. The current discussion is written assuming a single island for simplicity, but it generalizes straightforwardly to multiple islands.

## Newton's Method

Starting from an estimate $q_i^{(k)}$, Newton's method evaluates the residual and its derivative,
$$
\begin{aligned}
r_i^{(k)} &= r_i(q_i^{(k)};Y_i^0,\Delta t_i)~, \\
K_i^{(k)} &\approx \left.\frac{\partial r_i}{\partial q_i}\right|_{q_i^{(k)}}~,
\end{aligned}
$$
then computes a correction $\Delta q_i^{(k)}$ by solving the linear system
$$
K_i^{(k)}\Delta q_i^{(k)}=-r_i^{(k)}~.
$$
The next estimate is
$$
q_i^{(k+1)}=q_i^{(k)}+\lambda_k\Delta q_i^{(k)}~,
$$
where $\lambda_k$ is a line-search scaling factor. When $K_i^{(k)}$ is the exact residual Jacobian, $\lambda_k = 1$ reproduces classical Newton–Raphson iteration, but full steps are prone to divergence in practical scenarios. [Line-search](#line-search) methods modify this factor (typically reducing it) to improve solver robustness. For configurations on nonlinear manifolds (e.g., rotations of rigid bodies), the addition on the right-hand side above denotes a tangent-space update, which is then mapped back onto the manifold through a retraction (e.g., the exponential map for rotations).

This iteration continues until either a [convergence criterion](#convergence-criteria) or [other stopping criterion](#other-stopping-conditions) is reached, where reaching a maximum iteration limit is the most common non-converged stopping criterion.
In many high-fidelity simulators used for scientific and engineering applications, the simulation will either reduce its time step size or terminate with an error if the maximum number of iterations is reached without meeting a convergence criterion.
However, this is impractical for applications in real-time interactive simulation of robot teleoperation or large-scale control policy training, where speed and robustness must be prioritized over absolute accuracy.
The default behavior of SuperDex Physics is to continue to the next time integration step (or stage) after reaching the maximum iteration count, which is the common pragmatic solution in the computer graphics literature on physics-based simulation, and typically produces qualitatively reasonable approximate solutions when an appropriate line search strategy is used to stabilize the Newton solve.
Applications requiring stricter convergence may instead query convergence status with [`Scene::GetSolverStats()`](pathname:///generated/api/v1.0/cpp/classsuperdex_1_1Scene.html) ([`get_solver_stats`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.Scene.get_solver_stats) in Python), using custom time stepping logic and state capture/restore functionality to retry unconverged steps.

There are several reasons why the residual Jacobian $K_i^{(k)}$ may be approximate (as indicated by the use of $\approx$ above), resulting in a quasi-Newton method. These include the following:

- Removing nonsymmetric terms allows more efficient linear solvers to be used for the linear problem within each Newton iteration.
- Projecting Jacobian contributions to be positive semidefinite can improve robustness of the nonlinear iteration and, again, allows more efficient linear solvers to be used.
- The full Jacobian may include terms that are expensive to compute but provide little or no improvement in the convergence of the iteration.

Examples include [PSD-enforcement strategies](./actors/soft/materials/overview.md#psd-enforcement) for nonlinear soft-material tangents and the Gauss–Newton approximation used for Hessians of [constraint](./constraints.mdx#mathematical-formulation) energies.

[`NonLinearSolverParams::psdProjMode`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1NonLinearSolverParams.html) controls when the nonlinear solver requests PSD approximations of applicable Jacobian contributions, including material tangents. The default, `PsdProjectionMode::Always`, requests these approximations whenever the Jacobian is assembled. `PsdProjectionMode::Never` is the policy for never requesting them. `PsdProjectionMode::IfFailRetry` retries a failed nonlinear iteration with PSD approximations, while `PsdProjectionMode::IfFailAlways` does the same and keeps them enabled for the remaining iterations of that solve. Python exposes the corresponding [`psd_proj_mode`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.NonLinearSolverParams.psd_proj_mode) property and [`PsdProjectionMode`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.PsdProjectionMode) enum.

## Line Search

The classical Newton–Raphson iteration with $\lambda_k = 1$ is only provably guaranteed to converge under very narrow conditions on the residual function and initial guess. These conditions are rarely satisfied in practice. A full Newton step is therefore not always the best choice of solution update. A line search attempts to find a better choice of $\lambda_k$ by searching for an improved $q_i^{(k+1)}$ along a line parameterized by $\lambda_k$. This is typically an iterative search involving several evaluations of the residual and/or [incremental potential](./dynamics.md#incremental-potential-form) at different values of $\lambda_k$. Line-search trial evaluations avoid evaluating the Newton Jacobian, which is often the most expensive part of a Newton step.

The default line search strategy is `LineSearchType::ResidualNorm`, which only evaluates the residual, while some other strategies make use of the incremental potential. Potential-based line searches such as `LineSearchType::WolfeStrong` may improve results, especially in settings where the dominant residual contributions derive exactly from a potential (e.g., the elastic energy of a large deformable actor). Python exposes these choices through the [`LineSearchType`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.LineSearchType) enum.

## Linear Solvers

Every (quasi-)Newton iteration requires the solution of a linear system. Linear solvers fall into two broad categories:

- **Direct solvers:** These use factorizations like LDLT (for symmetric matrices) or LU (for general matrices), which are equivalent to Gaussian elimination, producing exact solutions up to floating-point error. These methods are often efficient for small and medium-sized linear systems (e.g., moderate numbers of **rigid** and/or **articulated** actors), and their accuracy makes nonlinear convergence more robust, but their memory and compute costs grow rapidly with problem size, making them impractical for problems involving large deformable actors.
- **Iterative solvers:** These solve problems approximately, with an accuracy vs. cost trade-off that can be tuned through tolerance and/or iteration count parameters. SuperDex Physics primarily uses the Krylov family of iterative methods, defaulting to conjugate gradient (CG) for large systems. CG assumes a symmetric positive-definite (SPD) $K_i^{(k)}$. In the formulations implemented by SuperDex Physics, SPD approximations of Jacobians are favored over indefinite and/or nonsymmetric exact Jacobians that would require more costly solvers. However, the MINRES (symmetric indefinite) and GMRES (general nonsymmetric) solvers are supported and may be useful in certain problems. Krylov iterative solvers like CG are the most practical choice for most **deformable** actors, because they can reach acceptable accuracy at much lower computational cost than direct solvers.

The default [`LinearSolverParams::solverType`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1LinearSolverParams.html) value, `LinearSolverType::Auto`, chooses a linear solver based on the system size, favoring LDLT factorization for smaller systems and CG for larger ones. Users can instead specify the solver type directly, if desired. Python exposes the corresponding [`solver_type`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.LinearSolverParams.solver_type) property and [`LinearSolverType`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.LinearSolverType) enum.

The performance of iterative solvers also depends strongly on using an appropriate **preconditioner**, i.e., a cheap approximation $M$ of $K_i^{(k)}$ whose inverse is applied within the iterative method. This allows the iterative method to reach a better approximate solution in fewer iterations. Multiple preconditioner types are available, and the default `PreconditionerType::PerActor` option applies a separate preconditioner to each actor's diagonal contribution to $K_i^{(k)}$, selected from actor-specific hints or the structure of that contribution. Preconditioners are ignored for direct solvers, because they are redundant in that setting. Python exposes these choices through [`PreconditionerType`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.PreconditionerType).

## Convergence Criteria

Suppressing the fixed stage index $i$, let $r^{(k)}$ denote the residual at nonlinear iteration $k$. The absolute residual tolerance is $\epsilon_\textrm{abs}$ (defaulting to $10^{-3}$), and the relative residual tolerance is $\epsilon_\textrm{rel}$ (defaulting to $10^{-6}$). [`NonLinearSolverParams::convergenceMode`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1NonLinearSolverParams.html) determines how these tolerances are applied. The `NonLinearSolverConvergenceMode` enum defines the modes:

- **`NonLinearSolverConvergenceMode::PerActorWeighted` (default):** For actor $a$, define

  $$
  \rho_a^{(k)}
  =\left\lVert r_a^{(k)}\right\rVert_W
  =\sqrt{\sum_j w_{a,j}\left(r_{a,j}^{(k)}\right)^2}~,
  $$

  where $r_{a,j}^{(k)}$ is residual component $j$ associated with actor $a$, and $w_{a,j}$ is its convergence weight. Actor $a$ satisfies the convergence criterion when
  $$
  \rho_a^{(k)}\leq\epsilon_\textrm{abs}
  $$
  or, after the initial iteration,
  $$
  \rho_a^{(k)}\leq\epsilon_\textrm{rel}\,\rho_a^{(0)}~.
  $$
  Every actor must individually satisfy at least one of these criteria.

  The weights are derived from characteristic inertial force and torque scales so that the weighted norms are dimensionless where such physical normalization is available, and less sensitive to actor mass, size, and mesh resolution. The weighted actor norms are designed so that residual contributions from characteristic loading (e.g., gravity) will have a norm of $\approx 1$, so the default absolute tolerance of $10^{-3}$ represents a small dimensionless residual relative to that scale. This mode is intended for general-purpose use in dynamic scenarios.

- **`NonLinearSolverConvergenceMode::Global`:** This mode uses the raw Euclidean ($\ell^2$) norm of the full residual to test for convergence:
  $$
  \rho^{(k)}=\left\lVert r^{(k)}\right\rVert_2~.
  $$
  Residual convergence is reached when
  $$
  \rho^{(k)}\leq\epsilon_\textrm{abs}
  $$
  or, after the initial iteration,
  $$
  \rho^{(k)}\leq\epsilon_\textrm{rel}\,\rho^{(0)}~.
  $$
  The raw algebraic norm does not account for dimensional consistency, but this mode may be useful when a single global residual scale is meaningful, or in near-static problems for which inertia-based weighting may not be appropriate. The default tolerances are not generally appropriate for this mode.

### Other Stopping Conditions

The nonlinear solve can also stop for the following reasons:

- **Relative step tolerance:** Let $\Delta q_i^{(k)}$ be the correction returned by the linear solve before line-search scaling. The solve stops when
  $$
  \left\lVert\Delta q_i^{(k)}\right\rVert_2
  \leq
  \epsilon_\textrm{relStep}\left\lVert q_i^{(k)}\right\rVert_2~.
  $$
  This check always uses an unweighted Euclidean norm. The default is $\epsilon_\textrm{relStep} = 10\epsilon_\textrm{mach}$, where $\epsilon_\textrm{mach}$ is machine precision for the floating-point type (`real`) that SuperDex Physics is built with.

- **Maximum iterations:** The solve stops after the configured maximum number of nonlinear updates. The default is 4.
- **Maximum elapsed time:** The solve stops after the configured wall-clock limit. The default is 0, which acts as a sentinel to disable the limit.
- **No improvement:** When this option is enabled, the solve stops after a line search fails to improve its monitored quantity. It is disabled by default.

As mentioned above, the convergence status is returned in [`SolverStats`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1SolverStats.html) by [`Scene::GetSolverStats()`](pathname:///generated/api/v1.0/cpp/classsuperdex_1_1Scene.html). Satisfying an absolute or relative residual criterion produces a `ConvergenceStatus::Converged` status. Reaching the relative step, iteration, elapsed-time, or no-improvement limit produces a `ConvergenceStatus::Stopped` status: the residual has not met its convergence tolerance. With explosion control enabled (the default), a non-finite or excessively growing residual produces a `ConvergenceStatus::Diverged` status. Applications that require converged stage solves should monitor the reported solver status. Nonlinear iterations, residual norms, the final stopping reason, and other solver data can be logged by setting [`NonLinearSolverParams::verbosity`](pathname:///generated/api/v1.0/cpp/structsuperdex_1_1NonLinearSolverParams.html) to `VerbosityLevel::Verbose`. Python exposes the containing [`SolverStats`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.SolverStats) and [`ConvergenceStatus`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.ConvergenceStatus) types, plus the [`verbosity`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.NonLinearSolverParams.verbosity) property and [`VerbosityLevel`](pathname:///generated/api/v1.0/python/api/physics.html#superdex.physics.VerbosityLevel) enum.

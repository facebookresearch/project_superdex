# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import gc

from test.conftest import mochi, MochiTestBase, np, np_real


class TestStructs(MochiTestBase):
    def test_contact_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.ContactParams()
        self.assertEqual(0, params.distance_error_bound)
        self.assertEqual(1, params.obj_scale)

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.ContactParams(
            penalty_coefficient=2e9,
            obj_scale=2,
            coulomb_friction_coefficient=0.8,
            penalty_smoothing_half_distance=0.01,
            penalty_threshold_default=0.002,
            penalty_threshold_extra_padding=0.001,
            max_alignment_normals=0.5,
            viscous_friction_coefficient=0.1,
            friction_falloff_vel=0.02,
            distance_error_bound=0.001,
            friction_with_collider_normal=False,
        )
        self.assertAlmostEqual(2e9, params_kw_all.penalty_coefficient)
        self.assertAlmostEqual(0.01, params_kw_all.penalty_smoothing_half_distance)
        self.assertAlmostEqual(0.002, params_kw_all.penalty_threshold_default)
        self.assertAlmostEqual(0.001, params_kw_all.penalty_threshold_extra_padding)
        self.assertFalse(params_kw_all.friction_with_collider_normal)
        self.assertAlmostEqual(0.5, params_kw_all.max_alignment_normals)
        self.assertAlmostEqual(0.1, params_kw_all.viscous_friction_coefficient)
        self.assertAlmostEqual(0.8, params_kw_all.coulomb_friction_coefficient)
        self.assertAlmostEqual(0.02, params_kw_all.friction_falloff_vel)
        self.assertAlmostEqual(0.001, params_kw_all.distance_error_bound)
        self.assertAlmostEqual(2.0, params_kw_all.obj_scale)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.ContactParams(
            coulomb_friction_coefficient=0,
            penalty_coefficient=5e8,
        )
        self.assertAlmostEqual(5e8, params_kw_partial.penalty_coefficient)
        self.assertEqual(0, params_kw_partial.coulomb_friction_coefficient)
        self.assertEqual(
            params.friction_with_collider_normal,
            params_kw_partial.friction_with_collider_normal,
        )
        self.assertAlmostEqual(params.obj_scale, params_kw_partial.obj_scale)

        # Assign every field (to show we can)
        params.penalty_coefficient = 1e9
        params.penalty_smoothing_half_distance = 0.005
        params.friction_with_collider_normal = True
        params.max_alignment_normals = 0
        params.viscous_friction_coefficient = 0
        params.coulomb_friction_coefficient = 0.5
        params.friction_falloff_vel = 0.01
        params.distance_error_bound = 0
        params.obj_scale = 1
        params.penalty_threshold_default = 0.001
        params.penalty_threshold_extra_padding = 0

    def test_grid_sdf_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.GridSdfParams()
        self.assertEqual(mochi.GridSdfResolutionMode.MEAN_EDGE, params.resolution_mode)
        self.assertAlmostEqual(0.005, params.boundary_padding_dist)
        self.assertEqual(3, len(params.resolution_delta))
        self.assertAlmostEqual(0.25, params.resolution_delta[0])
        self.assertAlmostEqual(0.25, params.resolution_delta[1])
        self.assertAlmostEqual(0.25, params.resolution_delta[2])

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.GridSdfParams(
            resolution_mode=mochi.GridSdfResolutionMode.EXPLICIT,
            resolution_delta=[0.5, 0.6, 0.7],
            boundary_padding_dist=0.1,
        )
        self.assertEqual(
            mochi.GridSdfResolutionMode.EXPLICIT, params_kw_all.resolution_mode
        )
        self.assertAlmostEqual(0.5, params_kw_all.resolution_delta[0])
        self.assertAlmostEqual(0.6, params_kw_all.resolution_delta[1])
        self.assertAlmostEqual(0.7, params_kw_all.resolution_delta[2])
        self.assertAlmostEqual(0.1, params_kw_all.boundary_padding_dist)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.GridSdfParams(
            resolution_mode=mochi.GridSdfResolutionMode.LARGEST_AXIS,
            boundary_padding_dist=0.05,
        )
        self.assertEqual(
            mochi.GridSdfResolutionMode.LARGEST_AXIS, params_kw_partial.resolution_mode
        )
        self.assertAlmostEqual(0.05, params_kw_partial.boundary_padding_dist)
        self.assertEqual(params.resolution_delta, params_kw_partial.resolution_delta)

        # Assign every field (to show we can)
        params.resolution_mode = mochi.GridSdfResolutionMode.EXPLICIT
        params.resolution_delta = [0.25, 0.25, 0.25]
        params.boundary_padding_dist = 0

    def test_soft_material_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.SoftMaterialParams()
        self.assertAlmostEqual(1e5, params.neo_hookean.youngs_modulus)
        self.assertAlmostEqual(1000, params.density)

        # Test keyword argument constructor with NeoHookean
        snh_params = mochi.NeoHookeanMaterialParams(
            youngs_modulus=2e5,
            poisson_ratio=0.3,
            psd_strategy=mochi.MaterialPsdStrategy.PROJECTION,
        )
        params_kw_snh = mochi.SoftMaterialParams(
            type=mochi.SoftMaterialType.NEO_HOOKEAN,
            neo_hookean=snh_params,
            density=2000,
        )
        self.assertEqual(mochi.SoftMaterialType.NEO_HOOKEAN, params_kw_snh.type)
        self.assertAlmostEqual(2e5, params_kw_snh.neo_hookean.youngs_modulus)
        self.assertAlmostEqual(0.3, params_kw_snh.neo_hookean.poisson_ratio)
        self.assertAlmostEqual(2000, params_kw_snh.density)
        self.assertEqual(
            mochi.MaterialPsdStrategy.PROJECTION,
            params_kw_snh.neo_hookean.psd_strategy,
        )

        # Test keyword argument constructor with LinearElastic
        le_params = mochi.LinearElasticMaterialParams(
            youngs_modulus=2e5,
            poisson_ratio=0.3,
        )
        params_kw_le = mochi.SoftMaterialParams(
            type=mochi.SoftMaterialType.LINEAR_ELASTIC,
            linear_elastic=le_params,
            density=2000,
        )
        self.assertEqual(mochi.SoftMaterialType.LINEAR_ELASTIC, params_kw_le.type)
        self.assertAlmostEqual(2e5, params_kw_le.linear_elastic.youngs_modulus)
        self.assertAlmostEqual(0.3, params_kw_le.linear_elastic.poisson_ratio)
        self.assertAlmostEqual(2000, params_kw_le.density)

        # Test keyword argument constructor with ActiveShapeTargetingArap
        astrap_params = mochi.ActiveShapeTargetingArapMaterialParams(
            stiffness=2000,
            shape_target_tensor=[1.0, -2, -3.0, -4.0, 5, -6.0],
            psd_strategy=mochi.MaterialPsdStrategy.PROJECTION,
        )
        params_kw_astrap = mochi.SoftMaterialParams(
            type=mochi.SoftMaterialType.ACTIVE_SHAPE_TARGETING_ARAP,
            active_shape_targeting_arap=astrap_params,
            density=2000,
        )
        self.assertAlmostEqual(
            2000, params_kw_astrap.active_shape_targeting_arap.stiffness
        )
        self.assertAlmostEqual(
            mochi.Real6(1.0, -2, -3.0, -4.0, 5, -6.0),
            params_kw_astrap.active_shape_targeting_arap.shape_target_tensor,
        )

        # Test keyword argument constructor with ActiveNeoHookean
        asnh_params = mochi.ActiveNeoHookeanMaterialParams(
            passive_isotropic=mochi.NeoHookeanMaterialParams(
                youngs_modulus=1e5,
                poisson_ratio=0.45,
            ),
            active_anisotropic=mochi.ActiveAnisoArapMaterialParams(
                alpha=1000,
                length=1,
                aniso_dir=[1, 0, 0],
            ),
        )
        params_asnh = mochi.SoftMaterialParams(
            type=mochi.SoftMaterialType.ACTIVE_NEO_HOOKEAN,
            active_neo_hookean=asnh_params,
            density=1.23,
        )
        self.assertAlmostEqual(
            1e5,
            params_asnh.active_neo_hookean.passive_isotropic.youngs_modulus,
        )
        self.assertAlmostEqual(
            1000,
            params_asnh.active_neo_hookean.active_anisotropic.alpha,
        )
        self.assertAlmostEqual(1.23, params_asnh.density)

        # Test keyword argument constructor with Arap
        arap = mochi.ArapMaterialParams(
            stiffness=1000,
            psd_strategy=mochi.MaterialPsdStrategy.PROJECTION,
        )
        params_kw_arap = mochi.SoftMaterialParams(
            type=mochi.SoftMaterialType.ARAP,
            arap=arap,
            density=1500,
        )
        self.assertAlmostEqual(1500, params_kw_arap.density)
        self.assertAlmostEqual(1000, params_kw_arap.arap.stiffness)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.SoftMaterialParams(
            neo_hookean=mochi.NeoHookeanMaterialParams(
                youngs_modulus=3e5,
            ),
            density=1500,
        )
        self.assertAlmostEqual(3e5, params_kw_partial.neo_hookean.youngs_modulus)
        self.assertAlmostEqual(1500, params_kw_partial.density)
        self.assertEqual(
            params.neo_hookean.poisson_ratio,
            params_kw_partial.neo_hookean.poisson_ratio,
        )

        # Assign fields via sub-struct access (to show we can)
        params.type = mochi.SoftMaterialType.NEO_HOOKEAN
        params.neo_hookean.youngs_modulus = 1e5
        params.neo_hookean.poisson_ratio = 0.45
        params.density = 1.23

    def test_non_linear_solver_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.NonLinearSolverParams()
        self.assertEqual(mochi.NonLinearSolverType.DEFAULT, params.solver_type)
        self.assertEqual(4, params.max_iter)
        self.assertTrue(params.explosion_control)

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.NonLinearSolverParams(
            solver_type=mochi.NonLinearSolverType.BFGS,
            gradient_descent_fallback=True,
            abs_tol=1e-5,
            line_search_alpha=0.6,
            d_residual_assembly_period=2,
            max_iter=10,
            explosion_control=False,
            verbosity=mochi.VerbosityLevel.VERBOSE,
            line_search_max_rel_increase=1.0,
            line_search_wolfe1=2e-4,
            abs_div_tol=2e9,
            max_elapsed_time_seconds=5,
            rel_tol=1e-5,
            rel_step_tol=1e-6,
            stop_if_no_improvement=True,
            line_search_type=mochi.LineSearchType.ARMIJO,
            psd_proj_mode=mochi.PsdProjectionMode.ALWAYS,
            rel_div_tol=2e4,
            line_search_max_iter=8,
            line_search_wolfe2=0.95,
            linear_tolerance_strategy=mochi.LinearToleranceStrategy.EISENSTAT_WALKER1,
        )
        self.assertEqual(mochi.NonLinearSolverType.BFGS, params_kw_all.solver_type)
        self.assertEqual(2, params_kw_all.d_residual_assembly_period)
        self.assertEqual(10, params_kw_all.max_iter)
        self.assertAlmostEqual(5.0, params_kw_all.max_elapsed_time_seconds)
        self.assertAlmostEqual(1e-5, params_kw_all.abs_tol)
        self.assertAlmostEqual(1e-5, params_kw_all.rel_tol)
        self.assertAlmostEqual(1e-6, params_kw_all.rel_step_tol)
        self.assertTrue(params_kw_all.stop_if_no_improvement)
        self.assertEqual(mochi.PsdProjectionMode.ALWAYS, params_kw_all.psd_proj_mode)
        self.assertTrue(params_kw_all.gradient_descent_fallback)
        self.assertFalse(params_kw_all.explosion_control)
        self.assertAlmostEqual(2e9, params_kw_all.abs_div_tol)
        self.assertAlmostEqual(2e4, params_kw_all.rel_div_tol)
        self.assertEqual(8, params_kw_all.line_search_max_iter)
        self.assertAlmostEqual(0.6, params_kw_all.line_search_alpha)
        self.assertAlmostEqual(2e-4, params_kw_all.line_search_wolfe1)
        self.assertAlmostEqual(0.95, params_kw_all.line_search_wolfe2)
        self.assertAlmostEqual(1.0, params_kw_all.line_search_max_rel_increase)
        self.assertEqual(mochi.LineSearchType.ARMIJO, params_kw_all.line_search_type)
        self.assertEqual(
            mochi.LinearToleranceStrategy.EISENSTAT_WALKER1,
            params_kw_all.linear_tolerance_strategy,
        )
        self.assertEqual(mochi.VerbosityLevel.VERBOSE, params_kw_all.verbosity)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.NonLinearSolverParams(
            max_iter=8,
            verbosity=mochi.VerbosityLevel.SILENT,
        )
        self.assertEqual(8, params_kw_partial.max_iter)
        self.assertEqual(mochi.VerbosityLevel.SILENT, params_kw_partial.verbosity)
        self.assertEqual(params.solver_type, params_kw_partial.solver_type)
        self.assertEqual(params.explosion_control, params_kw_partial.explosion_control)

        # Assign every field (to show we can)
        params.solver_type = mochi.NonLinearSolverType.DEFAULT
        params.d_residual_assembly_period = 1
        params.max_iter = 4
        params.max_elapsed_time_seconds = 0
        params.abs_tol = 1e-4
        params.rel_tol = 1e-4
        params.rel_step_tol = 0
        params.stop_if_no_improvement = False
        params.psd_proj_mode = mochi.PsdProjectionMode.DEFAULT
        params.gradient_descent_fallback = False
        params.explosion_control = True
        params.abs_div_tol = 1e9
        params.rel_div_tol = 1e4
        params.line_search_max_iter = 4
        params.line_search_alpha = 0.5
        params.line_search_wolfe1 = 1e-4
        params.line_search_wolfe2 = 0.9
        params.line_search_max_rel_increase = 0.0
        params.line_search_type = mochi.LineSearchType.DEFAULT
        params.linear_tolerance_strategy = mochi.LinearToleranceStrategy.DEFAULT
        params.verbosity = mochi.VerbosityLevel.WARNING

    def test_linear_solver_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.LinearSolverParams()
        self.assertEqual(mochi.LinearSolverType.DEFAULT, params.solver_type)
        self.assertEqual(1000, params.restart_size)

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.LinearSolverParams(
            solver_type=mochi.LinearSolverType.GMRES,
            abort_if_not_spd=True,
            rel_div_tol=1e11,
            preconditioner_type=mochi.PreconditionerType.JACOBI,
            norm_type=mochi.LinearSolverConvergenceNorm.RESIDUAL_L2,
            rel_tol=1e-6,
            max_iter=500,
            restart_size=500,
            abs_tol=1e-6,
            verbosity=mochi.VerbosityLevel.VERBOSE,
        )
        self.assertEqual(mochi.LinearSolverType.GMRES, params_kw_all.solver_type)
        self.assertEqual(
            mochi.PreconditionerType.JACOBI, params_kw_all.preconditioner_type
        )
        self.assertEqual(
            mochi.LinearSolverConvergenceNorm.RESIDUAL_L2, params_kw_all.norm_type
        )
        self.assertAlmostEqual(1e-6, params_kw_all.abs_tol)
        self.assertAlmostEqual(1e-6, params_kw_all.rel_tol)
        self.assertLess(abs(1e11 - params_kw_all.rel_div_tol), 1e-5 * 1e11)
        self.assertEqual(500, params_kw_all.max_iter)
        self.assertEqual(500, params_kw_all.restart_size)
        self.assertTrue(params_kw_all.abort_if_not_spd)
        self.assertEqual(mochi.VerbosityLevel.VERBOSE, params_kw_all.verbosity)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.LinearSolverParams(
            max_iter=2000,
            abs_tol=1e-7,
        )
        self.assertEqual(2000, params_kw_partial.max_iter)
        self.assertAlmostEqual(1e-7, params_kw_partial.abs_tol)
        self.assertEqual(params.solver_type, params_kw_partial.solver_type)
        self.assertEqual(
            params.preconditioner_type, params_kw_partial.preconditioner_type
        )

        # Assign every field (to show we can)
        params.solver_type = mochi.LinearSolverType.DEFAULT
        params.preconditioner_type = mochi.PreconditionerType.DEFAULT
        params.norm_type = mochi.LinearSolverConvergenceNorm.DEFAULT
        params.abs_tol = 1e-5
        params.rel_tol = 1e-5
        params.rel_div_tol = 1e10
        params.max_iter = 1000
        params.restart_size = 1000
        params.abort_if_not_spd = False
        params.verbosity = mochi.VerbosityLevel.WARNING

    def test_experimental_eval_params(self):
        # Test the default values so we're convinced that the C++ constructor ran
        params = mochi.ExperimentalEvalParams()
        self.assertFalse(params.explicit_normals)
        self.assertTrue(params.fade_friction)
        self.assertFalse(params.implicit_normal_force_for_dissipation)
        self.assertTrue(params.fitted_saturation_hessian.contact_friction)
        self.assertFalse(params.fitted_saturation_hessian.joint_friction)
        self.assertTrue(params.fitted_saturation_hessian.constraint_saturation)

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.ExperimentalEvalParams(
            fitted_saturation_hessian=mochi.SaturationHessianParams(
                contact_friction=False,
                joint_friction=False,
                constraint_saturation=False,
            ),
            fade_friction=False,
            implicit_normal_force_for_dissipation=True,
            explicit_normals=True,
        )
        self.assertTrue(params_kw_all.explicit_normals)
        self.assertFalse(params_kw_all.fade_friction)
        self.assertTrue(params_kw_all.implicit_normal_force_for_dissipation)
        self.assertFalse(params_kw_all.fitted_saturation_hessian.contact_friction)
        self.assertFalse(params_kw_all.fitted_saturation_hessian.joint_friction)
        self.assertFalse(params_kw_all.fitted_saturation_hessian.constraint_saturation)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.ExperimentalEvalParams(
            explicit_normals=True,
        )
        self.assertTrue(params_kw_partial.explicit_normals)
        self.assertEqual(params.fade_friction, params_kw_partial.fade_friction)
        self.assertEqual(
            params.fitted_saturation_hessian.contact_friction,
            params_kw_partial.fitted_saturation_hessian.contact_friction,
        )
        self.assertEqual(
            params.fitted_saturation_hessian.joint_friction,
            params_kw_partial.fitted_saturation_hessian.joint_friction,
        )
        self.assertEqual(
            params.fitted_saturation_hessian.constraint_saturation,
            params_kw_partial.fitted_saturation_hessian.constraint_saturation,
        )

        # Assign every field (to show we can)
        params.explicit_normals = True
        params.fade_friction = False
        params.implicit_normal_force_for_dissipation = True
        params.fitted_saturation_hessian = mochi.SaturationHessianParams(
            contact_friction=False, joint_friction=False, constraint_saturation=False
        )

    def test_solver_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.SolverParams()
        self.assertEqual(4, params.non_linear_solver.max_iter)
        self.assertEqual(1000, params.linear_solver.restart_size)

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.SolverParams(
            non_linear_solver=mochi.NonLinearSolverParams(max_iter=10),
            experimental_eval=mochi.ExperimentalEvalParams(),
            integration_method=mochi.IntegrationMethod.BACKWARD_EULER,
            linear_solver=mochi.LinearSolverParams(max_iter=500),
        )
        self.assertEqual(10, params_kw_all.non_linear_solver.max_iter)
        self.assertEqual(500, params_kw_all.linear_solver.max_iter)
        self.assertEqual(
            mochi.IntegrationMethod.BACKWARD_EULER,
            params_kw_all.integration_method,
        )

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.SolverParams(
            integration_method=mochi.IntegrationMethod.BDF2,
        )
        self.assertEqual(
            mochi.IntegrationMethod.BDF2, params_kw_partial.integration_method
        )
        self.assertEqual(4, params_kw_partial.non_linear_solver.max_iter)
        self.assertEqual(1000, params_kw_partial.linear_solver.restart_size)

        # Test keyword argument constructor with nested param customization in shuffled order
        params_kw_nested = mochi.SolverParams(
            integration_method=mochi.IntegrationMethod.BDF3,
            linear_solver=mochi.LinearSolverParams(max_iter=2000, abs_tol=1e-6),
            non_linear_solver=mochi.NonLinearSolverParams(max_iter=20, abs_tol=1e-3),
        )
        self.assertEqual(20, params_kw_nested.non_linear_solver.max_iter)
        self.assertAlmostEqual(
            1e-3, params_kw_nested.non_linear_solver.abs_tol, places=5
        )
        self.assertEqual(2000, params_kw_nested.linear_solver.max_iter)
        self.assertAlmostEqual(1e-6, params_kw_nested.linear_solver.abs_tol, places=7)
        self.assertEqual(
            mochi.IntegrationMethod.BDF3, params_kw_nested.integration_method
        )

        # Assign every field (to show we can)
        params.non_linear_solver = mochi.NonLinearSolverParams()
        params.linear_solver = mochi.LinearSolverParams()
        params.integration_method = mochi.IntegrationMethod.DEFAULT
        params.experimental_eval = mochi.ExperimentalEvalParams()

    def test_recentering_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.RecenteringParams()
        self.assertTrue(params.use_recentering)
        self.assertEqual(0, params.rotation_epsilon_deg)

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.RecenteringParams(
            use_recentering=False,
            translation_epsilon=0.01,
            rotation_epsilon_deg=5,
        )
        self.assertFalse(params_kw_all.use_recentering)
        self.assertAlmostEqual(5.0, params_kw_all.rotation_epsilon_deg, places=5)
        self.assertAlmostEqual(0.01, params_kw_all.translation_epsilon, places=5)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.RecenteringParams(
            use_recentering=False,
        )
        self.assertFalse(params_kw_partial.use_recentering)
        self.assertEqual(
            params.rotation_epsilon_deg, params_kw_partial.rotation_epsilon_deg
        )

        # Assign every field (to show we can)
        params.use_recentering = True
        params.rotation_epsilon_deg = 0
        params.translation_epsilon = 0

    def test_boundary_subsampling_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.BoundarySubsamplingParams()
        self.assertAlmostEqual(1.0, params.subsampling_density)
        self.assertEqual(
            mochi.BoundarySubsamplingStrategy.UNIFORM_PROBABILITY, params.strategy
        )

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.BoundarySubsamplingParams(
            strategy=mochi.BoundarySubsamplingStrategy.AREA_PROPORTIONAL,
            subsampling_density=0.5,
        )
        self.assertAlmostEqual(0.5, params_kw_all.subsampling_density)
        self.assertEqual(
            mochi.BoundarySubsamplingStrategy.AREA_PROPORTIONAL, params_kw_all.strategy
        )

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.BoundarySubsamplingParams(
            subsampling_density=0.75,
        )
        self.assertAlmostEqual(0.75, params_kw_partial.subsampling_density)
        self.assertEqual(
            params.strategy,
            params_kw_partial.strategy,
        )

        # Assign every field (to show we can)
        params.subsampling_density = 1.0
        params.strategy = mochi.BoundarySubsamplingStrategy.UNIFORM_PROBABILITY

    def test_pose_tracking_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.PoseTrackingParams()
        self.assertEqual(0, params.stiffness)
        self.assertAlmostEqual(-1, params.saturation)

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.PoseTrackingParams(
            stiffness=1000,
            damping=10,
            saturation=0.1,
        )
        self.assertAlmostEqual(1000.0, params_kw_all.stiffness, places=5)
        self.assertAlmostEqual(10.0, params_kw_all.damping, places=5)
        self.assertAlmostEqual(0.1, params_kw_all.saturation, places=5)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.PoseTrackingParams(
            stiffness=500.0,
        )
        self.assertAlmostEqual(500.0, params_kw_partial.stiffness, places=5)
        self.assertEqual(params.damping, params_kw_partial.damping)
        self.assertEqual(params.saturation, params_kw_partial.saturation)

        # Assign every field (to show we can)
        params.stiffness = 0
        params.damping = 0
        params.saturation = -1

    def test_constraint_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.ConstraintParams()
        self.assertAlmostEqual(1e6, params.stiffness)
        self.assertAlmostEqual(-1, params.saturation)

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.ConstraintParams(
            stiffness=2e6,
            saturation=0.5,
            damping=100,
        )
        self.assertAlmostEqual(2e6, params_kw_all.stiffness)
        self.assertAlmostEqual(100.0, params_kw_all.damping)
        self.assertAlmostEqual(0.5, params_kw_all.saturation)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.ConstraintParams(
            stiffness=5e5,
        )
        self.assertAlmostEqual(5e5, params_kw_partial.stiffness)
        self.assertEqual(params.damping, params_kw_partial.damping)
        self.assertEqual(params.saturation, params_kw_partial.saturation)

        # Assign every field (to show we can)
        params.stiffness = 1e6
        params.damping = 0
        params.saturation = -1

    def test_ik_solver_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.experimental.IKSolverParams()
        self.assertEqual(20, params.max_iter)
        self.assertAlmostEqual(1e-2, params.abs_tol)

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.experimental.IKSolverParams(
            max_iter=50,
            position_error_thres=1e-3,
            verbosity=mochi.VerbosityLevel.SILENT,
            abs_tol=1e-3,
            rotation_error_thres=1e-3,
            rel_tol=1e-9,
        )
        self.assertEqual(50, params_kw_all.max_iter)
        self.assertEqual(mochi.VerbosityLevel.SILENT, params_kw_all.verbosity)
        self.assertAlmostEqual(1e-3, params_kw_all.abs_tol)
        self.assertAlmostEqual(1e-9, params_kw_all.rel_tol)
        self.assertAlmostEqual(1e-3, params_kw_all.position_error_thres)
        self.assertAlmostEqual(1e-3, params_kw_all.rotation_error_thres)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.experimental.IKSolverParams(
            max_iter=200,
            abs_tol=1e-4,
        )
        self.assertEqual(200, params_kw_partial.max_iter)
        self.assertAlmostEqual(1e-4, params_kw_partial.abs_tol)
        self.assertEqual(params.verbosity, params_kw_partial.verbosity)
        self.assertEqual(params.rel_tol, params_kw_partial.rel_tol)

        # Assign every field (to show we can)
        params.max_iter = 100
        params.verbosity = mochi.VerbosityLevel.ERROR
        params.abs_tol = 1e-2
        params.rel_tol = 1e-8
        params.position_error_thres = 1e-2
        params.rotation_error_thres = 1e-2

    def test_recording_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.RecordingParams()
        self.assertTrue(params.record_actor_meshes)
        self.assertFalse(params.record_actor_local_to_global_map)
        self.assertFalse(params.record_actor_mass_matrix)

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.RecordingParams(
            record_actor_meshes=False,
            record_dynamic_actor_state=False,
            record_contact_points=True,
            record_actor_local_to_global_map=True,
            record_actor_mass_matrix=True,
            record_target_state=False,
            record_static_actor_state=True,
            record_sdf_distances=True,
            record_node_contact_forces=True,
        )
        self.assertFalse(params_kw_all.record_actor_meshes)
        self.assertTrue(params_kw_all.record_actor_local_to_global_map)
        self.assertTrue(params_kw_all.record_actor_mass_matrix)
        self.assertFalse(params_kw_all.record_target_state)
        self.assertFalse(params_kw_all.record_dynamic_actor_state)
        self.assertTrue(params_kw_all.record_static_actor_state)
        self.assertTrue(params_kw_all.record_contact_points)
        self.assertTrue(params_kw_all.record_node_contact_forces)
        self.assertTrue(params_kw_all.record_sdf_distances)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.RecordingParams(
            record_contact_points=True,
            record_sdf_distances=True,
        )
        self.assertTrue(params_kw_partial.record_contact_points)
        self.assertTrue(params_kw_partial.record_sdf_distances)
        self.assertEqual(
            params.record_actor_meshes, params_kw_partial.record_actor_meshes
        )

        # Assign every field (to show we can)
        params.record_actor_meshes = True
        params.record_actor_local_to_global_map = False
        params.record_actor_mass_matrix = False
        params.record_target_state = True
        params.record_dynamic_actor_state = True
        params.record_static_actor_state = False
        params.record_contact_points = False
        params.record_node_contact_forces = False
        params.record_sdf_distances = False

    def test_skinning_data(self):
        # Check default values
        data = mochi.SkinningData()
        self.assertEqual(0, data.weights_per_node)
        self.assertEqual(0, len(data.weights))
        self.assertEqual(0, len(data.indices))
        self.assertEqual(data, mochi.SkinningData())

        # Set some values
        data.weights_per_node = 3
        data.weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        data.indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.assertEqual(3, data.weights_per_node)
        self.assertListEqual(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], list(data.weights)
        )
        self.assertListEqual([0, 1, 2, 3, 4, 5, 6, 7, 8], list(data.indices))
        self.assertNotEqual(data, mochi.SkinningData())

        # Check default value of SkinningDataView
        view = mochi.SkinningDataView()
        self.assertEqual(0, view.weights_per_node)
        self.assertEqual(0, len(view.weights))
        self.assertEqual(0, len(view.indices))
        self.assertEqual(view, mochi.SkinningDataView())

        # Check conversions
        view = mochi.SkinningDataView(src=data)
        data2 = mochi.SkinningData(src=view)
        view2 = mochi.SkinningDataView(src=data2)
        self.assertEqual(data, data2)
        self.assertEqual(view, view2)

        # Modify data and observe the changes in view
        data.indices[0] = 123
        self.assertListEqual([123, 1, 2, 3, 4, 5, 6, 7, 8], list(data.indices))
        self.assertListEqual([123, 1, 2, 3, 4, 5, 6, 7, 8], list(view.indices))

        # Modify data2 and observe the changes in view2 (independent of data and view)
        data2.indices[0] = 456
        self.assertListEqual([456, 1, 2, 3, 4, 5, 6, 7, 8], list(data2.indices))
        self.assertListEqual([456, 1, 2, 3, 4, 5, 6, 7, 8], list(view2.indices))

        # The SkinningData object should not be garbage collected while the SkinningDataView is alive.
        gc.collect()
        self.assertListEqual([123, 1, 2, 3, 4, 5, 6, 7, 8], list(view.indices))
        self.assertListEqual([456, 1, 2, 3, 4, 5, 6, 7, 8], list(view2.indices))

    def test_per_element_soft_material_data(self):
        # Set some values
        data = mochi.PerElementSoftMaterialData()
        data.type = mochi.SoftMaterialType.LINEAR_ELASTIC
        data.psd_strategy = mochi.MaterialPsdStrategy.PROJECTION
        data.youngs_modulus = [1.0, 2.0, 3.0]
        data.poisson_ratio = [0.1, 0.2, 0.3]
        data.aniso_alpha = [10.0, 20.0, 30.0]
        data.aniso_length = [0.5, 0.6, 0.7]
        data.aniso_theta = [0.0, 0.1, 0.2]
        data.aniso_phi = [0.3, 0.4, 0.5]
        data.arap_stiffness = [100.0, 200.0, 300.0]
        data.shape_target_tensor = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        # Read them back
        self.assertEqual(mochi.SoftMaterialType.LINEAR_ELASTIC, data.type)
        self.assertEqual(mochi.MaterialPsdStrategy.PROJECTION, data.psd_strategy)
        self.assertListEqual(
            list(np.array([1.0, 2.0, 3.0], dtype=np_real)),
            list(data.youngs_modulus),
        )
        self.assertListEqual(
            list(np.array([0.1, 0.2, 0.3], dtype=np_real)),
            list(data.poisson_ratio),
        )
        self.assertListEqual(
            list(np.array([10.0, 20.0, 30.0], dtype=np_real)),
            list(data.aniso_alpha),
        )
        self.assertListEqual(
            list(np.array([0.5, 0.6, 0.7], dtype=np_real)),
            list(data.aniso_length),
        )
        self.assertListEqual(
            list(np.array([0.0, 0.1, 0.2], dtype=np_real)),
            list(data.aniso_theta),
        )
        self.assertListEqual(
            list(np.array([0.3, 0.4, 0.5], dtype=np_real)),
            list(data.aniso_phi),
        )
        self.assertListEqual(
            list(np.array([100.0, 200.0, 300.0], dtype=np_real)),
            list(data.arap_stiffness),
        )
        self.assertListEqual(
            list(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np_real)),
            list(data.shape_target_tensor),
        )
        self.assertNotEqual(data, mochi.PerElementSoftMaterialData())

        # Check conversions for PerElementSoftMaterialData <--> PerElementSoftMaterialDataView
        view = mochi.PerElementSoftMaterialDataView(src=data)
        data2 = mochi.PerElementSoftMaterialData(src=view)
        view2 = mochi.PerElementSoftMaterialDataView(src=data2)
        self.assertEqual(data, data2)
        self.assertEqual(view, view2)

        # Modify data and observe the changes in view
        data.youngs_modulus[0] = 123.0
        self.assertAlmostEqual(123.0, list(data.youngs_modulus)[0])
        self.assertAlmostEqual(123.0, list(view.youngs_modulus)[0])

        # Modify data2 and observe the changes in view2 (independent of data and view)
        data2.youngs_modulus[0] = 456.0
        self.assertAlmostEqual(456.0, list(data2.youngs_modulus)[0])
        self.assertAlmostEqual(456.0, list(view2.youngs_modulus)[0])

        # The PerElementSoftMaterialData object should not be garbage collected while the
        # PerElementSoftMaterialDataView is alive.
        gc.collect()
        self.assertAlmostEqual(123.0, list(view.youngs_modulus)[0])
        self.assertAlmostEqual(456.0, list(view2.youngs_modulus)[0])

    def test_blending_data(self):
        # Check default values
        data = mochi.BlendingData()
        self.assertEqual("", data.source_shape)
        self.assertEqual(0, len(data.indices))
        self.assertEqual(0, len(data.weights))
        self.assertEqual(data, mochi.BlendingData())

        # Set some values
        data.source_shape = "my_source_shape"
        data.indices = [0, 1, 2, 3, 4, 5]
        data.weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        self.assertEqual("my_source_shape", data.source_shape)
        self.assertListEqual([0, 1, 2, 3, 4, 5], list(data.indices))
        self.assertListEqual(
            list(np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np_real)),
            list(data.weights),
        )
        self.assertNotEqual(data, mochi.BlendingData())

        # Keyword construction
        data2 = mochi.BlendingData(
            source_shape="other_shape",
            indices=[10, 20],
            weights=[0.7, 0.8],
        )
        self.assertEqual("other_shape", data2.source_shape)
        self.assertListEqual([10, 20], list(data2.indices))

        # Check default value of BlendingDataView
        view = mochi.BlendingDataView()
        self.assertEqual("", view.source_shape)
        self.assertEqual(0, len(view.indices))
        self.assertEqual(0, len(view.weights))
        self.assertEqual(view, mochi.BlendingDataView())

        # Check conversions
        view = mochi.BlendingDataView(src=data)
        data_copy = mochi.BlendingData(src=view)
        view2 = mochi.BlendingDataView(src=data_copy)
        self.assertEqual(data, data_copy)
        self.assertEqual(view, view2)

        # Modify data and observe the changes in view
        data.indices[0] = 123
        self.assertListEqual([123, 1, 2, 3, 4, 5], list(data.indices))
        self.assertListEqual([123, 1, 2, 3, 4, 5], list(view.indices))

        # Modify data_copy and observe the changes in view2 (independent of data and view)
        data_copy.indices[0] = 456
        self.assertListEqual([456, 1, 2, 3, 4, 5], list(data_copy.indices))
        self.assertListEqual([456, 1, 2, 3, 4, 5], list(view2.indices))

        # The BlendingData object should not be garbage collected while the BlendingDataView is alive.
        gc.collect()
        self.assertListEqual([123, 1, 2, 3, 4, 5], list(view.indices))
        self.assertListEqual([456, 1, 2, 3, 4, 5], list(view2.indices))

        # copy.copy / copy.deepcopy: nested DynamicArrays are deep-copied via
        # the C++ copy ctor, so copies are fully independent of the source.
        original = mochi.BlendingData(
            source_shape="orig", indices=[1, 2, 3], weights=[0.5, 0.6, 0.7]
        )
        for c in (copy.copy(original), copy.deepcopy(original)):
            self.assertIsInstance(c, mochi.BlendingData)
            self.assertEqual(c.source_shape, "orig")
            self.assertEqual(c.weights[0], np_real(0.5))
            # In-place mutation of the source must not affect the copy.
            original.weights[0] = 99.0
            self.assertEqual(c.weights[0], np_real(0.5))
            original.weights[0] = 0.5  # restore for next iteration

        # BlendingDataView contains non-owning members (Span, StringView), so
        # it can't be deep-copied.
        view = mochi.BlendingDataView(src=original)
        with self.assertRaises(TypeError):
            copy.copy(view)  # We could support shallow copy, but choose not to for now.
        with self.assertRaises(TypeError):
            copy.deepcopy(view)

    def test_mesh_data(self):
        # Check default values
        data = mochi.MeshData()
        self.assertEqual(0, data.nodes_per_element)
        self.assertEqual(0, len(data.coordinates))
        self.assertEqual(0, len(data.connectivity))
        self.assertIsNone(data.skinning)
        self.assertEqual(data, mochi.MeshData())
        self.assertEqual(0, data.get_num_elements())
        self.assertEqual(0, data.get_num_nodes())

        # Set some values
        data.nodes_per_element = 3
        data.coordinates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        data.connectivity = [0, 1, 2]
        self.assertEqual(3, data.nodes_per_element)
        self.assertListEqual(
            list(
                np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np_real)
            ),
            list(data.coordinates),
        )
        self.assertListEqual([0, 1, 2], list(data.connectivity))
        self.assertEqual(1, data.get_num_elements())
        self.assertEqual(3, data.get_num_nodes())
        self.assertNotEqual(data, mochi.MeshData())

        # Set skinning data
        skinning = mochi.SkinningData(
            weights_per_node=2,
            indices=[0, 1, 2, 3],
            weights=[0.5, 0.5, 0.3, 0.7],
        )
        data.skinning = skinning
        self.assertIsNotNone(data.skinning)
        self.assertEqual(2, data.skinning.weights_per_node)

        # Keyword construction
        data2 = mochi.MeshData(
            nodes_per_element=4,
            coordinates=[1.0, 2.0, 3.0],
            connectivity=[0, 1, 2, 3],
        )
        self.assertEqual(4, data2.nodes_per_element)
        self.assertListEqual([0, 1, 2, 3], list(data2.connectivity))

        # Check default value of MeshDataView
        view = mochi.MeshDataView()
        self.assertEqual(0, view.nodes_per_element)
        self.assertEqual(0, len(view.coordinates))
        self.assertEqual(0, len(view.connectivity))
        self.assertEqual(0, view.get_num_elements())
        self.assertEqual(0, view.get_num_nodes())
        self.assertIsNone(view.skinning)
        self.assertEqual(view, mochi.MeshDataView())

        # Check conversions
        view = mochi.MeshDataView(src=data)
        data2 = mochi.MeshData(src=view)
        view2 = mochi.MeshDataView(src=data2)
        self.assertEqual(data, data2)
        self.assertEqual(view, view2)
        self.assertEqual(1, view.get_num_elements())
        self.assertEqual(3, view.get_num_nodes())
        self.assertEqual(1, view2.get_num_elements())
        self.assertEqual(3, view2.get_num_nodes())

        # Modify data and observe the changes in view
        data.connectivity[0] = 123
        self.assertListEqual([123, 1, 2], list(data.connectivity))
        self.assertListEqual([123, 1, 2], list(view.connectivity))

        # Modify data2 and observe the changes in view2 (independent of data and view)
        data2.connectivity[0] = 456
        self.assertListEqual([456, 1, 2], list(data2.connectivity))
        self.assertListEqual([456, 1, 2], list(view2.connectivity))

        # The MeshData object should not be garbage collected while the MeshDataView is alive.
        gc.collect()
        self.assertListEqual([123, 1, 2], list(view.connectivity))
        self.assertListEqual([456, 1, 2], list(view2.connectivity))

    def test_grid_sdf_data(self):
        # Check default values
        data = mochi.GridSdfData()
        self.assertEqual(mochi.Int3(0, 0, 0), data.dims)
        self.assertEqual(0, len(data.values))
        self.assertEqual(mochi.Aabb(), data.bounds)
        self.assertEqual(mochi.Aabb(), data.negative_value_bounds)
        self.assertIsNone(data.scale)
        self.assertIsNone(data.rotation)
        self.assertIsNone(data.translation)
        self.assertEqual(data, mochi.GridSdfData())

        # Set some values
        data.dims = mochi.Int3(2, 3, 4)
        data.values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        data.bounds = mochi.Aabb([-1, -2, -3], [4, 5, 6])
        data.negative_value_bounds = mochi.Aabb([-0.5, -1.0, -1.5], [2, 3, 4])
        data.scale = [1.2, 2.3, 3.4]
        data.rotation = mochi.Quaternion(0, 0, 0, 1)
        data.translation = mochi.Real3(1, 2, 3)
        self.assertEqual(mochi.Int3(2, 3, 4), data.dims)
        self.assertEqual(8, len(data.values))
        self.assertEqual(mochi.Aabb([-1, -2, -3], [4, 5, 6]), data.bounds)
        self.assertAlmostEqual([1.2, 2.3, 3.4], data.scale)
        self.assertEqual(mochi.Quaternion(0, 0, 0, 1), data.rotation)
        self.assertEqual(mochi.Real3(1, 2, 3), data.translation)
        self.assertNotEqual(data, mochi.GridSdfData())

        # Test clearing optional fields
        data.scale = None
        data.rotation = None
        data.translation = None
        self.assertIsNone(data.scale)
        self.assertIsNone(data.rotation)
        self.assertIsNone(data.translation)

        # Keyword construction
        data = mochi.GridSdfData(
            dims=mochi.Int3(2, 2, 2),
            values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            bounds=mochi.Aabb([0, 0, 0], [1, 1, 1]),
            negative_value_bounds=mochi.Aabb([0, 0, 0], [1, 1, 1]),
            scale=[0.1, -0.2, 0.3],
            rotation=mochi.Quaternion(0, 0, 0, 1),
            translation=mochi.Real3(0, 0, 0),
        )
        self.assertEqual(mochi.Int3(2, 2, 2), data.dims)
        self.assertEqual(8, len(data.values))
        self.assertAlmostEqual([0.1, -0.2, 0.3], data.scale)

        # Check default value of GridSdfDataView
        view = mochi.GridSdfDataView()
        self.assertEqual(mochi.Int3(0, 0, 0), view.dims)
        self.assertEqual(0, len(view.values))
        self.assertIsNone(view.scale)
        self.assertIsNone(view.rotation)
        self.assertIsNone(view.translation)
        self.assertEqual(view, mochi.GridSdfDataView())

        # Check conversions
        view = mochi.GridSdfDataView(src=data)
        data2 = mochi.GridSdfData(src=view)
        view2 = mochi.GridSdfDataView(src=data2)
        self.assertEqual(data, data2)
        self.assertEqual(view, view2)

        # Modify data and observe the changes in view
        data.values[0] = 99.0
        self.assertAlmostEqual(99.0, float(data.values[0]))
        self.assertAlmostEqual(99.0, float(view.values[0]))

        # Modify data2 and observe the changes in view2 (independent of data and view)
        data2.values[0] = 77.0
        self.assertAlmostEqual(77.0, float(data2.values[0]))
        self.assertAlmostEqual(77.0, float(view2.values[0]))

        # The GridSdfData object should not be garbage collected while the GridSdfDataView is alive.
        gc.collect()
        self.assertAlmostEqual(99.0, float(view.values[0]))
        self.assertAlmostEqual(77.0, float(view2.values[0]))

    def test_box(self):
        # Default constructor
        box = mochi.Box()
        self.assertEqual(mochi.Real3(0, 0, 0), box.center)
        self.assertEqual(mochi.Real3(1, 1, 1), box.half_extents)
        self.assertEqual(mochi.Quaternion(0, 0, 0, 1), box.rotation)

        # Keyword construction
        box = mochi.Box(
            center=mochi.Real3(1, 2, 3),
            half_extents=mochi.Real3(0.5, 1.0, 1.5),
            rotation=mochi.Quaternion(0, 0, 0, 1),
        )
        self.assertEqual(mochi.Real3(1, 2, 3), box.center)
        self.assertEqual(mochi.Real3(0.5, 1.0, 1.5), box.half_extents)
        self.assertEqual(mochi.Quaternion(0, 0, 0, 1), box.rotation)

        # Set values
        box.center = mochi.Real3(4, 5, 6)
        box.half_extents = mochi.Real3(2, 3, 4)
        box.rotation = mochi.Quaternion(0.1, 0.2, 0.3, 0.4)
        self.assertEqual(mochi.Real3(4, 5, 6), box.center)
        self.assertEqual(mochi.Real3(2, 3, 4), box.half_extents)
        self.assertEqual(mochi.Quaternion(0.1, 0.2, 0.3, 0.4), box.rotation)

        # Equality
        box2 = mochi.Box(
            center=mochi.Real3(4, 5, 6),
            half_extents=mochi.Real3(2, 3, 4),
            rotation=mochi.Quaternion(0.1, 0.2, 0.3, 0.4),
        )
        self.assertEqual(box, box2)
        box2.center = mochi.Real3(0, 0, 0)
        self.assertNotEqual(box, box2)

    def test_plane(self):
        # Default constructor
        plane = mochi.Plane()
        self.assertEqual(mochi.Real3(0, 1, 0), plane.normal)
        self.assertAlmostEqual(0.0, plane.distance)

        # Keyword construction
        plane = mochi.Plane(
            normal=mochi.Real3(0, 0, 1),
            distance=5.0,
        )
        self.assertEqual(mochi.Real3(0, 0, 1), plane.normal)
        self.assertAlmostEqual(5.0, plane.distance)

        # Set values
        plane.normal = mochi.Real3(1, 0, 0)
        plane.distance = -3.0
        self.assertEqual(mochi.Real3(1, 0, 0), plane.normal)
        self.assertAlmostEqual(-3.0, plane.distance)

        # Equality
        plane2 = mochi.Plane(normal=mochi.Real3(1, 0, 0), distance=-3.0)
        self.assertEqual(plane, plane2)
        plane2.distance = 10.0
        self.assertNotEqual(plane, plane2)

    def test_sphere(self):
        # Default constructor
        sphere = mochi.Sphere()
        self.assertEqual(mochi.Real3(0, 0, 0), sphere.center)
        self.assertAlmostEqual(0.0, sphere.radius)

        # Keyword construction
        sphere = mochi.Sphere(
            center=mochi.Real3(1, 2, 3),
            radius=5.0,
        )
        self.assertEqual(mochi.Real3(1, 2, 3), sphere.center)
        self.assertAlmostEqual(5.0, sphere.radius)

        # Set values
        sphere.center = mochi.Real3(-1, -2, -3)
        sphere.radius = 10.0
        self.assertEqual(mochi.Real3(-1, -2, -3), sphere.center)
        self.assertAlmostEqual(10.0, sphere.radius)

        # Equality
        sphere2 = mochi.Sphere(center=mochi.Real3(-1, -2, -3), radius=10.0)
        self.assertEqual(sphere, sphere2)
        sphere2.radius = 1.0
        self.assertNotEqual(sphere, sphere2)

    def test_model_data(self):
        # Check default values
        data = mochi.ModelData()
        self.assertIsNone(data.mesh)
        self.assertIsNone(data.visual_mesh)
        self.assertIsNone(data.blending)
        self.assertIsNone(data.constrained_nodes)
        self.assertIsNone(data.element_frame_axes)
        self.assertIsNone(data.box)
        self.assertIsNone(data.plane)
        self.assertIsNone(data.sphere)
        self.assertIsNone(data.sdf)
        self.assertIsNone(data.material)
        self.assertEqual(data, mochi.ModelData())

        # Set mesh data
        mesh = mochi.MeshData(
            nodes_per_element=3,
            coordinates=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            connectivity=[0, 1, 2],
        )
        data.mesh = mesh
        self.assertIsNotNone(data.mesh)
        self.assertEqual(3, data.mesh.nodes_per_element)
        self.assertListEqual([0, 1, 2], list(data.mesh.connectivity))

        # Set visual mesh
        visual_mesh = mochi.MeshData(
            nodes_per_element=3,
            coordinates=[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            connectivity=[0, 1, 2],
        )
        data.visual_mesh = visual_mesh
        self.assertIsNotNone(data.visual_mesh)

        # Set blending data
        blending = [
            mochi.BlendingData(
                source_shape="shape1",
                indices=[0, 1],
                weights=[0.5, 0.5],
            ),
            mochi.BlendingData(
                source_shape="shape2",
                indices=[2, 3],
                weights=[0.3, 0.7],
            ),
        ]
        data.blending = blending
        self.assertIsNotNone(data.blending)
        self.assertEqual(2, len(data.blending))
        self.assertEqual("shape1", data.blending[0].source_shape)
        self.assertEqual("shape2", data.blending[1].source_shape)

        # Set constrained nodes
        data.constrained_nodes = [0, 1, 2]
        self.assertIsNotNone(data.constrained_nodes)
        self.assertListEqual([0, 1, 2], list(data.constrained_nodes))

        # Set element frame axes
        data.element_frame_axes = [1.0, 0.0, 0.0]
        self.assertIsNotNone(data.element_frame_axes)

        # Set implicit geometry
        data.box = mochi.Box(half_extents=mochi.Real3(1, 2, 3))
        self.assertIsNotNone(data.box)
        self.assertEqual(mochi.Real3(1, 2, 3), data.box.half_extents)

        data.plane = mochi.Plane(normal=mochi.Real3(0, 1, 0), distance=5.0)
        self.assertIsNotNone(data.plane)

        data.sphere = mochi.Sphere(center=mochi.Real3(0, 0, 0), radius=1.0)
        self.assertIsNotNone(data.sphere)

        # Set SDF
        sdf = mochi.GridSdfData(
            dims=mochi.Int3(2, 2, 2),
            values=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            bounds=mochi.Aabb([-1, -1, -1], [1, 1, 1]),
        )
        data.sdf = sdf
        self.assertIsNotNone(data.sdf)
        self.assertEqual(mochi.Int3(2, 2, 2), data.sdf.dims)

        # Test clearing optional fields
        data.mesh = None
        data.visual_mesh = None
        data.blending = None
        data.constrained_nodes = None
        data.element_frame_axes = None
        data.box = None
        data.plane = None
        data.sphere = None
        data.sdf = None
        data.material = None
        self.assertIsNone(data.mesh)
        self.assertIsNone(data.visual_mesh)
        self.assertIsNone(data.blending)
        self.assertIsNone(data.constrained_nodes)
        self.assertIsNone(data.element_frame_axes)
        self.assertIsNone(data.box)
        self.assertIsNone(data.plane)
        self.assertIsNone(data.sphere)
        self.assertIsNone(data.sdf)
        self.assertIsNone(data.material)

        # Check default value of ModelDataView
        view = mochi.ModelDataView()
        self.assertIsNone(view.mesh)
        self.assertIsNone(view.visual_mesh)
        self.assertIsNone(view.blending)
        self.assertIsNone(view.constrained_nodes)
        self.assertIsNone(view.element_frame_axes)
        self.assertIsNone(view.box)
        self.assertIsNone(view.plane)
        self.assertIsNone(view.sphere)
        self.assertIsNone(view.sdf)
        self.assertIsNone(view.material)
        self.assertEqual(view, mochi.ModelDataView())

        # Check conversions with populated data
        data2 = mochi.ModelData()
        data2.mesh = mesh
        data2.box = mochi.Box(half_extents=mochi.Real3(1, 1, 1))
        view = mochi.ModelDataView(src=data2)
        data_copy = mochi.ModelData(src=view)
        view2 = mochi.ModelDataView(src=data_copy)
        self.assertEqual(data2, data_copy)
        self.assertEqual(view, view2)

        # Verify view reflects original data
        self.assertIsNotNone(view.mesh)
        self.assertEqual(3, view.mesh.nodes_per_element)
        self.assertIsNotNone(view.box)
        self.assertEqual(mochi.Real3(1, 1, 1), view.box.half_extents)

        # The ModelData object should not be garbage collected while the ModelDataView is alive.
        gc.collect()
        self.assertIsNotNone(view.mesh)
        self.assertIsNotNone(view2.mesh)

    def test_rigid_actor_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.RigidActorParams()
        self.assertEqual("", params.name)
        self.assertEqual(mochi.TransformRT(), params.world_from_local)
        self.assertEqual(mochi.ColliderType.AUTO, params.collider_type)
        self.assertFalse(params.is_static)
        self.assertIsNone(params.density)
        self.assertIsNone(params.mass)
        self.assertIsNone(params.center_of_mass)
        self.assertIsNone(params.moment_of_inertia)
        self.assertIsNone(params.boundary_subsampling)

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.RigidActorParams(
            has_gravity=False,
            is_static=True,
            name="TestRigid",
            angular_velocity=None,
            collider_type=mochi.ColliderType.BOX,
            layer="MyLayer",
            shape=mochi.ShapeHandle(),
            world_from_local=mochi.TransformRT(translation=[1, 2, 3]),
            contact=mochi.ContactParams(penalty_coefficient=5e8),
            sdf=mochi.GridSdfParams(resolution_delta=[0.1, 0.2, 0.3]),
            boundary_element_type=mochi.ActorBoundaryElementType.P1Q6,
            center_of_mass=[0.1, 0.2, 0.3],
        )
        self.assertEqual("TestRigid", params_kw_all.name)
        self.assertEqual("MyLayer", params_kw_all.layer)
        self.assertEqual(
            mochi.Real3(1, 2, 3), params_kw_all.world_from_local.translation
        )
        self.assertEqual(mochi.ColliderType.BOX, params_kw_all.collider_type)
        self.assertTrue(params_kw_all.is_static)
        self.assertFalse(params_kw_all.has_gravity)
        self.assertAlmostEqual(5e8, params_kw_all.contact.penalty_coefficient)
        self.assertAlmostEqual(0.1, params_kw_all.sdf.resolution_delta[0])
        self.assertAlmostEqual(0.2, params_kw_all.sdf.resolution_delta[1])
        self.assertAlmostEqual(0.3, params_kw_all.sdf.resolution_delta[2])
        self.assertEqual(
            mochi.ActorBoundaryElementType.P1Q6, params_kw_all.boundary_element_type
        )
        self.assertEqual(mochi.Real3(0.1, 0.2, 0.3), params_kw_all.center_of_mass)
        self.assertEqual(None, params_kw_all.angular_velocity)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.RigidActorParams(
            name="PartialRigid",
            is_static=True,
        )
        self.assertEqual("PartialRigid", params_kw_partial.name)
        self.assertTrue(params_kw_partial.is_static)
        self.assertEqual(params.layer, params_kw_partial.layer)
        self.assertEqual(params.has_gravity, params_kw_partial.has_gravity)

        # Assign every field (to show we can)
        params.name = "My Name"
        params.layer = "My Layer"
        params.shape = mochi.ShapeHandle()
        params.world_from_local = mochi.TransformRT()
        params.collider_type = mochi.ColliderType.BOX
        params.is_static = False
        params.contact = mochi.ContactParams()
        params.sdf = mochi.GridSdfParams()
        params.has_gravity = True
        params.density = 1.23
        params.mass = 2.34
        params.center_of_mass = mochi.Real3()
        params.moment_of_inertia = mochi.Real6()
        params.boundary_element_type = mochi.ActorBoundaryElementType.DEFAULT
        params.boundary_subsampling = mochi.BoundarySubsamplingParams()
        params.boundary_subsampling.subsampling_density = 1.0
        params.boundary_subsampling.strategy = (
            mochi.BoundarySubsamplingStrategy.UNIFORM_PROBABILITY
        )

    def test_soft_actor_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.SoftActorParams()
        self.assertEqual("", params.name)
        self.assertEqual(mochi.TransformRT(), params.world_from_local)
        self.assertTrue(params.has_gravity)
        self.assertTrue(params.has_inertia)
        self.assertTrue(params.has_stress)

        # Test keyword argument constructor with all fields in shuffled order.
        params_kw_all = mochi.SoftActorParams(
            has_stress=False,
            name="TestSoft",
            layer="SoftLayer",
            world_from_local=mochi.TransformRT(translation=[1, 2, 3]),
            contact=mochi.ContactParams(penalty_coefficient=5e8),
            boundary_element_type=mochi.ActorBoundaryElementType.P1Q6,
            has_gravity=False,
            has_inertia=False,
        )
        self.assertEqual("TestSoft", params_kw_all.name)
        self.assertEqual("SoftLayer", params_kw_all.layer)
        self.assertEqual(
            mochi.Real3(1, 2, 3), params_kw_all.world_from_local.translation
        )
        self.assertFalse(params_kw_all.has_gravity)
        self.assertFalse(params_kw_all.has_inertia)
        self.assertFalse(params_kw_all.has_stress)
        self.assertAlmostEqual(5e8, params_kw_all.contact.penalty_coefficient)
        self.assertEqual(
            mochi.ActorBoundaryElementType.P1Q6, params_kw_all.boundary_element_type
        )

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.SoftActorParams(
            name="PartialSoft",
            has_gravity=False,
        )
        self.assertEqual("PartialSoft", params_kw_partial.name)
        self.assertFalse(params_kw_partial.has_gravity)
        self.assertEqual(params.layer, params_kw_partial.layer)

        # Assign every field (to show we can)
        params.name = "My Name"
        params.layer = "My Layer"
        params.world_from_local = mochi.TransformRT()
        params.shape = mochi.ShapeHandle()
        params.material = mochi.SoftMaterialParams()
        params.contact = mochi.ContactParams()
        params.has_gravity = False
        params.has_inertia = False
        params.has_stress = False
        params.boundary_element_type = mochi.ActorBoundaryElementType.DEFAULT

    def test_articulated_joint_friction_params(self):
        # Test default construction is possible, but don't hard-code all the defaults here.
        params = mochi.ArticulatedJointFrictionParams()
        self.assertAlmostEqual(0.0, params.coulomb)

        # Test keyword constructor with all fields
        params_kw = mochi.ArticulatedJointFrictionParams(
            viscous=1.5,
            coulomb=2.5,
            falloff_vel=0.01,
            stiction_extra=0.3,
            stribeck_vel=0.05,
        )
        self.assertAlmostEqual(1.5, params_kw.viscous)
        self.assertAlmostEqual(2.5, params_kw.coulomb)
        self.assertAlmostEqual(0.01, params_kw.falloff_vel)
        self.assertAlmostEqual(0.3, params_kw.stiction_extra)
        self.assertAlmostEqual(0.05, params_kw.stribeck_vel)

        # Test equality operators
        params_copy = mochi.ArticulatedJointFrictionParams(
            viscous=1.5,
            coulomb=2.5,
            falloff_vel=0.01,
            stiction_extra=0.3,
            stribeck_vel=0.05,
        )
        self.assertEqual(params_kw, params_copy)
        self.assertNotEqual(params, params_kw)

        # Assign every field
        params.viscous = 10.0
        params.coulomb = 20.0
        params.falloff_vel = 0.1
        params.stiction_extra = 5.0
        params.stribeck_vel = 3.0
        self.assertAlmostEqual(10.0, params.viscous)
        self.assertAlmostEqual(20.0, params.coulomb)
        self.assertAlmostEqual(0.1, params.falloff_vel)
        self.assertAlmostEqual(5.0, params.stiction_extra)
        self.assertAlmostEqual(3.0, params.stribeck_vel)

    def test_articulated_joint_params(self):
        # Test default construction is possible, but don't hard-code all the defaults here.
        params = mochi.ArticulatedJointParams()
        self.assertEqual("", params.name)

        # Test keyword constructor with all fields
        friction = mochi.ArticulatedJointFrictionParams(viscous=0.5, coulomb=1.0)
        params_kw = mochi.ArticulatedJointParams(
            name="MyJoint",
            type=mochi.ArticulatedJointType.REVOLUTE,
            parent_link_from_joint=mochi.TransformRT(translation=[1, 2, 3]),
            axis=[0, 0, 1],
            friction=friction,
            inertia=2.5,
            min_limit=[-1.0, 0, 0],
            max_limit=[1.0, 0, 0],
            limit_stiffness=100.0,
            limit_damping=10.0,
        )
        self.assertEqual("MyJoint", params_kw.name)
        self.assertEqual(mochi.ArticulatedJointType.REVOLUTE, params_kw.type)
        self.assertEqual(
            mochi.Real3(1, 2, 3), params_kw.parent_link_from_joint.translation
        )
        self.assertEqual(mochi.Real3(0, 0, 1), params_kw.axis)
        self.assertEqual(friction, params_kw.friction)
        self.assertAlmostEqual(2.5, params_kw.inertia)
        self.assertEqual(mochi.Real3(-1, 0, 0), params_kw.min_limit)
        self.assertEqual(mochi.Real3(1, 0, 0), params_kw.max_limit)
        self.assertAlmostEqual(100.0, params_kw.limit_stiffness)
        self.assertAlmostEqual(10.0, params_kw.limit_damping)

        # Test partial keyword construction
        params_partial = mochi.ArticulatedJointParams(
            name="PartialJoint",
            type=mochi.ArticulatedJointType.PRISMATIC,
        )
        self.assertEqual("PartialJoint", params_partial.name)
        self.assertEqual(mochi.ArticulatedJointType.PRISMATIC, params_partial.type)
        self.assertIsNone(params_partial.inertia)

        # Assign every field
        params.name = "Updated"
        params.type = mochi.ArticulatedJointType.SPHERICAL
        params.parent_link_from_joint.translation = [4, 5, 6]
        params.axis = mochi.Real3(1, 0, 0)
        params.friction = mochi.ArticulatedJointFrictionParams(viscous=3.0)
        params.inertia = 5.0
        params.min_limit = mochi.Real3(-2, 0, 0)
        params.max_limit = mochi.Real3(2, 0, 0)
        params.limit_stiffness = 200.0
        params.limit_damping = 20.0
        self.assertEqual("Updated", params.name)
        self.assertEqual(mochi.ArticulatedJointType.SPHERICAL, params.type)
        self.assertAlmostEqual(5.0, params.inertia)
        self.assertAlmostEqual(200.0, params.limit_stiffness)

    def test_articulated_link_params(self):
        # Test default construction is possible, but don't hard-code all the defaults here.
        params = mochi.ArticulatedLinkParams()
        self.assertEqual("", params.name)

        # Test keyword constructor with all fields
        contact = mochi.ContactParams(penalty_coefficient=5e8)
        params_kw = mochi.ArticulatedLinkParams(
            name="MyLink",
            parent_link=2,
            parent_joint_from_link=mochi.TransformRT(translation=[1, 0, 0]),
            shape=mochi.ShapeHandle(value=42),
            layer="LinkLayer",
            collider_type=mochi.ColliderType.BOX,
            contact=contact,
            density=1000.0,
            mass=5.0,
            center_of_mass=[0.1, 0.2, 0.3],
            moment_of_inertia=[1, 0, 0, 1, 0, 1],
            boundary_element_type=mochi.ActorBoundaryElementType.P1Q6,
            boundary_subsampling=mochi.BoundarySubsamplingParams(),
        )
        self.assertEqual("MyLink", params_kw.name)
        self.assertEqual(2, params_kw.parent_link)
        self.assertEqual(
            mochi.Real3(1, 0, 0), params_kw.parent_joint_from_link.translation
        )
        self.assertEqual(mochi.ShapeHandle(value=42), params_kw.shape)
        self.assertEqual("LinkLayer", params_kw.layer)
        self.assertEqual(mochi.ColliderType.BOX, params_kw.collider_type)
        self.assertAlmostEqual(5e8, params_kw.contact.penalty_coefficient)
        self.assertAlmostEqual(1000.0, params_kw.density)
        self.assertAlmostEqual(5.0, params_kw.mass)
        self.assertEqual(mochi.Real3(0.1, 0.2, 0.3), params_kw.center_of_mass)
        self.assertEqual(mochi.Real6(1, 0, 0, 1, 0, 1), params_kw.moment_of_inertia)
        self.assertEqual(
            mochi.ActorBoundaryElementType.P1Q6, params_kw.boundary_element_type
        )
        self.assertIsNotNone(params_kw.boundary_subsampling)

        # Test partial keyword construction
        params_partial = mochi.ArticulatedLinkParams(
            name="PartialLink",
            parent_link=0,
        )
        self.assertEqual("PartialLink", params_partial.name)
        self.assertEqual(0, params_partial.parent_link)
        self.assertIsNone(params_partial.density)

        # Assign every field
        params.name = "Updated"
        params.parent_link = 3
        params.parent_joint_from_link.translation = [7, 8, 9]
        params.shape = mochi.ShapeHandle(value=99)
        params.layer = "NewLayer"
        params.collider_type = mochi.ColliderType.MESH
        params.contact = mochi.ContactParams()
        params.density = 2000.0
        params.mass = 10.0
        params.center_of_mass = mochi.Real3(0, 0, 0)
        params.moment_of_inertia = mochi.Real6(2, 0, 0, 2, 0, 2)
        params.boundary_element_type = mochi.ActorBoundaryElementType.DEFAULT
        params.boundary_subsampling = mochi.BoundarySubsamplingParams()
        self.assertEqual("Updated", params.name)
        self.assertEqual(3, params.parent_link)
        self.assertAlmostEqual(2000.0, params.density)

    def test_articulated_skin_params(self):
        # Test default construction is possible, but don't hard-code all the defaults here.
        params = mochi.ArticulatedSkinParams()
        self.assertEqual("", params.layer)

        # Test keyword constructor with all fields
        params_kw = mochi.ArticulatedSkinParams(
            shape=mochi.ShapeHandle(value=55),
            layer="SkinLayer",
            contact=mochi.ContactParams(penalty_coefficient=1e9),
            boundary_element_type=mochi.ActorBoundaryElementType.P1Q6,
            boundary_subsampling=mochi.BoundarySubsamplingParams(),
        )
        self.assertEqual(mochi.ShapeHandle(value=55), params_kw.shape)
        self.assertEqual("SkinLayer", params_kw.layer)
        self.assertAlmostEqual(1e9, params_kw.contact.penalty_coefficient)
        self.assertEqual(
            mochi.ActorBoundaryElementType.P1Q6, params_kw.boundary_element_type
        )
        self.assertIsNotNone(params_kw.boundary_subsampling)

        # Test equality operators
        params_copy = mochi.ArticulatedSkinParams(
            shape=mochi.ShapeHandle(value=55),
            layer="SkinLayer",
            contact=mochi.ContactParams(penalty_coefficient=1e9),
            boundary_element_type=mochi.ActorBoundaryElementType.P1Q6,
            boundary_subsampling=mochi.BoundarySubsamplingParams(),
        )
        self.assertEqual(params_kw, params_copy)
        self.assertNotEqual(params, params_kw)

        # Assign every field
        params.shape = mochi.ShapeHandle(value=77)
        params.layer = "Updated"
        params.contact = mochi.ContactParams()
        params.boundary_element_type = mochi.ActorBoundaryElementType.P1Q6
        params.boundary_subsampling = mochi.BoundarySubsamplingParams()
        self.assertEqual(mochi.ShapeHandle(value=77), params.shape)
        self.assertEqual("Updated", params.layer)

    def test_articulated_actor_params(self):
        # Test default construction is possible, but don't hard-code all the defaults here.
        params = mochi.ArticulatedActorParams()
        self.assertEqual("", params.name)

        # Test keyword constructor with all fields
        joint = mochi.ArticulatedJointParams(
            name="TestJoint",
            type=mochi.ArticulatedJointType.FREE,
        )
        link = mochi.ArticulatedLinkParams(
            name="TestLink",
            parent_link=-1,
        )
        skin = mochi.ArticulatedSkinParams(
            shape=mochi.ShapeHandle(value=88),
            layer="SkinLayer",
        )
        params_kw = mochi.ArticulatedActorParams(
            name="TestArticulatedNew",
            world_from_root=mochi.TransformRT(translation=[1, 2, 3]),
            joints=[joint],
            links=[link],
            skin=skin,
            joint_velocities=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        self.assertEqual("TestArticulatedNew", params_kw.name)
        self.assertEqual(mochi.Real3(1, 2, 3), params_kw.world_from_root.translation)
        self.assertEqual(1, len(params_kw.joints))
        self.assertEqual("TestJoint", params_kw.joints[0].name)
        self.assertEqual(mochi.ArticulatedJointType.FREE, params_kw.joints[0].type)
        self.assertEqual(1, len(params_kw.links))
        self.assertEqual("TestLink", params_kw.links[0].name)
        self.assertEqual(-1, params_kw.links[0].parent_link)
        self.assertIsNotNone(params_kw.skin)
        self.assertEqual("SkinLayer", params_kw.skin.layer)
        self.assertIsNotNone(params_kw.joint_velocities)
        self.assertEqual(6, len(params_kw.joint_velocities))

        # Test partial keyword construction
        params_partial = mochi.ArticulatedActorParams(
            name="PartialNew",
        )
        self.assertEqual("PartialNew", params_partial.name)
        self.assertEqual(0, len(params_partial.joints))
        self.assertIsNone(params_partial.skin)

        # Assign every field
        params.name = "AssignedName"
        params.world_from_root = mochi.TransformRT(translation=[4, 5, 6])
        params.joints = [
            mochi.ArticulatedJointParams(type=mochi.ArticulatedJointType.REVOLUTE),
            mochi.ArticulatedJointParams(type=mochi.ArticulatedJointType.PRISMATIC),
        ]
        params.links = [
            mochi.ArticulatedLinkParams(parent_link=-1),
            mochi.ArticulatedLinkParams(parent_link=0),
        ]
        params.skin = mochi.ArticulatedSkinParams()
        params.joint_velocities = [0.1, 0.2]
        self.assertEqual("AssignedName", params.name)
        self.assertEqual(2, len(params.joints))
        self.assertEqual(mochi.ArticulatedJointType.REVOLUTE, params.joints[0].type)
        self.assertEqual(mochi.ArticulatedJointType.PRISMATIC, params.joints[1].type)
        self.assertEqual(2, len(params.links))
        self.assertIsNotNone(params.skin)
        self.assertEqual(2, len(params.joint_velocities))

    def test_soft_skinned_actor_params(self):
        # Test a few default values to prove the C++ constructor ran
        params = mochi.SoftSkinnedActorParams()
        self.assertFalse(params.enable_colliding_links)
        self.assertFalse(params.has_gravity)
        self.assertFalse(params.has_inertia)
        self.assertFalse(params.has_stress)

        # Test keyword argument constructor with all fields in shuffled order.
        skeleton_params = mochi.ArticulatedActorParams(name="Skeleton")
        soft_params = mochi.SoftActorParams(name="Soft")
        params_kw_all = mochi.SoftSkinnedActorParams(
            has_stress=True,
            enable_colliding_links=True,
            has_gravity=True,
            skeleton_params=skeleton_params,
            soft_params=[soft_params],
            soft_attach_links=["link1", "link2"],
            has_inertia=True,
        )
        self.assertTrue(params_kw_all.enable_colliding_links)
        self.assertTrue(params_kw_all.has_gravity)
        self.assertTrue(params_kw_all.has_inertia)
        self.assertTrue(params_kw_all.has_stress)
        self.assertEqual(2, len(params_kw_all.soft_attach_links))
        self.assertEqual("link1", params_kw_all.soft_attach_links[0])
        self.assertEqual("link2", params_kw_all.soft_attach_links[1])
        self.assertEqual("Soft", params_kw_all.soft_params[0].name)
        self.assertEqual("Skeleton", params_kw_all.skeleton_params.name)

        # Test keyword argument constructor with some fields (rest use defaults)
        params_kw_partial = mochi.SoftSkinnedActorParams(
            has_gravity=True,
        )
        self.assertTrue(params_kw_partial.has_gravity)
        self.assertEqual(
            params.enable_colliding_links, params_kw_partial.enable_colliding_links
        )
        self.assertEqual(params.has_inertia, params_kw_partial.has_inertia)
        self.assertEqual(params.has_stress, params_kw_partial.has_stress)

        # Assign every field (to show we can)
        params.skeleton_params = mochi.ArticulatedActorParams()
        params.soft_params = [mochi.SoftActorParams()]
        params.soft_attach_links = ["test"]
        params.enable_colliding_links = True
        params.has_gravity = False
        params.has_inertia = False
        params.has_stress = False

    def test_linear_transmission_params(self):
        params = mochi.experimental.LinearTransmissionParams()
        self.assertEqual(0, len(params.joint_indices))
        self.assertEqual(0, len(params.joint_coefficients))

        params_kw = mochi.experimental.LinearTransmissionParams(
            joint_indices=[0, 1, 2],
            joint_coefficients=[0.5, -0.3, 0.2],
        )
        self.assertEqual(3, len(params_kw.joint_indices))
        self.assertEqual(3, len(params_kw.joint_coefficients))

        params.joint_indices = [0, 1]
        params.joint_coefficients = [0.4, 0.6]
        self.assertEqual(2, len(params.joint_indices))

    def test_displacement_control_actuator_params(self):
        params = mochi.experimental.DisplacementControlActuatorParams()
        self.assertEqual(0.0, params.target_displacement)
        self.assertEqual(1e9, params.stiffness)
        self.assertEqual(0.0, params.damping)
        self.assertEqual(False, params.allow_compressive_force)

        params_kw = mochi.experimental.DisplacementControlActuatorParams(
            target_displacement=0.1,
            stiffness=1e7,
            damping=100.0,
            allow_compressive_force=True,
        )
        self.assertAlmostEqual(0.1, params_kw.target_displacement)
        self.assertAlmostEqual(1e7, params_kw.stiffness)
        self.assertAlmostEqual(100.0, params_kw.damping)
        self.assertEqual(True, params_kw.allow_compressive_force)

        params_partial = mochi.experimental.DisplacementControlActuatorParams(
            stiffness=5e6,
        )
        self.assertEqual(0.0, params_partial.target_displacement)
        self.assertAlmostEqual(5e6, params_partial.stiffness)
        self.assertEqual(0.0, params_partial.damping)
        self.assertEqual(False, params_partial.allow_compressive_force)

        params.target_displacement = -0.25
        params.stiffness = 2e8
        params.damping = 50.0
        params.allow_compressive_force = True
        self.assertAlmostEqual(-0.25, params.target_displacement)
        self.assertAlmostEqual(2e8, params.stiffness)
        self.assertAlmostEqual(50.0, params.damping)
        self.assertEqual(True, params.allow_compressive_force)

    def test_force_control_actuator_params(self):
        params = mochi.experimental.ForceControlActuatorParams()
        self.assertEqual(0.0, params.force)
        self.assertEqual(False, params.allow_compressive_force)

        params_kw = mochi.experimental.ForceControlActuatorParams(
            force=42.0, allow_compressive_force=True
        )
        self.assertAlmostEqual(42.0, params_kw.force)
        self.assertEqual(True, params_kw.allow_compressive_force)

        params.force = -7.5
        params.allow_compressive_force = True
        self.assertAlmostEqual(-7.5, params.force)
        self.assertEqual(True, params.allow_compressive_force)

    def test_mckibben_actuator_params(self):
        params = mochi.experimental.McKibbenActuatorParams()
        self.assertEqual(0.0, params.pressure)
        self.assertEqual(0.0, params.minimum_pressure)
        self.assertEqual(0.0, params.thread_length)
        self.assertEqual(0.0, params.number_of_wraps)
        self.assertEqual(0.0, params.deflated_stiffness)
        self.assertEqual(0.0, params.deflated_equilibrium_length)

        params_kw = mochi.experimental.McKibbenActuatorParams(
            pressure=1e5,
            minimum_pressure=1e3,
            thread_length=0.15,
            number_of_wraps=3.0,
            deflated_stiffness=500.0,
            deflated_equilibrium_length=0.1,
        )
        self.assertAlmostEqual(1e5, params_kw.pressure)
        self.assertAlmostEqual(1e3, params_kw.minimum_pressure)
        self.assertAlmostEqual(0.15, params_kw.thread_length)
        self.assertAlmostEqual(3.0, params_kw.number_of_wraps)
        self.assertAlmostEqual(500.0, params_kw.deflated_stiffness)
        self.assertAlmostEqual(0.1, params_kw.deflated_equilibrium_length)

        params.pressure = 2e5
        params.thread_length = 0.2
        self.assertAlmostEqual(2e5, params.pressure)
        self.assertAlmostEqual(0.2, params.thread_length)

    def test_debug_draw_data_transitive_deepcopy(self):
        # DebugDrawData has nested structs  (DebugDrawLineVertices, DebugDrawSpheres) which contain
        # Span members. Therefore `DebugDrawData` cannot be deep copied.
        data = mochi.DebugDrawData()
        with self.assertRaises(TypeError):
            copy.copy(data)  # We could support shallow copy, but choose not to for now.
        with self.assertRaises(TypeError):
            copy.deepcopy(data)

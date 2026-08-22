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

"""
Generates data for tests using a third party library.
To run using docker simply spin up a container with
```
docker run --rm -ti -v $(pwd):/home/fenics/shared -w \
  /home/fenics/shared quay.io/dolfinadjoint/pyadjoint:2019.1.0
```
and run
```
python3 GenerateTestData.py
```
Output will print out relevant information.
"""

import numpy as np
import numpy.linalg
from dolfin import (
    assemble,
    Cell,
    Constant,
    derivative,
    det,
    DirichletBC,
    dot,
    ds,
    dx,
    Expression,
    FacetNormal,
    Function,
    FunctionSpace,
    grad,
    Identity,
    interpolate,
    ln,
    Mesh,
    MeshEditor,
    norm,
    parameters,
    Point,
    project,
    set_log_level,
    TestFunction,
    tr,
    TrialFunction,
    VectorFunctionSpace,
    vertex_to_dof_map,
)

# Optimization options for the form compiler
parameters["allow_extrapolation"] = True
parameters["form_compiler"]["cpp_optimize"] = True

ffc_options = {
    "optimize": True,
    "eliminate_zeros": True,
    "precompute_basis_const": True,
    "precompute_ip_const": True,
    "quadrature_degree": 2,
}

# level = 50 # errors that may lead to data corruption and suchlike
# level = 40 # things that go boom
level = 30  # things that may go boom later
# level = 20 # information of general interest
# level = 16 # what's happening (broadly)
# level = 13 # what's happening (in detail)
# level = 10 # sundr

set_log_level(level)


# A solid unit cube with one corner at (0,0,0)
#
#         6 ------- 7
#       / |       / |
#      /  |      /  |
#     2 ------- 3   |
#     |   4 ----|-- 5
#     |  /      |  /
#     | /       | /
#     0 ------- 1
#
class UnitCube:
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [1.0, 1.0, 0.0],  # 3
            [0.0, 0.0, 1.0],  # 4
            [1.0, 0.0, 1.0],  # 5
            [0.0, 1.0, 1.0],  # 6
            [1.0, 1.0, 1.0],  # 7
        ]
    )
    connectivity = np.array(
        [
            [2, 6, 3, 0],  # corner vert 2
            [7, 3, 6, 5],  # corner vert 7
            [1, 3, 5, 0],  # corner vert 1
            [4, 0, 5, 6],  # corner vert 4
            [6, 0, 3, 5],
        ]
    )  # the one fully interior tetrahedron


class SingleTet:
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],  # 0
            [1.0, 0.0, 0.0],  # 1
            [0.0, 1.0, 0.0],  # 2
            [0.0, 0.0, 1.0],  # 3
        ]
    )
    connectivity = np.array(
        [
            [0, 1, 2, 3],
        ]
    )  # the one fully interior tetrahedron


def createMesh(coordinates, connectivity):
    # Load the mesh
    mesh = Mesh()
    editor = MeshEditor()
    editor.open(mesh, "tetrahedron", 3, 3)  # top. and geom. dimension are both 3
    editor.init_vertices(len(coordinates))  # number of vertices
    editor.init_cells(len(connectivity))  # number of cells
    for i, v in enumerate(coordinates):
        editor.add_vertex(i, v)
    for i, e in enumerate(connectivity):
        editor.add_cell(i, e)
    editor.close()
    mesh = Mesh(mesh)
    return mesh


def forward(
    coordinates,
    connectivity,
    boundary_conditions,
    initial=Constant((0, 0, 0)),
    youngs=Constant(10.0),
    poisson=Constant(0.3),
    body_force=Constant((0, 0, 0)),
    traction=Constant((0, 0, 0)),
    time=0,
):
    # Load the mesh
    mesh = createMesh(coordinates, connectivity)

    # Create mesh and define function space
    V = VectorFunctionSpace(mesh, "Lagrange", 1)

    # Define functions
    du = TrialFunction(V)  # Incremental displacement
    v = TestFunction(V)  # Test function
    u = Function(V)  # Displacement from previous iteration
    B = Function(V)  # Body force per unit volume
    u.rename("u", "u")

    # Kinematics
    d = u.geometric_dimension()
    I = Identity(d)  # Identity tensor
    F = I + grad(u)  # Deformation gradient
    C = F.T * F  # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J = det(F)

    # Initial guess
    u.vector()[:] = interpolate(initial, V).vector()[:]

    # Material parameters
    lmbda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = youngs / (2.0 * (1.0 + poisson))
    alpha = 1 + mu / lmbda - mu / (4 * lmbda)

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu / 2) * (Ic - 3) + lmbda / 2 * (J - alpha) ** 2 - mu / 2 * ln(Ic + 1)

    # Total potential energy
    Pi = psi * dx - (dot(B, u) * dx + dot(traction, u) * ds)

    # Compute first variation of Pi
    F = derivative(Pi, u, v)

    # Compute Jacobian of F
    J = derivative(F, u, du)

    # Evaluate the bc
    bcs = boundary_conditions(V)

    R = assemble(F)
    DR = assemble(J)
    energyVal = assemble(Pi)
    print("Test values for AssemblyTest.cpp ")
    print(
        "Strain energy: %.6f Residual norm: %.6f Hessian norm: %.6f"
        % tuple([energyVal, norm(R), np.linalg.norm(DR.array().astype(np.float32))])
    )

    return u


def forwardDynamic(
    coordinates,
    connectivity,
    boundary_conditions,
    youngs=Constant(10.0),
    poisson=Constant(0.3),
    body_force=Constant((0, 0, 0)),
    traction=Constant((0, 0, 0)),
    time=0,
):
    # Load the mesh
    mesh = createMesh(coordinates, connectivity)

    # Create mesh and define function space
    V = VectorFunctionSpace(mesh, "Lagrange", 1)

    # Define functions
    du = TrialFunction(V)  # Incremental displacement
    v = TestFunction(V)  # Test function
    u = Function(V)  # Displacement from previous iteration
    u_n = Function(V)  # Displacement from previous time step
    dudt_n = Function(V)  # Displacement from previous time step
    B = Function(V)  # Body force per unit volume
    u.rename("u", "u")
    u_n = interpolate(Expression(("x[0]", "x[1]", "x[2]"), degree=1), V)
    dudt_n = interpolate(Expression(("x[2]", "x[1]", "x[0]"), degree=1), V)
    u.vector()[:] *= 0

    # Kinematics
    d = u.geometric_dimension()
    I = Identity(d)  # Identity tensor
    F = I + grad(u)  # Deformation gradient
    C = F.T * F  # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = tr(C)
    J = det(F)

    # Material parameters
    lmbda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = youngs / (2.0 * (1.0 + poisson))
    alpha = 1 + mu / lmbda - mu / (4 * lmbda)

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (mu / 2) * (Ic - 3) + lmbda / 2 * (J - alpha) ** 2 - mu / 2 * ln(Ic + 1)

    # Total potential energy
    delta_time = Constant(0.7)
    density = Constant(1.0 / 3)
    a = u - u_n - delta_time * dudt_n
    ah = Function(V)
    ah.vector()[:] = u.vector()[:] - u_n.vector()[:] - delta_time * dudt_n.vector()[:]
    print(assemble(1.0 * dx(mesh)))
    KE = density * dot(a, a) / pow(delta_time, 2) / 2 * dx
    Pi = psi * dx - (dot(B, u) * dx + dot(traction, u) * ds)

    # Compute first variation of Pi
    F = derivative(KE, u, v)

    # Compute Jacobian of F
    J = derivative(F, u, du)
    bcs = boundary_conditions(V)
    R = assemble(F)
    DR = assemble(J)
    v_d = vertex_to_dof_map(V)

    # Get the mass matrix
    print("Test values for LagrangianTest.cpp using a unit tetrahedron as the domain")
    I = R[v_d]
    print(
        "Coefficient used: density = %.5f, time_step = %.5f"
        % (float(density), float(delta_time))
    )
    print(
        "Initial conditions: initialDisplacements(X) = {X[0],X[1],X[2]}, initialVelocities(X) = {X[2],X[1],X[0]}"
    )
    print("\n\nResidual:\n")
    print("{ ", end="")
    for i in I:
        print(i, "\b, ", end="")
    print("\b\b },")

    print("\n\nDResidual (aka Mass Matrix):\n")
    M = DR.array().astype(np.float32)

    for m in M[v_d]:
        print("\t real12{", end="")
        for n in m[v_d]:
            print(n, "\b, " % n, end="")
        print("\b\b },")

    print("\n\n Merit:\n")
    print(assemble(KE))
    return u


def forwardTraction(coordinates, connectivity, time=0):
    # Load the mesh
    mesh = createMesh(coordinates, connectivity)

    # Create mesh and define function space
    V = VectorFunctionSpace(mesh, "Lagrange", 1)
    v_d = vertex_to_dof_map(V)

    # Define functions
    du = TrialFunction(V)  # Incremental displacement
    v = TestFunction(V)  # Test function
    u = Function(V)  # Displacement from previous iteration
    u.rename("u", "u")

    X = Expression(("x[0]", "x[1]", "x[2]"), degree=1)
    phi = X + u

    initial = Expression(("1.1*x[0]+1", "0.5*x[1]-0.5", "2*x[2]+5"), degree=1)
    u.vector()[:] = project(initial, V).vector()[:]

    n = FacetNormal(mesh)
    c = Constant((1.1, 0.23, 6))
    traction = c * dot(phi, n)

    xh = project(phi, V)

    # Compute first variation of Pi
    F = -dot(traction, v) * ds

    # Compute Jacobian of F
    J = derivative(F, u, du)

    R = assemble(F)
    DR = assemble(J)
    print("Test values for AssemblyTest.cpp ")
    print(
        "Residual norm: %.6f Hessian norm: %.6f"
        % tuple([norm(R), np.linalg.norm(DR.array().astype(np.float32))])
    )

    I = R[v_d]
    print("{ ", end="")
    for i in I:
        print(i, "\b_r, ", end="")
    print("\b\b },")

    print("\n\nDResidual (aka Mass Matrix):\n")
    M = DR.array().astype(np.float32)

    for m in M[v_d]:
        print("\t real12{", end="")
        for n in m[v_d]:
            print(n, "\b_r, " % n, end="")
        print("\b\b },")
    return


if __name__ == "__main__":
    boundary_conditions = lambda V: DirichletBC(
        V, Constant((0, 0, 0)), "fabs(x[2]) < 1.e-9"
    )

    forward(
        UnitCube.coordinates,
        UnitCube.connectivity,
        boundary_conditions,
        youngs=Constant(10.0),
        poisson=Constant(0.3),
    )

    forwardDynamic(
        SingleTet.coordinates,
        SingleTet.connectivity,
        boundary_conditions,
        youngs=Constant(10.0),
        poisson=Constant(0.3),
    )

    forwardTraction(SingleTet.coordinates, SingleTet.connectivity)

    # Print out basis function values
    mesh = createMesh(SingleTet.coordinates, SingleTet.connectivity)
    V = FunctionSpace(mesh, "CG", 1)
    el = V.element()

    # Where to evaluate
    X = np.array(
        [
            [0.58541020, 0.13819660, 0.13819660],
            [0.13819660, 0.58541020, 0.13819660],
            [0.13819660, 0.13819660, 0.58541020],
            [0.13819660, 0.13819660, 0.13819660],
        ]
    )
    cell_id = mesh.bounding_box_tree().compute_first_entity_collision(Point(*X[0]))
    cell = Cell(mesh, cell_id)
    coordinate_dofs = cell.get_vertex_coordinates()
    print(coordinate_dofs)
    # Find the cell with point
    print("Basis function values for Tet at quadrature points for 4 point rule")
    for x in X:
        values = el.evaluate_basis_all(x, coordinate_dofs, cell_id)
        print(x, values)

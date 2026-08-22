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

# fmt: off
import dolfin
import numpy as np

# Prints out the strain energy, the first Piola-Kirchhoff stress
# and its tangent for a set of predefined deformation gradients
# given the material strain energy as a function of the deformation gradient.
# This script uses
def main(get_material_model):

    # The polynomial order
    poly_order = 1

    # The unit tetrahedron nodes
    nodes = (
        np.array([[0.1, 0.1, 0.1], [1.0, -0.1, 0.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
        * 10
    )

    # The centroid of the element for convenience
    centroid = np.mean(nodes, axis=0)

    # The unit tet conectivity
    cells = np.array([[0, 1, 2, 3]], dtype=np.uintp)

    # Create the mesh object for the one-element mesh
    mesh = dolfin.Mesh()
    editor = dolfin.MeshEditor()
    topological_dimension = 3
    spatial_dimensions = 3
    cell_type = "tetrahedron"
    editor.open(mesh, cell_type, topological_dimension, spatial_dimensions)
    editor.init_vertices(4)
    editor.init_cells(1)

    [editor.add_vertex(i, n) for i, n in enumerate(nodes)]
    [editor.add_cell(i, n) for i, n in enumerate(cells)]
    editor.close()

    # Create mesh and define function space
    V = dolfin.VectorFunctionSpace(mesh, "Lagrange", poly_order)
    S = dolfin.FunctionSpace(mesh, "DG", 0)
    W = dolfin.TensorFunctionSpace(mesh, "DG", 0, shape=(3, 3))
    Z = dolfin.TensorFunctionSpace(mesh, "DG", 0, shape=(3, 3, 3, 3))

    # Define functions
    u = dolfin.Function(V)  # Displacement from previous iteration

    # Kinematics
    d = u.geometric_dimension()
    Identity = dolfin.Identity(d)  # Identity tensor
    F = dolfin.variable(Identity + dolfin.grad(u))  # Deformation gradient

    # Construct affine map Ax + b
    x = dolfin.Expression(("x[0]", "x[1]", "x[2]"), degree=1)

    tests = ["identity", "random", "isochoric", "symmetric", "dilation", "shear"]
    for test in tests:
        print("// Test : ", end="")
        # If random
        if test == "random":
            print("random")
            A = np.eye(3) + np.random.random((3, 3)) - np.random.random()
        # If random but isochoric
        elif test == "isochoric":
            print("isochoric")
            A = np.eye(3) + np.random.random((3, 3))
            A *= pow(np.linalg.det(A), -1.0 / 3)
        # symmetric
        elif test == "symmetric":
            print("symmetric")
            A = np.eye(3) + np.random.random((3, 3))
            A = np.dot(A.real, A)
        # if pure dilation
        elif test == "dilation":
            print("dilation")
            A = np.eye(3) * 4
        elif test == "shear":
            print("shear")
            A1 = np.array([[1.0, 0, 0.0], [0, 1.0, 0.5], [0.0, 0.5, 1.0]])
            A2 = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
            A = np.dot(A1, A2)
        elif test == "identity":
            A = np.eye(3)

        A = dolfin.Constant(A)

        ut = dolfin.project(dolfin.dot(A, x) - x, V)
        u.vector()[:] = ut.vector()[:]

        psi = get_material_model(F)
        PK1e = dolfin.variable(dolfin.diff(psi, F))
        Te = dolfin.diff(PK1e, F)

        # Project onto constant space for evaluation
        psie = dolfin.project(psi, S)
        PK1e = dolfin.project(PK1e, W)
        Fe = dolfin.project(F, W)
        Te = dolfin.project(Te, Z)
        strainEnergy = psie(centroid)
        PK1e = np.array(PK1e(centroid).reshape((3, 3)))
        Fe = np.array(Fe(centroid).reshape((3, 3)))
        Te = np.array(Te(centroid).reshape((3, 3, 3, 3)))

        # Create the material object
        deformation_gradient = Fe
        print("// Deformation gradient %s" % test)
        print("TestData{")
        print_second_order_tensor(deformation_gradient)
        print(",")
        print("// Strain Energy")
        print("%.5e_r," % strainEnergy)
        print("// Material PK1 %s" % test)
        print_second_order_tensor(PK1e)
        print(",")
        print("// Material Tangent %s" % test)
        print_fourth_order_tensor(Te)
        print("}, // End of TestData  ")  # end of test data

# Print the fourth order tensor in a convenient
# way for ease of copy and paste to C++
def print_second_order_tensor(A):
    row = "real3{%.6e_r,  %.6e_r, %.6e_r},"
    print("NdArray<real, 3, 3>{")
    for i in range(3):
        print(row % tuple(A[i]), end="\n" if not i == 2 else "\b")
    print("}", end="")

# Print the fourth order tensor in a convenient
# way for ease of copy and paste to C++
def print_fourth_order_tensor(A):
    print("NdArray<real, 3, 3, 3, 3>{")
    for i in range(3):
        print("NdArray<real, 3, 3, 3>{")
        for j in range(3):
            print_second_order_tensor(A[i][j])
            print("," if j < 2 else "},")
    print("}")

# Get invariants of strain measures
def get_invariants(F):
    C = F.T * F
    Ic = dolfin.tr(C)
    J = dolfin.det(F)
    return Ic, J

# Get the strain energy for the stable Neo-Hookean material
def get_stable_neo_hookean_strain_energy(F):
    Ic, J = get_invariants(F)
    E, nu = 1.0, 0.4
    mu, lmbda = dolfin.Constant(E / (2 * (1 + nu))), dolfin.Constant(
        E * nu / ((1 + nu) * (1 - 2 * nu))
    )
    lmbda += mu * (5.0 / 6.0)
    mu *= 4.0 / 3.0
    alpha = dolfin.Constant(1 + mu / lmbda - mu / (4 * lmbda))
    psi = (mu / 2) * (Ic - 3) + lmbda / 2 * (J - alpha) ** 2 - mu / 2 * dolfin.ln(Ic + 1)
    return psi

# Get the strain energy for the stable Neo-Hookean material
def get_neo_hookean_strain_energy(F):
    Ic, J = get_invariants(F)
    E, nu = 1.0, 0.4
    mu, lmbda = dolfin.Constant(E / (2 * (1 + nu))), dolfin.Constant(
        E * nu / ((1 + nu) * (1 - 2 * nu))
    )
    psi = (mu / 2) * (Ic - 3) + lmbda / 2 * dolfin.ln(J) ** 2 - mu  * dolfin.ln(J)
    return psi


# Get the strain energy for St. Venant-Kirchhoff stess
def get_st_venant_kirchoff_strain_energy(F):
    E, nu = 1.0, 0.4
    mu, lmbda = dolfin.Constant(E / (2 * (1 + nu))), dolfin.Constant(
        E * nu / ((1 + nu) * (1 - 2 * nu))
    )
    C = F.T * F  # Right Cauchy-Green tensor
    GLS = 1.0 / 2.0 * (C - dolfin.Identity(3))
    psi = lmbda / 2.0 * pow(dolfin.tr(GLS), 2) + mu * dolfin.inner(GLS, GLS)
    return psi


if __name__ == "__main__":

    main(get_neo_hookean_strain_energy)

# fmt: on

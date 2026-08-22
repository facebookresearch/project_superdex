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

import numpy as np

# ____________________________________________________________________________


def generate_random_PSD_system(n, sparsity=0.5):
    # Generate random system matrix.
    A = np.random.uniform(0, 1, size=(n, n)).astype(np.float32)
    A = 0.5 * (A.T + A)

    # Drop terms according to desired sparsity.
    m = int(0.5 * (n**2))
    drop = np.random.uniform(0, n - 1, size=(m, m)).astype(dtype=np.int32)
    A[drop[:, 0], drop[:, 1]] = 0
    A[drop[:, 1], drop[:, 0]] = 0

    # Ensure PSD
    A += (1e-4 - np.min(np.linalg.eigvalsh(A))) * np.eye(n)

    # Generate random solution.
    x = np.random.uniform(size=n).astype(np.float32)

    # Generate rhs.
    b = A @ x
    return A, b, x


def output(file, name, A, b, x, pc):
    if A is not None:
        nnz = np.sum(A != 0)
        triplets = []

        for i in range(A.shape[0]):
            for j, v in enumerate(A[i, :]):
                if v != 0:
                    triplets.append((i, j, v))

        rows = np.array([i for i, j, v in triplets], dtype=np.int32)
        cols = np.array([j for i, j, v in triplets], dtype=np.int32)
        vals = np.array([v for i, j, v in triplets], dtype=np.float32)

        file.write("constexpr int k" + name + "Nnz = %i;\n" % nnz)
        file.write("\n")
        file.write("constexpr int k" + name + "Size[] = { %i, %i };\n" % (*A.shape,))
        file.write("\n")
        file.write(
            "constexpr int k"
            + name
            + "Rows[] = {"
            + ", ".join(["%i" % i for i in rows])
            + " };\n"
        )
        file.write("\n")
        file.write(
            "constexpr int k"
            + name
            + "Cols[] = {"
            + ", ".join(["%i" % i for i in cols])
            + " };\n"
        )
        file.write("\n")
        file.write(
            "constexpr real k"
            + name
            + "Vals[] = {"
            + ", ".join(["%e_r" % i for i in vals])
            + " };\n"
        )
        file.write("\n")

    if pc is not None:
        file.write(
            "constexpr real k"
            + name
            + "InvDiag[] = {"
            + ", ".join(["%e_r" % i for i in pc])
            + " };\n"
        )
        file.write("\n")

    if b is not None:
        file.write(
            "constexpr real k"
            + name
            + "Rhs[] = {"
            + ", ".join(["%e_r" % i for i in b])
            + " };\n"
        )
        file.write("\n")

    if x is not None:
        file.write(
            "constexpr real k"
            + name
            + "Out[] = { "
            + ", ".join(["%e_r" % i for i in x])
            + " };\n"
        )
        file.write("\n")


# _____________________________________________________________________________

np.random.seed(0)
A, b, x = generate_random_PSD_system(100)
pc = 1 / np.diag(A)
pc = 0.4 + 0.6 * pc  # Otherwise Jacobi will make this non-PSD!

with open("krylov_solver_test_data.h", "w") as file:
    file.write("#pragma once\n")
    file.write("\n")
    output(file, "PsdMatrix", A, b, x, pc)
    file.write("\n")
    output(file, "PsdMatrix_00_", A[:20, :40], None, None, None)
    output(file, "PsdMatrix_01_", A[:20, 40:], None, None, None)
    output(file, "PsdMatrix_10_", A[20:, :40], None, None, None)
    output(file, "PsdMatrix_11_", A[20:, 40:], None, None, None)

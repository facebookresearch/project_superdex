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
import scipy as sp
import scipy.spatial
from helpers import export_ndarray

# _____________________________________________________________________________


def encode_contact_row(C):
    y = np.array([(x << i) for i, x in enumerate(C)], dtype=np.uint32)
    return np.sum(y)


def encode_contact_matrix(C):
    return np.array(
        [
            encode_contact_row(c2) << 16 | encode_contact_row(c1)
            for c1, c2 in zip(C[0::2], C[1::2])
        ],
        dtype=np.uint32,
    )


# _____________________________________________________________________________

np.random.seed(0)
r = 0.15
X = np.random.uniform(size=(10, 16, 3))

C = np.zeros((10, 10, 8), dtype=np.uint32)
SC = np.zeros((10, 8), dtype=np.uint32)
# SC = np.array([ sp.spatial.distance_matrix(x, x) < r for x in X ], dtype=np.int32)

for i, Xi in enumerate(X):
    sc = sp.spatial.distance_matrix(Xi, Xi) < 2 * r
    sc = encode_contact_matrix(sc)
    SC[i] = sc

    for j, Xj in enumerate(X):
        c = sp.spatial.distance_matrix(Xi, Xj) < 2 * r
        c = encode_contact_matrix(c)
        C[i, j] = c

# _____________________________________________________________________________
# %%

with open("bvh_tree_sphere_cloud_data.h", "w") as file:
    file.write("#pragma once\n")
    file.write("\n")
    file.write("namespace mochi {\n\n")
    file.write("real constexpr kSphereCloud_Radius = %e_r;\n\n" % r)
    file.write(export_ndarray(X, name="kSphereCloud_Points"))
    file.write("\n\n")
    file.write(export_ndarray(SC, name="kSphereCloud_SelfOverlapTest"))
    file.write("\n\n")
    file.write(export_ndarray(C, name="kSphereCloud_OverlapTest"))
    file.write("\n\n")
    file.write("} // namespace mochi\n\n")

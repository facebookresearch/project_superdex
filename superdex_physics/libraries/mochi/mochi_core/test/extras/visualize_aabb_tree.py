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

# -*- coding: utf-8 -*-

import json

import meshio as mio
import numpy as np
import polyscope as ps

# _____________________________________________________________________________

mesh = mio.read("../assets/BvhTreeTest_MeshB.obj")
tree = json.load(open("../out.json"))
X, T = mesh.points, mesh.cells_dict["triangle"]

# _____________________________________________________________________________


def process_aabb_node(X, T, node, depth, idx, support):
    node["idx"] = idx
    node["depth"] = depth
    support["levels"][depth].append(node)

    if "elements" in node:
        t = T[node["elements"], :]
        i = np.array([x for y in t for x in y])
        x = X[i, :]
        node["min_x"] = np.min(x, axis=0)
        node["max_x"] = np.max(x, axis=0)
        support["leaves"].append(node)
    else:
        idx = process_aabb_node(X, T, node["left"], depth + 1, idx + 1, support)
        idx = process_aabb_node(X, T, node["right"], depth + 1, idx + 1, support)
        node["min_x"] = np.minimum(node["left"]["min_x"], node["right"]["min_x"])
        node["max_x"] = np.maximum(node["left"]["max_x"], node["right"]["max_x"])

    return idx


def generate_aabb_tree_meshes(node):
    # Generate curve network
    depth, idx = node["depth"], node["idx"]
    min_x, max_x = node["min_x"], node["max_x"]
    x0, y0, z0 = min_x
    x1, y1, z1 = max_x

    points = np.array(
        [
            [x0, y0, z0],  # 0
            [x1, y0, z0],  # 1
            [x1, y0, z1],  # 2
            [x0, y0, z1],  # 3
            [x0, y1, z0],  # 4
            [x1, y1, z0],  # 5
            [x1, y1, z1],  # 6
            [x0, y1, z1],
        ]
    )  # 7

    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.int32,
    )

    node["cn"] = ps.register_curve_network(
        f"AABB ({depth}, {idx})", points, edges, radius=0.002
    )
    node["cn"].set_enabled(False)

    # Generate children curve networks
    if "left" in node:
        generate_aabb_tree_meshes(node["left"])
    if "right" in node:
        generate_aabb_tree_meshes(node["right"])


def show_aabb_leaves(support):
    for node in support["leaves"]:
        node["cn"].set_enabled(True)


def show_aabb_level(support, depth):
    for node in support["levels"][depth]:
        node["cn"].set_enabled(True)


def hide_aabb_level(support, depth):
    for node in support["levels"][depth]:
        node["cn"].set_enabled(False)


def identify_face_clusters(T, support):
    c = np.zeros(len(T))
    for i, l in enumerate(support["leaves"]):
        c[l["elements"]] = (31 * i) % len(support["leaves"])
    return c


# _____________________________________________________________________________

support = {
    "levels": [[] for i in range(tree["statistics"]["max-depth-reached"])],
    "leaves": [],
}

process_aabb_node(X, T, tree["structure"], 0, 0, support)
clusters = identify_face_clusters(T, support)

# _____________________________________________________________________________

# Initialize polyscope
ps.set_SSAA_factor(4)
ps.set_autocenter_structures(False)
ps.set_autoscale_structures(False)
ps.init()
ps.remove_all_structures()

# Register mesh
ps_mesh = ps.register_surface_mesh("Mesh", X, T, smooth_shade=True)
ps_mesh.add_scalar_quantity("Leaves", clusters, defined_on="faces")

# Register AABB nodes
generate_aabb_tree_meshes(tree["structure"])
show_aabb_leaves(support)

ps.show()

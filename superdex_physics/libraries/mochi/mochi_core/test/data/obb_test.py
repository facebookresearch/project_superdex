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

import matplotlib.pyplot as plt
import numpy as np

# _____________________________________________________________________________


def oobb_fit(X):
    # Compute centroid.
    mu = np.mean(X, axis=0)

    # Build covariance matrix.
    U = X - mu
    C = np.zeros((3, 3))
    for u in U:
        C += np.outer(u, u)
    C /= X.shape[0]

    # Estimate directions of maximum spread.
    vals, dirs = np.linalg.eigh(C)
    sort = np.argsort(vals)[::-1]
    vals = vals[sort]
    dirs = dirs[:, sort]

    if np.linalg.det(dirs) < 0:
        dirs = -dirs

    # Find extents.
    projs = np.array([dirs.T @ x for x in X])
    min_extents = np.min(projs, axis=0)
    max_extents = np.max(projs, axis=0)

    center = 0.5 * dirs @ (min_extents + max_extents)
    extents = 0.5 * (max_extents - min_extents)
    return center, dirs, extents


def oobb_corners(center, dirs, extents):
    return np.array(
        [
            center + dirs @ (extents * [-1, -1, -1]),
            center + dirs @ (extents * [1, -1, -1]),
            center + dirs @ (extents * [1, 1, -1]),
            center + dirs @ (extents * [-1, 1, -1]),
            center + dirs @ (extents * [-1, -1, 1]),
            center + dirs @ (extents * [1, -1, 1]),
            center + dirs @ (extents * [1, 1, 1]),
            center + dirs @ (extents * [-1, 1, 1]),
        ]
    )


def oobb_merge(center1, center2, dirs1, dirs2, extents1, extents2):
    c1 = oobb_corners(center1, dirs1, extents1)
    c2 = oobb_corners(center2, dirs2, extents2)
    return oobb_fit(np.vstack((c1, c2)))


def oobb_project(L, t, R, e):
    corners = oobb_corners(t, R, e)
    distances = [np.dot(L, c) for c in corners]
    return np.min(distances), np.max(distances)


def oobb_test_overlap(L, t1, t2, R1, R2, e1, e2):
    min1, max1 = oobb_project(L, t1, R1, e1)
    min2, max2 = oobb_project(L, t2, R2, e2)
    return max1 >= min2 and min2 <= max1


def oobb_intersect(t1, t2, R1, R2, e1, e2):
    # Express the second OOBB in terms of the first one.
    t = R1.T @ (t2 - t1)
    R = R1.T @ R2
    Rabs = np.abs(R) + 1e-7

    # Test 3 axes of a.
    for i in range(3):
        ra = e1[i]
        rb = np.dot(e2, Rabs[i, :])
        d = np.abs(t[i])
        if d > (ra + rb):
            return False

    # Test 3 axes of b.
    for i in range(3):
        ra = np.dot(e1, Rabs[:, i])
        rb = e2[i]
        d = np.abs(np.dot(t, R[:, i]))
        if d > (ra + rb):
            return False

    # Test other axes
    ra00 = e1[1] * Rabs[2, 0] + e1[2] * Rabs[1, 0]
    ra01 = e1[1] * Rabs[2, 1] + e1[2] * Rabs[1, 1]
    ra02 = e1[1] * Rabs[2, 2] + e1[2] * Rabs[1, 2]
    ra10 = e1[0] * Rabs[2, 0] + e1[2] * Rabs[0, 0]
    ra11 = e1[0] * Rabs[2, 1] + e1[2] * Rabs[0, 1]
    ra12 = e1[0] * Rabs[2, 2] + e1[2] * Rabs[0, 2]
    ra20 = e1[0] * Rabs[1, 0] + e1[1] * Rabs[0, 0]
    ra21 = e1[0] * Rabs[1, 1] + e1[1] * Rabs[0, 1]
    ra22 = e1[0] * Rabs[1, 2] + e1[1] * Rabs[0, 2]

    rb00 = e2[1] * Rabs[0, 2] + e2[2] * Rabs[0, 1]
    rb01 = e2[0] * Rabs[0, 2] + e2[2] * Rabs[0, 0]
    rb02 = e2[0] * Rabs[0, 1] + e2[1] * Rabs[0, 0]
    rb10 = e2[1] * Rabs[1, 2] + e2[2] * Rabs[1, 1]
    rb11 = e2[0] * Rabs[1, 2] + e2[2] * Rabs[1, 0]
    rb12 = e2[0] * Rabs[1, 1] + e2[1] * Rabs[1, 0]
    rb20 = e2[1] * Rabs[2, 2] + e2[2] * Rabs[2, 1]
    rb21 = e2[0] * Rabs[2, 2] + e2[2] * Rabs[2, 0]
    rb22 = e2[0] * Rabs[2, 1] + e2[1] * Rabs[2, 0]

    d00 = np.abs(t[2] * R[1, 0] - t[1] * R[2, 0])
    d01 = np.abs(t[2] * R[1, 1] - t[1] * R[2, 1])
    d02 = np.abs(t[2] * R[1, 2] - t[1] * R[2, 2])
    d10 = np.abs(t[0] * R[2, 0] - t[2] * R[0, 0])
    d11 = np.abs(t[0] * R[2, 1] - t[2] * R[0, 1])
    d12 = np.abs(t[0] * R[2, 2] - t[2] * R[0, 2])
    d20 = np.abs(t[1] * R[0, 0] - t[0] * R[1, 0])
    d21 = np.abs(t[1] * R[0, 1] - t[0] * R[1, 1])
    d22 = np.abs(t[1] * R[0, 2] - t[0] * R[1, 2])

    if d00 > ra00 + rb00:
        return False
    if d01 > ra01 + rb01:
        return False
    if d02 > ra02 + rb02:
        return False
    if d10 > ra10 + rb10:
        return False
    if d11 > ra11 + rb11:
        return False
    if d12 > ra12 + rb12:
        return False
    if d20 > ra20 + rb20:
        return False
    if d21 > ra21 + rb21:
        return False
    if d22 > ra22 + rb22:
        return False

    # They must be intersecting
    return True


# _____________________________________________________________________________


def generate_oobb_test_dataset():
    np.random.seed(3)
    points = np.random.uniform(-0.9, 0.9, size=(5, 1, 3)) + np.random.uniform(
        size=(5, 10, 3)
    )
    fits = [oobb_fit(x) for x in points]
    return points, fits


def export_ndarray(p, name=False, depth=0):
    # Determine typename of this ndarray level.
    if len(p.shape) == 1 and len(p) == 3:
        typename = "real3"
    elif len(p.shape) == 2 and p.shape[0] == 3 and p.shape[1] == 3:
        typename = "Matrix3x3"
    else:
        typename = "NdArray<real, %s>" % (", ".join(["%i" % d for d in p.shape]))

    # Export each.
    if len(p.shape) == 1:
        return typename + "{ " + ", ".join(["%e_r" % x for x in p]) + " }"
    else:
        vals = ", ".join([export_ndarray(x, depth=depth + 1) for x in p])
        prefix = typename + (" constexpr %s = " % name) if name else ""
        posfix = ";" if name else ""
        return prefix + typename + ("{ %s }" % vals) + posfix


def export_oobb_pointclouds(points, file):
    file.write(export_ndarray(points, name="kOobbTest_Points"))
    file.write("\n\n")


def export_oobb_fits(fits, file, prefix=""):
    centers = np.array([c for (c, _, _) in fits])
    rotations = np.array([r for (_, r, _) in fits])
    extents = np.array([e for (_, _, e) in fits])

    # Centers
    file.write(export_ndarray(centers, name=f"kOobbTest_{prefix}Translations"))
    file.write("\n\n")

    # Rotations
    file.write(export_ndarray(rotations, name=f"kOobbTest_{prefix}Rotations"))
    file.write("\n\n")

    # Extents
    file.write(export_ndarray(extents, name=f"kOobbTest_{prefix}Extents"))
    file.write("\n\n")


def export_oobb_tests(fits, file):
    nfits = len(fits)
    tests = []

    for center1, dirs1, extents1 in fits:
        t = [
            oobb_intersect(center1, center2, dirs1, dirs2, extents1, extents2)
            for center2, dirs2, extents2 in fits
        ]
        t = ", ".join(["true" if tt else "false" for tt in t])
        t = "{ " + t + " }"
        tests.append(t)

    file.write("bool constexpr kOobbTest_Overlaps[%i][%i] = {  " % (nfits, nfits))
    file.write(",  ".join(tests))
    file.write(" };\n\n")


def export_oobb_merges(fits, file):
    merges = []

    for center1, dirs1, extents1 in fits:
        for center2, dirs2, extents2 in fits:
            merges.append(
                oobb_merge(center1, center2, dirs1, dirs2, extents1, extents2)
            )

    export_oobb_fits(merges, file, prefix="Merge")


# _____________________________________________________________________________


def plot_oobb(ax, c, col):
    e = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    for a, b in e:
        ax.plot([c[a, 0], c[b, 0]], [c[a, 1], c[b, 1]], [c[a, 2], c[b, 2]], c=col)


def plot_oobb_overlap_tests(points, fits):
    fig = plt.figure()
    noobbs = len(points)

    for i in range(noobbs):
        for j in range(i + 1, noobbs):
            x, (center1, dirs1, extents1) = points[i], fits[i]
            y, (center2, dirs2, extents2) = points[j], fits[j]
            corners1 = oobb_corners(center1, dirs1, extents1)
            corners2 = oobb_corners(center2, dirs2, extents2)
            # center3, dirs3, extents3 = oobb_merge(center1, center2, dirs1, dirs2, extents1, extents2)

            test = oobb_intersect(center1, center2, dirs1, dirs2, extents1, extents2)
            color = "r" if test else "g"

            ax = fig.add_subplot(noobbs, noobbs, 1 + i + j * noobbs, projection="3d")
            plot_oobb(ax, corners1, color)
            plot_oobb(ax, corners2, color)
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=color)
            ax.scatter(y[:, 0], y[:, 1], y[:, 2], c=color)
            ax.set_xlim([0, 2])
            ax.set_ylim([0, 2])
            ax.set_zlim([0, 2])
            ax.set_title(test)

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()


def plot_oobb_merge_tests(fits):
    fig = plt.figure()
    noobbs = len(points)

    for i in range(noobbs):
        for j in range(i + 1, noobbs):
            center1, dirs1, extents1 = fits[i]
            center2, dirs2, extents2 = fits[j]
            corners1 = oobb_corners(center1, dirs1, extents1)
            corners2 = oobb_corners(center2, dirs2, extents2)

            center3, dirs3, extents3 = oobb_merge(
                center1, center2, dirs1, dirs2, extents1, extents2
            )
            corners3 = oobb_corners(center3, dirs3, extents3)

            ax = fig.add_subplot(noobbs, noobbs, 1 + i + j * noobbs, projection="3d")
            plot_oobb(ax, corners1, "r")
            plot_oobb(ax, corners2, "g")
            plot_oobb(ax, corners3, "b")
            ax.set_xlim([0, 2])
            ax.set_ylim([0, 2])
            ax.set_zlim([0, 2])

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()


points, fits = generate_oobb_test_dataset()
# plot_oobb_overlap_tests(points, fits)
# plot_oobb_merge_tests(fits)

with open("OobbTest_Data.h", "w") as handle:
    handle.write("#pragma once\n")
    handle.write("\n")
    handle.write("// clang-format off\n\n")
    handle.write("namespace mochi {\n\n")
    handle.write(f"int constexpr kOobbTest_NumFits = {points.shape[0]};\n")
    handle.write(f"int constexpr kOobbTest_NumPoints = {points.shape[1]};\n\n")
    export_oobb_pointclouds(points, handle)
    export_oobb_fits(fits, handle)
    export_oobb_tests(fits, handle)
    export_oobb_merges(fits, handle)
    handle.write("} // namespace mochi\n\n")
    handle.write("// clang-format on\n\n")

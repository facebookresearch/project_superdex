/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <mochi_core/utils/debug.h>
#include <mochi_core/utils/span.h>

namespace mochi {

/**
  Lagrange polynomial basis functions.

  Args:
    nodes(ndarray) : the 1D coordinates of the nodes
    index(int) : the index of the basis function
    order(int) : the order of the polynomial
    x(float) : the coordinate where to evaluate the basis

  Returns :
    float : the value of the index - th basis function at x
*/
template <typename T>
[[nodiscard]] inline constexpr T
GetLagrangeBasis(Span<T const> const& nodes, int index, int order, T x) {
  MOCHI_ASSERT_VERBOSE(nodes.size() == (order + 1), "Invalid number of nodes");

  // Create the initial value of the function
  T ell = T{1};

  // Loop over all nodes
  for (int i = 0; i < order + 1; ++i) {
    // If this node is the same as the support node of the basis function, skip it
    if (i == index) {
      continue;
    }

    // Otherwise perform the multiplication
    ell *= (x - nodes[i]) / (nodes[index] - nodes[i]);
  }

  return ell;
}

/**
  ..math::
  \sum_{ i = 0,i\neq j } \left[1. / (x_j - x_i) \prod_{ m = 0,m\neq i,j } \frac{ x - x_m }{x_j - x_m
  }

  Args:
  nodes(ndarray) : the coordinates of the nodes
  index(int) : the index of the basis function
  order(int) : the order of the polynomial basis
  x(float) : the coordinate where to evaluate the basis

  Returns :
  float : the value of the index - th basis function at point x
*/
template <typename T>
[[nodiscard]] inline constexpr T GetDLagrangeBasis(Span<T const> nodes, int index, int order, T x) {
  // Check that we have the right number of nodes
  MOCHI_ASSERT_VERBOSE(nodes.size() == (order + 1), "Invalid number of nodes");

  // Create the initial value of the function
  T d_ell = T{0};

  // Loop over all nodes
  for (int i = 0; i <= order; ++i) {
    if (i == index) {
      continue;
    }

    T a = T{1} / (nodes[index] - nodes[i]);

    // Loop over all nodes
    for (int j = 0; j <= order; ++j) {
      // Skip index because the python code deleted it from iter_index
      if (j == index) {
        continue;
      }

      // If this node is the same as the support node of the basis function, skip it
      if (j == i) {
        continue;
      }

      // Otherwise perform the multiplication
      a *= (x - nodes[j]) / (nodes[index] - nodes[j]);
    }

    d_ell += a;
  }

  return d_ell;
}

} // namespace mochi

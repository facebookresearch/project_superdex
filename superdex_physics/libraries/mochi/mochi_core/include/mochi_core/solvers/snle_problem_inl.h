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

#include <mochi_core/solvers/snle_problem.h>
#include <mochi_core/utils/basic_utils.h>
#include <mochi_core/utils/profile.h>

namespace mochi {

template <typename T>
SnleProblem<T>::SnleProblem(int dofsSize, int solutionSize, SnleProblemFunctions<T>&& functions)
    : solution(solutionSize), increment(dofsSize), _functions(std::move(functions)) {}

template <typename T>
int SnleProblem<T>::GetDofsSize() const {
  return this->increment.size();
}

template <typename T>
int SnleProblem<T>::GetSolutionSize() const {
  return this->solution.size();
}

template <typename T>
ColumnVectorView<T const> SnleProblem<T>::GetSolution() const {
  return this->solution;
}

template <typename T>
ColumnVectorView<T const> SnleProblem<T>::GetIncrement() const {
  return this->increment;
}

template <typename T>
void SnleProblem<T>::ScaleIncrement(T alpha) {
  MOCHI_PROFILE_SCOPE();
  this->increment *= alpha;
}

template <typename T>
double SnleProblem<T>::GetObjective() const {
  return this->objective;
}

template <typename T>
ColumnVectorView<T const> SnleProblem<T>::GetResidual() const {
  return AsConstView(_fullRes);
}

template <typename T>
bool SnleProblem<T>::HasSolutionChangedSinceLastAssembly() const {
  // The solution has changed since the last assembly if and only if _dirtyAssembly is true.
  return _dirtyAssembly;
}

template <typename T>
void SnleProblem<T>::InvalidateCachedData() {
  _dirtyAssembly = _dirtyObj = _dirtyRes = _dirtyFullDRes = true;
}

template <typename T>
void SnleProblem<T>::SetAssemblyFunction(
    std::function<void(SnleProblem<T>& problem, AssemblyParams const& params)> assemble) {
  _functions.assemble = assemble;
}

extern template struct SnleProblemFunctions<real>;
extern template class SnleProblem<real>;

} // namespace mochi

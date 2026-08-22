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

#include <mochi_physics/mochi_physics.h>
#include <mochi_physics/mochi_physics_experimental.h>

namespace mochi {

class SceneImpl;

class NewtonEulerTermsImpl : public experimental::NewtonEulerTerms {
 public:
  NewtonEulerTermsImpl(Actor* robot, Error& error);
  ~NewtonEulerTermsImpl() override;
  NewtonEulerTermsImpl(NewtonEulerTermsImpl const&) = delete;
  NewtonEulerTermsImpl& operator=(NewtonEulerTermsImpl const&) = delete;
  NewtonEulerTermsImpl(NewtonEulerTermsImpl&&) = delete;
  NewtonEulerTermsImpl& operator=(NewtonEulerTermsImpl&&) = delete;

  void Compute(
      real dt,
      Span<real const> q,
      Span<real const> dq,
      Span<real> outM,
      Span<real> outC,
      Span<real> outJ,
      Span<real> outJtF,
      Error& error) override;

 private:
  Actor* _robot = nullptr;
  SceneImpl* _scene = nullptr;
};

} // namespace mochi

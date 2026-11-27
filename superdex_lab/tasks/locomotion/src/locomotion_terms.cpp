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

// Ports unitree_rl_mjlab's velocity rewards (Apache-2.0, @1425b15,
// src/tasks/velocity/mdp/rewards.py).

#include "locomotion_terms.h"
#include "locomotion_names.h"

#include <superdex_lab/lab_kinds.h>
#include <superdex_lab/names.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace superdex::lab {

namespace {

constexpr std::string_view kTrackLinearVelocity = "track_linear_velocity";
constexpr std::string_view kTrackAngularVelocity = "track_angular_velocity";
constexpr std::string_view kFeetGait = "feet_gait";
constexpr std::string_view kFeetClearance = "feet_clearance";
constexpr std::string_view kFeetSlip = "feet_slip";
constexpr std::string_view kSoftLanding = "soft_landing";
constexpr std::string_view kVariablePosture = "variable_posture";
constexpr std::string_view kStandStill = "stand_still";
constexpr std::string_view kIsTerminated = "is_terminated";
constexpr std::string_view kProjectedGravityXy = "projected_gravity_xy";
constexpr std::string_view kBodyAngularVelocityXy = "body_angular_velocity_xy";
constexpr std::string_view kJointAccL2 = "joint_acc_l2";

} // namespace

namespace locomotion {
namespace {

/* track_linear_velocity's and track_angular_velocity's params (see locomotion_terms.h). */
struct TrackVelocityParams {
  std::string command;
  std::string velocity;
  double stdDev = 0.0;

  MOCHI_STRUCT_BEGIN(superdex::lab::locomotion::TrackVelocityParams)
  MOCHI_FIELD(command)
  MOCHI_FIELD(velocity)
  MOCHI_FIELD_NAME(stdDev, "std")
  MOCHI_STRUCT_END()
};

/* The foot terms' and stand_still's params (see locomotion_terms.h): each takes the ones it needs
   and refuses none of the others' (FootTermParams::Check names what each needs). */
struct FootTermParams {
  std::string command;
  std::string contact;
  std::string joints;
  double commandThreshold = 0.1;
  double period = 0.0;
  std::vector<double> offset;
  double threshold = 0.0;
  double targetHeight = 0.0;
  std::vector<double> stdStanding;
  std::vector<double> stdWalking;
  std::vector<double> stdRunning;
  double walkingThreshold = 0.5;
  double runningThreshold = 1.5;

  MOCHI_STRUCT_BEGIN(superdex::lab::locomotion::FootTermParams)
  MOCHI_FIELD(command)
  MOCHI_FIELD(contact)
  MOCHI_FIELD(joints)
  MOCHI_FIELD(commandThreshold)
  MOCHI_FIELD(period)
  MOCHI_FIELD(offset)
  MOCHI_FIELD(threshold)
  MOCHI_FIELD(targetHeight)
  MOCHI_FIELD(stdStanding)
  MOCHI_FIELD(stdWalking)
  MOCHI_FIELD(stdRunning)
  MOCHI_FIELD(walkingThreshold)
  MOCHI_FIELD(runningThreshold)
  MOCHI_STRUCT_END()
};

struct FeatureParams {
  std::string rotation;
  std::string velocity;
  std::string baseRotation;
  std::string velocities;

  MOCHI_STRUCT_BEGIN(superdex::lab::locomotion::FeatureParams)
  MOCHI_FIELD(rotation)
  MOCHI_FIELD(velocity)
  MOCHI_FIELD(baseRotation)
  MOCHI_FIELD(velocities)
  MOCHI_STRUCT_END()
};

} // namespace
} // namespace locomotion

namespace {

using mochi::Span;

double Squared(double x) {
  return x * x;
}

/* The command's |(vx, vy)| + |wz|, as mjlab's foot terms gate on. */
double CommandSpeed(Span<float const> c) {
  return std::hypot<double>(c[0], c[1]) + std::abs(c[2]);
}

/* The bindings every foot term shares: the command, the foot_contact outputs it reads, and its
   gate. */
struct FootBinding {
  int command = -1;
  double commandThreshold = 0.1;

  /* Binds the command and checks the gate; fails `bind` naming `needs` on a missing param. */
  static FootBinding Bind(
      TermBind& bind,
      locomotion::FootTermParams const& p,
      bool needsContact,
      std::string_view needs) {
    FootBinding b;
    if (!bind.Failed() &&
        (p.command.empty() || (needsContact && p.contact.empty()) ||
         !(p.commandThreshold >= 0.0))) {
      bind.Fail(
          std::string{"needs a command"} + (needsContact ? ", a contact" : "") +
          " and a commandThreshold of at least 0" + std::string{needs});
    }
    if (!bind.Failed()) {
      b.command = bind.Feature(names::FeatureOf(p.command, locomotion::kCommandOutput), 3);
      b.commandThreshold = p.commandThreshold;
    }
    return b;
  }
  [[nodiscard]] bool Active(TermView const& view) const {
    return CommandSpeed(view.Feature(command)) > commandThreshold;
  }
};

int ContactOutput(TermBind& bind, std::string const& contact, std::string_view output, int* width) {
  return bind.Feature(names::FeatureOf(contact, output), width);
}

class FeetGait final : public Term {
 public:
  static std::unique_ptr<Term> Make(TermBind& bind) {
    auto const p = bind.Params<locomotion::FootTermParams>();
    FootBinding const gate =
        FootBinding::Bind(bind, p, true, ", a period above 0 and an offset per foot");
    if (!bind.Failed() && !(p.period > 0.0)) {
      bind.Fail("needs a period above 0");
    }
    if (bind.Failed()) {
      return nullptr;
    }
    auto term = std::make_unique<FeetGait>();
    term->_gate = gate;
    int feet = 0;
    term->_found = ContactOutput(bind, p.contact, locomotion::kFoundOutput, &feet);
    if (!bind.Failed() && static_cast<int>(p.offset.size()) != feet) {
      bind.Fail("needs one offset per foot of its contact");
      return nullptr;
    }
    term->_period = p.period;
    term->_offset = p.offset;
    term->_threshold = p.threshold;
    term->_dt = bind.ControlDt();
    return bind.Failed() ? nullptr : std::move(term);
  }
  [[nodiscard]] double Value(TermView const& view, Span<float> /*state*/) const override {
    if (!_gate.Active(view)) {
      return 0.0;
    }
    Span<float const> const found = view.Feature(_found);
    double const phase = view.Steps() * _dt / _period;
    int matches = 0;
    for (std::size_t i = 0; i < _offset.size(); ++i) {
      double const leg = std::fmod(phase + _offset[i], 1.0);
      matches += (leg < _threshold) == (found[i] > 0.0f) ? 1 : 0;
    }
    return static_cast<double>(matches) / static_cast<double>(_offset.size());
  }

 private:
  FootBinding _gate;
  int _found = -1;
  double _period = 0.6;
  std::vector<double> _offset;
  double _threshold = 0.5;
  double _dt = 0.02;
};

/* feet_clearance, feet_slip and soft_landing: sums over feet of products of foot_contact outputs.
 */
enum class FootSum { Clearance, Slip, Landing };

template <FootSum Kind>
class FootSumTerm final : public Term {
 public:
  static std::unique_ptr<Term> Make(TermBind& bind) {
    auto const p = bind.Params<locomotion::FootTermParams>();
    FootBinding const gate = FootBinding::Bind(bind, p, true, "");
    if (bind.Failed()) {
      return nullptr;
    }
    auto term = std::make_unique<FootSumTerm>();
    term->_gate = gate;
    term->_target = p.targetHeight;
    if constexpr (Kind == FootSum::Clearance) {
      term->_a = ContactOutput(bind, p.contact, locomotion::kHeightOutput, nullptr);
      term->_b = ContactOutput(bind, p.contact, locomotion::kSpeedXyOutput, nullptr);
    } else if constexpr (Kind == FootSum::Slip) {
      term->_a = ContactOutput(bind, p.contact, locomotion::kFoundOutput, nullptr);
      term->_b = ContactOutput(bind, p.contact, locomotion::kSpeedXyOutput, nullptr);
    } else {
      term->_a = ContactOutput(bind, p.contact, locomotion::kFirstContactOutput, nullptr);
      term->_b = ContactOutput(bind, p.contact, locomotion::kForceOutput, nullptr);
    }
    return bind.Failed() ? nullptr : std::move(term);
  }
  [[nodiscard]] double Value(TermView const& view, Span<float> /*state*/) const override {
    if (!_gate.Active(view)) {
      return 0.0;
    }
    Span<float const> const a = view.Feature(_a);
    Span<float const> const b = view.Feature(_b);
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if constexpr (Kind == FootSum::Clearance) {
        sum += std::abs(a[i] - _target) * b[i];
      } else if constexpr (Kind == FootSum::Slip) {
        sum += a[i] * Squared(b[i]);
      } else {
        sum += a[i] * b[i];
      }
    }
    return sum;
  }

 private:
  FootBinding _gate;
  double _target = 0.0;
  int _a = -1;
  int _b = -1;
};

class VariablePosture final : public Term {
 public:
  static std::unique_ptr<Term> Make(TermBind& bind) {
    auto const p = bind.Params<locomotion::FootTermParams>();
    FootBinding const gate = FootBinding::Bind(bind, p, false, ", joints and three std lists");
    if (!bind.Failed() && p.joints.empty()) {
      bind.Fail("needs joints");
    }
    if (bind.Failed()) {
      return nullptr;
    }
    auto term = std::make_unique<VariablePosture>();
    term->_gate = gate;
    int joints = 0;
    term->_joints = bind.Feature(p.joints, &joints);
    auto const fits = [joints](std::vector<double> const& std) {
      return static_cast<int>(std.size()) == joints &&
          std::ranges::all_of(std, [](double s) { return s > 0.0; });
    };
    if (!bind.Failed() &&
        (!fits(p.stdStanding) || !fits(p.stdWalking) || !fits(p.stdRunning) ||
         !(p.walkingThreshold <= p.runningThreshold))) {
      bind.Fail(
          "needs stdStanding, stdWalking and stdRunning with one value above 0 per joint, and a "
          "walkingThreshold no above runningThreshold");
      return nullptr;
    }
    term->_std = {p.stdStanding, p.stdWalking, p.stdRunning};
    term->_walking = p.walkingThreshold;
    term->_running = p.runningThreshold;
    return bind.Failed() ? nullptr : std::move(term);
  }
  [[nodiscard]] double Value(TermView const& view, Span<float> /*state*/) const override {
    double const speed = CommandSpeed(view.Feature(_gate.command));
    std::vector<double> const& std = _std[speed < _walking ? 0 : (speed < _running ? 1 : 2)];
    Span<float const> const offset = view.Feature(_joints);
    double sum = 0.0;
    for (std::size_t i = 0; i < offset.size(); ++i) {
      sum += Squared(offset[i] / std[i]);
    }
    return std::exp(-sum / static_cast<double>(offset.size()));
  }

 private:
  FootBinding _gate;
  int _joints = -1;
  std::array<std::vector<double>, 3> _std;
  double _walking = 0.5;
  double _running = 1.5;
};

class StandStill final : public Term {
 public:
  static std::unique_ptr<Term> Make(TermBind& bind) {
    auto const p = bind.Params<locomotion::FootTermParams>();
    FootBinding const gate = FootBinding::Bind(bind, p, false, " and joints");
    if (!bind.Failed() && p.joints.empty()) {
      bind.Fail("needs joints");
    }
    if (bind.Failed()) {
      return nullptr;
    }
    auto term = std::make_unique<StandStill>();
    term->_gate = gate;
    term->_joints = bind.Feature(p.joints);
    return bind.Failed() ? nullptr : std::move(term);
  }
  [[nodiscard]] double Value(TermView const& view, Span<float> /*state*/) const override {
    if (_gate.Active(view)) {
      return 0.0;
    }
    double sum = 0.0;
    for (float const x : view.Feature(_joints)) {
      sum += Squared(x);
    }
    return sum;
  }

 private:
  FootBinding _gate;
  int _joints = -1;
};

class IsTerminated final : public Term {
 public:
  static std::unique_ptr<Term> Make(TermBind& bind) {
    (void)bind.Params<locomotion::FeatureParams>();
    return bind.Failed() ? nullptr : std::make_unique<IsTerminated>();
  }
  [[nodiscard]] double Value(TermView const& view, Span<float> /*state*/) const override {
    return view.Outcome() == TaskOutcome::Failure ? 1.0 : 0.0;
  }
};

/* The quaternion (x, y, z, w) `q`'s rotation matrix, row-major. */
std::array<std::array<double, 3>, 3> Matrix(Span<float const> q) {
  double const x = q[0], y = q[1], z = q[2], w = q[3];
  return {{
      {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
      {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
      {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)},
  }};
}

class ProjectedGravityXy final : public Term {
 public:
  static std::unique_ptr<Term> Make(TermBind& bind) {
    auto const p = bind.Params<locomotion::FeatureParams>();
    if (!bind.Failed() && p.rotation.empty()) {
      bind.Fail("needs a rotation");
    }
    if (bind.Failed()) {
      return nullptr;
    }
    auto term = std::make_unique<ProjectedGravityXy>();
    term->_rotation = bind.Feature(p.rotation, 4);
    return bind.Failed() ? nullptr : std::move(term);
  }
  [[nodiscard]] double Value(TermView const& view, Span<float> /*state*/) const override {
    // The unit gravity (0, 0, -1) in the link's axes is -(the third row of R).
    auto const r = Matrix(view.Feature(_rotation));
    return Squared(r[2][0]) + Squared(r[2][1]);
  }

 private:
  int _rotation = -1;
};

class BodyAngularVelocityXy final : public Term {
 public:
  static std::unique_ptr<Term> Make(TermBind& bind) {
    auto const p = bind.Params<locomotion::FeatureParams>();
    if (!bind.Failed() && (p.velocity.empty() || p.baseRotation.empty())) {
      bind.Fail("needs a velocity and a baseRotation");
    }
    if (bind.Failed()) {
      return nullptr;
    }
    auto term = std::make_unique<BodyAngularVelocityXy>();
    term->_velocity = bind.Feature(p.velocity, 3);
    term->_rotation = bind.Feature(p.baseRotation, 4);
    return bind.Failed() ? nullptr : std::move(term);
  }
  [[nodiscard]] double Value(TermView const& view, Span<float> /*state*/) const override {
    auto const r = Matrix(view.Feature(_rotation));
    Span<float const> const v = view.Feature(_velocity);
    double const x = r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2];
    double const y = r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2];
    return Squared(x) + Squared(y);
  }

 private:
  int _velocity = -1;
  int _rotation = -1;
};

class JointAccL2 final : public Term {
 public:
  static std::unique_ptr<Term> Make(TermBind& bind) {
    auto const p = bind.Params<locomotion::FeatureParams>();
    if (!bind.Failed() && p.velocities.empty()) {
      bind.Fail("needs velocities");
    }
    if (bind.Failed()) {
      return nullptr;
    }
    auto term = std::make_unique<JointAccL2>();
    int joints = 0;
    term->_velocities = bind.Feature(p.velocities, &joints);
    term->_dt = bind.ControlDt();
    // The previous velocities, then 1 once they hold a decision's.
    bind.State(joints + 1);
    return bind.Failed() ? nullptr : std::move(term);
  }
  [[nodiscard]] double Value(TermView const& view, Span<float> state) const override {
    Span<float const> const v = view.Feature(_velocities);
    bool const primed = state[v.size()] != 0.0f;
    double sum = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
      sum += Squared((v[i] - state[i]) / _dt);
      state[i] = v[i];
    }
    state[v.size()] = 1.0f;
    return primed ? sum : 0.0;
  }

 private:
  int _velocities = -1;
  double _dt = 0.02;
};

/* exp(-error / std²) of a base velocity against a twist_command's (vx, vy, wz): `Angular` picks
   mjlab's track_angular_velocity's error, else track_linear_velocity's. */
template <bool Angular>
class TrackVelocity final : public Term {
 public:
  static std::unique_ptr<Term> Make(TermBind& bind) {
    auto const p = bind.Params<locomotion::TrackVelocityParams>();
    if (!bind.Failed() && (p.command.empty() || p.velocity.empty() || !(p.stdDev > 0.0))) {
      bind.Fail("needs a command, a velocity feature and a std above 0");
    }
    if (bind.Failed()) {
      return nullptr;
    }
    auto term = std::make_unique<TrackVelocity>();
    term->_variance = p.stdDev * p.stdDev;
    term->_command = bind.Feature(names::FeatureOf(p.command, locomotion::kCommandOutput), 3);
    term->_velocity = bind.Feature(p.velocity, 3);
    return term;
  }

  [[nodiscard]] double Value(TermView const& view, Span<float> /*state*/) const override {
    Span<float const> const c = view.Feature(_command);
    Span<float const> const v = view.Feature(_velocity);
    auto const sq = [](double x) { return x * x; };
    double const error = Angular ? sq(c[2] - v[2]) + 0.05 * (sq(v[0]) + sq(v[1]))
                                 : sq(c[0] - v[0]) + sq(c[1] - v[1]) + 2.0 * sq(v[2]);
    return std::exp(-error / _variance);
  }

 private:
  double _variance = 1.0;
  int _command = -1;
  int _velocity = -1;
};

} // namespace

void RegisterLocomotionTerms(LabKinds& kinds) {
  kinds.Register(
      TermKind{.name = kTrackLinearVelocity, .make = &TrackVelocity</*Angular*/ false>::Make});
  kinds.Register(
      TermKind{.name = kTrackAngularVelocity, .make = &TrackVelocity</*Angular*/ true>::Make});
  kinds.Register(TermKind{.name = kFeetGait, .make = &FeetGait::Make});
  kinds.Register(TermKind{.name = kFeetClearance, .make = &FootSumTerm<FootSum::Clearance>::Make});
  kinds.Register(TermKind{.name = kFeetSlip, .make = &FootSumTerm<FootSum::Slip>::Make});
  kinds.Register(TermKind{.name = kSoftLanding, .make = &FootSumTerm<FootSum::Landing>::Make});
  kinds.Register(TermKind{.name = kVariablePosture, .make = &VariablePosture::Make});
  kinds.Register(TermKind{.name = kStandStill, .make = &StandStill::Make});
  kinds.Register(TermKind{.name = kIsTerminated, .make = &IsTerminated::Make});
  kinds.Register(TermKind{.name = kProjectedGravityXy, .make = &ProjectedGravityXy::Make});
  kinds.Register(TermKind{.name = kBodyAngularVelocityXy, .make = &BodyAngularVelocityXy::Make});
  kinds.Register(TermKind{.name = kJointAccL2, .make = &JointAccL2::Make});
}

} // namespace superdex::lab

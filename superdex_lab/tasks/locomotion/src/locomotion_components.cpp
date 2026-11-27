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

// Ports unitree_rl_mjlab's UniformVelocityCommand (Apache-2.0, @1425b15,
// src/tasks/velocity/mdp/velocity_command.py) and its `phase` observation (observations.py).

#include "locomotion_components.h"
#include "locomotion_names.h"

#include <superdex_lab/lab_kinds.h>
#include <superdex_lab/names.h>

#include <superdex_robotics/internal/domain_randomization/random.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <numbers>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace superdex::lab {

namespace {

constexpr std::string_view kTwistCommand = "twist_command";
constexpr std::string_view kFootContact = "foot_contact";
constexpr std::string_view kPushRobot = "push_robot";

using Range = std::array<double, 2>;
using Globals = std::array<std::string, 2>;

} // namespace

namespace locomotion {
namespace {

/* twist_command's params (see locomotion_components.h). */
struct TwistCommandParams {
  std::string base;
  Range linVelX{0.0, 0.0};
  Range linVelY{0.0, 0.0};
  Range angVelZ{0.0, 0.0};
  Range heading{-std::numbers::pi, std::numbers::pi};
  Range resamplingS{3.0, 8.0};
  double standingShare = 0.0;
  double headingShare = 1.0;
  bool headingCommand = false;
  double headingStiffness = 1.0;
  double minNorm = 0.1;
  double phasePeriodS = 0.6;
  std::optional<Globals> linVelXGlobals;
  std::optional<Globals> linVelYGlobals;
  std::optional<Globals> angVelZGlobals;

  MOCHI_STRUCT_BEGIN(superdex::lab::locomotion::TwistCommandParams)
  MOCHI_FIELD(base)
  MOCHI_FIELD(linVelX)
  MOCHI_FIELD(linVelY)
  MOCHI_FIELD(angVelZ)
  MOCHI_FIELD(heading)
  MOCHI_FIELD(resamplingS)
  MOCHI_FIELD(standingShare)
  MOCHI_FIELD(headingShare)
  MOCHI_FIELD(headingCommand)
  MOCHI_FIELD(headingStiffness)
  MOCHI_FIELD(minNorm)
  MOCHI_FIELD(phasePeriodS)
  MOCHI_FIELD(linVelXGlobals)
  MOCHI_FIELD(linVelYGlobals)
  MOCHI_FIELD(angVelZGlobals)
  MOCHI_STRUCT_END()
};

/* foot_contact's params (see locomotion_components.h). */
struct FootContactParams {
  std::vector<std::string> links;
  std::vector<std::string> sites;
  std::string ground;
  std::string base;
  double forceThreshold = 1.0;

  MOCHI_STRUCT_BEGIN(superdex::lab::locomotion::FootContactParams)
  MOCHI_FIELD(links)
  MOCHI_FIELD(sites)
  MOCHI_FIELD(ground)
  MOCHI_FIELD(base)
  MOCHI_FIELD(forceThreshold)
  MOCHI_STRUCT_END()
};

/* push_robot's params (see locomotion_components.h). */
struct PushRobotParams {
  Range intervalS{5.0, 6.0};
  std::array<Range, 3> linear{};
  std::array<Range, 3> angular{};

  MOCHI_STRUCT_BEGIN(superdex::lab::locomotion::PushRobotParams)
  MOCHI_FIELD(intervalS)
  MOCHI_FIELD(linear)
  MOCHI_FIELD(angular)
  MOCHI_STRUCT_END()
};

} // namespace
} // namespace locomotion

namespace {

/* The yaw (rad) of the x axis of the frame whose rotation is the quaternion (x, y, z, w): mjlab's
   heading_w. */
double Heading(mochi::Span<float const> q) {
  double const x = q[0], y = q[1], z = q[2], w = q[3];
  double const forwardX = 1.0 - 2.0 * (y * y + z * z);
  double const forwardY = 2.0 * (x * y + z * w);
  return std::atan2(forwardY, forwardX);
}

/* `angle` wrapped to [-pi, pi). */
double WrapToPi(double angle) {
  double const twoPi = 2.0 * std::numbers::pi;
  return angle - twoPi * std::floor((angle + std::numbers::pi) / twoPi);
}

/* The base link's rotation in the world, as a feature name: "<base>.rotation_in_task_root.quat". */
std::string RotationInWorld(std::string const& base) {
  return names::FeatureOf(
      base, std::string{names::kRotationInPrefix} + std::string{names::kTaskRoot} + ".quat");
}

/* `v` (base axes) in world axes, the base's rotation the quaternion (x, y, z, w) `q`. */
std::array<double, 3> ToWorld(mochi::Span<float const> q, mochi::Span<float const> v) {
  double const x = q[0], y = q[1], z = q[2], w = q[3];
  double const r[3][3] = {
      {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
      {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
      {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
  std::array<double, 3> out{};
  for (int i = 0; i < 3; ++i) {
    out[i] = r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2];
  }
  return out;
}

class TwistCommand final : public Component {
 public:
  static std::unique_ptr<Component> Make(BindContext& ctx) {
    auto const p = ctx.Params<locomotion::TwistCommandParams>();
    if (ctx.Failed()) {
      return nullptr;
    }
    auto const fits = [](Range const& r) { return r[0] <= r[1]; };
    if (p.base.empty() || !fits(p.linVelX) || !fits(p.linVelY) || !fits(p.angVelZ) ||
        !fits(p.heading) || !(p.resamplingS[0] > 0.0) || !fits(p.resamplingS) ||
        !(p.standingShare >= 0.0 && p.standingShare <= 1.0) ||
        !(p.headingShare >= 0.0 && p.headingShare <= 1.0) || !(p.minNorm >= 0.0) ||
        !(p.phasePeriodS > 0.0)) {
      ctx.Fail(
          "needs a base link, linVelX, linVelY, angVelZ, heading and resamplingS [lo, hi] ranges "
          "(resamplingS above 0), standingShare and headingShare in [0, 1], a minNorm of at least "
          "0 and a phasePeriodS above 0");
      return nullptr;
    }
    auto command = std::make_unique<TwistCommand>();
    command->_ranges = {p.linVelX, p.linVelY, p.angVelZ};
    std::array<std::optional<Globals> const*, 3> const globals{
        &p.linVelXGlobals, &p.linVelYGlobals, &p.angVelZGlobals};
    for (int axis = 0; axis < 3; ++axis) {
      if (*globals[axis]) {
        command->_globals[axis] = {
            ctx.Feature(names::FeatureOf("global", (**globals[axis])[0])),
            ctx.Feature(names::FeatureOf("global", (**globals[axis])[1]))};
      }
    }
    command->_heading = p.heading;
    command->_resampling = p.resamplingS;
    command->_standingShare = p.standingShare;
    command->_headingShare = p.headingShare;
    command->_headingCommand = p.headingCommand;
    command->_headingStiffness = p.headingStiffness;
    command->_minNorm = p.minNorm;
    command->_phasePeriod = p.phasePeriodS;
    command->_rotation = ctx.Feature(RotationInWorld(p.base));
    command->_command = ctx.Output(locomotion::kCommandOutput, 3);
    command->_phase = ctx.Output(locomotion::kPhaseOutput, 2);
    command->_drawKey = ctx.Random("draw");
    return ctx.Failed() ? nullptr : std::move(command);
  }

  void Reset(StepContext& ctx) override {
    _draws = 0;
    _elapsed = 0.0;
    Resample(ctx);
    WritePhase(ctx);
  }

  void Update(StepContext& ctx) override {
    _elapsed += ctx.Dt();
    _timeLeft -= ctx.Dt();
    if (_timeLeft <= 0.0) {
      Resample(ctx);
    }
    mochi::Span<float> const command = ctx.Output(_command);
    if (_headingCommand && _headingEnv) {
      double const error = WrapToPi(_headingTarget - Heading(ctx.Feature(_rotation)));
      command[2] =
          static_cast<float>(std::clamp(_headingStiffness * error, _current[2][0], _current[2][1]));
    }
    if (_standingEnv) {
      std::fill(command.begin(), command.end(), 0.0f);
    }
    WritePhase(ctx);
  }

 private:
  /* mjlab's _resample: the period, then _resample_command, from the ranges now (the globals'
     values for an axis that reads globals). */
  void Resample(StepContext& ctx) {
    for (int axis = 0; axis < 3; ++axis) {
      _current[axis] = _ranges[axis];
      if (_globals[axis][0] >= 0) {
        _current[axis] = {ctx.Feature(_globals[axis][0])[0], ctx.Feature(_globals[axis][1])[0]};
      }
    }
    uint64_t random = ctx.Random(_drawKey, _draws++);
    _timeLeft = robotics::Uniform(random, _resampling[0], _resampling[1]);
    mochi::Span<float> const command = ctx.Output(_command);
    double norm = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      double const value = robotics::Uniform(random, _current[axis][0], _current[axis][1]);
      command[axis] = static_cast<float>(value);
      norm += value * value;
    }
    if (std::sqrt(norm) <= _minNorm) {
      std::fill(command.begin(), command.end(), 0.0f);
    }
    _headingTarget = robotics::Uniform(random, _heading[0], _heading[1]);
    _headingEnv = robotics::Uniform(random, 0.0, 1.0) <= _headingShare;
    _standingEnv = robotics::Uniform(random, 0.0, 1.0) <= _standingShare;
  }

  /* mjlab velocity's phase observation at the episode's time, zeros for a command under 0.1. */
  void WritePhase(StepContext& ctx) const {
    mochi::Span<float const> const command = ctx.Output(_command);
    double const norm = std::hypot<double>(command[0], command[1], command[2]);
    mochi::Span<float> const phase = ctx.Output(_phase);
    if (norm < 0.1) {
      phase[0] = 0.0f;
      phase[1] = 0.0f;
      return;
    }
    double const cycle = std::fmod(_elapsed, _phasePeriod) / _phasePeriod;
    phase[0] = static_cast<float>(std::sin(2.0 * std::numbers::pi * cycle));
    phase[1] = static_cast<float>(std::cos(2.0 * std::numbers::pi * cycle));
  }

  std::array<Range, 3> _ranges{};
  /// The ranges of the current command, as drawn.
  std::array<Range, 3> _current{};
  std::array<std::array<int, 2>, 3> _globals{{{-1, -1}, {-1, -1}, {-1, -1}}};
  Range _heading{};
  Range _resampling{};
  double _standingShare = 0.0;
  double _headingShare = 1.0;
  bool _headingCommand = false;
  double _headingStiffness = 1.0;
  double _minNorm = 0.1;
  double _phasePeriod = 0.6;
  int _rotation = -1;
  int _command = -1;
  int _phase = -1;
  int _drawKey = -1;
  /// This episode's draws so far, and its time (s): mjlab's episode_length_buf * step_dt.
  int64_t _draws = 0;
  double _elapsed = 0.0;
  double _timeLeft = 0.0;
  double _headingTarget = 0.0;
  bool _headingEnv = false;
  bool _standingEnv = false;
};

class FootContact final : public Component {
 public:
  static std::unique_ptr<Component> Make(BindContext& ctx) {
    auto const p = ctx.Params<locomotion::FootContactParams>();
    if (ctx.Failed()) {
      return nullptr;
    }
    if (p.links.empty() || p.links.size() != p.sites.size() || p.ground.empty() || p.base.empty() ||
        !(p.forceThreshold >= 0.0)) {
      ctx.Fail(
          "needs one link set and one site per foot, a ground actor, a base link and a "
          "forceThreshold of at least 0");
      return nullptr;
    }
    auto contact = std::make_unique<FootContact>();
    int const feet = static_cast<int>(p.links.size());
    contact->_threshold = p.forceThreshold;
    contact->_rotation = ctx.Feature(RotationInWorld(p.base));
    for (int i = 0; i < feet; ++i) {
      contact->_forces.push_back(ctx.SenseFeature(p.links[i] + ".contact_force_sum." + p.ground));
      contact->_heights.push_back(ctx.Feature(
          names::FeatureOf(
              p.sites[i],
              std::string{names::kTranslationInPrefix} + std::string{names::kTaskRoot})));
      contact->_velocities.push_back(
          ctx.Feature(names::FeatureOf(p.sites[i], names::kLinearVelocityFromBase)));
    }
    contact->_found = ctx.Output(locomotion::kFoundOutput, feet);
    contact->_force = ctx.Output(locomotion::kForceOutput, feet);
    contact->_forceLog1p = ctx.Output(locomotion::kForceLog1pOutput, feet);
    contact->_airTime = ctx.Output(locomotion::kAirTimeOutput, feet);
    contact->_contactTime = ctx.Output(locomotion::kContactTimeOutput, feet);
    contact->_lastAirTime = ctx.Output(locomotion::kLastAirTimeOutput, feet);
    contact->_firstContact = ctx.Output(locomotion::kFirstContactOutput, feet);
    contact->_height = ctx.Output(locomotion::kHeightOutput, feet);
    contact->_speedXy = ctx.Output(locomotion::kSpeedXyOutput, feet);
    return ctx.Failed() ? nullptr : std::move(contact);
  }

  void Reset(StepContext& ctx) override {
    ReadSites(ctx);
  }

  void Sense(StepContext& ctx) override {
    double const dt = ctx.Dt();
    mochi::Span<float> const found = ctx.Output(_found);
    mochi::Span<float> const force = ctx.Output(_force);
    mochi::Span<float> const forceLog1p = ctx.Output(_forceLog1p);
    mochi::Span<float> const air = ctx.Output(_airTime);
    mochi::Span<float> const contact = ctx.Output(_contactTime);
    mochi::Span<float> const lastAir = ctx.Output(_lastAirTime);
    mochi::Span<float> const first = ctx.Output(_firstContact);
    for (std::size_t i = 0; i < _forces.size(); ++i) {
      force[i] = ctx.Feature(_forces[i])[0];
      forceLog1p[i] = std::log1p(force[i]);
      bool const touching = force[i] > _threshold;
      bool const landed = touching && found[i] == 0.0f;
      if (landed) {
        lastAir[i] = air[i];
      }
      found[i] = touching ? 1.0f : 0.0f;
      first[i] = landed ? 1.0f : 0.0f;
      contact[i] = touching ? contact[i] + static_cast<float>(dt) : 0.0f;
      air[i] = touching ? 0.0f : air[i] + static_cast<float>(dt);
    }
    ReadSites(ctx);
  }

 private:
  void ReadSites(StepContext& ctx) const {
    mochi::Span<float const> const rotation = ctx.Feature(_rotation);
    mochi::Span<float> const height = ctx.Output(_height);
    mochi::Span<float> const speed = ctx.Output(_speedXy);
    for (std::size_t i = 0; i < _heights.size(); ++i) {
      height[i] = ctx.Feature(_heights[i])[2];
      std::array<double, 3> const v = ToWorld(rotation, ctx.Feature(_velocities[i]));
      speed[i] = static_cast<float>(std::hypot(v[0], v[1]));
    }
  }

  double _threshold = 1.0;
  int _rotation = -1;
  std::vector<int> _forces;
  std::vector<int> _heights;
  std::vector<int> _velocities;
  int _found = -1;
  int _force = -1;
  int _forceLog1p = -1;
  int _airTime = -1;
  int _contactTime = -1;
  int _lastAirTime = -1;
  int _firstContact = -1;
  int _height = -1;
  int _speedXy = -1;
};

class PushRobot final : public Component {
 public:
  static std::unique_ptr<Component> Make(BindContext& ctx) {
    auto const p = ctx.Params<locomotion::PushRobotParams>();
    if (ctx.Failed()) {
      return nullptr;
    }
    auto const fits = [](Range const& r) { return r[0] <= r[1]; };
    if (!(p.intervalS[0] > 0.0) || !fits(p.intervalS) || !std::ranges::all_of(p.linear, fits) ||
        !std::ranges::all_of(p.angular, fits)) {
      ctx.Fail("needs intervalS [lo, hi] above 0, and linear and angular [lo, hi] per axis");
      return nullptr;
    }
    auto push = std::make_unique<PushRobot>();
    push->_interval = p.intervalS;
    push->_linear = p.linear;
    push->_angular = p.angular;
    push->_random = ctx.Random("push");
    push->_velocity = ctx.Output("velocity", 6);
    return ctx.Failed() ? nullptr : std::move(push);
  }

  void Reset(StepContext& ctx) override {
    _draws = 0;
    uint64_t state = ctx.Random(_random, _draws++);
    _timeLeft = robotics::Uniform(state, _interval[0], _interval[1]);
  }

  /* mjlab's interval event: counts down, then adds a twist drawn per axis to the robot's root
     velocity (world axes) and draws the next interval. */
  void Advance(StepContext& ctx) override {
    _timeLeft -= ctx.Dt();
    if (_timeLeft > 1e-6) {
      return;
    }
    uint64_t state = ctx.Random(_random, _draws++);
    _timeLeft = robotics::Uniform(state, _interval[0], _interval[1]);
    mochi::Span<float> const velocity = ctx.Output(_velocity);
    mochi::Real3 linear{}, angular{};
    for (int i = 0; i < 3; ++i) {
      linear[i] = static_cast<mochi::real>(robotics::Uniform(state, _linear[i][0], _linear[i][1]));
      angular[i] =
          static_cast<mochi::real>(robotics::Uniform(state, _angular[i][0], _angular[i][1]));
      velocity[i] = linear[i];
      velocity[i + 3] = angular[i];
    }
    ctx.PushRobot(linear, angular);
  }

 private:
  Range _interval{};
  std::array<Range, 3> _linear{};
  std::array<Range, 3> _angular{};
  int _random = -1;
  int _velocity = -1;
  int64_t _draws = 0;
  double _timeLeft = 0.0;
};

} // namespace

void RegisterLocomotionComponents(LabKinds& kinds) {
  kinds.Register(ComponentKind{.name = kTwistCommand, .make = &TwistCommand::Make});
  kinds.Register(ComponentKind{.name = kFootContact, .make = &FootContact::Make});
  kinds.Register(ComponentKind{.name = kPushRobot, .make = &PushRobot::Make, .event = true});
}

} // namespace superdex::lab

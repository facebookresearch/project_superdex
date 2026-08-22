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

"""Benchmark environments.

Environments here are found and registered automatically by
:mod:`superdex.lab.gym.utils.env_discovery` (call ``register_all_envs()``); this package
intentionally has no import-time side effects. Import concrete classes from their
modules, e.g. ``from superdex.lab.gym.envs.benchmarks.cartpole_env import CartPoleEnv``.
"""

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

#include <superdex_robotics/sensors/camera_sensor.h>
#include <superdex_robotics/utils/file_utils.h>

using namespace mochi;
using namespace superdex::robotics;

#if MOCHI_USE_REFLECTION

CameraSensorParams CameraSensorParams::LoadFromFile(std::string_view path, Error& error) {
  return LoadParamsFromFile<CameraSensorParams>(path, error);
}

void CameraSensorParams::SaveToFile(std::string_view path, Error& error) const {
  SaveParamsToFile(*this, path, error);
}

#else

CameraSensorParams CameraSensorParams::LoadFromFile(std::string_view, Error& error) {
  MOCHI_ERROR_SET(error, "LoadFromFile requires MOCHI_USE_REFLECTION");
  return {};
}

void CameraSensorParams::SaveToFile(std::string_view, Error& error) const {
  MOCHI_ERROR_SET(error, "SaveToFile requires MOCHI_USE_REFLECTION");
}

#endif

CameraSensor::CameraSensor(Actor* linkActor, CameraSensor::Params const& params, Error& error)
    : SensorBase(linkActor, error) {
  MOCHI_ERROR_RETURN(error);
  _params = params;
}

CameraSensor::CameraSensor(Actor* linkActor, std::string_view paramArgs, Error& error)
    : SensorBase(linkActor, error) {
  MOCHI_ERROR_RETURN(error);
  _params = LoadParamsFromPathOrJson<CameraSensorParams>(paramArgs, error);
  MOCHI_ERROR_RETURN(error);
}

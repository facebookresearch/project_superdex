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

#include "TeleopNodeZenohSettings.h"

#include "Misc/CommandLine.h"
#include "Misc/Parse.h"
#include "TeleopNodeZenohModule.h"

FZenohSessionConfig UTeleopNodeZenohSettings::GetSessionConfig() const {
  FZenohSessionConfig Config;
  Config.Mode = Mode;
  Config.ConnectEndpoints = ConnectEndpoints;
  Config.ListenEndpoint = ListenEndpoint;
  Config.bMulticastScouting = bMulticastScouting;

  const TCHAR* CommandLine = FCommandLine::Get();
  FString Value;
  if (FParse::Value(CommandLine, TEXT("-ZenohMode="), Value)) {
    if (Value.Equals(TEXT("peer"), ESearchCase::IgnoreCase)) {
      Config.Mode = EZenohSessionMode::Peer;
    } else if (Value.Equals(TEXT("client"), ESearchCase::IgnoreCase)) {
      Config.Mode = EZenohSessionMode::Client;
    } else {
      UE_LOG(LogZenoh, Warning, TEXT("Ignoring -ZenohMode=%s (expected peer or client)"), *Value);
    }
  }
  if (FParse::Value(CommandLine, TEXT("-ZenohConnect="), Value, /*bShouldStopOnSeparator=*/false)) {
    Value.ParseIntoArray(Config.ConnectEndpoints, TEXT(","), /*InCullEmpty=*/true);
  }
  if (FParse::Value(CommandLine, TEXT("-ZenohListen="), Value)) {
    Config.ListenEndpoint = Value;
  }
  FParse::Bool(CommandLine, TEXT("-ZenohScouting="), Config.bMulticastScouting);
  return Config;
}

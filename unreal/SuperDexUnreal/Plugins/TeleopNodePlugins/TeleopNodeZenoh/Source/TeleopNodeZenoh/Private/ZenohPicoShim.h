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

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ZenohPicoShim_Session ZenohPicoShim_Session;
typedef struct ZenohPicoShim_Publisher ZenohPicoShim_Publisher;
typedef struct ZenohPicoShim_Subscriber ZenohPicoShim_Subscriber;
// A received sample, valid only during the sample callback.
typedef struct ZenohPicoShim_Sample ZenohPicoShim_Sample;

typedef enum ZenohPicoShim_Mode {
  ZenohPicoShim_Mode_Client,
  ZenohPicoShim_Mode_Peer,
} ZenohPicoShim_Mode;

typedef enum ZenohPicoShim_Priority {
  ZenohPicoShim_Priority_RealTime,
  ZenohPicoShim_Priority_InteractiveHigh,
  ZenohPicoShim_Priority_InteractiveLow,
  ZenohPicoShim_Priority_DataHigh,
  ZenohPicoShim_Priority_Data,
  ZenohPicoShim_Priority_DataLow,
  ZenohPicoShim_Priority_Background,
} ZenohPicoShim_Priority;

typedef enum ZenohPicoShim_CongestionControl {
  ZenohPicoShim_CongestionControl_Drop,
  ZenohPicoShim_CongestionControl_Block,
} ZenohPicoShim_CongestionControl;

typedef struct ZenohPicoShim_SessionConfig {
  ZenohPicoShim_Mode mode;
  const char* const* connectEndpoints;
  size_t numConnectEndpoints;
  // NULL or "" to not listen.
  const char* listenEndpoint;
  bool multicastScouting;
} ZenohPicoShim_SessionConfig;

typedef struct ZenohPicoShim_PublisherOptions {
  // NULL or "" for none.
  const char* encoding;
  ZenohPicoShim_Priority priority;
  ZenohPicoShim_CongestionControl congestionControl;
  bool express;
} ZenohPicoShim_PublisherOptions;

const char* ZenohPicoShim_ErrorDescription(int result);

// On failure *outSession is NULL.
int ZenohPicoShim_OpenSession(
    const ZenohPicoShim_SessionConfig* config,
    ZenohPicoShim_Session** outSession);
// Every publisher and subscriber of the session must be undeclared first.
void ZenohPicoShim_CloseSession(ZenohPicoShim_Session* session);

// keyExpr has no leading '/', e.g. "teleop/left/arm_joint_states".
// On failure *outPublisher is NULL.
int ZenohPicoShim_DeclarePublisher(
    ZenohPicoShim_Session* session,
    const char* keyExpr,
    const ZenohPicoShim_PublisherOptions* options,
    ZenohPicoShim_Publisher** outPublisher);
void ZenohPicoShim_UndeclarePublisher(ZenohPicoShim_Publisher* publisher);
int ZenohPicoShim_Put(ZenohPicoShim_Publisher* publisher, const uint8_t* data, size_t numBytes);

size_t ZenohPicoShim_SamplePayloadSize(const ZenohPicoShim_Sample* sample);
// Copies the payload into dst, which must hold ZenohPicoShim_SamplePayloadSize bytes.
void ZenohPicoShim_SampleReadPayload(const ZenohPicoShim_Sample* sample, uint8_t* dst);

// Called on zenoh-pico's read thread.
typedef void (*ZenohPicoShim_SampleCallback)(const ZenohPicoShim_Sample* sample, void* context);
// Called exactly once when context is no longer used: after undeclaring, or right away if
// declaring fails.
typedef void (*ZenohPicoShim_DropCallback)(void* context);

// On failure *outSubscriber is NULL.
int ZenohPicoShim_DeclareSubscriber(
    ZenohPicoShim_Session* session,
    const char* keyExpr,
    ZenohPicoShim_SampleCallback onSample,
    ZenohPicoShim_DropCallback onDrop,
    void* context,
    ZenohPicoShim_Subscriber** outSubscriber);
void ZenohPicoShim_UndeclareSubscriber(ZenohPicoShim_Subscriber* subscriber);

#ifdef __cplusplus
} // extern "C"
#endif

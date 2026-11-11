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

#include "ZenohPicoShim.h"

#include <stdlib.h>

#include "zenoh-pico.h"

struct ZenohPicoShim_Session {
  z_owned_session_t session;
};

struct ZenohPicoShim_Publisher {
  z_owned_publisher_t publisher;
};

struct ZenohPicoShim_Subscriber {
  z_owned_subscriber_t subscriber;
};

// ZenohPicoShim_Sample is not defined: a sample pointer is a z_loaned_sample_t pointer.

typedef struct SubscriberContext {
  ZenohPicoShim_SampleCallback onSample;
  ZenohPicoShim_DropCallback onDrop;
  void* context;
} SubscriberContext;

static z_priority_t ToZenohPriority(ZenohPicoShim_Priority priority) {
  switch (priority) {
    case ZenohPicoShim_Priority_RealTime:
      return Z_PRIORITY_REAL_TIME;
    case ZenohPicoShim_Priority_InteractiveHigh:
      return Z_PRIORITY_INTERACTIVE_HIGH;
    case ZenohPicoShim_Priority_InteractiveLow:
      return Z_PRIORITY_INTERACTIVE_LOW;
    case ZenohPicoShim_Priority_DataHigh:
      return Z_PRIORITY_DATA_HIGH;
    case ZenohPicoShim_Priority_Data:
      return Z_PRIORITY_DATA;
    case ZenohPicoShim_Priority_DataLow:
      return Z_PRIORITY_DATA_LOW;
    case ZenohPicoShim_Priority_Background:
      return Z_PRIORITY_BACKGROUND;
  }
  return Z_PRIORITY_DEFAULT;
}

static z_congestion_control_t ToZenohCongestionControl(
    ZenohPicoShim_CongestionControl congestionControl) {
  switch (congestionControl) {
    case ZenohPicoShim_CongestionControl_Drop:
      return Z_CONGESTION_CONTROL_DROP;
    case ZenohPicoShim_CongestionControl_Block:
      return Z_CONGESTION_CONTROL_BLOCK;
  }
  return Z_CONGESTION_CONTROL_DEFAULT;
}

static bool IsSet(const char* s) {
  return s != NULL && s[0] != '\0';
}

const char* ZenohPicoShim_ErrorDescription(int result) {
  switch (result) {
    case _Z_ERR_TRANSPORT_OPEN_FAILED:
      return "could not open a link to any endpoint (is the router or peer running and reachable? "
             "On Windows, start zenohd with an IPv4 listener, e.g. -l tcp/0.0.0.0:7447)";
    case _Z_ERR_SCOUT_NO_RESULTS:
      return "no router found by multicast scouting; set a connect endpoint";
    case _Z_ERR_CONFIG_LOCATOR_INVALID:
    case _Z_ERR_CONFIG_LOCATOR_SCHEMA_UNKNOWN:
      return "invalid endpoint";
    case _Z_ERR_CONFIG_INVALID_MODE:
      return "invalid mode";
    case _Z_ERR_CONFIG_FAILED_INSERT:
      return "could not apply the session configuration";
    case _Z_ERR_SYSTEM_OUT_OF_MEMORY:
      return "out of memory";
    case _Z_ERR_SESSION_CLOSED:
      return "session closed";
    default:
      return "see zenoh-pico result codes";
  }
}

int ZenohPicoShim_OpenSession(
    const ZenohPicoShim_SessionConfig* config,
    ZenohPicoShim_Session** outSession) {
  *outSession = NULL;

  z_owned_config_t zconfig;
  z_config_default(&zconfig);
  z_result_t result = zp_config_insert(
      z_config_loan_mut(&zconfig),
      Z_CONFIG_MODE_KEY,
      config->mode == ZenohPicoShim_Mode_Peer ? "peer" : "client");
  for (size_t i = 0; result == Z_OK && i < config->numConnectEndpoints; ++i) {
    result = zp_config_insert(
        z_config_loan_mut(&zconfig), Z_CONFIG_CONNECT_KEY, config->connectEndpoints[i]);
  }
  if (result == Z_OK && IsSet(config->listenEndpoint)) {
    result =
        zp_config_insert(z_config_loan_mut(&zconfig), Z_CONFIG_LISTEN_KEY, config->listenEndpoint);
  }
  if (result == Z_OK) {
    result = zp_config_insert(
        z_config_loan_mut(&zconfig),
        Z_CONFIG_MULTICAST_SCOUTING_KEY,
        config->multicastScouting ? "true" : "false");
  }
  if (result != Z_OK) {
    z_config_drop(z_config_move(&zconfig));
    return result;
  }

  ZenohPicoShim_Session* session = malloc(sizeof(ZenohPicoShim_Session));
  if (session == NULL) {
    z_config_drop(z_config_move(&zconfig));
    return _Z_ERR_SYSTEM_OUT_OF_MEMORY;
  }
  result = z_open(&session->session, z_config_move(&zconfig), NULL);
  if (result != Z_OK) {
    free(session);
    return result;
  }
  *outSession = session;
  return Z_OK;
}

void ZenohPicoShim_CloseSession(ZenohPicoShim_Session* session) {
  if (session != NULL) {
    z_session_drop(z_session_move(&session->session));
    free(session);
  }
}

int ZenohPicoShim_DeclarePublisher(
    ZenohPicoShim_Session* session,
    const char* keyExpr,
    const ZenohPicoShim_PublisherOptions* options,
    ZenohPicoShim_Publisher** outPublisher) {
  *outPublisher = NULL;

  z_view_keyexpr_t key;
  z_result_t result = z_view_keyexpr_from_str(&key, keyExpr);
  if (result != Z_OK) {
    return result;
  }

  z_publisher_options_t zoptions;
  z_publisher_options_default(&zoptions);
  zoptions.priority = ToZenohPriority(options->priority);
  zoptions.congestion_control = ToZenohCongestionControl(options->congestionControl);
  zoptions.is_express = options->express;
  z_owned_encoding_t encoding;
  if (IsSet(options->encoding)) {
    result = z_encoding_from_str(&encoding, options->encoding);
    if (result != Z_OK) {
      return result;
    }
    zoptions.encoding = z_encoding_move(&encoding);
  }

  ZenohPicoShim_Publisher* publisher = malloc(sizeof(ZenohPicoShim_Publisher));
  if (publisher == NULL) {
    if (zoptions.encoding != NULL) {
      z_encoding_drop(zoptions.encoding);
    }
    return _Z_ERR_SYSTEM_OUT_OF_MEMORY;
  }
  result = z_declare_publisher(
      z_session_loan(&session->session),
      &publisher->publisher,
      z_view_keyexpr_loan(&key),
      &zoptions);
  if (result != Z_OK) {
    free(publisher);
    return result;
  }
  *outPublisher = publisher;
  return Z_OK;
}

void ZenohPicoShim_UndeclarePublisher(ZenohPicoShim_Publisher* publisher) {
  if (publisher != NULL) {
    z_undeclare_publisher(z_publisher_move(&publisher->publisher));
    free(publisher);
  }
}

int ZenohPicoShim_Put(ZenohPicoShim_Publisher* publisher, const uint8_t* data, size_t numBytes) {
  z_owned_bytes_t payload;
  z_result_t result = z_bytes_copy_from_buf(&payload, data, numBytes);
  if (result != Z_OK) {
    return result;
  }
  return z_publisher_put(z_publisher_loan(&publisher->publisher), z_bytes_move(&payload), NULL);
}

size_t ZenohPicoShim_SamplePayloadSize(const ZenohPicoShim_Sample* sample) {
  return z_bytes_len(z_sample_payload((const z_loaned_sample_t*)sample));
}

void ZenohPicoShim_SampleReadPayload(const ZenohPicoShim_Sample* sample, uint8_t* dst) {
  const z_loaned_bytes_t* payload = z_sample_payload((const z_loaned_sample_t*)sample);
  z_bytes_reader_t reader = z_bytes_get_reader(payload);
  z_bytes_reader_read(&reader, dst, z_bytes_len(payload));
}

static void OnSample(z_loaned_sample_t* sample, void* arg) {
  const SubscriberContext* context = arg;
  context->onSample((const ZenohPicoShim_Sample*)sample, context->context);
}

static void OnDrop(void* arg) {
  SubscriberContext* context = arg;
  if (context->onDrop != NULL) {
    context->onDrop(context->context);
  }
  free(context);
}

int ZenohPicoShim_DeclareSubscriber(
    ZenohPicoShim_Session* session,
    const char* keyExpr,
    ZenohPicoShim_SampleCallback onSample,
    ZenohPicoShim_DropCallback onDrop,
    void* context,
    ZenohPicoShim_Subscriber** outSubscriber) {
  *outSubscriber = NULL;

  SubscriberContext* subscriberContext = malloc(sizeof(SubscriberContext));
  ZenohPicoShim_Subscriber* subscriber = malloc(sizeof(ZenohPicoShim_Subscriber));
  if (subscriberContext == NULL || subscriber == NULL) {
    free(subscriberContext);
    free(subscriber);
    if (onDrop != NULL) {
      onDrop(context);
    }
    return _Z_ERR_SYSTEM_OUT_OF_MEMORY;
  }
  subscriberContext->onSample = onSample;
  subscriberContext->onDrop = onDrop;
  subscriberContext->context = context;

  z_view_keyexpr_t key;
  z_result_t result = z_view_keyexpr_from_str(&key, keyExpr);
  if (result != Z_OK) {
    OnDrop(subscriberContext);
    free(subscriber);
    return result;
  }

  // From here on zenoh-pico owns subscriberContext and calls OnDrop, also if declaring fails.
  z_owned_closure_sample_t closure;
  z_closure_sample(&closure, &OnSample, &OnDrop, subscriberContext);
  result = z_declare_subscriber(
      z_session_loan(&session->session),
      &subscriber->subscriber,
      z_view_keyexpr_loan(&key),
      z_closure_sample_move(&closure),
      NULL);
  if (result != Z_OK) {
    free(subscriber);
    return result;
  }
  *outSubscriber = subscriber;
  return Z_OK;
}

void ZenohPicoShim_UndeclareSubscriber(ZenohPicoShim_Subscriber* subscriber) {
  if (subscriber != NULL) {
    z_undeclare_subscriber(z_subscriber_move(&subscriber->subscriber));
    free(subscriber);
  }
}

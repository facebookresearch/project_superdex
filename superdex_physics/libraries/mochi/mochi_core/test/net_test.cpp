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

#include <mochi_core/net/client_socket.h>
#include <mochi_core/net/server_list.h>
#include <mochi_core/net/server_socket.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/rand_utils.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>

using namespace mochi;
using namespace mochi::net;

// -------------------------------------------------------------------------------------------------
// Transport selection
// -------------------------------------------------------------------------------------------------

// When MOCHI_TEST_REAL_SOCKETS is 1 every test below drives the real TCP transport
// (ServerSocket::Start / ClientSocket::Connect).
// When 0 the SAME tests run over the in-process loopback transport (StartInProc / ConnectInProc).
// The default is zero because CI machines may have restricted network privileges.
#ifndef MOCHI_TEST_REAL_SOCKETS
#define MOCHI_TEST_REAL_SOCKETS 0
#endif

// -------------------------------------------------------------------------------------------------
// Test Helpers
// -------------------------------------------------------------------------------------------------

namespace {

// Thread-safe collection of received messages.
struct Sink {
  struct Entry {
    ClientId client{0};
    DynamicArray<uint8_t> data;
  };

  Guarded<DynamicArray<Entry>> entries;

  void Append(ClientId client, void const* data, size_t size) {
    Entry e;
    e.client = client;
    e.data.resize_noinit(size);
    if (size > 0) {
      std::memcpy(e.data.data(), data, size);
    }
    entries.Mutate([&](DynamicArray<Entry>& v) { v.push_back(std::move(e)); });
  }

  [[nodiscard]] size_t Count() const {
    return entries.Read([](DynamicArray<Entry> const& v) { return v.size(); });
  }

  [[nodiscard]] Entry Get(size_t i) const {
    return entries.Read([&](DynamicArray<Entry> const& v) { return v[i]; });
  }
};

} // namespace

// Generate a payload of random bytes, seeded so distinct seeds produce distinct payloads.
static DynamicArray<uint8_t> MakePayload(size_t size, uint32_t seed = 0) {
  DynamicArray<uint8_t> data;
  data.resize_noinit(size);
  auto rng = RandomGenerator(seed);
  SetRandom(rng, uint8_t{0}, uint8_t{255}, MakeSpan(data));
  return data;
}

// Poll a predicate in a bounded loop (condition-based waiting; avoids fixed sleeps). Returns true
// as soon as the predicate holds, false if the timeout elapses first.
template <typename Predicate>
static bool PollUntil(Predicate&& pred, float timeoutSec = 60.0f) {
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::duration<float>(timeoutSec);
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return pred();
}

// Start a server on the active transport and route received messages into `sink`.
static void StartServer(ServerSocket& server, Sink& sink, int maxClients = 4) {
  server.SetReceiveCallback(
      [&sink](ClientId id, void const* data, size_t size) { sink.Append(id, data, size); });
#if MOCHI_TEST_REAL_SOCKETS
  // Tests may run in parallel on a single machine. Therefore, they cannot all use the same port.
  // Port 0 is specified here, which means "any available port".
  server.Start(/*preferredPort*/ 0, maxClients, "TestServer");
#else
  server.StartInProc(maxClients, "TestServer");
#endif
}

// Initiate a connection without asserting the outcome.
static void InitiateConnect(ClientSocket& client, ServerSocket& server, Sink& sink) {
  client.SetReceiveCallback([&sink](void const* data, size_t size) { sink.Append(0, data, size); });
#if MOCHI_TEST_REAL_SOCKETS
  client.Connect("127.0.0.1", server.GetPort());
#else
  client.ConnectInProc(server);
#endif
}

// Connect a client on the active transport and route received messages into `sink`, waiting for the
// connection to reach Connected. Use this only when the connection is expected to succeed.
static void ConnectClient(ClientSocket& client, ServerSocket& server, Sink& sink) {
  InitiateConnect(client, server, sink);
  EXPECT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Connected; }));
}

// Learn the server-side ClientId for `client` by round-tripping a unique probe. Only `client`
// should be sending while this runs, so the freshly appended server message is its probe.
static ClientId IdentifyClient(Sink& serverSink, ClientSocket& client, uint32_t salt) {
  size_t const before = serverSink.Count();
  auto const probe = MakePayload(8, 0xC0DE0000u + salt);
  EXPECT_TRUE(client.Send(probe.data(), probe.size()));
  EXPECT_TRUE(PollUntil([&] { return serverSink.Count() > before; }));
  auto const entry = serverSink.Get(before);
  EXPECT_EQ(entry.data, probe);
  return entry.client;
}

// -------------------------------------------------------------------------------------------------
// Connection + round trip
// -------------------------------------------------------------------------------------------------

TEST(Net, ClientServerRoundTrip) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  Sink clientSink;
  ConnectClient(client, server, clientSink);
  EXPECT_EQ(client.GetStatus(), SocketStatus::Connected);

  auto const up = MakePayload(5, 1);
  EXPECT_TRUE(client.Send(up.data(), up.size()));
  ASSERT_TRUE(PollUntil([&] { return serverSink.Count() >= 1; }));
  EXPECT_EQ(serverSink.Get(0).data, up);

  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 1u; }));
  auto const clients = server.GetClients();
  ASSERT_EQ(clients.size(), 1u);
  auto const down = MakePayload(5, 2);
  EXPECT_TRUE(server.SendTo(clients[0], down.data(), down.size()));
  ASSERT_TRUE(PollUntil([&] { return clientSink.Count() >= 1; }));
  EXPECT_EQ(clientSink.Get(0).data, down);

  client.Disconnect();
  server.Stop();
}

TEST(Net, MultipleClients) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client1;
  ClientSocket client2;
  Sink sink1;
  Sink sink2;
  ConnectClient(client1, server, sink1);
  ConnectClient(client2, server, sink2);
  EXPECT_EQ(client1.GetStatus(), SocketStatus::Connected);
  EXPECT_EQ(client2.GetStatus(), SocketStatus::Connected);

  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 2u; }));
  auto const clients = server.GetClients();
  ASSERT_EQ(clients.size(), 2u);

  // clients[i] is reported in connection order, so clients[0] is client1 and clients[1] is client2.
  auto const msg1 = MakePayload(11, 1);
  auto const msg2 = MakePayload(11, 2);
  EXPECT_TRUE(server.SendTo(clients[0], msg1.data(), msg1.size()));
  EXPECT_TRUE(server.SendTo(clients[1], msg2.data(), msg2.size()));

  ASSERT_TRUE(PollUntil([&] { return sink1.Count() >= 1 && sink2.Count() >= 1; }));
  // Each client received exactly the distinct payload addressed to it.
  EXPECT_EQ(sink1.Get(0).data, msg1);
  EXPECT_EQ(sink2.Get(0).data, msg2);

  client1.Disconnect();
  client2.Disconnect();
  server.Stop();
}

TEST(Net, Broadcast) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client1;
  ClientSocket client2;
  Sink sink1;
  Sink sink2;
  ConnectClient(client1, server, sink1);
  ConnectClient(client2, server, sink2);
  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 2u; }));

  auto const payload = MakePayload(13, 9);
  server.Broadcast(payload.data(), payload.size());

  ASSERT_TRUE(PollUntil([&] { return sink1.Count() >= 1 && sink2.Count() >= 1; }));
  EXPECT_EQ(sink1.Get(0).data, payload);
  EXPECT_EQ(sink2.Get(0).data, payload);

  client1.Disconnect();
  client2.Disconnect();
  server.Stop();
}

// A multi-kilobyte message survives a round trip intact in both directions, exercising the partial
// send/recv loops on the real-socket path (the message stays below the internal frame-size cap).
TEST(Net, LargeMessageRoundTrip) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  Sink clientSink;
  ConnectClient(client, server, clientSink);
  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 1u; }));
  auto const clients = server.GetClients();
  ASSERT_EQ(clients.size(), 1u);

  constexpr size_t kSize = 8u * 1024u;

  auto const up = MakePayload(kSize, 1);
  EXPECT_TRUE(client.Send(up.data(), up.size()));
  ASSERT_TRUE(PollUntil([&] { return serverSink.Count() >= 1; }));
  EXPECT_EQ(serverSink.Get(0).data, up);

  auto const down = MakePayload(kSize, 2);
  EXPECT_TRUE(server.SendTo(clients[0], down.data(), down.size()));
  ASSERT_TRUE(PollUntil([&] { return clientSink.Count() >= 1; }));
  EXPECT_EQ(clientSink.Get(0).data, down);

  client.Disconnect();
  server.Stop();
}

// A zero-length client->server message is delivered as an empty message, not dropped.
TEST(Net, ZeroLengthMessage) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  Sink clientSink;
  ConnectClient(client, server, clientSink);

  EXPECT_TRUE(client.Send(nullptr, 0));

  ASSERT_TRUE(PollUntil([&] { return serverSink.Count() >= 1; }));
  EXPECT_EQ(serverSink.Get(0).data.size(), 0u);

  client.Disconnect();
  server.Stop();
}

// Zero-length server->client messages (via SendTo and Broadcast) are delivered as empty messages.
TEST(Net, ServerToClientZeroLengthMessage) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  Sink clientSink;
  ConnectClient(client, server, clientSink);
  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 1u; }));
  auto const clients = server.GetClients();
  ASSERT_EQ(clients.size(), 1u);

  EXPECT_TRUE(server.SendTo(clients[0], nullptr, 0));
  server.Broadcast(nullptr, 0);

  ASSERT_TRUE(PollUntil([&] { return clientSink.Count() >= 2; }));
  EXPECT_EQ(clientSink.Get(0).data.size(), 0u);
  EXPECT_EQ(clientSink.Get(1).data.size(), 0u);

  client.Disconnect();
  server.Stop();
}

// A connected client and server that send nothing observe no spurious messages.
TEST(Net, NoSpuriousMessages) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  Sink clientSink;
  ConnectClient(client, server, clientSink);

  float constexpr kShortTimeOut = 0.1f; // seconds
  EXPECT_FALSE(
      PollUntil([&] { return serverSink.Count() > 0 || clientSink.Count() > 0; }, kShortTimeOut));

  client.Disconnect();
  server.Stop();
}

// A message sent from a background thread is delivered to the server's receive callback.
TEST(Net, ServerReceivesMessageCrossThread) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  Sink clientSink;
  ConnectClient(client, server, clientSink);

  auto const payload = MakePayload(16, 0x51);
  std::thread sender([&] { client.Send(payload.data(), payload.size()); });

  ASSERT_TRUE(PollUntil([&] { return serverSink.Count() >= 1; }));
  EXPECT_EQ(serverSink.Get(0).data, payload);

  sender.join();
  client.Disconnect();
  server.Stop();
}

// -------------------------------------------------------------------------------------------------
// Sending from within callbacks
// -------------------------------------------------------------------------------------------------
//
// It must be legal to call Send / SendTo / Broadcast from inside any transport callback. On TCP
// this is naturally safe (sends are async and callbacks run on dedicated threads). On the
// in-process transport the send is deferred until the running callback returns, so the peer's
// callback never runs synchronously nested inside it. Where a "no reentry" property is asserted, it
// is checked by comparing thread ids: in-proc runs the peer callback after the current one returns
// (same thread, not nested); TCP runs it on another thread (concurrency, not reentry).

// A client that sends from its status callback (on Connected) reaches the server.
TEST(Net, ClientSendsFromStatusCallback) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  auto const payload = MakePayload(16, 0xA1);
  ClientSocket client;
  client.SetStatusCallback([&](SocketStatus s) {
    if (s == SocketStatus::Connected) {
      EXPECT_TRUE(client.Send(payload.data(), payload.size()));
    }
  });
#if MOCHI_TEST_REAL_SOCKETS
  client.Connect("127.0.0.1", server.GetPort());
#else
  client.ConnectInProc(server);
#endif

  ASSERT_TRUE(PollUntil([&] { return serverSink.Count() >= 1; }));
  EXPECT_EQ(serverSink.Get(0).data, payload);

  client.Disconnect();
  server.Stop();
}

// Multiple sends from one callback arrive in the order they were issued (deferred-delivery FIFO).
TEST(Net, MultipleSendsFromCallbackPreserveFifoOrder) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  auto const p0 = MakePayload(3, 0xF0);
  auto const p1 = MakePayload(3, 0xF1);
  auto const p2 = MakePayload(3, 0xF2);
  ClientSocket client;
  client.SetStatusCallback([&](SocketStatus s) {
    if (s == SocketStatus::Connected) {
      EXPECT_TRUE(client.Send(p0.data(), p0.size()));
      EXPECT_TRUE(client.Send(p1.data(), p1.size()));
      EXPECT_TRUE(client.Send(p2.data(), p2.size()));
    }
  });
#if MOCHI_TEST_REAL_SOCKETS
  client.Connect("127.0.0.1", server.GetPort());
#else
  client.ConnectInProc(server);
#endif

  ASSERT_TRUE(PollUntil([&] { return serverSink.Count() >= 3; }));
  EXPECT_EQ(serverSink.Get(0).data, p0);
  EXPECT_EQ(serverSink.Get(1).data, p1);
  EXPECT_EQ(serverSink.Get(2).data, p2);

  client.Disconnect();
  server.Stop();
}

// A client that replies from its receive callback reaches the server, without the server's receive
// callback running synchronously nested inside the client's.
TEST(Net, ClientSendsFromReceiveCallbackWithoutReentry) {
  ServerSocket server;
  Sink serverSink;
  Guarded<std::thread::id> clientCbThread;
  std::atomic<bool> reentry{false};
  server.SetReceiveCallback([&](ClientId id, void const* d, size_t n) {
    if (clientCbThread.Read(
            [](std::thread::id const& t) { return t == std::this_thread::get_id(); })) {
      reentry.store(true);
    }
    serverSink.Append(id, d, n);
  });
#if MOCHI_TEST_REAL_SOCKETS
  server.Start(0, 4, "TestServer");
#else
  server.StartInProc(4, "TestServer");
#endif

  auto const reply = MakePayload(4, 0xB1);
  ClientSocket client;
  Sink clientSink;
  client.SetReceiveCallback([&](void const* d, size_t n) {
    clientCbThread.Store(std::this_thread::get_id());
    clientSink.Append(0, d, n);
    EXPECT_TRUE(client.Send(reply.data(), reply.size()));
    clientCbThread.Store({});
  });
#if MOCHI_TEST_REAL_SOCKETS
  client.Connect("127.0.0.1", server.GetPort());
#else
  client.ConnectInProc(server);
#endif
  ASSERT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Connected; }));
  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 1u; }));

  auto const down = MakePayload(4, 0xB2);
  ASSERT_TRUE(server.SendTo(server.GetClients()[0], down.data(), down.size()));

  ASSERT_TRUE(PollUntil([&] { return serverSink.Count() >= 1; }));
  EXPECT_EQ(serverSink.Get(0).data, reply);
  EXPECT_FALSE(reentry.load());

  client.Disconnect();
  server.Stop();
}

// A server that replies via SendTo from its receive callback reaches the client, without the
// client's receive callback running synchronously nested inside the server's.
TEST(Net, ServerSendsToFromReceiveCallbackWithoutReentry) {
  ServerSocket server;
  Sink serverSink;
  auto const echo = MakePayload(4, 0xC1);
  Guarded<std::thread::id> serverCbThread;
  std::atomic<bool> reentry{false};
  server.SetReceiveCallback([&](ClientId id, void const* d, size_t n) {
    serverCbThread.Store(std::this_thread::get_id());
    serverSink.Append(id, d, n);
    EXPECT_TRUE(server.SendTo(id, echo.data(), echo.size()));
    serverCbThread.Store({});
  });
#if MOCHI_TEST_REAL_SOCKETS
  server.Start(0, 4, "TestServer");
#else
  server.StartInProc(4, "TestServer");
#endif

  ClientSocket client;
  Sink clientSink;
  client.SetReceiveCallback([&](void const* d, size_t n) {
    if (serverCbThread.Read(
            [](std::thread::id const& t) { return t == std::this_thread::get_id(); })) {
      reentry.store(true);
    }
    clientSink.Append(0, d, n);
  });
#if MOCHI_TEST_REAL_SOCKETS
  client.Connect("127.0.0.1", server.GetPort());
#else
  client.ConnectInProc(server);
#endif
  ASSERT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Connected; }));

  auto const up = MakePayload(4, 0xC0);
  EXPECT_TRUE(client.Send(up.data(), up.size()));

  ASSERT_TRUE(PollUntil([&] { return clientSink.Count() >= 1; }));
  EXPECT_EQ(clientSink.Get(0).data, echo);
  EXPECT_FALSE(reentry.load());

  client.Disconnect();
  server.Stop();
}

// A server that broadcasts from its receive callback reaches every client, without any client's
// receive callback running synchronously nested inside the server's.
TEST(Net, ServerBroadcastsFromReceiveCallbackWithoutReentry) {
  ServerSocket server;
  Sink serverSink;
  auto const echo = MakePayload(4, 0xE1);
  Guarded<std::thread::id> serverCbThread;
  std::atomic<bool> reentry{false};
  server.SetReceiveCallback([&](ClientId id, void const* d, size_t n) {
    serverCbThread.Store(std::this_thread::get_id());
    serverSink.Append(id, d, n);
    server.Broadcast(echo.data(), echo.size());
    serverCbThread.Store({});
  });
#if MOCHI_TEST_REAL_SOCKETS
  server.Start(0, 4, "TestServer");
#else
  server.StartInProc(4, "TestServer");
#endif

  auto const observeReentry = [&] {
    if (serverCbThread.Read(
            [](std::thread::id const& t) { return t == std::this_thread::get_id(); })) {
      reentry.store(true);
    }
  };
  ClientSocket client1;
  ClientSocket client2;
  Sink sink1;
  Sink sink2;
  client1.SetReceiveCallback([&](void const* d, size_t n) {
    observeReentry();
    sink1.Append(0, d, n);
  });
  client2.SetReceiveCallback([&](void const* d, size_t n) {
    observeReentry();
    sink2.Append(0, d, n);
  });
#if MOCHI_TEST_REAL_SOCKETS
  client1.Connect("127.0.0.1", server.GetPort());
  client2.Connect("127.0.0.1", server.GetPort());
#else
  client1.ConnectInProc(server);
  client2.ConnectInProc(server);
#endif
  ASSERT_TRUE(PollUntil([&] { return client1.GetStatus() == SocketStatus::Connected; }));
  ASSERT_TRUE(PollUntil([&] { return client2.GetStatus() == SocketStatus::Connected; }));
  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 2u; }));

  auto const trigger = MakePayload(4, 0xE0);
  EXPECT_TRUE(client1.Send(trigger.data(), trigger.size()));

  ASSERT_TRUE(PollUntil([&] { return sink1.Count() >= 1 && sink2.Count() >= 1; }));
  EXPECT_EQ(sink1.Get(0).data, echo);
  EXPECT_EQ(sink2.Get(0).data, echo);
  EXPECT_FALSE(reentry.load());

  client1.Disconnect();
  client2.Disconnect();
  server.Stop();
}

// A server that greets each new client via SendTo from its client-event (Connected) callback
// reaches the client.
TEST(Net, ServerSendsFromClientEventCallback) {
  ServerSocket server;
  Sink serverSink;
  server.SetReceiveCallback(
      [&serverSink](ClientId id, void const* d, size_t n) { serverSink.Append(id, d, n); });
  auto const greeting = MakePayload(6, 0xD1);
  std::atomic<bool> greeted{false};
  server.SetClientCallback([&](ClientId id, ClientEvent event) {
    if (event == ClientEvent::Connected) {
      EXPECT_TRUE(server.SendTo(id, greeting.data(), greeting.size()));
      greeted.store(true);
    }
  });
#if MOCHI_TEST_REAL_SOCKETS
  server.Start(0, 4, "TestServer");
#else
  server.StartInProc(4, "TestServer");
#endif

  ClientSocket client;
  Sink clientSink;
  client.SetReceiveCallback([&clientSink](void const* d, size_t n) { clientSink.Append(0, d, n); });
#if MOCHI_TEST_REAL_SOCKETS
  client.Connect("127.0.0.1", server.GetPort());
#else
  client.ConnectInProc(server);
#endif

  ASSERT_TRUE(PollUntil([&] { return clientSink.Count() >= 1; }));
  EXPECT_EQ(clientSink.Get(0).data, greeting);
  EXPECT_TRUE(greeted.load());

  client.Disconnect();
  server.Stop();
}

// -------------------------------------------------------------------------------------------------
// Status + callbacks
// -------------------------------------------------------------------------------------------------

TEST(Net, StatusTransitions) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  EXPECT_EQ(client.GetStatus(), SocketStatus::None);

  Sink clientSink;
  ConnectClient(client, server, clientSink);
  EXPECT_EQ(client.GetStatus(), SocketStatus::Connected);

  client.Disconnect();
  EXPECT_EQ(client.GetStatus(), SocketStatus::None);

  server.Stop();
}

TEST(Net, StatusCallback) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  std::atomic<int> callbackCount{0};
  std::atomic<SocketStatus> lastStatus{SocketStatus::None};

  ClientSocket client;
  client.SetStatusCallback([&](SocketStatus s) {
    lastStatus.store(s);
    callbackCount.fetch_add(1);
  });

  Sink clientSink;
  ConnectClient(client, server, clientSink);
  EXPECT_TRUE(PollUntil([&] { return lastStatus.load() == SocketStatus::Connected; }));
  EXPECT_GE(callbackCount.load(), 1);

  client.Disconnect();
  EXPECT_EQ(lastStatus.load(), SocketStatus::None);

  server.Stop();
}

TEST(Net, ClientEventCallback) {
  ServerSocket server;
  Sink serverSink;

  std::atomic<int> connectCount{0};
  std::atomic<int> disconnectCount{0};
  server.SetClientCallback([&](ClientId, ClientEvent event) {
    if (event == ClientEvent::Connected) {
      connectCount.fetch_add(1);
    } else {
      disconnectCount.fetch_add(1);
    }
  });

  StartServer(server, serverSink);

  ClientSocket client;
  Sink clientSink;
  ConnectClient(client, server, clientSink);
  EXPECT_TRUE(PollUntil([&] { return connectCount.load() == 1; }));
  EXPECT_EQ(disconnectCount.load(), 0);

  client.Disconnect();
  EXPECT_TRUE(PollUntil([&] { return disconnectCount.load() == 1; }));

  server.Stop();
}

// When the server stops while a client is still connected, the server's client callback fires
// Disconnected for that client. Covers both transports: TcpServerTransport::Stop() and
// InProcServerTransport teardown.
TEST(Net, ServerFiresDisconnectOnServerStop) {
  ServerSocket server;
  Sink serverSink;

  std::atomic<int> connectCount{0};
  std::atomic<int> disconnectCount{0};
  server.SetClientCallback([&](ClientId, ClientEvent event) {
    if (event == ClientEvent::Connected) {
      connectCount.fetch_add(1);
    } else {
      disconnectCount.fetch_add(1);
    }
  });

  StartServer(server, serverSink);

  ClientSocket client;
  Sink clientSink;
  ConnectClient(client, server, clientSink);
  ASSERT_TRUE(PollUntil([&] { return connectCount.load() == 1; }));
  EXPECT_EQ(disconnectCount.load(), 0);

  // Stopping the server (not the client) must still deliver the server-side disconnect event.
  server.Stop();
  EXPECT_TRUE(PollUntil([&] { return disconnectCount.load() == 1; }));

  client.Disconnect();
}

// When the server stops, a connected client transitions to Lost and a subsequent Send fails.
TEST(Net, ClientObservesLostOnServerStop) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  Sink clientSink;
  ConnectClient(client, server, clientSink);
  EXPECT_EQ(client.GetStatus(), SocketStatus::Connected);

  server.Stop();
  EXPECT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Lost; }));

  auto const payload = MakePayload(16, 0x41);
  EXPECT_FALSE(client.Send(payload.data(), payload.size()));

  client.Disconnect();
}

// -------------------------------------------------------------------------------------------------
// Routing and sender identity
// -------------------------------------------------------------------------------------------------

// The server tags each received message with the sender's ClientId, matching the ids reported by
// GetClients().
TEST(Net, ServerReceiveIdentifiesSender) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client1;
  ClientSocket client2;
  Sink sink1;
  Sink sink2;
  ConnectClient(client1, server, sink1);
  ConnectClient(client2, server, sink2);

  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 2u; }));
  auto const knownIds = server.GetClients();
  ASSERT_EQ(knownIds.size(), 2u);

  ClientId const id1 = IdentifyClient(serverSink, client1, 11);
  ClientId const id2 = IdentifyClient(serverSink, client2, 22);

  EXPECT_NE(id1, 0u);
  EXPECT_NE(id2, 0u);
  EXPECT_NE(id1, id2);

  auto const isKnown = [&](ClientId id) {
    for (auto known : knownIds) {
      if (known == id) {
        return true;
      }
    }
    return false;
  };
  EXPECT_TRUE(isKnown(id1));
  EXPECT_TRUE(isKnown(id2));

  client1.Disconnect();
  client2.Disconnect();
  server.Stop();
}

// SendTo delivers only to the targeted client; the other client receives nothing.
TEST(Net, SendToIsolatedToTargetClient) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client1;
  ClientSocket client2;
  Sink sink1;
  Sink sink2;
  ConnectClient(client1, server, sink1);
  ConnectClient(client2, server, sink2);

  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 2u; }));
  ClientId const id1 = IdentifyClient(serverSink, client1, 1);
  ClientId const id2 = IdentifyClient(serverSink, client2, 2);
  EXPECT_NE(id1, 0u);
  EXPECT_NE(id2, 0u);
  EXPECT_NE(id1, id2);

  auto const payload = MakePayload(24, 7);
  EXPECT_TRUE(server.SendTo(id1, payload.data(), payload.size()));

  // The targeted client receives the payload; the other client receives nothing.
  ASSERT_TRUE(PollUntil([&] { return sink1.Count() >= 1; }));
  float constexpr kShortTimeOut = 0.1f; // seconds
  EXPECT_FALSE(PollUntil([&] { return sink2.Count() > 0; }, kShortTimeOut));
  ASSERT_EQ(sink1.Count(), 1u);
  EXPECT_EQ(sink1.Get(0).data, payload);
  EXPECT_EQ(sink2.Count(), 0u);

  client1.Disconnect();
  client2.Disconnect();
  server.Stop();
}

// Two independent servers, each with its own client. Messages route only to the server the client
// connected to. In-proc servers report port 0; real-socket servers bind distinct non-zero ports.
TEST(Net, TwoServersRouting) {
  ServerSocket serverA;
  ServerSocket serverB;
  Sink sinkA;
  Sink sinkB;
  StartServer(serverA, sinkA);
  StartServer(serverB, sinkB);

  ClientSocket clientA;
  ClientSocket clientB;
  Sink clientSinkA;
  Sink clientSinkB;
  ConnectClient(clientA, serverA, clientSinkA);
  ConnectClient(clientB, serverB, clientSinkB);
  EXPECT_EQ(clientA.GetStatus(), SocketStatus::Connected);
  EXPECT_EQ(clientB.GetStatus(), SocketStatus::Connected);

  auto const payloadA = MakePayload(32, 0xA);
  auto const payloadB = MakePayload(32, 0xB);
  EXPECT_TRUE(clientA.Send(payloadA.data(), payloadA.size()));
  EXPECT_TRUE(clientB.Send(payloadB.data(), payloadB.size()));

  // Each server received only its own client's payload (no cross-routing).
  ASSERT_TRUE(PollUntil([&] { return sinkA.Count() >= 1 && sinkB.Count() >= 1; }));
  EXPECT_EQ(sinkA.Get(0).data, payloadA);
  EXPECT_EQ(sinkB.Get(0).data, payloadB);

#if MOCHI_TEST_REAL_SOCKETS
  // Real-socket servers bind distinct TCP ports.
  EXPECT_NE(serverA.GetPort(), 0);
  EXPECT_NE(serverB.GetPort(), 0);
  EXPECT_NE(serverA.GetPort(), serverB.GetPort());
#else
  // In-proc servers never bind a TCP port.
  EXPECT_EQ(serverA.GetPort(), 0);
  EXPECT_EQ(serverB.GetPort(), 0);
#endif

  clientA.Disconnect();
  clientB.Disconnect();
  serverA.Stop();
  serverB.Stop();
}

// -------------------------------------------------------------------------------------------------
// Lifecycle
// -------------------------------------------------------------------------------------------------

// The client list shrinks when a client disconnects, leaving only the remaining client's id.
TEST(Net, GetClientsShrinksOnClientDisconnect) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client1;
  ClientSocket client2;
  Sink sink1;
  Sink sink2;
  ConnectClient(client1, server, sink1);
  ConnectClient(client2, server, sink2);
  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 2u; }));

  ClientId const id2 = IdentifyClient(serverSink, client2, 2);

  client1.Disconnect();

  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 1u; }));
  auto const remaining = server.GetClients();
  ASSERT_EQ(remaining.size(), 1u);
  EXPECT_EQ(remaining[0], id2);

  client2.Disconnect();
  server.Stop();
}

// Stop() and Disconnect() are safe to call when never started, and safe to call repeatedly.
TEST(Net, StopAndDisconnectIdempotent) {
  ServerSocket neverStarted;
  neverStarted.Stop(); // no-op on a server that was never started

  ClientSocket neverConnected;
  neverConnected.Disconnect(); // no-op on a client that never connected
  EXPECT_EQ(neverConnected.GetStatus(), SocketStatus::None);

  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  Sink clientSink;
  ConnectClient(client, server, clientSink);
  EXPECT_EQ(client.GetStatus(), SocketStatus::Connected);

  client.Disconnect();
  client.Disconnect(); // idempotent
  server.Stop();
  server.Stop(); // idempotent

  EXPECT_EQ(client.GetStatus(), SocketStatus::None);
}

// Sending before a connection has been initiated is a no-op (returns false, no transport yet) and
// leaves no stale message; a post-connect send is delivered normally.
TEST(Net, SendWhileNotConnectedIsNoOp) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  EXPECT_EQ(client.GetStatus(), SocketStatus::None);

  auto const stale = MakePayload(16, 7);
  EXPECT_FALSE(client.Send(stale.data(), stale.size())); // no channel yet -> not delivered

  Sink clientSink;
  ConnectClient(client, server, clientSink);
  EXPECT_EQ(client.GetStatus(), SocketStatus::Connected);

  // No stale pre-connect message leaked into the server.
  float constexpr kShortTimeOut = 0.1f; // seconds
  EXPECT_FALSE(PollUntil([&] { return serverSink.Count() > 0; }, kShortTimeOut));

  auto const good = MakePayload(16, 9);
  EXPECT_TRUE(client.Send(good.data(), good.size()));
  ASSERT_TRUE(PollUntil([&] { return serverSink.Count() >= 1; }));
  EXPECT_EQ(serverSink.Get(0).data, good);

  client.Disconnect();
  server.Stop();
}

// SendTo reports whether the target client is currently connected, and a Broadcast with no
// recipients is dropped rather than retained for future clients.
TEST(Net, SendToMissingClientReturnsFalseAndBroadcastNoRecipientsDrops) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  auto const payload = MakePayload(16, 0x71);
  EXPECT_TRUE(server.GetClients().empty());
  server.Broadcast(payload.data(), payload.size());
  EXPECT_FALSE(server.SendTo(12345, payload.data(), payload.size()));

  ClientSocket client;
  Sink clientSink;
  ConnectClient(client, server, clientSink);
  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 1u; }));

  // The broadcast issued before the client connected was dropped, not retained.
  static constexpr float kShortTimeOut = 0.1f; // seconds
  EXPECT_FALSE(PollUntil([&] { return clientSink.Count() >= 1; }, kShortTimeOut));

  auto const clients = server.GetClients();
  ASSERT_EQ(clients.size(), 1u);
  EXPECT_TRUE(server.SendTo(clients[0], payload.data(), payload.size()));
  ASSERT_TRUE(PollUntil([&] { return clientSink.Count() >= 1; }));
  EXPECT_EQ(clientSink.Get(0).data, payload);

  client.Disconnect();
  EXPECT_TRUE(PollUntil([&] { return server.GetClients().empty(); }));
  EXPECT_FALSE(server.SendTo(clients[0], payload.data(), payload.size()));

  server.Stop();
}

// Destroying the ServerSocket while a client is still connected is safe — the client transitions to
// Lost and a subsequent Send returns false. The shared_ptr/weak_ptr design enforces this rather
// than relying on the caller to keep the server alive.
TEST(Net, DestroyServerWhileClientConnected) {
  ClientSocket client;
  Sink clientSink;
  {
    ServerSocket server;
    Sink serverSink;
    StartServer(server, serverSink);
    ConnectClient(client, server, clientSink);
    EXPECT_EQ(client.GetStatus(), SocketStatus::Connected);
    // server destroyed here (scope exit) while the client is still connected.
  }

  EXPECT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Lost; }));

  auto const payload = MakePayload(16, 0x55);
  EXPECT_FALSE(client.Send(payload.data(), payload.size()));

  client.Disconnect();
}

// A client whose connection ended (Disconnect) can be reused to connect to a new server, and the
// receive callback set once keeps delivering across both connections.
TEST(Net, ReconnectAfterDisconnect) {
  Sink clientSink;
  ClientSocket client;
  // Set the receive callback exactly once, up front, to prove the holder is reused across both
  // connections; connect manually below so we never re-set it.
  client.SetReceiveCallback([&clientSink](void const* d, size_t n) { clientSink.Append(0, d, n); });

  auto const connect = [&](ServerSocket& server) {
#if MOCHI_TEST_REAL_SOCKETS
    client.Connect("127.0.0.1", server.GetPort());
    PollUntil([&] { return client.GetStatus() == SocketStatus::Connected; });
#else
    client.ConnectInProc(server);
#endif
  };

  ServerSocket serverA;
  Sink serverSinkA;
  StartServer(serverA, serverSinkA);

  connect(serverA);
  ASSERT_EQ(client.GetStatus(), SocketStatus::Connected);
  auto const first = MakePayload(16, 0xA1);
  EXPECT_TRUE(client.Send(first.data(), first.size()));
  ASSERT_TRUE(PollUntil([&] { return serverSinkA.Count() >= 1; }));
  EXPECT_EQ(serverSinkA.Get(0).data, first);

  client.Disconnect();
  EXPECT_EQ(client.GetStatus(), SocketStatus::None);

  // Reconnect to a fresh server; the reused receive callback still delivers in both directions.
  ServerSocket serverB;
  Sink serverSinkB;
  StartServer(serverB, serverSinkB);

  connect(serverB);
  ASSERT_EQ(client.GetStatus(), SocketStatus::Connected);

  auto const second = MakePayload(16, 0xB2);
  EXPECT_TRUE(client.Send(second.data(), second.size()));
  ASSERT_TRUE(PollUntil([&] { return serverSinkB.Count() >= 1; }));
  EXPECT_EQ(serverSinkB.Get(0).data, second);

  ASSERT_TRUE(PollUntil([&] { return serverB.GetClients().size() == 1u; }));
  auto const clients = serverB.GetClients();
  ASSERT_EQ(clients.size(), 1u);
  auto const down = MakePayload(16, 0xB3);
  EXPECT_TRUE(serverB.SendTo(clients[0], down.data(), down.size()));
  ASSERT_TRUE(PollUntil([&] { return clientSink.Count() >= 1; }));
  EXPECT_EQ(clientSink.Get(0).data, down);

  client.Disconnect();
  serverA.Stop();
  serverB.Stop();
}

// A second client beyond maxClients is refused: it ends Lost and the server keeps exactly one
// client. In-proc rejects synchronously; the real client may briefly TCP-connect before the server
// drops it, so the terminal status is polled rather than read once.
TEST(Net, MaxClientsEnforced) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink, /*maxClients*/ 1);

  ClientSocket client1;
  Sink sink1;
  ConnectClient(client1, server, sink1);
  EXPECT_EQ(client1.GetStatus(), SocketStatus::Connected);
  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 1u; }));

  ClientSocket client2;
  Sink sink2;
  InitiateConnect(client2, server, sink2);
  EXPECT_TRUE(PollUntil([&] { return client2.GetStatus() == SocketStatus::Lost; }));
  EXPECT_EQ(server.GetClients().size(), 1u);

  client1.Disconnect();
  client2.Disconnect();
  server.Stop();
}

// -------------------------------------------------------------------------------------------------
// Concurrency (documented thread-safety contracts)
// -------------------------------------------------------------------------------------------------

// SendTo is documented as thread-safe. Two threads concurrently sending to the same client must
// deliver every message and preserve each sender's relative order.
TEST(Net, ConcurrentSendToSameClient) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  Sink clientSink;
  ConnectClient(client, server, clientSink);

  ASSERT_TRUE(PollUntil([&] { return server.GetClients().size() == 1u; }));
  auto const clients = server.GetClients();
  ASSERT_EQ(clients.size(), 1u);
  ClientId const clientId = clients[0];

  constexpr int kMessagesPerThread = 50;

  // Thread A sends values 0..49, thread B sends values 100..149.
  std::thread threadA([&] {
    for (int i = 0; i < kMessagesPerThread; ++i) {
      auto const val = static_cast<uint32_t>(i);
      server.SendTo(clientId, &val, sizeof(val));
    }
  });
  std::thread threadB([&] {
    for (int i = 0; i < kMessagesPerThread; ++i) {
      auto const val = static_cast<uint32_t>(100 + i);
      server.SendTo(clientId, &val, sizeof(val));
    }
  });
  threadA.join();
  threadB.join();

  ASSERT_TRUE(PollUntil([&] { return clientSink.Count() >= 2u * kMessagesPerThread; }));

  // Split the received values back into per-thread subsequences; each must arrive in send order.
  DynamicArray<uint32_t> fromA;
  DynamicArray<uint32_t> fromB;
  size_t const total = clientSink.Count();
  for (size_t i = 0; i < total; ++i) {
    auto const entry = clientSink.Get(i);
    ASSERT_EQ(entry.data.size(), sizeof(uint32_t));
    uint32_t val = 0;
    std::memcpy(&val, entry.data.data(), sizeof(val));
    (val < 100 ? fromA : fromB).push_back(val);
  }

  DynamicArray<uint32_t> expectedA;
  DynamicArray<uint32_t> expectedB;
  for (int i = 0; i < kMessagesPerThread; ++i) {
    expectedA.push_back(static_cast<uint32_t>(i));
    expectedB.push_back(static_cast<uint32_t>(100 + i));
  }
  EXPECT_EQ(fromA, expectedA);
  EXPECT_EQ(fromB, expectedB);

  client.Disconnect();
  server.Stop();
}

// Racing client.Disconnect() against server.Stop() from two threads must never crash or hang. The
// terminal status is last-writer-wins (None if Disconnect wins, Lost if Stop wins); both are valid.
// Repeated to shake out ordering races.
TEST(Net, ConcurrentDisconnectAndServerStop) {
  constexpr int kIterations = 50;
  for (int i = 0; i < kIterations; ++i) {
    ServerSocket server;
    Sink serverSink;
    StartServer(server, serverSink);

    ClientSocket client;
    Sink clientSink;
    ConnectClient(client, server, clientSink);
    ASSERT_EQ(client.GetStatus(), SocketStatus::Connected);

    std::thread stopper([&] { server.Stop(); });
    std::thread disconnecter([&] { client.Disconnect(); });
    stopper.join();
    disconnecter.join();

    SocketStatus const status = client.GetStatus();
    EXPECT_NE(status, SocketStatus::Connected) << "iteration " << i;
    EXPECT_NE(status, SocketStatus::Pending) << "iteration " << i;
  }
}

// -------------------------------------------------------------------------------------------------
// Connection versioning
// -------------------------------------------------------------------------------------------------

// Matching user-defined versions on both sides connect successfully. On the real-socket path this
// exercises the connection handshake carrying the user-defined version.
TEST(Net, VersionMatchConnects) {
  ServerSocket server;
  server.SetVersion(42);
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  client.SetVersion(42);
  Sink clientSink;
  ConnectClient(client, server, clientSink);

  EXPECT_EQ(client.GetStatus(), SocketStatus::Connected);
  EXPECT_TRUE(PollUntil([&] { return server.GetClients().size() == 1u; }));

  client.Disconnect();
  server.Stop();
}

// Mismatched user-defined versions are rejected: the client goes Lost and the server registers no
// client. On real sockets the client momentarily reaches Connected before Lost, so the terminal
// state is polled rather than read once.
TEST(Net, VersionMismatchRejected) {
  ServerSocket server;
  server.SetVersion(42);
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  client.SetVersion(43);
  Sink clientSink;
  InitiateConnect(client, server, clientSink);

  EXPECT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Lost; }));
  EXPECT_TRUE(PollUntil([&] { return server.GetClients().empty(); }));

  client.Disconnect();
  server.Stop();
}

// A client that sets a non-zero version is rejected by a server left at the default 0, confirming
// "no SetVersion" behaves like SetVersion(0).
TEST(Net, VersionClientNonZeroServerDefaultRejected) {
  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink); // no SetVersion -> version 0

  ClientSocket client;
  client.SetVersion(1);
  Sink clientSink;
  InitiateConnect(client, server, clientSink);

  EXPECT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Lost; }));
  EXPECT_TRUE(PollUntil([&] { return server.GetClients().empty(); }));

  client.Disconnect();
  server.Stop();
}

// -------------------------------------------------------------------------------------------------
// ServerInfo serialization
// -------------------------------------------------------------------------------------------------

// ServerInfo round-trips through reflection-based JSON serialization with every field preserved.
// Exercises the reflection markup directly; no sockets involved, so it always runs.
TEST(Net, ServerInfoJsonRoundTrip) {
  ServerInfo original;
  original.label = DynamicString("MyServer");
  original.address = DynamicString("10.1.2.3");
  original.port = 4321;
  original.numClients = 7;
  original.maxClients = 16;
  original.version = 0xABCDEF0123456789ull;

  std::string const json = SReflect::ToJsonString(original, /*pretty*/ false);

  ServerInfo parsed;
  ASSERT_TRUE(SReflect::FromJsonString(parsed, json));

  EXPECT_EQ(
      std::string_view(parsed.label.data(), parsed.label.size()),
      std::string_view(original.label.data(), original.label.size()));
  EXPECT_EQ(
      std::string_view(parsed.address.data(), parsed.address.size()),
      std::string_view(original.address.data(), original.address.size()));
  EXPECT_EQ(parsed.port, original.port);
  EXPECT_EQ(parsed.numClients, original.numClients);
  EXPECT_EQ(parsed.maxClients, original.maxClients);
  EXPECT_EQ(parsed.version, original.version);
}

// -------------------------------------------------------------------------------------------------
// Real-socket-only behavior (public API only)
//
// These tests exercise behavior that is meaningful only on the asynchronous real-socket transport
// (a Pending connect state, outbound queueing before connect, UDP discovery, the sender-side frame
// cap, IPv4-literal parsing, and reconnect after a server restart). They are compiled and run only
// when MOCHI_TEST_REAL_SOCKETS is enabled, and they still go exclusively through net.h's public API
// — no raw sockets.
// -------------------------------------------------------------------------------------------------

#if MOCHI_TEST_REAL_SOCKETS

namespace {

// RAII: the mochi_core unit-test harness fails any test that logs to the Error/Warning channels. A
// few real-socket tests intentionally exercise paths that log an error (oversized frame, invalid
// address); while this is in scope, Error-channel logs are swallowed and the previous callback is
// restored on destruction.
struct SuppressErrorLogs {
  LogFn previous{GetLogCallback()};
  SuppressErrorLogs() {
    SetLogCallback(
        [prev = previous](LogChannel channel, char const* msg, char const* file, int line) {
          if (channel != LogChannel::Error && prev) {
            prev(channel, msg, file, line);
          }
        });
  }
  ~SuppressErrorLogs() {
    SetLogCallback(previous);
  }
};

} // namespace

// A fixed, pre-known port for the one test that must connect before any server is bound
// (SendWhilePendingQueuesUntilConnected). All other real-socket servers use Start(0, ...) so the OS
// allocates an ephemeral port, avoiding cross-process collisions under parallel test runs.
static constexpr uint16_t kTestPort = 19876;
static constexpr uint16_t kTestDiscoveryPort = 17331;
static constexpr uint16_t kOtherTestDiscoveryPort = 17332;

// Connect a real-socket client to `port` and wait for the TCP handshake to complete.
static void ConnectReal(ClientSocket& client, uint16_t port, Sink& sink) {
  client.SetReceiveCallback([&sink](void const* data, size_t size) { sink.Append(0, data, size); });
  client.Connect("127.0.0.1", port);
  EXPECT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Connected; }));
}

// Disconnect interrupts a pending TCP connect to a blackholed target instead of waiting for the OS
// connect timeout. This private-network address is commonly unrouted from dev hosts, leaving
// connect() pending long enough to exercise the in-flight TCP SYN path rather than a fast refusal.
TEST(Net, DisconnectDuringPendingConnectReturnsPromptly) {
  auto client = std::make_shared<ClientSocket>();
  client->Connect("10.255.255.1", kTestPort);
  ASSERT_TRUE(PollUntil([&] { return client->GetStatus() == SocketStatus::Pending; }));

  auto disconnectDone = std::make_shared<std::atomic<bool>>(false);
  std::thread disconnectThread([client, disconnectDone] {
    client->Disconnect();
    disconnectDone->store(true);
  });

  bool const completed = PollUntil([&] { return disconnectDone->load(); });
  EXPECT_TRUE(completed);
  if (completed) {
    disconnectThread.join();
    EXPECT_EQ(client->GetStatus(), SocketStatus::None);
  } else {
    disconnectThread.detach();
  }
}

// A send while the initial TCP connection is pending is queued and delivered once connected. Uses
// the fixed kTestPort because the client deliberately connects before any server is bound, so the
// port must be known in advance.
TEST(Net, SendWhilePendingQueuesUntilConnected) {
  ClientSocket client;
  Sink clientSink;
  client.SetReceiveCallback([&clientSink](void const* d, size_t n) { clientSink.Append(0, d, n); });
  client.Connect("127.0.0.1", kTestPort);
  ASSERT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Pending; }));

  auto const payload = MakePayload(16, 0x91);
  EXPECT_TRUE(client.Send(payload.data(), payload.size()));

  ServerSocket server;
  Sink serverSink;
  server.SetReceiveCallback(
      [&serverSink](ClientId id, void const* d, size_t n) { serverSink.Append(id, d, n); });
  server.Start(kTestPort, 4, "S");
  EXPECT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Connected; }));

  ASSERT_TRUE(PollUntil([&] { return serverSink.Count() >= 1; }));
  EXPECT_EQ(serverSink.Get(0).data, payload);

  client.Disconnect();
  server.Stop();
}

// The client's status callback observes Connected then Lost when the server stops.
TEST(Net, ClientStatusCallbackFiresLost) {
  std::mutex mutex;
  DynamicArray<SocketStatus> sequence;

  ServerSocket server;
  Sink serverSink;
  StartServer(server, serverSink);

  ClientSocket client;
  Sink clientSink;
  client.SetReceiveCallback([&clientSink](void const* d, size_t n) { clientSink.Append(0, d, n); });
  client.SetStatusCallback([&](SocketStatus s) {
    std::lock_guard<std::mutex> lock(mutex);
    sequence.push_back(s);
  });

  client.Connect("127.0.0.1", server.GetPort());
  EXPECT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Connected; }));

  server.Stop();
  EXPECT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Lost; }));

  {
    std::lock_guard<std::mutex> lock(mutex);
    size_t connectedIdx = sequence.size();
    bool lostAfterConnected = false;
    for (size_t i = 0; i < sequence.size(); ++i) {
      if (sequence[i] == SocketStatus::Connected && connectedIdx == sequence.size()) {
        connectedIdx = i;
      } else if (sequence[i] == SocketStatus::Lost && connectedIdx != sequence.size()) {
        lostAfterConnected = true;
      }
    }
    EXPECT_NE(connectedIdx, sequence.size());
    EXPECT_TRUE(lostAfterConnected);
  }

  client.Disconnect();
}

// A lost client stays Lost after the server restarts; reconnect policy belongs above ClientSocket.
TEST(Net, ClientDoesNotReconnectAfterServerRestart) {
  ServerSocket server;
  Sink serverSink;
  server.SetReceiveCallback(
      [&serverSink](ClientId id, void const* d, size_t n) { serverSink.Append(id, d, n); });
  server.Start(0, 4, "S");
  uint16_t const port = server.GetPort();

  ClientSocket client;
  Sink clientSink;
  ConnectReal(client, port, clientSink);
  EXPECT_EQ(client.GetStatus(), SocketStatus::Connected);

  server.Stop();
  EXPECT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Lost; }));

  ServerSocket server2;
  Sink serverSink2;
  server2.SetReceiveCallback(
      [&serverSink2](ClientId id, void const* d, size_t n) { serverSink2.Append(id, d, n); });
  server2.Start(port, 4, "S");
  ASSERT_EQ(server2.GetPort(), port);

  float constexpr kShortTimeOut = 0.1f; // seconds
  EXPECT_FALSE(
      PollUntil([&] { return client.GetStatus() == SocketStatus::Connected; }, kShortTimeOut));
  EXPECT_EQ(client.GetStatus(), SocketStatus::Lost);
  EXPECT_FALSE(PollUntil([&] { return !server2.GetClients().empty(); }, kShortTimeOut));

  auto const payload = MakePayload(32, 3);
  EXPECT_FALSE(client.Send(payload.data(), payload.size()));

  client.Disconnect();
  server2.Stop();
}

// After Stop(), GetPort() reports 0 per the documented contract (the server is no longer
// listening).
TEST(Net, GetPortZeroAfterStop) {
  ServerSocket server;
  server.Start(0, 4, "S");
  EXPECT_NE(server.GetPort(), 0);

  server.Stop();
  EXPECT_EQ(server.GetPort(), 0);
}

// A client whose connection was lost can be reused — Disconnect() then Connect() re-establishes a
// working connection. Regression test for the outbound queue's shutdown flag being reset on
// reconnect (otherwise the send thread would busy-spin and never flush).
TEST(Net, ClientCanReconnectAfterServerStops) {
  ServerSocket server1;
  Sink serverSink1;
  server1.SetReceiveCallback(
      [&serverSink1](ClientId id, void const* d, size_t n) { serverSink1.Append(id, d, n); });
  server1.Start(0, 4, "S");
  uint16_t const port = server1.GetPort();
  ASSERT_NE(port, 0);

  ClientSocket client;
  Sink clientSink;
  ConnectReal(client, port, clientSink);
  EXPECT_EQ(client.GetStatus(), SocketStatus::Connected);

  auto const first = MakePayload(16, 1);
  EXPECT_TRUE(client.Send(first.data(), first.size()));
  ASSERT_TRUE(PollUntil([&] { return serverSink1.Count() >= 1; }));
  EXPECT_EQ(serverSink1.Get(0).data, first);

  // Server goes away; client observes the loss.
  server1.Stop();
  EXPECT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Lost; }));

  // A new server comes up on the same port; the same client reconnects and round-trips again.
  ServerSocket server2;
  Sink serverSink2;
  server2.SetReceiveCallback(
      [&serverSink2](ClientId id, void const* d, size_t n) { serverSink2.Append(id, d, n); });
  server2.Start(port, 4, "S");

  client.Disconnect();
  ConnectReal(client, port, clientSink);
  EXPECT_EQ(client.GetStatus(), SocketStatus::Connected);

  auto const second = MakePayload(16, 2);
  EXPECT_TRUE(client.Send(second.data(), second.size()));
  ASSERT_TRUE(PollUntil([&] { return serverSink2.Count() >= 1; }));
  EXPECT_EQ(serverSink2.Get(0).data, second);

  client.Disconnect();
  server2.Stop();
}

// Connecting to a malformed (non-IPv4-literal) address fails fast to Lost instead of retrying
// forever in Pending. IPv4 literals only (no DNS).
TEST(Net, ConnectInvalidAddressGoesLost) {
  SuppressErrorLogs suppressErrors; // the connect loop logs an error for the unparseable address

  ClientSocket client;
  client.Connect("not.an.ip", kTestPort);
  EXPECT_TRUE(PollUntil([&] { return client.GetStatus() == SocketStatus::Lost; }));

  client.Disconnect();
}

// A hosted server is discoverable via ServerList. The discovery probe is sent to the LAN broadcast
// address and to loopback, so same-host discovery works even on hosts with no broadcast-capable
// route (e.g. devservers).
TEST(Net, DiscoveryEndToEnd) {
  ServerSocket server;
  server.SetDiscoveryPort(kTestDiscoveryPort);
  server.Start(0, 4, "DiscoveryTarget");
  uint16_t const port = server.GetPort();
  ASSERT_NE(port, 0);

  ServerList list(kTestDiscoveryPort);
  list.Refresh();

  bool const found = PollUntil(
      [&] {
        DynamicArray<ServerInfo> servers;
        list.GetServers(servers);
        for (auto const& info : servers) {
          if (std::string_view(info.label.data(), info.label.size()) == "DiscoveryTarget" &&
              info.port == port) {
            return true;
          }
        }
        return false;
      },
      2.0f);
  EXPECT_TRUE(found);

  server.Stop();
}

TEST(Net, ServerList) {
  // Create two servers on the same host.
  ServerSocket serverA;
  serverA.SetDiscoveryPort(kTestDiscoveryPort);
  ServerSocket serverB;
  serverB.SetDiscoveryPort(kTestDiscoveryPort);
  ServerSocket hiddenServer;
  hiddenServer.SetDiscoveryPort(0);
  serverA.SetVersion(123);
  serverB.SetVersion(456);
  hiddenServer.SetVersion(789);
  serverA.Start(/*port*/ 0, /*maxClients*/ 2, "DiscoveryA");
  serverB.Start(/*port*/ 0, /*maxClients*/ 3, "DiscoveryB");
  hiddenServer.Start(/*port*/ 0, /*maxClients*/ 1, "HiddenDiscovery");
  uint16_t const portA = serverA.GetPort();
  uint16_t const portB = serverB.GetPort();
  uint16_t const hiddenPort = hiddenServer.GetPort();
  ASSERT_NE(portA, 0);
  ASSERT_NE(portB, 0);
  ASSERT_NE(hiddenPort, 0);
  ASSERT_NE(portA, portB);
  ASSERT_NE(portA, hiddenPort);
  ASSERT_NE(portB, hiddenPort);

  // Connect one client to serverB
  ClientSocket clientB;
  clientB.SetVersion(456);
  clientB.Connect("127.0.0.1", portB);
  bool const connected =
      PollUntil([&]() { return clientB.GetStatus() == net::SocketStatus::Connected; }, 10.0f);
  EXPECT_TRUE(connected);

  // Create server list (initially empty)
  ServerList list(kTestDiscoveryPort);
  DynamicArray<ServerInfo> servers;
  servers.push_back({}); // will be cleared bet GetServers
  list.GetServers(servers);
  EXPECT_EQ(0, servers.size());

  // Send UDP discovery broadcast
  list.Refresh();

  // Wait for results
  bool const foundBoth = PollUntil(
      [&] {
        list.GetServers(servers);
        bool foundA = false;
        bool foundB = false;
        for (auto const& s : servers) {
          if (s.port == portA) {
            EXPECT_FALSE(foundA);
            EXPECT_STREQ("DiscoveryA", s.label.c_str());
            EXPECT_STREQ("127.0.0.1", s.address.c_str());
            EXPECT_EQ(123, s.version);
            EXPECT_EQ(0, s.numClients);
            EXPECT_EQ(2, s.maxClients);
            foundA = true;
          } else if (s.port == portB) {
            EXPECT_FALSE(foundB);
            EXPECT_STREQ("DiscoveryB", s.label.c_str());
            EXPECT_STREQ("127.0.0.1", s.address.c_str());
            EXPECT_EQ(456, s.version);
            EXPECT_EQ(1, s.numClients);
            EXPECT_EQ(3, s.maxClients);
            foundB = true;
          } else {
            printf("Found Unknown: %s:%u %s\n", s.address.c_str(), s.port, s.label.c_str());
          }
        }
        return foundA && foundB;
      },
      10.0f);
  EXPECT_TRUE(foundBoth);

  bool const foundHidden = PollUntil(
      [&] {
        list.GetServers(servers);
        for (auto const& s : servers) {
          if (s.port == hiddenPort) {
            return true;
          }
        }
        return false;
      },
      2.0f);
  EXPECT_FALSE(foundHidden);

  ServerList otherList(kOtherTestDiscoveryPort);
  otherList.Refresh();
  bool const foundOnOtherPort = PollUntil(
      [&] {
        DynamicArray<ServerInfo> otherServers;
        otherList.GetServers(otherServers);
        for (auto const& s : otherServers) {
          if (s.port == portA || s.port == portB || s.port == hiddenPort) {
            return true;
          }
        }
        return false;
      },
      2.0f);
  EXPECT_FALSE(foundOnOtherPort);
}

#endif // MOCHI_TEST_REAL_SOCKETS

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
#include <mochi_core/net/message_client.h>
#include <mochi_core/net/message_dispatcher.h>
#include <mochi_core/net/message_serialization.h>
#include <mochi_core/net/message_server.h>
#include <mochi_core/net/server_socket.h>
#include <mochi_core/test/log_suppression.h>
#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/test/wait_until.h>
#include <mochi_core/utils/defer.h>
#include <mochi_core/utils/dynamic_array.h>
#include <mochi_core/utils/guarded.h>
#include <mochi_core/utils/log.h>
#include <mochi_core/utils/stream.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <thread>
#include <utility>

using namespace mochi;
using namespace mochi::net;

namespace {

struct TestReply;
struct TestReply2;

struct TestRequest : RequestMessage {
  using Reply = TestReply;
  TestRequest() = default;
  explicit TestRequest(int v) : value(v) {}

  int value = 0;

  MOCHI_STRUCT_BEGIN(TestRequest)
  MOCHI_BASE_CLASS(RequestMessage)
  MOCHI_FIELD(value)
  MOCHI_STRUCT_END()
};

struct TestReply : ReplyMessage {
  TestReply() = default;
  explicit TestReply(std::string s) : str(std::move(s)) {}
  explicit TestReply(TestRequest const& req) : ReplyMessage(req), value(req.value) {}

  std::string str;
  int value = 0;

  MOCHI_STRUCT_BEGIN(TestReply)
  MOCHI_BASE_CLASS(ReplyMessage)
  MOCHI_FIELD(str)
  MOCHI_FIELD(value)
  MOCHI_STRUCT_END()
};

// A second request/reply pair to exercise type-keyed reply routing.
struct TestRequest2 : RequestMessage {
  using Reply = TestReply2;
  TestRequest2() = default;
  explicit TestRequest2(int v) : value(v) {}

  int value = 0;

  MOCHI_STRUCT_BEGIN(TestRequest2)
  MOCHI_BASE_CLASS(RequestMessage)
  MOCHI_FIELD(value)
  MOCHI_STRUCT_END()
};

struct TestReply2 : ReplyMessage {
  TestReply2() = default;
  explicit TestReply2(TestRequest2 const& req) : ReplyMessage(req), value(req.value) {}

  int value = 0;

  MOCHI_STRUCT_BEGIN(TestReply2)
  MOCHI_BASE_CLASS(ReplyMessage)
  MOCHI_FIELD(value)
  MOCHI_STRUCT_END()
};

} // namespace

template <class ClientT, class ServerT>
static void Connect(ClientT& client, ServerT& server) {
  size_t const initialClients = server.GetClients().size();
  client.ConnectInProc(server);
  test::WaitUntil([&] { return client.GetStatus() == SocketStatus::Connected; });
  test::WaitUntil([&] { return server.GetClients().size() > initialClients; });
}

TEST(NetMessageClient, StatusCallback) {
  MessageServer server;
  MessageClient client;

  std::atomic<bool> sawConnected = false;
  std::atomic<bool> sawLost = false;
  client.SetStatusCallback([&](SocketStatus status) {
    sawConnected = sawConnected || status == SocketStatus::Connected;
    sawLost = sawLost || status == SocketStatus::Lost;
  });

  server.StartInProc(/*maxClients*/ 1, "StatusCallbackTest");
  Connect(client, server);

  // The callback observes the connection becoming established.
  test::WaitUntil([&] { return sawConnected.load(); });

  // Clearing the callback stops further notifications.
  client.SetStatusCallback({});
  server.Stop();

  // The client still transitions to Lost, but the cleared callback is not invoked.
  test::WaitUntil([&] { return client.GetStatus() == SocketStatus::Lost; });
  EXPECT_FALSE(sawLost.load());
}

TEST(NetMessageClient, ClientSend) {
  MessageServer server;
  MessageClient client;

  struct Received {
    bool gotIt = false;
    ClientId client = 0;
    int requestValue = 0;
    std::string replyStr;
  };
  Guarded<Received> received;

  // Server setup
  server.Register<TestRequest>([&](ClientId id, auto&& msg) {
    received.Mutate([&](Received& r) {
      r.gotIt = true;
      r.client = id;
      r.requestValue = msg.value;
    });
  });
  server.Register<TestReply>([&](ClientId id, auto&& msg) {
    received.Mutate([&](Received& r) {
      r.gotIt = true;
      r.client = id;
      r.replyStr = msg.str;
    });
  });
  server.StartInProc(/*maxClients*/ 1, "TestServer");
  EXPECT_EQ(0, isize(server.GetClients()));

  // Fail to send from client to server before connecting
  EXPECT_FALSE(client.Send(TestRequest{911}));

  // Connect and get the ClientId
  Connect(client, server);
  auto clients = server.GetClients();
  ASSERT_EQ(clients.size(), 1u);
  auto clientId = clients[0];

  // Send TestRequest from client to server
  EXPECT_TRUE(client.Send(TestRequest{42}));
  test::WaitUntil([&] { return received.Read(&Received::gotIt); });
  received.Mutate([&](Received& r) {
    EXPECT_EQ(clientId, r.client);
    EXPECT_EQ(42, r.requestValue);
    r = {}; // clear
  });

  // Send TestReply from client to server
  EXPECT_TRUE(client.Send(TestReply{"hello"}));
  test::WaitUntil([&] { return received.Read(&Received::gotIt); });
  received.Mutate([&](Received& r) {
    EXPECT_EQ(clientId, r.client);
    EXPECT_STREQ("hello", r.replyStr.c_str());
    r = {}; // clear
  });

  // Fail to send after disconnecting
  client.Disconnect();
  received.Store({}); // clear
  EXPECT_FALSE(client.Send(TestRequest{911}));
  EXPECT_FALSE(received.Read(&Received::gotIt));

  // Reconnect
  Connect(client, server);
  clients = server.GetClients();
  ASSERT_EQ(clients.size(), 1u);
  clientId = clients[0];

  // Send from client to server (successful again)
  EXPECT_TRUE(client.Send(TestRequest{123}));
  test::WaitUntil([&] { return received.Read(&Received::gotIt); });
  received.Mutate([&](Received& r) {
    EXPECT_EQ(clientId, r.client);
    EXPECT_EQ(123, r.requestValue);
    r = {}; // clear
  });
}

TEST(NetMessageServer, ServerSendTo) {
  MessageServer server;
  MessageClient client1, client2;

  struct Received {
    std::optional<int> value1;
    std::optional<int> value2;
  };
  Guarded<Received> received;

  // Client setup
  client1.Register<TestRequest>(
      [&](auto&& msg) { received.Mutate([&](Received& r) { r.value1 = msg.value; }); });
  client2.Register<TestRequest>(
      [&](auto&& msg) { received.Mutate([&](Received& r) { r.value2 = msg.value; }); });

  // Try to send to invalid ClientIds
  EXPECT_FALSE(server.SendTo(ClientId{0}, TestRequest{911}));
  EXPECT_FALSE(server.SendTo(ClientId{7}, TestRequest{911}));

  // Start server
  server.StartInProc(/*maxClients*/ 2, "TestServer");
  EXPECT_EQ(0, isize(server.GetClients()));

  // Try to send to invalid ClientIds again
  EXPECT_FALSE(server.SendTo(ClientId{0}, TestRequest{911}));
  EXPECT_FALSE(server.SendTo(ClientId{7}, TestRequest{911}));

  // Connect client1 and get the ClientId
  Connect(client1, server);
  auto clients = server.GetClients();
  ASSERT_EQ(1, isize(clients));
  auto clientId1 = clients[0];

  // Connect client2 and get the ClientId
  Connect(client2, server);
  clients = server.GetClients();
  ASSERT_EQ(2, isize(clients));
  EXPECT_NE(clients[0], clients[1]);
  auto clientId2 = clients[0] == clientId1 ? clients[1] : clients[0]; // The other one

  // Send from server to client1
  EXPECT_TRUE(server.SendTo(clientId1, TestRequest{123}));
  test::WaitUntil(
      [&] { return received.Read([](auto const& r) { return r.value1.has_value(); }); });
  received.Mutate([](auto& r) {
    ASSERT_TRUE(r.value1.has_value());
    EXPECT_EQ(*r.value1, 123);
    EXPECT_FALSE(r.value2.has_value());
    r = {}; // clear
  });

  // Fail to send to client1 after disconnecting
  client1.Disconnect();
  EXPECT_FALSE(server.SendTo(clientId1, TestRequest{123}));
  received.Mutate([](auto& r) {
    EXPECT_FALSE(r.value1.has_value());
    EXPECT_FALSE(r.value2.has_value());
    r = {}; // clear
  });

  // Send from server to client2
  EXPECT_TRUE(server.SendTo(clientId2, TestRequest{456}));
  test::WaitUntil(
      [&] { return received.Read([](auto const& r) { return r.value2.has_value(); }); });
  received.Mutate([](auto& r) {
    EXPECT_FALSE(r.value1.has_value());
    ASSERT_TRUE(r.value2.has_value());
    EXPECT_EQ(*r.value2, 456);
    r = {}; // clear
  });
}

TEST(NetMessageServer, Broadcast) {
  MessageServer server;
  MessageClient client1, client2;

  struct Received {
    std::optional<int> value1;
    std::optional<int> value2;
    std::optional<std::string> str1;
    std::optional<std::string> str2;
  };
  Guarded<Received> received;

  // Client setup
  client1.Register<TestRequest>([&](auto&& msg) {
    received.Mutate([&](Received& r) {
      EXPECT_FALSE(r.value1.has_value());
      r.value1 = msg.value;
    });
  });
  client1.Register<TestReply>([&](auto&& msg) {
    received.Mutate([&](Received& r) {
      EXPECT_FALSE(r.str1.has_value());
      r.str1 = msg.str;
    });
  });
  client2.Register<TestRequest>([&](auto&& msg) {
    received.Mutate([&](Received& r) {
      EXPECT_FALSE(r.value2.has_value());
      r.value2 = msg.value;
    });
  });
  client2.Register<TestReply>([&](auto&& msg) {
    received.Mutate([&](Received& r) {
      EXPECT_FALSE(r.str2.has_value());
      r.str2 = msg.str;
    });
  });

  // Broadcast before starting server
  server.Broadcast(TestRequest{911});

  // Start server
  server.StartInProc(/*maxClients*/ 2, "TestServer");
  EXPECT_EQ(0, isize(server.GetClients()));

  // Broadcast with no clients connected
  server.Broadcast(TestRequest{911});

  // Connect clients
  Connect(client1, server);
  Connect(client2, server);

  // Broadcast TestRequest to both clients
  server.Broadcast(TestRequest{123});
  test::WaitUntil(
      [&] { return received.Read([](auto const& r) { return r.value1 && r.value2; }); });
  received.Mutate([](auto& r) {
    ASSERT_TRUE(r.value1.has_value());
    EXPECT_EQ(123, *r.value1);
    ASSERT_TRUE(r.value2.has_value());
    EXPECT_EQ(123, *r.value2);
    r = {}; // clear
  });

  // Broadcast TestReply to both clients
  server.Broadcast(TestReply{"hello"});
  test::WaitUntil([&] { return received.Read([](auto const& r) { return r.str1 && r.str2; }); });
  received.Mutate([](auto& r) {
    ASSERT_TRUE(r.str1.has_value());
    EXPECT_STREQ("hello", r.str1->c_str());
    ASSERT_TRUE(r.str2.has_value());
    EXPECT_STREQ("hello", r.str2->c_str());
    r = {}; // clear
  });

  // Disconnect client1
  client1.Disconnect();

  // Broadcast to client2 (only)
  server.Broadcast(TestRequest{123});
  test::WaitUntil([&] { return received.Read([](auto const& r) { return r.value2; }); });
  received.Mutate([](auto& r) {
    EXPECT_FALSE(r.value1.has_value());
    ASSERT_TRUE(r.value2.has_value());
    EXPECT_EQ(*r.value2, 123);
    r = {}; // clear
  });
}

TEST(NetMessageClient, DropsMalformedPayload) {
  // Suppress warnings about invalid messages.
  auto noWarn = test::SuppressLogWarning();

  ServerSocket rawServer;
  rawServer.StartInProc(/*maxClients*/ 1, "ClientDropsInvalidPayload");

  struct Received {
    int value = 0;
  };
  Guarded<Received> received;

  MessageClient client;
  client.Register<TestRequest>([&](auto&& msg) {
    received.Mutate([&](Received& r) {
      EXPECT_EQ(0, r.value) << "Already received";
      r.value = msg.value;
    });
  });
  Connect(client, rawServer);

  // Serialize a message to send over the wire
  DynamicArray<uint8_t> bytes;
  SerializeMessage(TestRequest{42}, bytes);

  // Send various messages of illegal size. The client should not receive them.
  for (size_t sz = 0; sz < bytes.size(); ++sz) {
    rawServer.Broadcast(bytes.data(), sz);
  }

  // Now send one valid message
  rawServer.Broadcast(bytes.data(), bytes.size());

  // Only the valid message should have been received by the client
  test::WaitUntil([&] { return received.Read(&Received::value) != 0; });
  received.Mutate([](Received& r) {
    EXPECT_EQ(42, r.value);
    r = {}; // clear
  });
}

TEST(NetMessageServer, DropsMalformedPayload) {
  // Suppress warnings about invalid messages.
  auto noWarn = test::SuppressLogWarning();

  MessageServer server;
  server.StartInProc(/*maxClients*/ 1, "ServerDropsMalformedPayload");

  struct Received {
    int value = 0;
  };
  Guarded<Received> received;

  server.Register<TestRequest>([&](ClientId /*client*/, auto&& msg) {
    received.Mutate([&](Received& r) {
      EXPECT_EQ(0, r.value) << "Already received";
      r.value = msg.value;
    });
  });

  ClientSocket rawClient;
  Connect(rawClient, server.GetSocket_InternalUseOnly());

  // Serialize a message to send over the wire
  DynamicArray<uint8_t> bytes;
  SerializeMessage(TestRequest{42}, bytes);

  // Send various messages of illegal size. The server should not receive them.
  for (size_t sz = 0; sz < bytes.size(); ++sz) {
    rawClient.Send(bytes.data(), sz);
  }

  // Now send one valid message
  rawClient.Send(bytes.data(), bytes.size());

  // Only the valid message should have been received by the server
  test::WaitUntil([&] { return received.Read(&Received::value) != 0; });
  received.Mutate([](Received& r) {
    EXPECT_EQ(42, r.value);
    r = {}; // clear
  });
}

TEST(NetMessageClient, DropsUnregisteredMessage) {
  // Suppress warnings about invalid messages.
  auto noWarn = test::SuppressLogWarning();

  MessageServer server;
  MessageClient client;

  struct Received {
    std::string str;
  };
  Guarded<Received> received;

  client.Register<TestReply>([&](auto&& msg) {
    received.Mutate([&](auto& r) {
      EXPECT_TRUE(r.str.empty()) << "Already received";
      r.str = msg.str;
    });
  });

  server.StartInProc(/*maxClients*/ 1, "ClientDropsUnregisteredMessage");
  Connect(client, server);

  // Broadcast TestRequest, which the client has NOT registered for
  server.Broadcast(TestRequest{42});

  // Then Broadcast TestReply, which the client HAS registered for
  server.Broadcast(TestReply{"works"});

  // Receipt of TestRequest was not fatal. Receipt of TestReply was successful.
  test::WaitUntil([&]() { return !received.Read(&Received::str).empty(); });
  received.Mutate([&](auto& r) {
    EXPECT_STREQ("works", r.str.c_str());
    r = {}; // clear
  });
}

TEST(NetMessageServer, DropsUnregisteredMessage) {
  // Suppress warnings about invalid messages.
  auto noWarn = test::SuppressLogWarning();

  MessageServer server;
  MessageClient client;

  struct Received {
    int value = 0;
  };
  Guarded<Received> received;

  server.Register<TestRequest>([&](ClientId /*client*/, auto&& msg) {
    received.Mutate([&](auto& r) {
      EXPECT_EQ(0, r.value) << "Already received";
      r.value = msg.value;
    });
  });

  server.StartInProc(/*maxClients*/ 1, "ServerDropsUnregisteredMessage");
  Connect(client, server);

  // Send TestReply, which the server has NOT registered for
  client.Send(TestReply{"not for you"});

  // Then send TestRequest, which the server HAS registered for
  client.Send(TestRequest{123});

  // Receipt of TestReply was not fatal. Receipt of TestRequest was successful.
  test::WaitUntil([&]() { return received.Read(&Received::value) != 0; });
  received.Mutate([&](auto& r) {
    EXPECT_EQ(123, r.value);
    r = {}; // clear
  });
}

TEST(NetMessageServer, ClientVersionMismatch) {
  MessageServer server;
  MessageClient client;

  // Set different version numbers
  server.SetVersion(1);
  client.SetVersion(2);

  // Connection refused because of the version mismatch
  server.StartInProc(/*maxClients*/ 1, "VersionOneServer");
  EXPECT_EQ(SocketStatus::None, client.GetStatus());
  client.ConnectInProc(server.GetSocket_InternalUseOnly());
  EXPECT_NE(SocketStatus::None, client.GetStatus());
  test::WaitUntil([&]() { return client.GetStatus() == SocketStatus::Lost; });
}

TEST(NetMessageServer, CalcProtocolVersionHash) {
  // Prove that CalcProtocolVersionHash simply passes through to MessageDispatcher (which has its
  // own tests).
  MessageDispatcher<> dispatcher;
  MessageServer server;
  EXPECT_EQ(dispatcher.CalcProtocolVersionHash(), server.CalcProtocolVersionHash());
  dispatcher.Register<TestRequest>();
  server.Register<TestRequest>();
  EXPECT_EQ(dispatcher.CalcProtocolVersionHash(), server.CalcProtocolVersionHash());
  dispatcher.Register<TestReply>();
  server.Register<TestReply>();
  EXPECT_EQ(dispatcher.CalcProtocolVersionHash(), server.CalcProtocolVersionHash());
}

TEST(NetMessageClient, CalcProtocolVersionHash) {
  // Prove that CalcProtocolVersionHash simply passes through to MessageDispatcher (which has its
  // own tests).
  MessageDispatcher<> dispatcher;
  MessageClient client;
  EXPECT_EQ(dispatcher.CalcProtocolVersionHash(), client.CalcProtocolVersionHash());
  dispatcher.Register<TestRequest>();
  client.Register<TestRequest>();
  EXPECT_EQ(dispatcher.CalcProtocolVersionHash(), client.CalcProtocolVersionHash());
  dispatcher.Register<TestReply>();
  client.Register<TestReply>();
  EXPECT_EQ(dispatcher.CalcProtocolVersionHash(), client.CalcProtocolVersionHash());
}

//--------------------------------------------------------------------------------
// Test fixture for MessageClient::SendAndAwaitReply
//--------------------------------------------------------------------------------

namespace {
class NetMessageClient_RequestReply : public ::testing::Test {
 protected:
  static constexpr double kTimeout = 15.0; // Generous to avoid flaky test failures

  struct RequestInfo {
    uint64_t id = 0;
    int value = 0;
  };

  MessageServer _server;
  MessageClient _client;
  Guarded<DynamicArray<RequestInfo>> _serverRequests;
  std::atomic<int> _serverRequestCount = 0;
  std::atomic<bool> _serverAutoRespond = false;

  void SetUp() override {
    // Server responds to each request type (auto or deferred).
    ServerRegisterRequest<TestRequest>();
    ServerRegisterRequest<TestRequest2>();
    _server.StartInProc(/*maxClients*/ 1, "SendAndAwaitReplyTest");

    // Client registers reply types with no callback so replies route to the request/reply
    // machinery.
    _client.Register<TestReply>();
    _client.Register<TestReply2>();
    Connect(_client, _server);
  }

  template <class RequestT>
  void ServerRegisterRequest() {
    _server.Register<RequestT>([this](ClientId id, auto&& msg) {
      using ReplyT = typename RequestT::Reply;
      if (_serverAutoRespond) {
        _server.SendTo(id, ReplyT{msg});
      } else {
        _serverRequests.Mutate(
            [&](auto& list) { list.emplace_back(RequestInfo{msg.requestId, msg.value}); });
      }
      ++_serverRequestCount;
    });
  }

  template <class ReplyT>
  void ServerReply(RequestInfo const& info) {
    ReplyT reply;
    reply.requestId = info.id;
    reply.value = info.value;
    _server.Broadcast(reply); // Broadcast is OK because there is only one client
  }
};
} // namespace

TEST_F(NetMessageClient_RequestReply, BasicRoundTrip) {
  _serverAutoRespond = true;
  {
    auto reply = _client.SendAndAwaitReply(TestRequest{123}, kTimeout, test::ExpectOK{});
    EXPECT_EQ(123, reply.value);
    EXPECT_EQ(1, _serverRequestCount);
  }
  {
    auto reply = _client.SendAndAwaitReply(TestRequest2{456}, kTimeout, test::ExpectOK{});
    EXPECT_EQ(456, reply.value);
    EXPECT_EQ(2, _serverRequestCount);
  }
}

TEST_F(NetMessageClient_RequestReply, Timeout) {
  _serverAutoRespond = false;
  [[maybe_unused]] auto reply = _client.SendAndAwaitReply(
      TestRequest{123}, /*timeout*/ 0.001, test::ExpectNotOK{}); // 1ms timeout
  test::WaitUntil([&]() { return _serverRequestCount == 1; });

  // A reply after timeout will be ignored
  auto const requests = _serverRequests.Load();
  ASSERT_EQ(1, isize(requests));
  ServerReply<TestReply>(requests[0]);

  // Prove that the client is still in a healthy state
  _serverAutoRespond = true;
  auto reply2 = _client.SendAndAwaitReply(TestRequest2{456}, kTimeout, test::ExpectOK{});
  EXPECT_EQ(456, reply2.value);
}

TEST_F(NetMessageClient_RequestReply, ClientDisconnectCancels) {
  _serverAutoRespond = false;

  // Start request threads that block awaiting a reply
  DynamicArray<std::thread> threads;
  int constexpr kNumThreads = 4;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([this]() {
      [[maybe_unused]] auto reply =
          _client.SendAndAwaitReply(TestRequest{123}, kTimeout, test::ExpectNotOK{});
    });
  }

  // Wait until the requests have been received by the server
  test::WaitUntil([&]() { return _serverRequestCount == kNumThreads; });

  // Disconnecting cancels all in-flight requests
  _client.Disconnect();
  for (auto& t : threads) {
    t.join();
  }
}

TEST_F(NetMessageClient_RequestReply, ServerDisconnectCancels) {
  _serverAutoRespond = false;

  // Start request threads that block awaiting a reply
  DynamicArray<std::thread> threads;
  int constexpr kNumThreads = 4;
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([this]() {
      [[maybe_unused]] auto reply =
          _client.SendAndAwaitReply(TestRequest{123}, kTimeout, test::ExpectNotOK{});
    });
  }

  // Wait until the requests have been received by the server
  test::WaitUntil([&]() { return _serverRequestCount == kNumThreads; });

  // Stopping the server cancels all in-flight requests
  _server.Stop();
  for (auto& t : threads) {
    t.join();
  }
}

TEST_F(NetMessageClient_RequestReply, NotConnected) {
  _client.Disconnect();
  [[maybe_unused]] auto reply =
      _client.SendAndAwaitReply(TestRequest{123}, kTimeout, test::ExpectNotOK{});
  EXPECT_EQ(0, _serverRequestCount);
}

TEST_F(NetMessageClient_RequestReply, OutOfOrder) {
  _serverAutoRespond = false;

  // Start a few request threads
  int constexpr kNumThreads = 4;
  DynamicArray<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([this, i]() {
      auto reply = _client.SendAndAwaitReply(TestRequest{i}, kTimeout, test::ExpectOK{});
      EXPECT_EQ(i * 2, reply.value); // Doubled by the test when replying
    });
  }

  // Wait for the requests to come in
  test::WaitUntil([&]() { return _serverRequestCount == kNumThreads; });
  auto requests = _serverRequests.Load();
  ASSERT_EQ(kNumThreads, isize(requests));

  // Send replies in opposite order from the order received by the server
  for (int i = isize(requests) - 1; i >= 0; --i) {
    ServerReply<TestReply>(RequestInfo{requests[i].id, requests[i].value * 2});
  }

  // Wait for replies to be received
  for (auto& t : threads) {
    t.join();
  }
}

TEST_F(NetMessageClient_RequestReply, ReceiveReplyTwoWays) {
  // Request and reply messages can be used in two ways:
  //  1. Send request via Send. Receive the reply later via normal registered callback.
  //  2. Send request via SendAndAwaitReply. Receive the message synchronously, or time-out.
  //
  // These two methods can be used simultaneiously. It all depends on whether or not a valid
  // request ID was used.

  _serverAutoRespond = true;

  Guarded<int> receivedValue{0};

  // If the request is sent via Send, then the reply will be received this way.
  _client.Register<TestReply>([&](auto&& msg) {
    receivedValue.Mutate([&](auto& v) {
      EXPECT_NE(0, msg.value);
      EXPECT_EQ(0, v) << "Already received";
      EXPECT_EQ(0, msg.requestId);
      v = msg.value; // Store
    });
  });

  _client.Send(TestRequest{123});
  test::WaitUntil([&]() { return receivedValue.Load() != 0; });
  EXPECT_EQ(123, receivedValue.Load());
  receivedValue.Store(0); // clear

  // The the request is sent via SendAndAwaitReply, then we receive it here and the registered
  // callback will NOT be invoked.
  auto reply = _client.SendAndAwaitReply(TestRequest{456}, kTimeout, test::ExpectOK{});
  EXPECT_EQ(456, reply.value);
  EXPECT_NE(0, reply.requestId);
  EXPECT_EQ(0, receivedValue.Load()); // Not received via callback
}

TEST_F(NetMessageClient_RequestReply, Stress) {
  _serverAutoRespond = true;

  // Start a few request threads that will each issue many requests across both pairs
  int constexpr kNumThreads = 4;
  int constexpr kNumRequestsPerThread = 1000;
  DynamicArray<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int iThread = 0; iThread < kNumThreads; ++iThread) {
    threads.emplace_back([this, iThread]() {
      for (int i = 0; i < kNumRequestsPerThread; ++i) {
        int value = iThread * kNumRequestsPerThread + i;
        if (i & 1) { // Alternate request/reply message types
          auto reply = _client.SendAndAwaitReply(TestRequest{value}, kTimeout, test::ExpectOK{});
          EXPECT_EQ(value, reply.value);
        } else {
          auto reply = _client.SendAndAwaitReply(TestRequest2{value}, kTimeout, test::ExpectOK{});
          EXPECT_EQ(value, reply.value);
        }
      }
    });
  }

  // Wait
  for (auto& t : threads) {
    t.join();
  }
  EXPECT_EQ(kNumThreads * kNumRequestsPerThread, _serverRequestCount);
}

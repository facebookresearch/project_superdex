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

#include "console_test_utils.h"
#include "mochi_debugger_test.h"

#include <mochi_core/test/mochi_test_helpers.h>
#include <mochi_core/test/wait_until.h>
#include <mochi_core/utils/array_utils.h>
#include <mochi_core/utils/nd_array.h>
#include <mochi_core/utils/reflection_error.h>
#include <mochi_core/utils/string_utils.h>
#include <mochi_physics/src/mochi_common_components.h>
#include <mochi_physics/src/mochi_scene.h>

#include <picojson/picojson.h>
#include <simple_reflection/simple_reflection.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <future>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

using namespace mochi;
using namespace mochi::dbg;

namespace {

// Cover the console protocol and special routing paths here. Reflected API behavior and member
// coverage belong in those APIs' tests. Use test-owned roots when asserting particular members.
// Real API members are routing sentinels only. Do not enumerate each class's properties or
// methods here. Adding or changing reflected API members should not require console-test edits.

using Lines = DynamicArray<std::string>;

std::string NormalizeOutput(std::string_view command, std::string_view output) {
  bool const listing = command.substr(0, command.find_first_of(" \t\r\n\v\f")) == "ls";
  return listing ? test::RemoveUnquotedWhitespace(output) : std::string{output};
}

void ExpectOutput(std::string_view command, std::string_view expected, Lines const& actual) {
  ASSERT_EQ(1, actual.size());
  EXPECT_EQ(NormalizeOutput(command, expected), NormalizeOutput(command, actual[0]));
}

struct CommandCase {
  std::string_view cwd;
  std::string_view command;
  std::string output;
  std::string_view error;
};

struct ConsoleChild {
  std::string name = "child";
  int32_t value = 3;

  MOCHI_STRUCT_BEGIN(ConsoleChild)
  MOCHI_FIELD(name)
  MOCHI_FIELD(value)
  MOCHI_STRUCT_END()
};

struct ConsoleTestObject {
  ConsoleChild child;
  std::array<int32_t, 3> numbers{1, 2, 3};
  std::string text = "hello world";

  int32_t GetValue() const {
    return _value;
  }
  void SetValue(int32_t value) {
    _value = value;
  }
  int32_t Add(int32_t amount) {
    return _value += amount;
  }
  bool IsPositive() const {
    return _value > 0;
  }
  void Reset() {
    _value = 7;
  }

  MOCHI_STRUCT_BEGIN(ConsoleTestObject)
  MOCHI_FIELD(child)
  MOCHI_FIELD(numbers)
  MOCHI_FIELD(text)
  MOCHI_PROPERTY("value", &ConsoleTestObject::GetValue, &ConsoleTestObject::SetValue)
  MOCHI_METHOD("add(amount)", &ConsoleTestObject::Add)
  MOCHI_METHOD("positive()", &ConsoleTestObject::IsPositive)
  MOCHI_METHOD("reset()", &ConsoleTestObject::Reset)
  MOCHI_STRUCT_END()

 private:
  int32_t _value = 7;
};

class ConsoleTest : public MochiDebuggerTest {
 protected:
  static constexpr double kShortTimeout = 0.01;

  void SetUp() override {
    MochiDebuggerTest::SetUp();
    _sceneA = _context->CreateScene("A");
    _sceneB = _context->CreateScene("1/2");
    InstallConsoleRoot();
    StartServer();
    ConnectClient();
  }

  virtual void InstallConsoleRoot() {
    _server->SetConsoleRoot_ForTestingOnly(
        &_consoleObject, SReflect::GetTypeInfo<ConsoleTestObject>());
  }

  Lines TakePrinted() {
    return _clientLogs.Mutate([](auto& logs) {
      Lines lines;
      for (auto const& log : logs) {
        if (!log.file.empty()) {
          continue;
        }
        std::string_view text{log.message};
        if (text.ends_with('\n')) {
          text.remove_suffix(1);
        }
        lines.push_back((log.channel == LogChannel::Error ? "error: " : "") + std::string{text});
      }
      logs.clear();
      return lines;
    });
  }

  Lines Run(std::initializer_list<std::string_view> commands) {
    TakePrinted();
    for (std::string_view command : commands) {
      _client->QueueCommand(command);
    }
    auto finished = std::async(std::launch::async, [&] { _client->WaitForQueuedCommands(); });
    test::WaitUntil([&] {
      for (Scene* scene : {_sceneA, _sceneB}) {
        if (scene) {
          scene->UpdateDebugger();
        }
      }
      for (Scene* scene : _extraScenes) {
        scene->UpdateDebugger();
      }
      return finished.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
    });
    finished.get();
    return TakePrinted();
  }

  void SetDirectory(std::string_view path) {
    std::string const command = "cd " + SReflect::ToJsonString(std::string{path}, false);
    EXPECT_TRUE(Run({"cd /", command}).empty());
    EXPECT_EQ(path, _client->GetCommandPrompt());
  }

  DynamicArray<CommandCase> RemoteCases() const {
    std::string const methods =
        "add (int32 amount) -> int32\npositive () -> bool\nreset () -> void\n\n";
    std::string const root =
        methods + "child\nnumbers [1,2,3]\nscenes (size=2)\ntext \"hello world\"\nvalue 7";
    std::string const rootDetailed = methods + "child " +
        std::string{SReflect::GetTypeInfo<ConsoleChild>()._name} + "\nnumbers " +
        SReflect::GetTypeInfo<std::array<int32_t, 3>>()._name +
        " [1,2,3]\nscenes (size=2)\ntext string \"hello world\"\nvalue int32 7";
    std::string const child = "name \"child\"\nvalue 3";
    std::string const scenes = "1_2 (handle=" + std::to_string(_sceneB->GetHandle().value) +
        ")\nA (handle=" + std::to_string(_sceneA->GetHandle().value) + ")";
    std::string const scenesJson = R"({
  "1_2": {},
  "A": {}
})";
    return {
        {"/", "value", "7", ""},
        {"/scenes/A", "/value", "7", ""},
        {"/", "positive", "true", ""},
        {"/", "add 0", "7", ""},
        {"/", "text", "\"hello world\"", ""},
        {"/", "numbers/1", "2", ""},
        {"/", "ls", root, ""},
        {"/scenes/A", "ls /", root, ""},
        {"/", "ls -l", rootDetailed, ""},
        {"/scenes/A", "ls / -l", rootDetailed, ""},
        {"/", "ls scenes", scenes, ""},
        {"/scenes/A", "ls /scenes", scenes, ""},
        {"/scenes/A", "ls ../", scenes, ""},
        {"/scenes/A", "ls /scenes/my_scene/../../scenes", scenes, ""},
        {"/scenes/A", "ls /scenes/my_scene/../../scenes/", scenes, ""},
        {"/", "ls child", child, ""},
        {"/child", "ls", child, ""},
        {"/child", "ls\t-l", "name string \"child\"\nvalue int32 3", ""},
        {"/", "ls scenes -l", scenes, ""},
        {"/scenes/A", "ls '../' -l", scenes, ""},
        {"/", "/scenes/A/name", "\"A\"", ""},
        {"/scenes/A", "name", "\"A\"", ""},
        {"/scenes/A", "./name", "\"A\"", ""},
        {"/scenes/A", "../A//name/", "\"A\"", ""},
        {"/", "'scenes/\"1_2\"/name'", "\"1/2\"", ""},
        {"/", "missing", "", "not found"},
        {"/", "scenes/missing/name", "", "not found"},
        {"/child", "missing", "", "not found"},
        {"/child", "./echo hi", "", "not found"},
        {"/", "add", "", "argument"},
        {"/", "add 1 2", "", "argument"},
        {"/", "../value", "", "Cannot go above the root"},
        {"/", "ls /../scenes", "", "Cannot go above the root"},
        {"/", "scenes", scenesJson, ""},
        {"/", "scenes 1", "", "is read-only"},
        {"/", "ls value", "", "Not a directory"},
        {"/", "ls 'child/\"A'", "", "Unterminated quotation"}};
  }

  void CheckCases(Span<CommandCase const> cases) {
    for (auto const& c : cases) {
      SCOPED_TRACE(std::string{c.cwd} + "> " + std::string{c.command});
      SetDirectory(c.cwd);
      auto const lines = Run({c.command});
      ASSERT_EQ(1, lines.size());
      if (c.error.empty()) {
        ExpectOutput(c.command, c.output, lines);
      } else {
        EXPECT_TRUE(lines[0].starts_with("error: ")) << lines[0];
        EXPECT_NE(
            std::string::npos,
            NormalizeOutput(c.command, lines[0]).find(NormalizeOutput(c.command, c.error)))
            << lines[0];
      }
      EXPECT_EQ(c.cwd, _client->GetCommandPrompt());
    }
  }

  void SetTimeout(double timeout) {
    auto settings = _client->GetSettings();
    settings.commandTimeout = timeout;
    _client->SetSettings(settings);
  }

  void MakeServerUnresponsive() {
    _server->GetMessageServer_ForTestingOnly().GetSocket_InternalUseOnly().SetReceiveCallback(
        [this](auto, auto, auto) { _received = true; });
  }

  ConsoleTestObject _consoleObject;
  DynamicArray<Scene*> _extraScenes;
  Scene* _sceneA = nullptr;
  Scene* _sceneB = nullptr;
  std::atomic<bool> _received = false;
};

class ActorConsoleTest : public ConsoleTest {
 protected:
  void SetUp() override {
    ConsoleTest::SetUp();
    auto const [coordinates, connectivity] = test::CreateMinimalTetMeshUnitCube();
    _shape = _context->CreateTetMeshShape(
        Flatten(MakeConstSpan(coordinates)),
        Flatten(MakeConstSpan(connectivity)),
        test::ExpectOK{});
    _box = AddRigid(_sceneA, "box");
    _robot = AddRobot(_sceneA, "robot");
  }

  ActorHandle AddRigid(Scene* scene, std::string_view name) const {
    RigidActorParams params{};
    params.name = DynamicString{name};
    params.shape = _shape;
    params.colliderType = ColliderType::Box;
    return scene->CreateRigidActor(params, test::ExpectOK{})->GetHandle();
  }

  // An articulated actor with two links
  Actor* AddRobot(
      Scene* scene,
      std::string_view name,
      std::string_view link0 = "link0",
      std::string_view link1 = "link1") const {
    ArticulatedActorParams robot{};
    robot.name = DynamicString{name};
    robot.joints.resize(2, {});
    robot.joints[0].type = ArticulatedJointType::Free;
    robot.joints[1].type = ArticulatedJointType::Spherical;
    robot.links.resize(2, {});
    robot.links[0].name = DynamicString{link0};
    robot.links[0].shape = _shape;
    robot.links[0].colliderType = ColliderType::Box;
    robot.links[1].name = DynamicString{link1};
    robot.links[1].parentLink = 0;
    robot.links[1].shape = _shape;
    robot.links[1].colliderType = ColliderType::Box;
    return scene->CreateArticulatedActor(robot, test::ExpectOK{});
  }

  void ExpectDirectories(std::string_view path, DynamicArray<std::string> expected) {
    std::ranges::sort(expected);
    std::string const command = "ls " + std::string{path};
    auto const output = Run({command});
    ASSERT_EQ(1, output.size());
    Lines names;
    std::istringstream lines(output[0]);
    std::string line;
    while (std::getline(lines, line)) {
      std::istringstream row(line);
      std::string name;
      std::string handle;
      if (row >> name) {
        EXPECT_TRUE(bool(row >> handle));
        names.push_back(name);
      }
    }
    EXPECT_EQ(expected, names);
  }

  ShapeHandle _shape;
  ActorHandle _box;
  Actor* _robot = nullptr;
};

} // namespace

TEST_F(ConsoleTest, RemoteCommands) {
  CheckCases(RemoteCases());
}

TEST_F(ConsoleTest, ListingFlagsAndSpacing) {
  auto const detailed = Run({"ls child -l"});
  ASSERT_EQ(1, detailed.size());
  EXPECT_EQ(detailed, Run({"ls -l child"}));
  EXPECT_EQ(detailed, Run({"ls -l -- child"}));
  auto const listing = Run({"ls"});
  ASSERT_EQ(1, listing.size());
  EXPECT_TRUE(listing[0].starts_with("add"));
  EXPECT_NE(std::string::npos, listing[0].find("\n\nchild"));
  auto const unsupported = Run({"ls -m"});
  ASSERT_EQ(1, unsupported.size());
  EXPECT_TRUE(unsupported[0].starts_with("error: Usage: ls"));
}

TEST_F(ConsoleTest, ChangesDirectoryOnlyAfterValidation) {
  struct Case {
    std::string_view cwd;
    std::string_view path;
    std::string_view result;
    std::string_view error;
  };
  Case const cases[] = {
      {"/", "scenes/A", "/scenes/A", ""},
      {"/scenes/A", "/scenes", "/scenes", ""},
      {"/scenes/A", "../", "/scenes", ""},
      {"/scenes/A", "/scenes/my_scene/../../scenes", "/scenes", ""},
      {"/scenes/A", "/scenes/my_scene/../../scenes/", "/scenes", ""},
      {"/child", "value/../..", "/", ""},
      {"/", "./child//", "/child", ""},
      {"/", "'scenes/\"1_2\"'", "/scenes/\"1_2\"", ""},
      {"/child", "missing", "/child", "not found"},
      {"/scenes/A", "/scene", "/scenes/A", "not found"},
      {"/child", "value", "/child", "Not a directory"},
      {"/scenes/A", "../../..", "/scenes/A", "Cannot go above the root"},
      {"/scenes/A", "/../scenes", "/scenes/A", "Cannot go above the root"},
      {"/", "../../", "/", "Cannot go above the root"}};
  for (auto const& c : cases) {
    SCOPED_TRACE(std::string{c.cwd} + "> cd " + std::string{c.path});
    SetDirectory(c.cwd);
    auto const lines = Run({"cd " + std::string{c.path}});
    if (c.error.empty()) {
      EXPECT_TRUE(lines.empty());
    } else {
      ASSERT_EQ(1, lines.size());
      EXPECT_NE(std::string::npos, lines[0].find(c.error)) << lines[0];
    }
    EXPECT_EQ(c.result, _client->GetCommandPrompt());
  }
  for (std::string_view path :
       {"../", "/scenes/", "/scenes/my_scene/../../scenes", "/scenes/my_scene/../../scenes/"}) {
    SetDirectory("/scenes/\"1_2\"");
    DisconnectClient();
    EXPECT_EQ((Lines{"/scenes", "/"}), Run({"cd " + std::string{path}, "pwd", "cd /", "pwd"}));
    ConnectClient();
  }
  _context->DestroyScene(_sceneA);
  _context->DestroyScene(_sceneB);
  _sceneA = _sceneB = nullptr;
  EXPECT_TRUE(Run({"cd /scenes", "ls"}).empty()); // Empty directories are valid
  EXPECT_EQ("/scenes", _client->GetCommandPrompt());
}

TEST_F(ConsoleTest, OrdersLocalAndRemoteCommands) {
  auto const lines = Run({"cd child", "value 4", "value", "cd /", "value", "echo after"});
  ASSERT_EQ(4, lines.size());
  EXPECT_EQ("4", lines[0]);
  EXPECT_EQ(4, _consoleObject.child.value);
  EXPECT_EQ(lines[0], lines[1]);
  EXPECT_EQ("7", lines[2]);
  EXPECT_EQ("after", lines[3]);
  _client->QueueCommand([] { throw std::runtime_error("oops"); });
  _client->QueueCommand("echo next");
  _client->WaitForQueuedCommands();
  EXPECT_EQ((Lines{"error: Exception: oops", "next"}), TakePrinted());
  std::promise<void> nestedQueued;
  auto nestedReady = nestedQueued.get_future();
  _client->QueueCommand([&] {
    _client->QueueCommand("echo nested");
    nestedQueued.set_value();
  });
  nestedReady.wait();
  _client->WaitForQueuedCommands();
  EXPECT_EQ(Lines{"nested"}, TakePrinted());
}

TEST_F(ConsoleTest, MutatesTheInjectedObject) {
  EXPECT_EQ(
      (Lines{"11", "16", "-1", "false", "7", "true", "\"quoted value\"", "9"}),
      Run(
          {"value 11",
           "add 5",
           "value -1",
           "positive",
           "reset",
           "value",
           "positive",
           "text 'quoted value'",
           "numbers/1 9"}));
  EXPECT_EQ(7, _consoleObject.GetValue());
  EXPECT_EQ("quoted value", _consoleObject.text);
  EXPECT_EQ(9, _consoleObject.numbers[1]);
}

TEST_F(ConsoleTest, WaitOnlyFlushesEarlierCommands) {
  std::atomic<bool> keepQueuing = true;
  std::function<void()> command = [&] {
    if (keepQueuing) {
      _client->QueueCommand(command);
    }
  };
  _client->QueueCommand(command);
  auto finished = std::async(std::launch::async, [&] { _client->WaitForQueuedCommands(); });
  EXPECT_EQ(std::future_status::ready, finished.wait_for(std::chrono::seconds(2)));
  keepQueuing = false;
  finished.get();
  _client->WaitForQueuedCommands();
  _client->WaitForQueuedCommands();
}

TEST_F(ConsoleTest, LocalCommands) {
  struct Case {
    std::string_view command;
    std::string_view prefix;
    size_t count;
  };
  Case const cases[] = {
      {"", "", 0},
      {"echo 'hello world'", "hello world", 1},
      {"pwd", "/", 1},
      {"help", "Commands:", 1},
      {"ping", "ping:", 6},
      {"connect invalid", "error: Invalid address", 1},
      {"connect 127.0.0.1:bad", "error: Invalid port number", 1},
      {"connect a b", "error: Invalid number of arguments", 1},
      {"cd", "error: Invalid number of arguments", 1},
      {"ls a b", "error: Usage: ls [<path>] [-l]", 1},
      {"ls a b -l", "error: Usage: ls [<path>] [-l]", 1},
      {"disconnect extra", "error: Invalid number of arguments", 1},
      {"ping extra", "error: Invalid number of arguments", 1},
      {"echo \"a\\q\"", "error: Unsupported escape sequence", 1}};
  for (auto const& c : cases) {
    SCOPED_TRACE(c.command);
    auto const lines = Run({c.command});
    ASSERT_EQ(c.count, lines.size());
    for (auto const& line : lines) {
      EXPECT_TRUE(
          NormalizeOutput(c.command, line).starts_with(NormalizeOutput(c.command, c.prefix)))
          << line;
    }
  }
  auto const servers = Run({"servers"});
  ASSERT_EQ(1, servers.size());
  EXPECT_TRUE(servers[0].starts_with("Servers:") || servers[0] == "No servers found.");
  EXPECT_TRUE(Run({"quit"}).empty());
  EXPECT_TRUE(_client->WasExitRequested());
}

TEST_F(ConsoleTest, ConnectAndDisconnectCommands) {
  EXPECT_TRUE(Run({"disconnect"}).empty());
  EXPECT_EQ(net::SocketStatus::None, _client->GetStatus());
  _server->Stop();
  _server->Start(0);
  std::string const connect = "connect 127.0.0.1:" + std::to_string(_server->GetPort());
  Run({connect});
  test::WaitUntil([&] { return _client->GetStatus() == net::SocketStatus::Connected; });
  EXPECT_TRUE(_server->HasConnection());
  EXPECT_EQ(Lines{"7"}, Run({"value"}));
  Run({"disconnect"});
  test::WaitUntil([&] { return !_server->HasConnection(); });
  for (auto const& c : RemoteCases()) {
    SCOPED_TRACE(c.command);
    ExpectOutput(c.command, "error: No connection", Run({c.command}));
  }
  EXPECT_EQ(Lines{"error: No connection"}, Run({"ping"}));
  EXPECT_EQ(Lines{"error: No connection"}, Run({"cd scenes/A"}));
}

TEST_F(ConsoleTest, RemoteCommandsTimeOut) {
  SetTimeout(kShortTimeout);
  MakeServerUnresponsive();
  std::string const error = "error: No reply from the server: Timeout";
  for (auto const& c : RemoteCases()) {
    SCOPED_TRACE(c.command);
    ExpectOutput(c.command, error, Run({c.command}));
  }
  EXPECT_EQ((Lines{error, error}), Run({"cd scenes/A", "ping"}));
  EXPECT_EQ("/", _client->GetCommandPrompt());
  EXPECT_TRUE(_received);
}

TEST_F(ConsoleTest, DisconnectAndDestructionCancelPendingCommands) {
  TakePrinted();
  MakeServerUnresponsive();
  _client->QueueCommand("ls");
  test::WaitUntil([&] { return _received.load(); });
  _client->Disconnect();
  _client->WaitForQueuedCommands();
  auto lines = TakePrinted();
  ASSERT_EQ(1, lines.size());
  EXPECT_TRUE(NormalizeOutput("ls", lines[0])
                  .starts_with(NormalizeOutput("ls", "error: No reply from the server:")))
      << lines[0];
  ConnectClient();
  _received = false;
  TakePrinted();
  _client->QueueCommand("ls");
  test::WaitUntil([&] { return _received.load(); });
  _client->QueueCommand("echo dropped");
  _client.reset();
  _client = std::make_unique<DebugClient>();
  lines = TakePrinted();
  ASSERT_EQ(1, lines.size());
  EXPECT_TRUE(NormalizeOutput("ls", lines[0])
                  .starts_with(NormalizeOutput("ls", "error: No reply from the server:")))
      << lines[0];
}

TEST_F(ActorConsoleTest, ActorCommands) {
  std::string const actorsJson = R"({
  "box": {},
  "robot": {}
})";
  std::string const childrenJson = R"({
  "link0": {},
  "link1": {}
})";
  auto const links = _robot->GetNestedLinkActors(ErrorAssert{});
  std::string const actorRows = "box (handle=" + std::to_string(_box.value) +
      ")\nrobot (handle=" + std::to_string(_robot->GetHandle().value) + ")";
  std::string const childRows = "link0 (handle=" + std::to_string(links[0].value) +
      ")\nlink1 (handle=" + std::to_string(links[1].value) + ")";
  CommandCase const cases[] = {
      {"/scenes/A", "ls actors", actorRows, ""},
      {"/scenes/A/actors/box", "name", "\"box\"", ""},
      {"/scenes/A/actors/robot", "name", "\"robot\"", ""},
      {"/scenes/A/actors/robot", "ls children", childRows, ""},
      {"/scenes/A/actors/robot/children", "link0/name", "\"robot/link0\"", ""},
      {"/scenes/A/actors/robot/children", "link1/name", "\"robot/link1\"", ""},
      {"/scenes/A", "actors/box/name", "\"box\"", ""},
      {"/scenes/A", "actors/missing/name", "", "not found"},
      {"/scenes/A/actors/robot", "ls children/missing", "", "not found"},
      {"/scenes/A", "actors", actorsJson, ""},
      {"/scenes/A/actors/robot", "children", childrenJson, ""},
      {"/scenes/A", "actors 1", "", "Cannot set value because \"actors\" is read-only"},
      {"/scenes/A/actors/box", "ls children", "", "not found"}};
  CheckCases(cases);
}

TEST_F(ConsoleTest, SceneDebugNames) {
  std::pair<std::string_view, std::string_view> const names[] = {
      {"A2", "A2"},
      {"A", "A3"},
      {"", "scene"},
      {"my scene/x", "my_scene_x"},
      {"a\"b'c\td\ae", "a_b_c_d_e"},
      {"..", "scene2"},
      {".", "scene3"},
      {"[1]", "_1]"},
      {"{a", "_a"},
      {"@a", "_a2"}};
  DynamicArray<std::string> entries{
      "1_2 (handle=" + std::to_string(_sceneB->GetHandle().value) + ")",
      "A (handle=" + std::to_string(_sceneA->GetHandle().value) + ")"};
  for (auto const& [createdAs, name] : names) {
    Scene* scene = _context->CreateScene(createdAs);
    _extraScenes.push_back(scene);
    entries.emplace_back(
        std::string{name} + " (handle=" + std::to_string(scene->GetHandle().value) + ")");
  }
  std::ranges::sort(entries);
  ExpectOutput("ls scenes", Join(entries, "\n"), Run({"ls scenes"}));
  for (auto const& [createdAs, name] : names) {
    std::string const command = "cd /scenes/" + std::string{name};
    EXPECT_EQ(Lines{SReflect::ToJsonString(std::string{createdAs}, false)}, Run({command, "name"}));
  }
}

TEST_F(ActorConsoleTest, ActorDebugNames) {
  std::pair<std::string_view, std::string_view> const names[] = {
      {"", "Rigid"},
      {".", "Rigid2"},
      {"sceneInstance/nestedPrefab/robot", "sceneInstance_nestedPrefab_robot"},
      {"my box", "my_box"},
      {"[robot", "_robot"},
      {"@box", "_box"}};
  DynamicArray<std::string> expected;
  for (auto const& [createdAs, name] : names) {
    AddRigid(_sceneB, createdAs);
    expected.emplace_back(name);
  }
  AddRobot(_sceneB, "");
  expected.emplace_back("Articulated");
  ExpectDirectories("/scenes/1_2/actors", expected);
  ExpectDirectories("/scenes/1_2/actors/Articulated/children", {"link0", "link1"});
  EXPECT_EQ(
      Lines{"\"sceneInstance/nestedPrefab/robot\""},
      Run({"/scenes/1_2/actors/sceneInstance_nestedPrefab_robot/name"}));
}

TEST_F(ActorConsoleTest, DuplicateNamesStayStable) {
  ActorHandle const second = AddRigid(_sceneA, "box");
  Run({"ls /scenes/A/actors"});
  auto const [plain, suffixed] = std::minmax(_box, second);
  AddRigid(_sceneA, "box");
  ExpectDirectories("/scenes/A/actors", {"box", "box2", "box3", "robot"});

  // A freed name can be taken again, but numbers continue from the last one, so box2 is not reused
  _sceneA->DestroyActor(plain);
  _sceneA->DestroyActor(suffixed);
  AddRigid(_sceneA, "box");
  AddRigid(_sceneA, "box");
  ExpectDirectories("/scenes/A/actors", {"box", "box3", "box4", "robot"});
}

TEST_F(ActorConsoleTest, ChildrenHaveTheirOwnNameScope) {
  Actor const* const robot = AddRobot(_sceneB, "a/b/robot", "l 0", "l_0");
  AddRigid(_sceneB, "l_0");
  ExpectDirectories("/scenes/1_2/actors", {"a_b_robot", "l_0"});
  ExpectDirectories("/scenes/1_2/actors/a_b_robot/children", {"l_0", "l_02"});
  EXPECT_STREQ("a/b/robot", robot->GetName());
}

TEST_F(ActorConsoleTest, DisconnectRemovesDebugNames) {
  entt::registry const& reg = assert_cast<SceneImpl*>(_sceneA)->GetRegistry();
  auto const numNames = [&] { return static_cast<int>(reg.view<CDebugName const>().size()); };
  Run({"ls /scenes/A/actors"});
  EXPECT_EQ(_sceneA->GetNumActors(), numNames());
  DisconnectClient();
  test::WaitUntil([&] { return !_server->HasConnection(); });
  Run({});
  EXPECT_EQ(0, numNames());
}

namespace {

class DefaultRootConsoleTest : public ConsoleTest {
 protected:
  void InstallConsoleRoot() override {}
};

struct ShapeConsoleTestObject {
  Context* context = nullptr;

  ShapeHandle CreateShape(real radius, Error& error) const {
    return context->CreateSphereShape({}, radius, error);
  }
  void ReleaseShape(ShapeHandle const& handle) const {
    context->ReleaseShape(handle);
  }

  MOCHI_STRUCT_BEGIN(ShapeConsoleTestObject)
  MOCHI_METHOD_WITH_ERROR("createShape(radius)", &ShapeConsoleTestObject::CreateShape)
  MOCHI_METHOD("releaseShape(handle)", &ShapeConsoleTestObject::ReleaseShape)
  MOCHI_STRUCT_END()
};

class ShapeConsoleTest : public ConsoleTest {
 protected:
  void InstallConsoleRoot() override {
    _shapeObject.context = _context;
    _server->SetConsoleRoot_ForTestingOnly(
        &_shapeObject, SReflect::GetTypeInfo<ShapeConsoleTestObject>());
  }

  ShapeConsoleTestObject _shapeObject;
};

} // namespace

TEST_F(DefaultRootConsoleTest, ListsScenesAtDefaultRoot) {
  auto listing = Run({"ls"});
  ASSERT_EQ(1, listing.size());
  EXPECT_NE(std::string::npos, listing[0].find("scenes"));
  // port is a sentinel for /debug routing. Do not add tests for every future DebugServer member
  // or assert port's behavior here. DebugServer's own tests cover its API.
  auto const port = Run({"/debug/port"});
  ASSERT_EQ(1, port.size());
  EXPECT_FALSE(port[0].starts_with("error: ")) << port[0];
  SetDirectory("/scenes/A");
  ExpectOutput(
      "ls /scenes",
      "1_2 (handle=" + std::to_string(_sceneB->GetHandle().value) +
          ")\nA (handle=" + std::to_string(_sceneA->GetHandle().value) + ")",
      Run({"ls /scenes"}));
}

TEST_F(ShapeConsoleTest, RetainsHandlesReturnedByConsoleCalls) {
  auto result = Run({"createShape 1"});
  ASSERT_EQ(1, result.size());
  EXPECT_FALSE(result[0].starts_with("error:")) << result[0];
  ShapeHandle handle;
  ASSERT_TRUE(SReflect::FromJsonString(handle, result[0]));
  EXPECT_TRUE(handle.IsValid());
  EXPECT_EQ(1, _context->GetNumShapes());
  {
    auto native = _context->CreateSphereShape({}, 2_r, test::ExpectOK{});
    EXPECT_EQ(2, _context->GetNumShapes());
  }
  EXPECT_EQ(1, _context->GetNumShapes());
  auto failure = Run({"createShape -1"});
  ASSERT_EQ(1, failure.size());
  EXPECT_TRUE(failure[0].starts_with("error:")) << failure[0];
  EXPECT_EQ(1, _context->GetNumShapes());
  EXPECT_TRUE(Run({"releaseShape " + result[0]}).empty());
  EXPECT_EQ(0, _context->GetNumShapes());
}

TEST_F(ActorConsoleTest, ConstraintsListHandlesTypesAndAffectedActorNames) {
  RigidPivotPositionConstraintParams params;
  params.actor = _box;
  auto* first = _sceneA->CreateRigidPivotPositionConstraint(params, test::ExpectOK{});
  auto* second = _sceneA->CreateRigidPivotPositionConstraint(params, test::ExpectOK{});
  ConstraintHandle const firstHandle = first->GetHandle();
  ConstraintHandle const secondHandle = second->GetHandle();
  std::string const base = SReflect::EnumToString(first->GetType());
  RigidSphericalJointConstraintParams jointParams;
  jointParams.actorA = _box;
  jointParams.actorB = AddRigid(_sceneA, "other/actor");
  auto* joint = _sceneA->CreateRigidSphericalJointConstraint(jointParams, test::ExpectOK{});
  struct Row {
    std::string handle;
    std::string type;
    std::string actors;
  };
  std::array<Row, 3> rows{
      {{std::to_string(firstHandle.value), base, "box"},
       {std::to_string(secondHandle.value), base, "box"},
       {std::to_string(joint->GetHandle().value),
        SReflect::EnumToString(joint->GetType()),
        "box, other_actor"}}};
  std::ranges::sort(rows, {}, &Row::handle);
  std::string plain;
  std::string detailed;
  for (auto const& row : rows) {
    if (!plain.empty()) {
      plain += '\n';
      detailed += '\n';
    }
    std::string const hint = "(actors=[" + row.actors + "])";
    plain += row.handle + " " + hint;
    detailed += row.handle + " " + row.type + " " + hint;
  }
  ExpectOutput("ls /scenes/A/constraints", plain, Run({"ls /scenes/A/constraints"}));
  ExpectOutput("ls /scenes/A/constraints -l", detailed, Run({"ls /scenes/A/constraints -l"}));
  auto const& reg = assert_cast<SceneImpl*>(_sceneA)->GetRegistry();
  EXPECT_EQ(_sceneA->GetNumActors(), reg.view<CDebugName const>().size());
  std::string const namedPath = "ls constraints/" + base;
  CommandCase const missingName[]{{"/scenes/A", namedPath, "", "not found"}};
  CheckCases(missingName);
  SetDirectory("/scenes/A/constraints/" + std::to_string(firstHandle.value));
}

TEST_F(ActorConsoleTest, ActorsListEnumTypesAndLabeledHandles) {
  std::string const box = "(handle=" + std::to_string(_box.value) + ")";
  std::string const robot = "(handle=" + std::to_string(_robot->GetHandle().value) + ")";
  ExpectOutput(
      "ls /scenes/A/actors", "box " + box + "\nrobot " + robot, Run({"ls /scenes/A/actors"}));
  ExpectOutput(
      "ls /scenes/A/actors -l",
      "box Rigid " + box + "\nrobot Articulated " + robot,
      Run({"ls /scenes/A/actors -l"}));
}

TEST_F(ActorConsoleTest, DebugDrawAndConstraintDirectories) {
  RigidPivotPositionConstraintParams params;
  params.actor = _box;
  auto* constraint = _sceneA->CreateRigidPivotPositionConstraint(params, test::ExpectOK{});
  std::string const handle = std::to_string(constraint->GetHandle().value);
  CommandCase const cases[] = {
      {"/scenes/A", "ls constraints", handle + " (actors=[box])", ""},
      {"/scenes/A", "ls constraints/missing", "", "not found"},
      {"/scenes/A", "constraints", "{\n  \"" + handle + "\": {}\n}", ""},
      {"/scenes/A", "constraints 1", "", "is read-only"}};
  CheckCases(cases);
  SetDirectory("/scenes/A/constraints/" + handle);
  SetDirectory("/scenes/A/debugDraw");
}

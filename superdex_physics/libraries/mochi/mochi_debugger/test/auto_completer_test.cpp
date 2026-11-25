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

#include <mochi_debugger/lib/auto_completer.h>

#include <mochi_core/utils/dynamic_array.h>

#include <gtest/gtest.h>

#include <chrono>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>

using namespace mochi;
using namespace mochi::dbg;
using namespace std::chrono_literals;

namespace {

using Strings = DynamicArray<std::string>;

using Kind = SReflect::ConsoleCandidateKind;

struct ReplyOptions {
  size_t defaultIndex = 0;
  std::string note{};
  Strings laterNotes{};
  bool isTruncated = false;
  DynamicArray<Kind> kinds{}; // None, as from a server built before candidate kinds
  std::chrono::nanoseconds delay{}; // Time since the request was sent
  std::optional<uint64_t> timestamp{}; // Overrides the request's timestamp
};

class AutoCompleterTest : public ::testing::Test {
 protected:
  // preview, annotations joined by spaces, candidates, selected and isCycling
  using ViewTuple = std::tuple<std::string, std::string, Strings, size_t, bool>;

  // Reply to the last request
  bool Reply(Strings candidates, ReplyOptions options = {}) {
    EXPECT_FALSE(_sent.empty());
    protocol::AutoCompleteReply reply(_sent.back());
    reply.result._candidates.assign(candidates.begin(), candidates.end());
    reply.result._defaultIndex = options.defaultIndex;
    reply.result._note = std::move(options.note);
    reply.result._laterArgumentNotes.assign(options.laterNotes.begin(), options.laterNotes.end());
    reply.result._isTruncated = options.isTruncated;
    reply.result._candidateKinds.assign(options.kinds.begin(), options.kinds.end());
    reply.timestamp = options.timestamp.value_or(reply.timestamp);
    _now += options.delay;
    return _completer.OnReply(std::move(reply));
  }

  ViewTuple View() const {
    auto view = _completer.GetView();
    std::string annotations;
    for (auto const& annotation : view.annotations) {
      annotations += (annotations.empty() ? "" : " ") + annotation;
    }
    return {view.preview, annotations, view.candidates, view.selected, view.isCycling};
  }

  Strings SentCommands() const {
    Strings commands;
    for (auto const& request : _sent) {
      commands.emplace_back(request.command);
    }
    return commands;
  }

  DynamicArray<protocol::AutoCompleteRequest> _sent;
  std::string _workingDirectory = "/";
  uint64_t _connectionGeneration = 0;
  std::chrono::nanoseconds _now = 1s;
  AutoCompleter _completer{AutoCompleterHost{
      .send = [this](protocol::AutoCompleteRequest const& request) { _sent.push_back(request); },
      .workingDirectory = [this] { return _workingDirectory; },
      .commandKind =
          [](std::string_view word) {
            if (word == "cd" || word == "ls") {
              return ConsoleCommandKind::LocalWithPath;
            }
            return word == "echo" ? ConsoleCommandKind::Local : ConsoleCommandKind::Remote;
          },
      .connectionGeneration = [this] { return _connectionGeneration; },
      .now = [this] { return _now; }}};
};

TEST_F(AutoCompleterTest, RequestsOncePerStep) {
  for (std::string_view text : {"", " ", "/", "/s", "/sc", "/scenes/", "/scenes/A", "/scenes/A "}) {
    _completer.SetInput(text, true);
  }
  EXPECT_EQ(SentCommands(), (Strings{"/", "/scenes/", "/scenes/A "}));
  EXPECT_EQ(_sent.back().workingDirectory, "/");
  // Timestamps identify requests, so they increase even when the clock does not.
  EXPECT_GE(_sent[0].timestamp, std::chrono::nanoseconds{1s}.count());
  EXPECT_EQ(_sent[1].timestamp, _sent[0].timestamp + 1);

  // Backspacing into an earlier step requests it again.
  _completer.SetInput("/scenes/A", true);
  EXPECT_EQ(SentCommands().back(), "/scenes/");
}

TEST_F(AutoCompleterTest, ShowsTheDefaultForTypedText) {
  _completer.SetInput("/", true);
  EXPECT_TRUE(Reply({"scenes", "settings", "value"}, {.defaultIndex = 1}));

  // Nothing is previewed until something is typed in the step.
  EXPECT_EQ(View(), ViewTuple("", "", Strings{"scenes", "settings", "value"}, 1, false));

  // The preview shrinks as the text typed in the step matches more of it.
  _completer.SetInput("/s", true);
  EXPECT_EQ(View(), ViewTuple("ettings", "(+1)", Strings{"scenes", "settings"}, 1, false));
  _completer.SetInput("/sc", true);
  EXPECT_EQ(View(), ViewTuple("enes", "", Strings{"scenes"}, 0, false));
  _completer.SetInput("/x", true);
  EXPECT_EQ(View(), ViewTuple("", "", Strings{}, 0, false));

  // Only at the end of the line
  _completer.SetInput("/s", false);
  EXPECT_EQ(View(), ViewTuple("", "", Strings{}, 0, false));
  EXPECT_EQ(_sent.size(), 1);
}

TEST_F(AutoCompleterTest, CompletesTheFirstStepOfARelativePath) {
  _workingDirectory = "/scenes/main";
  _completer.SetInput("na", true);
  EXPECT_EQ(SentCommands(), Strings{""});
  EXPECT_EQ(_sent.back().workingDirectory, "/scenes/main");
  EXPECT_TRUE(Reply({"name", "numActors"}));
  EXPECT_EQ(View(), ViewTuple("me", "", Strings{"name"}, 0, false));
}

TEST_F(AutoCompleterTest, ShowsNotesAsPlaceholders) {
  _completer.SetInput("kind ", true);
  EXPECT_TRUE(Reply({"Rigid", "Soft"}, {.note = "Kind"}));
  EXPECT_EQ(View(), ViewTuple("", "<Kind>", Strings{"Rigid", "Soft"}, 0, false));
  _completer.SetInput("kind S", true);
  EXPECT_EQ(View(), ViewTuple("oft", "<Kind>", Strings{"Soft"}, 0, false));
  _completer.SetInput("kind X", true);
  EXPECT_EQ(View(), ViewTuple("", "<Kind>", Strings{}, 0, false));
}

TEST_F(AutoCompleterTest, ShowsTheArgumentsThatFollow) {
  _completer.SetInput("setKind ", true);
  EXPECT_TRUE(Reply(
      {"Rigid", "Rod", "Soft"}, {.note = "Kind kind", .laterNotes = {"float speed", "bool wake"}}));
  _completer.SetInput("setKind R", true);
  EXPECT_EQ(
      _completer.GetView().annotations,
      (Strings{"(+1)", "<Kind kind>", "<float speed>", "<bool wake>"}));

  // The count stays while cycling, which shows that Tab offers more.
  EXPECT_EQ(_completer.Cycle(false), "setKind Rigid");
  EXPECT_EQ(
      _completer.GetView().annotations,
      (Strings{"(+1)", "<Kind kind>", "<float speed>", "<bool wake>"}));
}

TEST_F(AutoCompleterTest, FormatsNoteTypesLikeListings) {
  _completer.SetInput("follow ", true);
  EXPECT_TRUE(Reply({}, {.note = "std::pair<ActorReflectionData,int> target"}));
  EXPECT_EQ(View(), ViewTuple("", "<std::pair<Actor, int> target>", Strings{}, 0, false));
}

TEST_F(AutoCompleterTest, IgnoresRepliesThatDoNotCoverTheInput) {
  _completer.SetInput("/", true);
  _completer.SetInput("/x/", true);
  std::swap(_sent[0], _sent[1]); // Reply to the first request
  EXPECT_FALSE(Reply({"x"}));
  std::swap(_sent[0], _sent[1]);
  EXPECT_TRUE(Reply({"a"}));

  // A reply counts only for its working directory.
  _completer.SetInput("/x/a", true);
  EXPECT_EQ(View(), ViewTuple("", "", Strings{"a"}, 0, false));
  _workingDirectory = "/other";
  _completer.SetInput("/x/a", true);
  EXPECT_EQ(View(), ViewTuple("", "", Strings{}, 0, false));
  EXPECT_EQ(_sent.back().workingDirectory, "/other");
}

TEST_F(AutoCompleterTest, KeepsTheNewestReply) {
  _completer.SetInput("/", true);
  uint64_t const first = _sent.back().timestamp;
  _now += 100ms;
  EXPECT_TRUE(Reply({"new"}, {.timestamp = first + 50}));
  EXPECT_FALSE(Reply({"old"}));
  EXPECT_EQ(View(), ViewTuple("", "", Strings{"new"}, 0, false));

  // An older reply that arrives late is not used either.
  EXPECT_FALSE(Reply({"old"}, {.delay = 600ms}));
  _completer.SetInput("/n", true);
  EXPECT_EQ(View(), ViewTuple("ew", "", Strings{"new"}, 0, false));
}

TEST_F(AutoCompleterTest, ForgetsRepliesAfterReconnecting) {
  _completer.SetInput("/s", true);
  EXPECT_TRUE(Reply({"scenes"}));
  ++_connectionGeneration;
  EXPECT_EQ(View(), ViewTuple("", "", Strings{}, 0, false));
  _completer.SetInput("/sc", true);
  EXPECT_EQ(SentCommands(), (Strings{"/", "/"}));

  // Only the reply to the request sent after reconnecting counts, even if the clock has not moved.
  auto const request = _sent.back();
  _sent.erase(_sent.end() - 1);
  EXPECT_FALSE(Reply({"scenes"}));
  _sent.push_back(request);
  EXPECT_TRUE(Reply({"scenes"}));
  EXPECT_EQ(View(), ViewTuple("enes", "", Strings{"scenes"}, 0, false));
}

TEST_F(AutoCompleterTest, ForgetsRepliesAtABlankLine) {
  _completer.SetInput("/s", true);
  EXPECT_TRUE(Reply({"scenes"}));
  _completer.SetInput("", true);
  EXPECT_EQ(View(), ViewTuple("", "", Strings{}, 0, false));

  // The next command requests its first step again, and ignores the previous command's replies.
  _completer.SetInput("/", true);
  _completer.SetInput("", true);
  _completer.SetInput("/s", true);
  EXPECT_EQ(SentCommands(), (Strings{"/", "/", "/"}));
  EXPECT_EQ(View(), ViewTuple("", "", Strings{}, 0, false));
  auto const request = _sent.back();
  _sent.erase(_sent.end() - 1);
  EXPECT_FALSE(Reply({"scenes"}));
  _sent.push_back(request);
  EXPECT_TRUE(Reply({"scenes"}));
}

TEST_F(AutoCompleterTest, ShowsLateReplies) {
  _completer.SetInput("/s", true);
  EXPECT_TRUE(Reply({"scenes"}, {.delay = 600ms}));
  EXPECT_EQ(View(), ViewTuple("cenes", "", Strings{"scenes"}, 0, false));
}

TEST_F(AutoCompleterTest, ResendsWhenUnansweredOrTruncated) {
  // At most one request per timeout while the server does not answer
  _completer.SetInput("items/", true);
  _now += 400ms;
  _completer.SetInput("items/1", true);
  _now += 200ms;
  _completer.SetInput("items/", true);
  EXPECT_EQ(SentCommands(), (Strings{"items/", "items/"}));

  // A truncated reply covers only the typed text it was requested with.
  EXPECT_TRUE(Reply({"0", "1"}, {.isTruncated = true}));
  _completer.SetInput("items/1", true);
  EXPECT_EQ(SentCommands(), (Strings{"items/", "items/", "items/1"}));
  EXPECT_EQ(View(), ViewTuple("", "", Strings{}, 0, false));
  EXPECT_TRUE(Reply({"1", "10"}));
  _completer.SetInput("items/10", true);
  EXPECT_EQ(View(), ViewTuple("", "", Strings{"10"}, 0, false));
}

TEST_F(AutoCompleterTest, ResendsTheTypedTextAfterATruncatedReplyToAnEarlierInput) {
  _completer.SetInput("items/", true);
  _completer.SetInput("items/1", true);
  EXPECT_FALSE(Reply({"0"}, {.isTruncated = true}));
  _completer.SetInput("items/12", true);
  EXPECT_EQ(SentCommands(), (Strings{"items/", "items/12"}));
  EXPECT_TRUE(Reply({"120", "121"}, {.isTruncated = true}));
  EXPECT_EQ(View(), ViewTuple("0", "(+1)", Strings{"120", "121"}, 0, false));
}

TEST_F(AutoCompleterTest, CompletesDirectoriesForPathCommands) {
  _completer.SetInput("cd ", true);
  _completer.SetInput("cd a ", true);
  _completer.SetInput("ls a -", true);
  _completer.SetInput("echo a", true);
  _completer.SetInput("cd", true);
  ASSERT_EQ(SentCommands(), (Strings{"cd ", ""}));
  EXPECT_EQ(_sent[0].pathTokenIndex, 1);
  EXPECT_TRUE(_sent[0].directoriesOnly);
  EXPECT_EQ(_sent[1].pathTokenIndex, 0); // The first token is a path while it is typed
}

TEST_F(AutoCompleterTest, CyclesThroughMatches) {
  _completer.SetInput("/s", true);
  EXPECT_EQ(_completer.Cycle(false), std::nullopt); // No reply yet
  Reply({"scenes", "settings", "value"}, {.defaultIndex = 1});
  EXPECT_EQ(_completer.Cycle(false), "/settings");
  EXPECT_EQ(View(), ViewTuple("", "(+1)", Strings{"scenes", "settings"}, 1, true));
  EXPECT_EQ(_completer.Cycle(false), "/scenes");
  EXPECT_EQ(_completer.Cycle(true), "/settings");

  // Reporting the text that cycling produced keeps the cycle going.
  _completer.SetInput("/settings", true);
  EXPECT_EQ(_completer.Cycle(false), "/scenes");

  // Any other input ends it.
  _completer.SetInput("/scenes", false);
  _completer.SetInput("/scenes", true);
  EXPECT_EQ(_completer.Cycle(false), "/scenes");
  _completer.SetInput("/scenes/", true);
  EXPECT_EQ(SentCommands(), (Strings{"/", "/scenes/"}));
}

TEST_F(AutoCompleterTest, TabFinishesTheOnlyMatch) {
  _completer.SetInput("/", true);
  Reply(
      {"gravity", "scenes", "settings", "speed"},
      {.kinds = {Kind::DirectoryOrLeaf, Kind::Directory, Kind::Directory, Kind::Leaf}});

  // A directory gets '/' and anything else a space, so the next Tab completes what follows.
  _completer.SetInput("/sc", true);
  EXPECT_EQ(_completer.Cycle(false), "/scenes/");
  EXPECT_FALSE(_completer.GetView().isCycling);
  _completer.SetInput("/sp", true);
  EXPECT_EQ(_completer.Cycle(false), "/speed ");
  _completer.SetInput("/speed ", true);
  EXPECT_EQ(SentCommands().back(), "/speed ");

  // A step that is a directory with a value can be followed by either, so the user decides.
  _completer.SetInput("/g", true);
  EXPECT_EQ(_completer.Cycle(false), "/gravity");
  EXPECT_TRUE(_completer.GetView().isCycling);

  // Several matches cycle, and typing '/' or a space accepts one.
  _completer.SetInput("/s", true);
  EXPECT_EQ(_completer.Cycle(false), "/scenes");
  EXPECT_EQ(View(), ViewTuple("", "(+2)", Strings{"scenes", "settings", "speed"}, 0, true));

  // Selecting in a menu never adds anything.
  _completer.SetInput("/sc", true);
  EXPECT_EQ(_completer.Select(0), "/scenes");

  // After a method argument, the next Tab completes the next one.
  _completer.SetInput("add ", true);
  Reply({"0"}, {.note = "float amount", .laterNotes = {"string note"}, .kinds = {Kind::Leaf}});
  EXPECT_EQ(_completer.Cycle(false), "add 0 ");
}

TEST_F(AutoCompleterTest, CountsOtherMatchesWhenTheTypedTextIsADefault) {
  _completer.SetInput("/scenes", true);
  EXPECT_TRUE(Reply({"scenes", "scenesB"}));
  EXPECT_EQ(View(), ViewTuple("", "(+1)", Strings{"scenes", "scenesB"}, 0, false));
}

TEST_F(AutoCompleterTest, TabFinishesOnlyACertainMatch) {
  // A truncated reply may leave out other matches.
  _completer.SetInput("items/", true);
  Reply({"10"}, {.isTruncated = true, .kinds = {Kind::Leaf}});
  EXPECT_EQ(_completer.Cycle(false), "items/10");

  // A server built before candidate kinds does not say what follows.
  _completer.SetInput("na", true);
  Reply({"name"});
  EXPECT_EQ(_completer.Cycle(false), "name");
}

TEST_F(AutoCompleterTest, TabNeedsTheCursorAtTheEnd) {
  _completer.SetInput("/s", true);
  Reply({"scenes"});
  EXPECT_EQ(_completer.Cycle(false), "/scenes");
  _completer.SetInput("/s", false);
  EXPECT_EQ(_completer.Cycle(false), std::nullopt);
}

TEST_F(AutoCompleterTest, ShiftTabStartsFromTheLastMatch) {
  _completer.SetInput("/", true);
  Reply({"a", "b", "c"});
  EXPECT_EQ(_completer.Cycle(true), "/c");
  EXPECT_EQ(_completer.Cycle(true), "/b");
}

TEST_F(AutoCompleterTest, SelectsMatches) {
  _completer.SetInput("/s", true);
  Reply({"scenes", "settings"});
  EXPECT_EQ(_completer.Select(2), std::nullopt);
  EXPECT_EQ(_completer.Select(1), "/settings");
  EXPECT_EQ(_completer.Cycle(false), "/scenes");
}

} // namespace

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

#include "mochi_physics_test_fixture.h"

#include <mochi_core/geometry/geometry_utils.h>

namespace mochi::test {

MochiContextTestBase::~MochiContextTestBase() {
  MOCHI_ASSERT(
      _mochiContext == nullptr,
      "Class inheriting from MochiContextTestBase has not correctly called TearDown()!"
      " This can result in memory leaks/corruption! Please clean up after yourself!");
}

/*******************************************************************************************
  MochiPhysicsTest (test fixture)
*/

void MochiContextTestBase::SetUp() {
  // Create mochi::Context
  _mochiContext = mochi::CreateContext(_numWorkerThreads);
  EXPECT_NE((Context*)nullptr, _mochiContext);

  // Take the logging callback that has been registered for this test project
  // and make sure it also applies with the mochi_physics DLL (does nothing
  // if mochi_physics is statically linked with this test).
  Context::SetLogCallback(mochi::GetLogCallback());
}

void mochi::test::MochiContextTestBase::TearDown() {
  mochi::DestroyContext(_mochiContext);
  _mochiContext = nullptr;
}

/*******************************************************************************************
  MochiContextTestWithParam (parameterized test fixture)
*/

void MochiContextTestWithParam::SetUp() {
  // Create mochi::Context with some numer of worker threads
  _numWorkerThreads = GetParam().numWorkerThreads;
  Class::SetUp();

  // Apply other parameters
  _mochiContext->SetIsSingleThreaded(GetParam().isSingleThreadedMode);
}

/*******************************************************************************************
  MochiSceneTest
*/

void MochiSceneTestBase::SetUp() {
  MochiContextTestBase::SetUp();

  _scene = _mochiContext->CreateScene("MochiPhysicsTest");
  EXPECT_NE((mochi::Scene*)nullptr, _scene);
}

void MochiSceneTestBase::TearDown() {
  EXPECT_NE((mochi::Scene*)nullptr, _scene);
  _mochiContext->DestroyScene(_scene);
  _scene = nullptr;

  MochiContextTestBase::TearDown();
}

/*******************************************************************************************
  MochiAsyncSceneTest
*/

MochiAsyncSceneTestBase::MochiAsyncSceneTestBase() {
  // Must have at least one worker thread to create an AsyncScene
  _numWorkerThreads = 1;
}

void MochiAsyncSceneTestBase::SetUp() {
  MochiContextTestBase::SetUp();

  _asyncScene = _mochiContext->CreateAsyncScene("MochiPhysicsTest", ExpectOK{});
  EXPECT_NE((mochi::AsyncScene*)nullptr, _asyncScene);
}

void MochiAsyncSceneTestBase::TearDown() {
  EXPECT_NE((mochi::AsyncScene*)nullptr, _asyncScene);
  _mochiContext->DestroyAsyncScene(_asyncScene);
  _asyncScene = nullptr;

  MochiContextTestBase::TearDown();
}

/*******************************************************************************************
  ExpectLoggingInScope
*/

ExpectLoggingInScope::ExpectLoggingInScope(Context* mochiContext, LogChannel channel)
    : _mochiContext(mochiContext),
      _channel(channel),
      _prevCallbackForThisModule(mochi::GetLogCallback()),
      _prevCallbackForMochiPhysics(mochiContext ? mochiContext->GetLogCallback() : LogFn{}) {
  LogFn ourCallback =
      [this](LogChannel channel, char const* /*message*/, char const* /*file*/, int /*line*/) {
        if (channel == _channel) {
          ++_messagesReceived;
        }
      };
  mochi::Context::SetLogCallback(ourCallback);
}

ExpectLoggingInScope::~ExpectLoggingInScope() {
  mochi::Context::SetLogCallback(_prevCallbackForMochiPhysics);
  if (_messagesReceived == 0) {
    ADD_FAILURE() << "Expected a log message, but it never happened.";
  }
}

} // namespace mochi::test
